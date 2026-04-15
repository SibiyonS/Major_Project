import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import amp, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.config import MODELS_DIR, SPLITS_DIR, TrainConfig
from src.data import MelDataset
from src.model import CnnBiLstmDetector
from src.utils import save_json, seed_everything
import gc

def clean_memory():
    gc.collect()
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        return (self.alpha * (1 - pt).pow(self.gamma) * bce_loss).mean()


class HybridLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, focal_alpha: float = 0.5, gamma: float = 2.0, focal_ratio: float = 0.35):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=gamma)
        self.focal_ratio = focal_ratio

    def forward(self, logits, targets):
        return (1.0 - self.focal_ratio) * self.bce(logits, targets) + self.focal_ratio * self.focal(logits, targets)


def run_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(device_type="cuda", enabled=(scaler is not None)):
            logits = model(x)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            loss = criterion(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    losses, probs, labels = [], [], []

    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()

        with amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            loss = criterion(logits, y)

        p = torch.sigmoid(logits).view(-1)
        y = y.view(-1)

        probs.extend(p.detach().cpu().numpy())
        labels.extend(y.detach().cpu().numpy())
        losses.append(loss.item())

    probs = np.array(probs, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)
    if len(probs) == 0:
        return {}

    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    probs = np.clip(probs, 0.0, 1.0)
    labels = np.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0)
    labels = (labels >= 0.5).astype(int)

    preds = (probs >= threshold).astype(int)

    try:
        roc_auc = roc_auc_score(labels, probs)
        if not np.isfinite(roc_auc):
            roc_auc = 0.5
    except ValueError:
        roc_auc = 0.5

    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc,
        "threshold": threshold,
    }


'''def _make_balanced_sampler(df: pd.DataFrame):
    class_counts = df["label"].value_counts().to_dict()
    sample_weights = df["label"].map(lambda v: 1.0 / class_counts.get(v, 1)).values
    return WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)'''

# ---------- BALANCED SAMPLING FUNCTION ----------
def stratified_sample(df, total_samples):
    per_class = total_samples // 2

    real_df = df[df["label"] == 0]
    fake_df = df[df["label"] == 1]

    real_sample = real_df.sample(min(len(real_df), per_class), random_state=42)
    fake_sample = fake_df.sample(min(len(fake_df), per_class), random_state=42)

    balanced_df = pd.concat([real_sample, fake_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    # ---------- APPLY SAMPLING ----------
    '''print("sampling to balance classes and reduce size for faster training...")
    train_df = stratified_sample(train_df, 200000)   # 100k real + 100k fake
    val_df = stratified_sample(val_df, 50000)        # 25k + 25k
    test_df = stratified_sample(test_df, 50000)      # 25k + 25k'''

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    workers = train_cfg.num_workers
    pin_mem = device.type == "cuda"
    #persistent = workers > 0 and device.type == "cuda"

    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": pin_mem,
        #"persistent_workers": True,
        #"prefetch_factor": 4,   # 🔥 increase prefetch
    }

    train_loader = DataLoader(
        MelDataset(train_df, augment=True),
        batch_size=train_cfg.batch_size,
        #sampler=_make_balanced_sampler(train_df),
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        MelDataset(val_df),
        batch_size=train_cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        MelDataset(test_df),
        batch_size=train_cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = CnnBiLstmDetector(hidden_size=train_cfg.hidden_size, dropout=train_cfg.dropout).to(device)

    fake_count = int((train_df["label"] == 1).sum())
    real_count = int((train_df["label"] == 0).sum())
    # alpha controls weight for positive class in focal loss.
    # If fake is rare, alpha should increase.
    alpha = min(0.95, max(0.25, real_count / max(1, (fake_count + real_count))))
    pos_weight = torch.tensor([real_count / max(1, fake_count)], device=device, dtype=torch.float32)
    criterion = HybridLoss(pos_weight=pos_weight, focal_alpha=alpha, gamma=2.0, focal_ratio=0.35)

    optimizer = AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.3)
    scaler = amp.GradScaler("cuda") if device.type == "cuda" else None

    best_f1 = -1.0
    patience, patience_counter = 7, 0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODELS_DIR / "best_cnn_bilstm.pt"
    history = []

    for epoch in range(1, train_cfg.epochs + 1):
        clean_memory()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        clean_memory()
        val_metrics = evaluate(model, val_loader, criterion, device)
        clean_memory()
        if not val_metrics:
            raise RuntimeError("Validation returned empty metrics.")

        scheduler.step(val_metrics["loss"])
        val_metrics["train_loss"] = train_loss
        val_metrics["epoch"] = epoch
        history.append(val_metrics)

        print(
            f"Epoch {epoch} | train_loss={train_loss:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | val_auc={val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            torch.save({"model": model.state_dict(), "config": train_cfg.__dict__}, best_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    
    clean_memory() 
    print("Testing is in process.Data fetch from test.csv...")
    test_metrics = evaluate(model, test_loader, criterion, device, threshold=0.6)
    clean_memory() 

    print("\n===== TEST RESULTS =====")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    pd.DataFrame(history).to_csv(MODELS_DIR / "history.csv", index=False)
    save_json(
        {
            "best_f1": best_f1,
            "best_threshold": 0.5,
            "test_metrics": test_metrics,
            "device": str(device),
            "class_balance": {"real": real_count, "fake": fake_count},
        },
        MODELS_DIR / "summary.json",
    )
    print("Training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=6)
    main(parser.parse_args())