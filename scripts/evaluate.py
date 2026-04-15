import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.config import EVAL_DIR
from src.data import MelDataset
from src.model import CnnBiLstmDetector
from src.utils import save_json


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())

    return labels, probs


def _compute_binary_metrics(labels, probs, threshold: float):
    preds = [1 if p >= threshold else 0 for p in probs]
    roc_auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0
    return {
        "threshold": threshold,
        "count": len(labels),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc,
    }, preds


def _build_per_source_report(df: pd.DataFrame, probs: list[float], threshold: float, min_count: int = 200) -> pd.DataFrame:
    eval_df = df.copy()
    eval_df["prob"] = np.asarray(probs, dtype=np.float64)
    eval_df["prob"] = np.clip(np.nan_to_num(eval_df["prob"], nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)

    source_col = "group_source" if "group_source" in eval_df.columns else ("source" if "source" in eval_df.columns else None)
    if source_col is None:
        return pd.DataFrame()

    rows = []
    for source_name, g in eval_df.groupby(source_col):
        if len(g) < min_count:
            continue
        labels = g["label"].astype(int).tolist()
        source_probs = g["prob"].tolist()
        m, _ = _compute_binary_metrics(labels, source_probs, threshold)
        m["source"] = str(source_name)
        rows.append(m)

    out = pd.DataFrame(rows).sort_values(["f1", "roc_auc"], ascending=[False, False])
    if not out.empty:
        out = out[
            ["source", "count", "accuracy", "precision", "recall", "f1", "roc_auc", "threshold"]
        ]
    return out


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    test_df = pd.read_csv(args.test_csv)

    test_loader = DataLoader(
        MelDataset(test_df, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        cfg = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        cfg = {}

    model = CnnBiLstmDetector(
        hidden_size=cfg.get("hidden_size", 64),
        dropout=cfg.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(state_dict)

    labels, probs = predict(model, test_loader, device)
    probs = [float(p) for p in probs]
    labels = [int(y) for y in labels]
    probs = [0.5 if not math.isfinite(p) else p for p in probs]
    probs = [min(1.0, max(0.0, p)) for p in probs]

    threshold = args.threshold
    metrics, preds = _compute_binary_metrics(labels, probs, threshold)
    metrics["report"] = classification_report(labels, preds, output_dict=True, zero_division=0)
    save_json(metrics, EVAL_DIR / "metrics.json")

    per_source = _build_per_source_report(test_df, probs, threshold, min_count=args.per_source_min_count)
    if not per_source.empty:
        per_source.to_csv(EVAL_DIR / "per_source_metrics.csv", index=False)
        save_json(
            {
                "worst_10_by_f1": per_source.sort_values("f1", ascending=True).head(10).to_dict(orient="records"),
                "best_10_by_f1": per_source.head(10).to_dict(orient="records"),
            },
            EVAL_DIR / "per_source_summary.json",
        )

    cm = confusion_matrix(labels, preds)
    cm = cm.astype(int)

    plt.figure(figsize=(6, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"])

    print("CM:\n", cm)
    print("Shape:", cm.shape)
    print("Labels:", set(labels))
    print("Preds:", set(preds))

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.savefig(EVAL_DIR / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "roc_curve.png", dpi=180)
    plt.close()

    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "pr_curve.png", dpi=180)
    plt.close()

    history_csv = Path(args.history_csv)
    if history_csv.exists():
        history = pd.read_csv(history_csv)
        plt.figure(figsize=(8, 4))
        sns.lineplot(data=history, x="epoch", y="train_loss", label="train_loss")
        sns.lineplot(data=history, x="epoch", y="f1", label="val_f1")
        plt.title("Training History")
        plt.tight_layout()
        plt.savefig(EVAL_DIR / "training_history.png", dpi=180)
        plt.close()

    print("Evaluation complete. Results saved in artifacts/evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="artifacts/splits/test.csv")
    parser.add_argument("--model_path", type=str, default="artifacts/models/best_cnn_bilstm.pt")
    parser.add_argument("--history_csv", type=str, default="artifacts/models/history.csv")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--per_source_min_count", type=int, default=200)
    main(parser.parse_args())