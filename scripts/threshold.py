import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model import CnnBiLstmDetector
from src.data import MelDataset
import pandas as pd

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.model import CnnBiLstmDetector
from src.data import MelDataset
import pandas as pd


@torch.no_grad()
def get_probs_labels(model, loader, device):
    model.eval()
    probs, labels = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)

        logits = model(x)
        p = torch.sigmoid(logits).cpu().numpy()

        probs.extend(p)
        labels.extend(y.numpy())

    return np.array(labels), np.array(probs)


def find_best_threshold_fast(labels, probs):
    thresholds = np.linspace(0.3, 0.8, 100)

    best_f1 = -1.0
    best_t = 0.5

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print(f"\n✅ Best threshold: {best_t:.3f} | F1: {best_f1:.4f}")
    return best_t


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load validation data
    val_df = pd.read_csv("artifacts/splits/val.csv")
    val_loader = DataLoader(
        MelDataset(val_df, augment=False),
        batch_size=24,
        shuffle=False
    )

    # Load model
    checkpoint = torch.load("artifacts/models/best_cnn_bilstm.pt", map_location=device)
    model = CnnBiLstmDetector().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print("🔍 Running model once to collect predictions...")

    # 🔥 KEY CHANGE: run model only once
    labels, probs = get_probs_labels(model, val_loader, device)

    print("⚡ Finding best threshold (fast)...")

    best_threshold = find_best_threshold_fast(labels, probs)

    print(f"\n🔥 Final Best Threshold: {best_threshold:.3f}")

if __name__ == "__main__":
    main()