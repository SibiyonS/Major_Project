import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        pooled = torch.sum(weights * x, dim=1)
        return pooled, weights.squeeze(-1)


class CnnBiLstmDetector(nn.Module):
    def __init__(self, hidden_size: int = 64, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )

        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        self.attention = AttentionPool(hidden_size * 2)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x, return_attention=False):
        feat = self.features(x)
        feat = torch.mean(feat, dim=2)
        feat = feat.permute(0, 2, 1)
        feat = torch.nn.functional.adaptive_avg_pool1d(feat.permute(0, 2, 1), 40).permute(0, 2, 1)

        seq_out, _ = self.bilstm(feat)
        pooled, attn = self.attention(seq_out)
        logits = self.classifier(pooled).squeeze(-1)

        if return_attention:
            return logits, attn
        return logits