from dataclasses import dataclass
from pathlib import Path
import math


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    duration_sec: float = 2.0

    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    fmin: int = 20
    fmax: int = 8000

    @property
    def chunk_frames(self) -> int:
        return math.ceil((self.sample_rate * self.duration_sec) / self.hop_length)

    @property
    def max_samples(self) -> int:
        return int(self.sample_rate * self.duration_sec)


@dataclass
class TrainConfig:
    batch_size: int = 25
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-3
    hidden_size: int = 64
    dropout: float = 0.35
    num_workers: int = 6
    seed: int = 42


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SPLITS_DIR = ARTIFACTS_DIR / "splits"
EVAL_DIR = ARTIFACTS_DIR / "evaluation"
XAI_DIR = ARTIFACTS_DIR / "xai"

for folder in [ARTIFACTS_DIR, MODELS_DIR, SPLITS_DIR, EVAL_DIR, XAI_DIR]:
    folder.mkdir(parents=True, exist_ok=True)