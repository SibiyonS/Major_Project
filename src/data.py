from pathlib import Path
from typing import List
import hashlib
import warnings

from joblib import Parallel, delayed
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from tqdm import tqdm

from src.config import AudioConfig

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def _infer_label_and_source(path: Path, folder_label_map: dict[str, int] | None = None):
    folder_label_map = folder_label_map or {}
    map_lower = {k.lower(): v for k, v in folder_label_map.items()}

    parts_lower = [p.lower() for p in path.parts]
    source = path.parent.name

    for part in reversed(parts_lower):
        if part in map_lower:
            return map_lower[part], part

    for part in reversed(parts_lower):
        if "real" in part:
            return 0, part
        if "fake" in part:
            return 1, part

    return None, source


def _infer_group_source(path: Path, dataset_root: Path, folder_label_map: dict[str, int] | None = None) -> str:
    folder_label_map = folder_label_map or {}
    map_lower = {k.lower() for k in folder_label_map}
    rel_parts = [p.lower() for p in path.relative_to(dataset_root).parts[:-1]]

    # pick first non-label token so groups are more meaningful than just "real/fake"
    for token in rel_parts:
        if token in map_lower:
            continue
        if "real" in token or "fake" in token:
            continue
        return token

    if rel_parts:
        return rel_parts[0]
    return path.parent.name.lower()


def collect_audio_files(dataset_root: Path, folder_label_map=None):
    rows = []
    for path in dataset_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        label, source = _infer_label_and_source(path, folder_label_map)
        if label is None:
            continue
        group_source = _infer_group_source(path, dataset_root, folder_label_map)

        rows.append(
            {
                "audio_path": str(path),
                "label": int(label),
                "source": source,
                "group_source": group_source,
            }
        )

    if not rows:
        raise ValueError(f"No audio files found in {dataset_root}")

    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


def preprocess_audio(y, sr):
    y, _ = librosa.effects.trim(y, top_db=30)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if len(y) > 1:
        y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    y = y / (np.max(np.abs(y)) + 1e-8)
    return y


def load_and_preprocess_audio(audio_path: str, cfg: AudioConfig) -> List[np.ndarray]:
    waveform, sr = sf.read(audio_path)

    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)

    if sr != cfg.sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=cfg.sample_rate)
        sr = cfg.sample_rate

    waveform = preprocess_audio(waveform, sr)

    window_size = cfg.max_samples
    hop_size = window_size // 2

    if len(waveform) < window_size:
        waveform = np.pad(waveform, (0, window_size - len(waveform)))

    features = []
    for start in range(0, len(waveform), hop_size):
        chunk = waveform[start:start + window_size]
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)))

        mel = librosa.feature.melspectrogram(
            y=chunk,
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            power=2.0,
        )

        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)
        features.append(log_mel.astype(np.float32))

    if not features:
        raise ValueError(f"No features extracted for {audio_path}")

    return features


def process_audio_row(row, cache_dir: Path, audio_cfg: AudioConfig):
    audio_path = row["audio_path"]
    label = int(row["label"])

    try:
        y, sr = sf.read(audio_path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        if sr != audio_cfg.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=audio_cfg.sample_rate)
            sr = audio_cfg.sample_rate

        if y is None or len(y) == 0:
            return None

        y = preprocess_audio(y, sr)
    except Exception:
        return None

    window = audio_cfg.max_samples
    hop = window // 2

    if len(y) < window:
        y = np.pad(y, (0, window - len(y)))

    rows = []
    file_id = hashlib.md5(audio_path.encode()).hexdigest()[:12]
    chunk_id = 0

    for start in range(0, len(y), hop):
        chunk = y[start:start + window]
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk)))

        mel = librosa.feature.melspectrogram(
            y=chunk,
            sr=sr,
            n_fft=audio_cfg.n_fft,
            hop_length=audio_cfg.hop_length,
            n_mels=audio_cfg.n_mels,
            fmin=audio_cfg.fmin,
            fmax=audio_cfg.fmax,
            power=2.0,
        )

        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

        out_path = cache_dir / f"{file_id}_chunk{chunk_id}.npy"
        np.save(out_path, log_mel.astype(np.float32))

        rows.append(
            {
                "feature_path": str(out_path),
                "label": label,
                "audio_path": audio_path,
                "source": row.get("source", "unknown"),
            }
        )
        chunk_id += 1

    return rows


def build_feature_cache(df: pd.DataFrame, cache_dir: Path, audio_cfg: AudioConfig, n_jobs: int = 4):
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_audio_row)(row, cache_dir, audio_cfg)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="feature-cache")
    )

    rows_out = []
    skipped = 0
    for res in results:
        if res is None:
            skipped += 1
        else:
            rows_out.extend(res)

    out = pd.DataFrame(rows_out)
    if out.empty:
        raise ValueError("Feature cache is empty after processing.")

    print(f"Skipped files: {skipped}")
    return out


def _score_split_balance(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, full_ratio: float) -> float:
    split_ratios = [train["label"].mean(), val["label"].mean(), test["label"].mean()]
    score = sum(abs(r - full_ratio) for r in split_ratios)

    # hard-penalize splits that collapse to a single class
    penalty = 0.0
    for split in (train, val, test):
        if split["label"].nunique() < 2:
            penalty += 10.0

    return score + penalty


from sklearn.model_selection import train_test_split
import pandas as pd


def make_splits(df: pd.DataFrame, val_size=0.15, test_size=0.15, random_state=42):
    """
    ✔ Uses stratified split (IMPORTANT)
    ✔ Keeps real/fake ratio same in all splits
    ✔ Prevents val/test having single class
    ✔ Simple and robust
    """

    # -------- Train + Temp --------
    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        stratify=df["label"],   # 🔥 THIS IS THE KEY
        random_state=random_state
    )

    # -------- Val + Test --------
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),
        stratify=temp_df["label"],   # 🔥 AGAIN IMPORTANT
        random_state=random_state
    )

    # -------- DEBUG CHECK (DON'T REMOVE) --------
    print("\n📊 SPLIT CHECK:")
    print("Train:\n", train_df["label"].value_counts(normalize=True))
    print("Val:\n", val_df["label"].value_counts(normalize=True))
    print("Test:\n", test_df["label"].value_counts(normalize=True))

    # Optional: strict safety check
    assert train_df["label"].nunique() == 2, "Train missing a class!"
    assert val_df["label"].nunique() == 2, "Val missing a class!"
    assert test_df["label"].nunique() == 2, "Test missing a class!"

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


import numpy as np
import torch
from torch.utils.data import Dataset

class MelDataset(Dataset):
    def __init__(self, df, augment=False, cache_size=10000):
        self.df = df.reset_index(drop=True)
        self.augment = augment

        # 🔥 SMART CACHE (limited size to avoid RAM crash)
        self.cache = {}
        self.cache_order = []
        self.cache_size = cache_size

    def __len__(self):
        return len(self.df)

    def _load_feature(self, path):
        # 🔥 If already cached → FAST
        if path in self.cache:
            return self.cache[path]

        # 🔥 Load from disk
        x = np.load(path).astype(np.float32)

        # 🔥 Add to cache
        self.cache[path] = x
        self.cache_order.append(path)

        # 🔥 Remove old cache if too big (FIFO)
        if len(self.cache_order) > self.cache_size:
            old = self.cache_order.pop(0)
            del self.cache[old]

        return x

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["feature_path"]

        # 🔥 LOAD (optimized)
        x = self._load_feature(path)

        # -------------------------
        # 🔥 LIGHT AUGMENTATION
        # -------------------------
        if self.augment:
            if np.random.rand() < 0.3:   # reduced from 0.5
                t = np.random.randint(0, max(1, x.shape[1] // 12))
                x[:, :t] = 0

            if np.random.rand() < 0.3:
                f = np.random.randint(0, max(1, x.shape[0] // 12))
                x[:f, :] = 0

        # -------------------------
        # 🔥 TORCH CONVERSION (FAST)
        # -------------------------
        x = torch.from_numpy(x).unsqueeze(0)  # faster than torch.tensor
        y = torch.tensor(row["label"], dtype=torch.float32)

        return x, y