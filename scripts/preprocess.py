import argparse
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.config import AudioConfig, SPLITS_DIR
from src.data import build_feature_cache, collect_audio_files, make_splits

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def convert_dataset(input_root: Path, output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    files = list(input_root.rglob("*"))

    for path in tqdm(files, desc="convert"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        relative_path = path.relative_to(input_root)
        out_path = output_root / relative_path.with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            y, _ = librosa.load(path, sr=16000, mono=True)
            sf.write(out_path, y, 16000)
        except Exception:
            print(f"Skipped: {path}")


def main(args):
    original_root = Path(args.dataset_root)
    if not original_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {original_root}")

    if args.convert_to_wav:
        converted_root = Path(args.converted_root)
        if not converted_root.exists() or args.force_reconvert:
            print("Converting dataset to WAV format...")
            convert_dataset(original_root, converted_root)

        print("Using Already converted dataset ...")
        dataset_root = converted_root
    else:
        print("Using original dataset without conversion")
        dataset_root = original_root

    print(f"Using dataset: {dataset_root}")

    audio_cfg = AudioConfig()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    folder_label_map = {
        "real": 0,
        "fake": 1,
        "FakeAudios": 1,
        "RealAudios": 0,
        "REAL": 0,
        "FAKE": 1
    }

    print("Collecting audio files...")
    index_df = collect_audio_files(dataset_root, folder_label_map)
    print(index_df["label"].value_counts())
    print("Total files:", len(index_df))

    print(index_df.sample(15))
    print("Unique files:", index_df["audio_path"].nunique())

    train_df, val_df, test_df = make_splits(index_df, val_size=args.val_size, test_size=args.test_size)
    print(f"Split sizes => Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_df = build_feature_cache(train_df, cache_dir, audio_cfg, n_jobs=args.n_jobs)
    val_df = build_feature_cache(val_df, cache_dir, audio_cfg, n_jobs=args.n_jobs)
    test_df = build_feature_cache(test_df, cache_dir, audio_cfg, n_jobs=args.n_jobs)

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    split_stats = pd.DataFrame(
        {
            "split": ["train", "val", "test"],
            "count": [len(train_df), len(val_df), len(test_df)],
            "fake_ratio": [
                train_df["label"].mean(),
                val_df["label"].mean(),
                test_df["label"].mean(),
            ],
        }
    )
    split_stats.to_csv(SPLITS_DIR / "split_stats.csv", index=False)

    print("Preprocessing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio dataset")
    parser.add_argument("--dataset_root", type=str, default=r"S:\Major_Project_New\dataset_root")
    parser.add_argument("--cache_dir", type=str, default="artifacts/features")
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--convert_to_wav", action="store_true")
    parser.add_argument("--converted_root", type=str, default="artifacts/dataset_wav")
    parser.add_argument("--force_reconvert", action="store_true")
    main(parser.parse_args())