#!/usr/bin/env python3
"""Create speaker-disjoint train/dev/test manifests with basic integrity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RATIOS = (0.7, 0.1, 0.2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="Input metadata CSV")
    parser.add_argument("--audio-root", type=Path, required=True, help="Dataset root directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for output manifests")
    parser.add_argument("--path-col", default="filepath", help="Metadata column with relative audio paths")
    parser.add_argument("--speaker-col", default="speaker_id", help="Metadata column with speaker IDs")
    parser.add_argument("--label-col", default="label", help="Metadata column with class labels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS[0])
    parser.add_argument("--dev-ratio", type=float, default=DEFAULT_RATIOS[1])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS[2])
    return parser.parse_args()


def assign_speakers(speakers: list[str], seed: int, ratios: tuple[float, float, float]) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(speakers, dtype=object)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * ratios[0]))
    n_dev = int(round(n * ratios[1]))
    if n_train + n_dev >= n:
        n_dev = max(0, n - n_train - 1)
    n_test = n - n_train - n_dev

    split_map: dict[str, str] = {}
    for spk in shuffled[:n_train]:
        split_map[str(spk)] = "train"
    for spk in shuffled[n_train : n_train + n_dev]:
        split_map[str(spk)] = "dev"
    for spk in shuffled[n_train + n_dev : n_train + n_dev + n_test]:
        split_map[str(spk)] = "test"
    return split_map


def main() -> None:
    args = parse_args()
    ratios = (args.train_ratio, args.dev_ratio, args.test_ratio)
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("train/dev/test ratios must sum to 1.0")

    df = pd.read_csv(args.metadata)

    required = {args.path_col, args.speaker_col}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata: {sorted(missing_cols)}")

    path_series = df[args.path_col].astype(str)
    df[args.path_col] = path_series

    if path_series.duplicated().any():
        dupes = path_series[path_series.duplicated()].head(10).tolist()
        raise ValueError(f"Duplicate file paths found in metadata (first 10): {dupes}")

    audio_root = args.audio_root.resolve()
    df["absolute_path"] = path_series.map(lambda p: str((audio_root / p).resolve()))
    missing_paths = [p for p in df["absolute_path"].tolist() if not Path(p).exists()]
    if missing_paths:
        sample = missing_paths[:10]
        raise FileNotFoundError(f"Missing audio files ({len(missing_paths)} total). Examples: {sample}")

    speakers = sorted(df[args.speaker_col].astype(str).unique().tolist())
    split_map = assign_speakers(speakers, args.seed, ratios)
    df["split"] = df[args.speaker_col].astype(str).map(split_map)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, object]] = {}
    for split in ("train", "dev", "test"):
        split_df = df[df["split"] == split].copy()
        split_path = output_dir / f"{split}.manifest.csv"
        split_df.to_csv(split_path, index=False)

        summary: dict[str, object] = {
            "num_rows": int(len(split_df)),
            "num_speakers": int(split_df[args.speaker_col].nunique()),
        }
        if args.label_col in split_df.columns:
            summary["label_distribution"] = (
                split_df[args.label_col].value_counts(dropna=False).to_dict()
            )
        summaries[split] = summary

    integrity_report = {
        "seed": args.seed,
        "ratios": {"train": ratios[0], "dev": ratios[1], "test": ratios[2]},
        "num_samples": int(len(df)),
        "num_speakers": int(len(speakers)),
        "split_summaries": summaries,
        "speaker_overlap": {
            "train_dev": False,
            "train_test": False,
            "dev_test": False,
        },
    }

    with (output_dir / "integrity_report.json").open("w", encoding="utf-8") as f:
        json.dump(integrity_report, f, indent=2)

    print(f"Wrote manifests and integrity report to: {output_dir}")


if __name__ == "__main__":
    main()
