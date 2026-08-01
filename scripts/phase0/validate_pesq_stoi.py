#!/usr/bin/env python3
"""Validate PESQ/STOI computation over paired clean/noisy audio samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from pesq import pesq
from pystoi import stoi


TARGET_SR = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", type=Path, required=True, help="CSV with clean and degraded file paths")
    parser.add_argument("--clean-col", default="clean_path", help="Column name for clean audio path")
    parser.add_argument("--degraded-col", default="degraded_path", help="Column name for degraded audio path")
    parser.add_argument("--fail-on-invalid", action="store_true", help="Exit non-zero if any value is out of range")
    return parser.parse_args()


def load_mono_16k(path: Path) -> np.ndarray:
    waveform, sr = torchaudio.load(str(path))
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    return waveform.squeeze(0).to(torch.float32).numpy()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.pairs_csv)
    for col in (args.clean_col, args.degraded_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    pesq_scores: list[float] = []
    stoi_scores: list[float] = []

    for _, row in df.iterrows():
        clean = load_mono_16k(Path(str(row[args.clean_col])))
        degraded = load_mono_16k(Path(str(row[args.degraded_col])))
        min_len = min(len(clean), len(degraded))
        if min_len == 0:
            raise ValueError("Encountered empty waveform after loading/resampling")
        clean = clean[:min_len]
        degraded = degraded[:min_len]

        pesq_score = float(pesq(TARGET_SR, clean, degraded, "wb"))
        stoi_score = float(stoi(clean, degraded, TARGET_SR, extended=False))
        pesq_scores.append(pesq_score)
        stoi_scores.append(stoi_score)

    pesq_arr = np.array(pesq_scores, dtype=np.float32)
    stoi_arr = np.array(stoi_scores, dtype=np.float32)

    pesq_valid = bool(np.all((pesq_arr >= -0.5) & (pesq_arr <= 4.5)))
    stoi_valid = bool(np.all((stoi_arr >= 0.0) & (stoi_arr <= 1.0)))

    print(f"Pairs evaluated: {len(df)}")
    print(f"PESQ mean/min/max: {pesq_arr.mean():.4f} / {pesq_arr.min():.4f} / {pesq_arr.max():.4f}")
    print(f"STOI mean/min/max: {stoi_arr.mean():.4f} / {stoi_arr.min():.4f} / {stoi_arr.max():.4f}")
    print(f"PESQ in valid range [-0.5, 4.5]: {pesq_valid}")
    print(f"STOI in valid range [0.0, 1.0]: {stoi_valid}")

    if args.fail_on_invalid and not (pesq_valid and stoi_valid):
        raise SystemExit("Invalid PESQ/STOI range detected")


if __name__ == "__main__":
    main()
