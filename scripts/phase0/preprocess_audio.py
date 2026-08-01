#!/usr/bin/env python3
"""Standardize audio to 16kHz mono, peak-normalized, and fixed duration."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm


TARGET_SR = 16000
TARGET_SECONDS = 4.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_SECONDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Input manifest CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Processed WAV output directory")
    parser.add_argument("--output-manifest", type=Path, required=True, help="Output manifest CSV")
    parser.add_argument("--path-col", default="absolute_path", help="Manifest column with input audio paths")
    parser.add_argument("--workers", type=int, default=0, help="Reserved for future parallel preprocessing")
    _ = parser
    return parser.parse_args()


def standardize_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SR)

    peak = waveform.abs().max()
    if torch.isfinite(peak) and peak > 0:
        waveform = waveform / peak

    num_samples = waveform.shape[-1]
    if num_samples < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    elif num_samples > TARGET_SAMPLES:
        waveform = waveform[..., :TARGET_SAMPLES]

    return waveform.to(torch.float32)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    if args.path_col not in df.columns:
        raise ValueError(f"Column '{args.path_col}' not found in manifest")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_paths = []
    for idx, input_path in enumerate(tqdm(df[args.path_col].astype(str).tolist(), desc="Preprocessing")):
        src = Path(input_path)
        if not src.exists():
            raise FileNotFoundError(f"Missing input audio file: {src}")

        waveform, sr = torchaudio.load(str(src))
        waveform = standardize_waveform(waveform, sr)

        dst = out_dir / f"sample_{idx:08d}.wav"
        torchaudio.save(str(dst), waveform, TARGET_SR, encoding="PCM_S", bits_per_sample=16)
        processed_paths.append(str(dst))

    df["preprocessed_path"] = processed_paths
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_manifest, index=False)
    print(f"Saved preprocessed manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
