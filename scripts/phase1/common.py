#!/usr/bin/env python3
"""Shared utilities for Phase 1 training/evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import Dataset


TARGET_SR = 16000


@dataclass
class RunArtifacts:
    run_dir: Path
    checkpoints_dir: Path
    metrics_path: Path
    config_path: Path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_run_dir(output_dir: Path, run_name: str, config: dict[str, Any]) -> RunArtifacts:
    run_dir = output_dir.resolve() / run_name
    checkpoints_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.snapshot.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return RunArtifacts(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        metrics_path=run_dir / "metrics.jsonl",
        config_path=config_path,
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class ManifestAudioDataset(Dataset):
    """Loads waveforms from manifest CSV and returns binary labels."""

    def __init__(
        self,
        manifest_path: Path,
        path_col: str = "preprocessed_path",
        fallback_path_col: str = "absolute_path",
        label_col: str = "label",
        target_sr: int = TARGET_SR,
    ) -> None:
        self.df = pd.read_csv(manifest_path)
        self.path_col = path_col if path_col in self.df.columns else fallback_path_col
        if self.path_col not in self.df.columns:
            raise ValueError(
                f"Neither '{path_col}' nor '{fallback_path_col}' found in {manifest_path}"
            )
        if label_col not in self.df.columns:
            raise ValueError(f"'{label_col}' column is required in {manifest_path}")
        self.label_col = label_col
        self.target_sr = target_sr

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        path = Path(str(row[self.path_col]))
        waveform, sr = torchaudio.load(str(path))
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
        waveform = waveform.squeeze(0).to(torch.float32)

        label_value = row[self.label_col]
        if isinstance(label_value, str):
            label = 1.0 if label_value.lower() in {"fake", "spoof", "1", "true"} else 0.0
        else:
            label = float(label_value)
            label = 1.0 if label >= 0.5 else 0.0

        return {
            "waveform": waveform,
            "label": torch.tensor(label, dtype=torch.float32),
        }


def collate_waveforms(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    waveforms = [item["waveform"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    lengths = torch.tensor([w.shape[-1] for w in waveforms], dtype=torch.long)
    max_len = int(lengths.max())
    padded = []
    for waveform in waveforms:
        if waveform.shape[-1] < max_len:
            pad = max_len - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        padded.append(waveform)
    return {
        "waveform": torch.stack(padded),
        "lengths": lengths,
        "label": labels,
    }


def compute_binary_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.float32)
    accuracy = float((preds == labels).mean())

    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")

    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        fnr = 1.0 - tpr
        eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    except ValueError:
        eer = float("nan")

    return {"eer": eer, "auc": auc, "accuracy": accuracy}


def append_metrics(metrics_path: Path, payload: dict[str, Any]) -> None:
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def ensure_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)
