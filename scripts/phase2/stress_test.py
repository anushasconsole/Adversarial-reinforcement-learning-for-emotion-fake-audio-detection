#!/usr/bin/env python3
"""Phase 2: cross-architecture stress-test matrix for defenders x attackers."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pesq import pesq
from pystoi import stoi
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from scripts.phase1.common import (
    ManifestAudioDataset,
    append_metrics,
    collate_waveforms,
    compute_binary_metrics,
    ensure_device,
    load_config,
    prepare_run_dir,
    save_checkpoint,
    set_seed,
)
from scripts.phase1.models import (
    MambaStyleAttacker,
    SqueezeformerStyleAttacker,
    build_defender_model,
)


TARGET_SR = 16000


class CnnUnetAttacker(torch.nn.Module):
    def __init__(self, epsilon: float) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.down1 = torch.nn.Sequential(torch.nn.Conv1d(1, 32, 5, stride=2, padding=2), torch.nn.ReLU())
        self.down2 = torch.nn.Sequential(torch.nn.Conv1d(32, 64, 5, stride=2, padding=2), torch.nn.ReLU())
        self.up1 = torch.nn.Sequential(torch.nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1), torch.nn.ReLU())
        self.up2 = torch.nn.ConvTranspose1d(32, 1, 4, stride=2, padding=1)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        x = waveforms.unsqueeze(1)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        u1 = self.up1(d2)
        u1 = u1[..., : d1.shape[-1]] + d1
        u2 = self.up2(u1)[..., : x.shape[-1]]
        return torch.clamp(torch.tanh(u2).squeeze(1) * self.epsilon, -self.epsilon, self.epsilon)


class BiLstmAttacker(torch.nn.Module):
    def __init__(self, epsilon: float, hidden: int = 64) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.rnn = torch.nn.LSTM(1, hidden, batch_first=True, bidirectional=True)
        self.head = torch.nn.Sequential(torch.nn.Linear(hidden * 2, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1), torch.nn.Tanh())

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        x = waveforms.unsqueeze(-1)
        x, _ = self.rnn(x)
        x = self.head(x).squeeze(-1)
        return torch.clamp(x * self.epsilon, -self.epsilon, self.epsilon)


@dataclass
class LoadedDefender:
    name: str
    model: torch.nn.Module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Phase 2 JSON config path")
    return parser.parse_args()


def load_model_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)


def load_defenders(cfg: dict, device: torch.device) -> list[LoadedDefender]:
    defenders = []
    for item in cfg["defenders"]:
        name = str(item["name"])
        model = build_defender_model(name, wavlm=cfg.get("wavlm"))
        load_model_checkpoint(model, Path(str(item["checkpoint"])), device)
        model.to(device).eval()
        defenders.append(LoadedDefender(name=name, model=model))
    return defenders


def load_attackers(cfg: dict, device: torch.device) -> dict[str, torch.nn.Module]:
    epsilon = float(cfg.get("epsilon", 0.01))
    attackers: dict[str, torch.nn.Module] = {
        "fgsm": None,
        "pgd": None,
        "squeezeformer": SqueezeformerStyleAttacker(epsilon),
        "mamba": MambaStyleAttacker(epsilon),
        "cnn_unet": CnnUnetAttacker(epsilon),
        "bi_lstm": BiLstmAttacker(epsilon),
    }

    ckpts = cfg.get("attacker_checkpoints", {})
    for name, model in list(attackers.items()):
        if model is None:
            continue
        ckpt = ckpts.get(name)
        if ckpt:
            load_model_checkpoint(model, Path(str(ckpt)), device)
        model.to(device).eval()
        attackers[name] = model
    return attackers


def build_loader(cfg: dict) -> DataLoader:
    ds = ManifestAudioDataset(
        manifest_path=Path(str(cfg["eval_manifest"])),
        path_col=str(cfg.get("path_col", "preprocessed_path")),
        fallback_path_col=str(cfg.get("fallback_path_col", "absolute_path")),
        label_col=str(cfg.get("label_col", "label")),
    )
    max_samples = int(cfg.get("max_eval_samples", 0))
    if max_samples > 0 and max_samples < len(ds):
        ds = Subset(ds, range(max_samples))
    return DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_waveforms,
    )


def pesq_stoi_batch(clean: torch.Tensor, degraded: torch.Tensor) -> tuple[float, float]:
    clean_np = clean.detach().cpu().numpy()
    degraded_np = degraded.detach().cpu().numpy()
    pesq_values = []
    stoi_values = []
    for c, d in zip(clean_np, degraded_np):
        n = min(len(c), len(d))
        if n == 0:
            continue
        c = c[:n]
        d = d[:n]
        try:
            pesq_values.append(float(pesq(TARGET_SR, c, d, "wb")))
        except Exception:
            pass
        try:
            stoi_values.append(float(stoi(c, d, TARGET_SR, extended=False)))
        except Exception:
            pass
    pesq_mean = float(np.mean(pesq_values)) if pesq_values else float("nan")
    stoi_mean = float(np.mean(stoi_values)) if stoi_values else float("nan")
    return pesq_mean, stoi_mean


def attack_with_grad(
    attacker_name: str,
    attacker: torch.nn.Module | None,
    defender: torch.nn.Module,
    waveforms: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    pgd_steps: int,
    pgd_alpha: float,
) -> torch.Tensor:
    if attacker_name == "fgsm":
        x = waveforms.detach().clone().requires_grad_(True)
        logits = defender(x, None) if defender.__class__.__name__.startswith("WavLM") else defender(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        grad = torch.autograd.grad(loss, x)[0]
        return torch.clamp(x + epsilon * grad.sign(), -1.0, 1.0).detach()

    if attacker_name == "pgd":
        x_orig = waveforms.detach()
        x = x_orig + (torch.rand_like(x_orig) * 2 - 1) * epsilon
        x = torch.clamp(x, -1.0, 1.0)
        for _ in range(pgd_steps):
            x = x.detach().requires_grad_(True)
            logits = defender(x, None) if defender.__class__.__name__.startswith("WavLM") else defender(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            grad = torch.autograd.grad(loss, x)[0]
            x = x + pgd_alpha * grad.sign()
            x = torch.max(torch.min(x, x_orig + epsilon), x_orig - epsilon)
            x = torch.clamp(x, -1.0, 1.0)
        return x.detach()

    if attacker is None:
        return waveforms.detach()

    with torch.no_grad():
        delta = attacker(waveforms)
        return torch.clamp(waveforms + delta, -1.0, 1.0)


def eval_pair(defender: LoadedDefender, attacker_name: str, attacker: torch.nn.Module | None, loader: DataLoader, cfg: dict, device: torch.device) -> dict:
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    pesq_values: list[float] = []
    stoi_values: list[float] = []

    epsilon = float(cfg.get("epsilon", 0.01))
    pgd_steps = int(cfg.get("pgd_steps", 20))
    pgd_alpha = float(cfg.get("pgd_alpha", epsilon / 10.0))

    for batch in tqdm(loader, desc=f"{defender.name} vs {attacker_name}", leave=False):
        waveform = batch["waveform"].to(device)
        labels = batch["label"].to(device)
        adv = attack_with_grad(attacker_name, attacker, defender.model, waveform, labels, epsilon, pgd_steps, pgd_alpha)

        with torch.no_grad():
            if defender.name.lower() == "wavlm":
                lengths = torch.full((adv.shape[0],), adv.shape[1], device=device, dtype=torch.long)
                logits = defender.model(adv, lengths)
            else:
                logits = defender.model(adv)

        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())

        if attacker_name != "clean":
            p, s = pesq_stoi_batch(waveform, adv)
            if np.isfinite(p):
                pesq_values.append(p)
            if np.isfinite(s):
                stoi_values.append(s)

    logits_np = np.concatenate(logits_all, axis=0)
    labels_np = np.concatenate(labels_all, axis=0)
    metrics = compute_binary_metrics(logits_np, labels_np)
    probs = 1 / (1 + np.exp(-logits_np))
    preds = (probs >= 0.5).astype(np.float32)
    attack_success = float((preds != labels_np).mean())

    return {
        "defender": defender.name,
        "attacker": attacker_name,
        "attack_success": attack_success,
        "pesq": float(np.mean(pesq_values)) if pesq_values else float("nan"),
        "stoi": float(np.mean(stoi_values)) if stoi_values else float("nan"),
        **metrics,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = ensure_device(str(cfg.get("device", "cuda")))

    artifacts = prepare_run_dir(Path(str(cfg["output_dir"])), str(cfg.get("run_name", "phase2_stress_test")), cfg)

    defenders = load_defenders(cfg, device)
    attackers = load_attackers(cfg, device)
    loader = build_loader(cfg)

    rows: list[dict] = []
    for defender in defenders:
        clean_row = eval_pair(defender, "clean", None, loader, cfg, device)
        rows.append(clean_row)
        append_metrics(artifacts.metrics_path, clean_row)

        for attacker_name, attacker in attackers.items():
            row = eval_pair(defender, attacker_name, attacker, loader, cfg, device)
            rows.append(row)
            append_metrics(artifacts.metrics_path, row)

    df = pd.DataFrame(rows)
    matrix_path = artifacts.run_dir / "vulnerability_matrix.csv"
    df.to_csv(matrix_path, index=False)

    summary = {
        "rows": int(len(df)),
        "defenders": sorted(df["defender"].dropna().unique().tolist()),
        "attackers": sorted(df["attacker"].dropna().unique().tolist()),
        "matrix_path": str(matrix_path),
    }
    summary_path = artifacts.run_dir / "phase2_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
