#!/usr/bin/env python3
"""Phase 1B: train attacker SFT models against frozen AASIST target."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from pesq import pesq
from pystoi import stoi
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.phase1.common import (
    ManifestAudioDataset,
    TARGET_SR,
    append_metrics,
    collate_waveforms,
    compute_binary_metrics,
    ensure_device,
    load_config,
    prepare_run_dir,
    save_checkpoint,
    set_seed,
)
from scripts.phase1.models import build_attacker_model, build_defender_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to attacker SFT JSON config")
    return parser.parse_args()


def pesq_stoi_batch(clean: torch.Tensor, degraded: torch.Tensor) -> tuple[float, float]:
    pesq_values: list[float] = []
    stoi_values: list[float] = []
    clean_np = clean.detach().cpu().numpy()
    degraded_np = degraded.detach().cpu().numpy()
    for c, d in zip(clean_np, degraded_np):
        length = min(len(c), len(d))
        if length == 0:
            continue
        c = c[:length]
        d = d[:length]
        try:
            pesq_values.append(float(pesq(TARGET_SR, c, d, "wb")))
        except Exception:
            continue
        try:
            stoi_values.append(float(stoi(c, d, TARGET_SR, extended=False)))
        except Exception:
            continue
    pesq_mean = float(np.mean(pesq_values)) if pesq_values else float("nan")
    stoi_mean = float(np.mean(stoi_values)) if stoi_values else float("nan")
    return pesq_mean, stoi_mean


def differentiable_quality_penalty(
    clean: torch.Tensor,
    degraded: torch.Tensor,
    pesq_threshold: float,
    stoi_threshold: float,
    lambda_pesq: float,
    lambda_stoi: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = torch.mean((degraded - clean) ** 2, dim=1)
    pesq_proxy = 4.5 - torch.clamp(mse * 100.0, min=0.0)

    clean_centered = clean - clean.mean(dim=1, keepdim=True)
    degraded_centered = degraded - degraded.mean(dim=1, keepdim=True)
    cosine = torch.nn.functional.cosine_similarity(clean_centered, degraded_centered, dim=1)
    stoi_proxy = torch.clamp((cosine + 1.0) / 2.0, 0.0, 1.0)

    pesq_penalty = torch.relu(torch.tensor(pesq_threshold, device=clean.device) - pesq_proxy).mean()
    stoi_penalty = torch.relu(torch.tensor(stoi_threshold, device=clean.device) - stoi_proxy).mean()

    weighted = lambda_pesq * pesq_penalty + lambda_stoi * stoi_penalty
    return weighted, pesq_proxy.mean(), stoi_proxy.mean()


def wgan_gp_penalty(clean: torch.Tensor, degraded: torch.Tensor, lambda_gp: float) -> torch.Tensor:
    delta = degraded - clean
    if delta.shape[1] < 2:
        return torch.tensor(0.0, device=clean.device)
    grad_t = delta[:, 1:] - delta[:, :-1]
    grad_norm = torch.sqrt(torch.sum(grad_t**2, dim=1) + 1e-8)
    return lambda_gp * ((grad_norm - 1.0) ** 2).mean()


def evaluate_attacker(
    attacker: torch.nn.Module,
    defender: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    attacker.eval()
    defender.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    pesq_scores: list[float] = []
    stoi_scores: list[float] = []

    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].to(device)
            labels = batch["label"].to(device)

            delta = attacker(waveform)
            adv = torch.clamp(waveform + delta, -1.0, 1.0)
            logits = defender(adv)

            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            pesq_value, stoi_value = pesq_stoi_batch(waveform, adv)
            if np.isfinite(pesq_value):
                pesq_scores.append(pesq_value)
            if np.isfinite(stoi_value):
                stoi_scores.append(stoi_value)

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    metrics = compute_binary_metrics(logits_np, labels_np)

    probs = 1.0 / (1.0 + np.exp(-logits_np))
    preds = (probs >= 0.5).astype(np.float32)
    attack_success = float((preds != labels_np).mean())

    metrics["attack_success"] = attack_success
    metrics["pesq"] = float(np.mean(pesq_scores)) if pesq_scores else float("nan")
    metrics["stoi"] = float(np.mean(stoi_scores)) if stoi_scores else float("nan")
    return metrics


def load_frozen_aasist(path: Path, device: torch.device) -> torch.nn.Module:
    model = build_defender_model("aasist")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    run_name = str(config.get("run_name", "phase1_attacker_sft"))
    output_dir = Path(str(config["output_dir"]))
    artifacts = prepare_run_dir(output_dir, run_name, config)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = ensure_device(str(config.get("device", "cuda")))

    train_ds = ManifestAudioDataset(
        manifest_path=Path(str(config["train_manifest"])),
        path_col=str(config.get("path_col", "preprocessed_path")),
        fallback_path_col=str(config.get("fallback_path_col", "absolute_path")),
        label_col=str(config.get("label_col", "label")),
    )
    dev_ds = ManifestAudioDataset(
        manifest_path=Path(str(config["dev_manifest"])),
        path_col=str(config.get("path_col", "preprocessed_path")),
        fallback_path_col=str(config.get("fallback_path_col", "absolute_path")),
        label_col=str(config.get("label_col", "label")),
    )

    batch_size = int(config.get("batch_size", 8))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collate_waveforms,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=collate_waveforms,
    )

    frozen_defender_path = Path(str(config["frozen_defender_checkpoint"]))
    defender = load_frozen_aasist(frozen_defender_path, device)

    attackers = config.get("models", ["squeezeformer", "mamba"])
    epochs = int(config.get("epochs", 3))
    lr = float(config.get("lr", 1e-4))
    amp = bool(config.get("amp", True))
    epsilon = float(config.get("epsilon", 0.01))
    lambda_pesq = float(config.get("lambda_pesq", 0.5))
    lambda_stoi = float(config.get("lambda_stoi", 0.5))
    pesq_threshold = float(config.get("pesq_threshold", 2.5))
    stoi_threshold = float(config.get("stoi_threshold", 0.75))
    lambda_gp = float(config.get("lambda_gp", 10.0))

    criterion = torch.nn.BCEWithLogitsLoss()

    for attacker_name in attackers:
        attacker = build_attacker_model(str(attacker_name), epsilon=epsilon).to(device)
        optimizer = torch.optim.AdamW(attacker.parameters(), lr=lr)
        scaler = GradScaler(enabled=amp and device.type == "cuda")

        best_attack_success = -1.0
        for epoch in range(1, epochs + 1):
            attacker.train()
            losses: list[float] = []
            for batch in tqdm(train_loader, desc=f"{attacker_name} epoch {epoch}"):
                waveform = batch["waveform"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=amp and device.type == "cuda"):
                    delta = attacker(waveform)
                    adv = torch.clamp(waveform + delta, -1.0, 1.0)

                    logits = defender(adv)
                    defender_loss = criterion(logits, labels)

                    quality_penalty, _, _ = differentiable_quality_penalty(
                        waveform,
                        adv,
                        pesq_threshold,
                        stoi_threshold,
                        lambda_pesq,
                        lambda_stoi,
                    )
                    gp_penalty = wgan_gp_penalty(waveform, adv, lambda_gp=lambda_gp)
                    total_loss = -defender_loss + quality_penalty + gp_penalty

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                losses.append(float(total_loss.detach().cpu().item()))

            metrics = evaluate_attacker(attacker, defender, dev_loader, device)
            payload = {
                "phase": "1B",
                "model": attacker_name,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "quality_constraints": {
                    "pesq_threshold": pesq_threshold,
                    "stoi_threshold": stoi_threshold,
                },
                **metrics,
            }
            append_metrics(artifacts.metrics_path, payload)
            print(payload)

            if metrics["attack_success"] > best_attack_success:
                best_attack_success = metrics["attack_success"]
                save_checkpoint(
                    artifacts.checkpoints_dir / f"{attacker_name}.best.pt",
                    attacker,
                    optimizer,
                    epoch,
                )


if __name__ == "__main__":
    main()
