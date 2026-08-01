#!/usr/bin/env python3
"""Phase 1A: train defender baselines with shared config."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
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
from scripts.phase1.models import build_defender_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to defender SFT JSON config")
    return parser.parse_args()


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["label"].to(device)

            if model.__class__.__name__.startswith("WavLM"):
                logits = model(waveform, lengths)
            else:
                logits = model(waveform)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    return compute_binary_metrics(logits_np, labels_np)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    run_name = str(config.get("run_name", "phase1_defender_sft"))
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

    models = config.get("models", ["wavlm", "aasist", "rawnet2", "lcnn"])
    epochs = int(config.get("epochs", 3))
    lr = float(config.get("lr", 5e-5))
    amp = bool(config.get("amp", True))

    for model_name in models:
        model = build_defender_model(str(model_name), wavlm=config.get("wavlm"))
        model = model.to(device)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
        )
        scaler = GradScaler(enabled=amp and device.type == "cuda")

        best_eer = float("inf")
        for epoch in range(1, epochs + 1):
            model.train()
            losses: list[float] = []

            for batch in tqdm(train_loader, desc=f"{model_name} epoch {epoch}"):
                waveform = batch["waveform"].to(device)
                lengths = batch["lengths"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=amp and device.type == "cuda"):
                    if model_name.lower() == "wavlm":
                        logits = model(waveform, lengths)
                    else:
                        logits = model(waveform)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                losses.append(float(loss.detach().cpu().item()))

            metrics = evaluate(model, dev_loader, device)
            payload = {
                "phase": "1A",
                "model": model_name,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                **metrics,
            }
            append_metrics(artifacts.metrics_path, payload)
            print(payload)

            if metrics["eer"] < best_eer:
                best_eer = metrics["eer"]
                save_checkpoint(
                    artifacts.checkpoints_dir / f"{model_name}.best.pt",
                    model,
                    optimizer,
                    epoch,
                )


if __name__ == "__main__":
    main()
