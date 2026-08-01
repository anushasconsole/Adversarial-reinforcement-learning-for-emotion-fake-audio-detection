#!/usr/bin/env python3
"""Phase 3: 5,000-step zero-sum style RL loop for attacker/defender co-training."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pesq import pesq
from pystoi import stoi
from torch.utils.data import DataLoader

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


@dataclass
class DefenderSlot:
    name: str
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer


@dataclass
class AttackerSlot:
    name: str
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_ckpt_if_exists(model: torch.nn.Module, path: str | None, device: torch.device) -> None:
    if not path:
        return
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt))


def create_train_loader(cfg: dict) -> DataLoader:
    ds = ManifestAudioDataset(
        manifest_path=Path(str(cfg["train_manifest"])),
        path_col=str(cfg.get("path_col", "preprocessed_path")),
        fallback_path_col=str(cfg.get("fallback_path_col", "absolute_path")),
        label_col=str(cfg.get("label_col", "label")),
    )
    return DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_waveforms,
        drop_last=True,
    )


def create_eval_loader(cfg: dict) -> DataLoader:
    ds = ManifestAudioDataset(
        manifest_path=Path(str(cfg["dev_manifest"])),
        path_col=str(cfg.get("path_col", "preprocessed_path")),
        fallback_path_col=str(cfg.get("fallback_path_col", "absolute_path")),
        label_col=str(cfg.get("label_col", "label")),
    )
    return DataLoader(
        ds,
        batch_size=int(cfg.get("eval_batch_size", cfg.get("batch_size", 8))),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_waveforms,
    )


def iter_batches(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def get_curriculum(step: int) -> tuple[float, int]:
    if step <= 1000:
        return 0.005, 4
    if step <= 3000:
        return 0.01, 4
    return 0.02, 8


def quality_metrics(clean: torch.Tensor, adv: torch.Tensor) -> tuple[float, float]:
    clean_np = clean.detach().cpu().numpy()
    adv_np = adv.detach().cpu().numpy()
    pesq_values = []
    stoi_values = []
    for c, d in zip(clean_np, adv_np):
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
    return (
        float(np.mean(pesq_values)) if pesq_values else float("nan"),
        float(np.mean(stoi_values)) if stoi_values else float("nan"),
    )


def evaluate(defender: torch.nn.Module, attacker: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 20, wavlm_name: bool = False) -> dict[str, float]:
    all_logits = []
    all_labels = []
    pesq_values = []
    stoi_values = []
    attacker.eval()
    defender.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            waveform = batch["waveform"].to(device)
            labels = batch["label"].to(device)
            delta = attacker(waveform)
            adv = torch.clamp(waveform + delta, -1.0, 1.0)

            if wavlm_name:
                lengths = torch.full((adv.shape[0],), adv.shape[1], device=device, dtype=torch.long)
                logits = defender(adv, lengths)
            else:
                logits = defender(adv)

            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            p, s = quality_metrics(waveform, adv)
            if np.isfinite(p):
                pesq_values.append(p)
            if np.isfinite(s):
                stoi_values.append(s)

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    metrics = compute_binary_metrics(logits_np, labels_np)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    preds = (probs >= 0.5).astype(np.float32)
    success = float((preds != labels_np).mean())
    metrics.update({
        "attack_success": success,
        "pesq": float(np.mean(pesq_values)) if pesq_values else float("nan"),
        "stoi": float(np.mean(stoi_values)) if stoi_values else float("nan"),
    })
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(int(cfg.get("seed", 42)))
    device = ensure_device(str(cfg.get("device", "cuda")))
    artifacts = prepare_run_dir(Path(str(cfg["output_dir"])), str(cfg.get("run_name", "phase3_rl")), cfg)

    wavlm_cfg = cfg.get("wavlm")

    defenders: dict[str, DefenderSlot] = {}
    for d in cfg["defenders"]:
        name = str(d["name"]).lower()
        model = build_defender_model(name, wavlm=wavlm_cfg).to(device)
        load_ckpt_if_exists(model, d.get("checkpoint"), device)
        for p in model.parameters():
            p.requires_grad = True
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("defender_lr", 5e-5)))
        defenders[name] = DefenderSlot(name=name, model=model, optimizer=opt)

    epsilon = float(cfg.get("epsilon", 0.01))
    squeezeformer = SqueezeformerStyleAttacker(epsilon).to(device)
    mamba = MambaStyleAttacker(epsilon).to(device)
    load_ckpt_if_exists(squeezeformer, cfg.get("attacker_checkpoints", {}).get("squeezeformer"), device)
    load_ckpt_if_exists(mamba, cfg.get("attacker_checkpoints", {}).get("mamba"), device)
    attackers = {
        "squeezeformer": AttackerSlot("squeezeformer", squeezeformer, torch.optim.AdamW(squeezeformer.parameters(), lr=float(cfg.get("attacker_lr", 1e-4)))),
        "mamba": AttackerSlot("mamba", mamba, torch.optim.AdamW(mamba.parameters(), lr=float(cfg.get("attacker_lr", 1e-4)))),
    }

    train_loader = create_train_loader(cfg)
    eval_loader = create_eval_loader(cfg)
    batch_iter = iter_batches(train_loader)

    steps = int(cfg.get("steps", 5000))
    group_size = int(cfg.get("group_size", 4))
    eval_interval = int(cfg.get("eval_interval", 100))
    checkpoint_interval = int(cfg.get("checkpoint_interval", 500))
    lambda_pesq = float(cfg.get("lambda_pesq", 0.5))
    lambda_stoi = float(cfg.get("lambda_stoi", 0.5))
    pesq_threshold = float(cfg.get("pesq_threshold", 2.5))
    stoi_threshold = float(cfg.get("stoi_threshold", 0.75))
    lambda_gp = float(cfg.get("lambda_gp", 10.0))

    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    for step in range(1, steps + 1):
        epsilon_step, _max_secs = get_curriculum(step)
        active_defender_name = "wavlm" if ((step - 1) // 100) % 2 == 0 else "aasist"
        active_attacker_name = "squeezeformer" if step <= 3000 else "mamba"

        defender_slot = defenders[active_defender_name]
        attacker_slot = attackers[active_attacker_name]

        defender = defender_slot.model
        attacker = attacker_slot.model
        defender.train()
        attacker.train()

        batch = next(batch_iter)
        waveform = batch["waveform"].to(device)
        labels = batch["label"].to(device)

        defender_slot.optimizer.zero_grad(set_to_none=True)
        attacker_slot.optimizer.zero_grad(set_to_none=True)

        group_rewards_attacker = []
        group_rewards_defender = []
        variants = []
        logits_group = []

        for _ in range(group_size):
            noise = torch.randn_like(waveform) * 0.001
            delta = attacker(waveform + noise)
            delta = torch.clamp(delta, -epsilon_step, epsilon_step)
            adv = torch.clamp(waveform + delta, -1.0, 1.0)
            variants.append(adv)

            if active_defender_name == "wavlm":
                lengths = torch.full((adv.shape[0],), adv.shape[1], device=device, dtype=torch.long)
                logits = defender(adv, lengths)
            else:
                logits = defender(adv)
            logits_group.append(logits)

            bce = criterion(logits, labels)
            defender_reward = -bce.detach()
            attacker_reward = bce.detach()
            group_rewards_defender.append(defender_reward)
            group_rewards_attacker.append(attacker_reward)

        rewards_att = torch.stack(group_rewards_attacker, dim=0)
        rewards_def = torch.stack(group_rewards_defender, dim=0)
        adv_att = (rewards_att - rewards_att.mean(dim=0, keepdim=True)) / (rewards_att.std(dim=0, keepdim=True) + 1e-8)
        adv_def = (rewards_def - rewards_def.mean(dim=0, keepdim=True)) / (rewards_def.std(dim=0, keepdim=True) + 1e-8)

        logits_stack = torch.stack(logits_group, dim=0)
        bce_stack = criterion(logits_stack, labels.unsqueeze(0).expand_as(logits_stack))

        defender_loss = (bce_stack * (-adv_def.detach())).mean()
        attacker_loss_main = (bce_stack * (-adv_att.detach())).mean()

        adv_concat = torch.cat(variants, dim=0)
        clean_concat = waveform.repeat(group_size, 1)
        mse = torch.mean((adv_concat - clean_concat) ** 2, dim=1)
        pesq_proxy = 4.5 - torch.clamp(mse * 100.0, min=0.0)

        clean_centered = clean_concat - clean_concat.mean(dim=1, keepdim=True)
        adv_centered = adv_concat - adv_concat.mean(dim=1, keepdim=True)
        cosine = torch.nn.functional.cosine_similarity(clean_centered, adv_centered, dim=1)
        stoi_proxy = torch.clamp((cosine + 1.0) / 2.0, 0.0, 1.0)

        quality_penalty = (
            lambda_pesq * torch.relu(torch.tensor(pesq_threshold, device=device) - pesq_proxy).mean()
            + lambda_stoi * torch.relu(torch.tensor(stoi_threshold, device=device) - stoi_proxy).mean()
        )

        delta = adv_concat - clean_concat
        if delta.shape[1] > 1:
            grad_t = delta[:, 1:] - delta[:, :-1]
            grad_norm = torch.sqrt(torch.sum(grad_t**2, dim=1) + 1e-8)
            gp_penalty = lambda_gp * ((grad_norm - 1.0) ** 2).mean()
        else:
            gp_penalty = torch.tensor(0.0, device=device)

        attacker_loss = attacker_loss_main + quality_penalty + gp_penalty

        defender_loss.backward(retain_graph=True)
        attacker_loss.backward()
        defender_slot.optimizer.step()
        attacker_slot.optimizer.step()

        if step % eval_interval == 0 or step == 1:
            eval_metrics = evaluate(
                defender,
                attacker,
                eval_loader,
                device,
                max_batches=int(cfg.get("eval_max_batches", 20)),
                wavlm_name=(active_defender_name == "wavlm"),
            )
            eps = max(epsilon_step, 1e-8)
            ci = (eval_metrics["attack_success"] * max(eval_metrics["pesq"], 0.0) * max(eval_metrics["stoi"], 0.0)) / eps
            payload = {
                "phase": "3",
                "step": step,
                "active_defender": active_defender_name,
                "active_attacker": active_attacker_name,
                "epsilon_step": epsilon_step,
                "defender_loss": float(defender_loss.detach().cpu().item()),
                "attacker_loss": float(attacker_loss.detach().cpu().item()),
                "quality_penalty": float(quality_penalty.detach().cpu().item()),
                "gp_penalty": float(gp_penalty.detach().cpu().item()),
                "attacker_reward_std": float(rewards_att.std().detach().cpu().item()),
                "defender_reward_std": float(rewards_def.std().detach().cpu().item()),
                "creativeness_index": float(ci),
                **eval_metrics,
            }
            append_metrics(artifacts.metrics_path, payload)
            print(json.dumps(payload))

        if step % checkpoint_interval == 0 or step == steps:
            save_checkpoint(artifacts.checkpoints_dir / f"defender_{active_defender_name}_step{step}.pt", defender, defender_slot.optimizer, step)
            save_checkpoint(artifacts.checkpoints_dir / f"attacker_{active_attacker_name}_step{step}.pt", attacker, attacker_slot.optimizer, step)


if __name__ == "__main__":
    main()
