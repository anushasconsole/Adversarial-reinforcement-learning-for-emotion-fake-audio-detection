#!/usr/bin/env python3
"""Phase 4: final evaluation with sanity checks, transfer hooks, and statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from scripts.phase1.common import ManifestAudioDataset, collate_waveforms, compute_binary_metrics, ensure_device, load_config
from scripts.phase1.models import MambaStyleAttacker, SqueezeformerStyleAttacker, build_defender_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_ckpt(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt))


def dataloader_from_manifest(path: Path, cfg: dict) -> DataLoader:
    ds = ManifestAudioDataset(
        manifest_path=path,
        path_col=str(cfg.get("path_col", "preprocessed_path")),
        fallback_path_col=str(cfg.get("fallback_path_col", "absolute_path")),
        label_col=str(cfg.get("label_col", "label")),
    )
    return DataLoader(ds, batch_size=int(cfg.get("batch_size", 8)), shuffle=False, num_workers=int(cfg.get("num_workers", 0)), collate_fn=collate_waveforms)


def eval_logits(defender_name: str, defender: torch.nn.Module, loader: DataLoader, device: torch.device, attacker: torch.nn.Module | None = None, fgsm_eps: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    defender.eval()
    if attacker is not None:
        attacker.eval()
    logits_all = []
    labels_all = []
    for batch in loader:
        waveform = batch["waveform"].to(device)
        labels = batch["label"].to(device)

        if attacker is not None:
            with torch.no_grad():
                adv = torch.clamp(waveform + attacker(waveform), -1.0, 1.0)
        elif fgsm_eps is not None:
            x = waveform.detach().clone().requires_grad_(True)
            if defender_name == "wavlm":
                lengths = torch.full((x.shape[0],), x.shape[1], device=device, dtype=torch.long)
                logits = defender(x, lengths)
            else:
                logits = defender(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            grad = torch.autograd.grad(loss, x)[0]
            adv = torch.clamp(x + fgsm_eps * grad.sign(), -1.0, 1.0).detach()
        else:
            adv = waveform

        with torch.no_grad():
            if defender_name == "wavlm":
                lengths = torch.full((adv.shape[0],), adv.shape[1], device=device, dtype=torch.long)
                logits = defender(adv, lengths)
            else:
                logits = defender(adv)

        logits_all.append(logits.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())

    return np.concatenate(logits_all, axis=0), np.concatenate(labels_all, axis=0)


def fast_delong_auc_cov(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    m = len(pos)
    n = len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan")

    v10 = np.array([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])
    v01 = np.array([np.mean(pos > n_) + 0.5 * np.mean(pos == n_) for n_ in neg])

    auc = float(np.mean(v10))
    sx = np.var(v10, ddof=1) / m if m > 1 else 0.0
    sy = np.var(v01, ddof=1) / n if n > 1 else 0.0
    return auc, float(sx + sy)


def delong_p_value(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    auc_a, var_a = fast_delong_auc_cov(y_true, scores_a)
    auc_b, var_b = fast_delong_auc_cov(y_true, scores_b)
    if not np.isfinite(auc_a) or not np.isfinite(auc_b):
        return float("nan")
    diff = auc_a - auc_b
    var = max(var_a + var_b, 1e-12)
    z = abs(diff) / np.sqrt(var)
    return float(2.0 * (1.0 - 0.5 * (1.0 + np.math.erf(z / np.sqrt(2.0)))))


def eer_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    probs = 1.0 / (1.0 + np.exp(-logits))
    fpr, tpr, _ = roc_curve(labels, probs)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def bootstrap_eer_ci(logits: np.ndarray, labels: np.ndarray, n_bootstrap: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    eers = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        sample_labels = labels[idx]
        if np.unique(sample_labels).size < 2:
            continue
        sample_logits = logits[idx]
        eers.append(eer_from_logits(sample_logits, sample_labels))
    if not eers:
        return float("nan"), float("nan")
    return float(np.percentile(eers, 2.5)), float(np.percentile(eers, 97.5))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = ensure_device(str(cfg.get("device", "cuda")))

    output_dir = Path(str(cfg["output_dir"])).resolve() / str(cfg.get("run_name", "phase4_final_eval"))
    output_dir.mkdir(parents=True, exist_ok=True)

    test_loader = dataloader_from_manifest(Path(str(cfg["test_manifest"])), cfg)

    defenders = {}
    for d in cfg["defenders"]:
        name = str(d["name"]).lower()
        model = build_defender_model(name, wavlm=cfg.get("wavlm")).to(device)
        load_ckpt(model, Path(str(d["checkpoint"])), device)
        defenders[name] = model

    rl_defender_name = str(cfg.get("rl_defender_name", "wavlm")).lower()
    sft_defender_name = str(cfg.get("sft_defender_name", "wavlm")).lower()

    rl_attacker_kind = str(cfg.get("rl_attacker", "mamba")).lower()
    epsilon = float(cfg.get("epsilon", 0.01))
    rl_attacker = MambaStyleAttacker(epsilon).to(device) if rl_attacker_kind == "mamba" else SqueezeformerStyleAttacker(epsilon).to(device)
    load_ckpt(rl_attacker, Path(str(cfg["rl_attacker_checkpoint"])), device)

    rows = []

    # FGSM sanity test on RL defender
    rl_defender = defenders[rl_defender_name]
    logits_fgsm, labels_fgsm = eval_logits(rl_defender_name, rl_defender, test_loader, device, attacker=None, fgsm_eps=float(cfg.get("fgsm_epsilon", 0.005)))
    fgsm_metrics = compute_binary_metrics(logits_fgsm, labels_fgsm)
    rows.append({"evaluation": "rl_defender_fgsm_sanity", **fgsm_metrics})

    # RL attacker transfer to LCNN
    lcnn = defenders.get("lcnn")
    if lcnn is not None:
        logits_transfer, labels_transfer = eval_logits("lcnn", lcnn, test_loader, device, attacker=rl_attacker)
        transfer_metrics = compute_binary_metrics(logits_transfer, labels_transfer)
        rows.append({"evaluation": "rl_attacker_to_lcnn", **transfer_metrics})

    # Main test comparison: SFT defender vs RL defender
    sft_defender = defenders[sft_defender_name]
    logits_sft_clean, labels = eval_logits(sft_defender_name, sft_defender, test_loader, device)
    logits_rl_clean, _ = eval_logits(rl_defender_name, rl_defender, test_loader, device)

    sft_metrics = compute_binary_metrics(logits_sft_clean, labels)
    rl_metrics = compute_binary_metrics(logits_rl_clean, labels)
    p_delong = delong_p_value(labels.astype(int), 1.0 / (1.0 + np.exp(-logits_sft_clean)), 1.0 / (1.0 + np.exp(-logits_rl_clean)))

    sft_eer_ci = bootstrap_eer_ci(logits_sft_clean, labels.astype(int), n_bootstrap=int(cfg.get("bootstrap_samples", 1000)), seed=int(cfg.get("seed", 42)))
    rl_eer_ci = bootstrap_eer_ci(logits_rl_clean, labels.astype(int), n_bootstrap=int(cfg.get("bootstrap_samples", 1000)), seed=int(cfg.get("seed", 42)) + 1)

    rows.append({
        "evaluation": "sft_vs_rl_clean",
        "sft_eer": sft_metrics["eer"],
        "sft_auc": sft_metrics["auc"],
        "rl_eer": rl_metrics["eer"],
        "rl_auc": rl_metrics["auc"],
        "delong_p_value": p_delong,
        "sft_eer_ci_low": sft_eer_ci[0],
        "sft_eer_ci_high": sft_eer_ci[1],
        "rl_eer_ci_low": rl_eer_ci[0],
        "rl_eer_ci_high": rl_eer_ci[1],
    })

    # Zero-day transfer hook if provided
    transfer_manifest = cfg.get("transfer_manifest")
    if transfer_manifest:
        transfer_loader = dataloader_from_manifest(Path(str(transfer_manifest)), cfg)
        for name, defender in defenders.items():
            logits_z, labels_z = eval_logits(name, defender, transfer_loader, device)
            z_metrics = compute_binary_metrics(logits_z, labels_z)
            rows.append({"evaluation": f"zero_day_{name}", **z_metrics})

    results_df = pd.DataFrame(rows)
    results_path = output_dir / "phase4_results.csv"
    results_df.to_csv(results_path, index=False)

    summary = {
        "results_path": str(results_path),
        "num_evaluations": int(len(results_df)),
    }
    with (output_dir / "phase4_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
