#!/usr/bin/env python3
"""Validate Phase 1 exit criteria from defender/attacker metric logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defender-metrics", type=Path, required=True, help="Phase 1A metrics.jsonl path")
    parser.add_argument("--attacker-metrics", type=Path, required=True, help="Phase 1B metrics.jsonl path")
    parser.add_argument("--output-summary", type=Path, required=True, help="Output path for JSON summary")
    parser.add_argument("--target-defender", default="aasist", help="Defender used for attacker SFT")
    parser.add_argument("--pesq-threshold", type=float, default=2.5)
    parser.add_argument("--stoi-threshold", type=float, default=0.75)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def latest_by_model(rows: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for row in rows:
        model = str(row.get("model", "")).lower()
        epoch = int(row.get("epoch", 0))
        if model not in latest or epoch >= int(latest[model].get("epoch", -1)):
            latest[model] = row
    return latest


def main() -> None:
    args = parse_args()

    defender_rows = read_jsonl(args.defender_metrics)
    attacker_rows = read_jsonl(args.attacker_metrics)

    if not defender_rows:
        raise ValueError("No defender metrics rows found")
    if not attacker_rows:
        raise ValueError("No attacker metrics rows found")

    defender_latest = latest_by_model(defender_rows)
    attacker_latest = latest_by_model(attacker_rows)

    target = args.target_defender.lower()
    if target not in defender_latest:
        raise ValueError(f"Target defender '{target}' not found in defender metrics")

    clean_eer = float(defender_latest[target].get("eer", np.nan))
    attacker_eval: dict[str, dict] = {}
    eer_improved_count = 0
    quality_ok_count = 0

    for model_name in ("squeezeformer", "mamba"):
        row = attacker_latest.get(model_name)
        if row is None:
            continue

        attacker_eer = float(row.get("eer", np.nan))
        pesq_value = float(row.get("pesq", np.nan))
        stoi_value = float(row.get("stoi", np.nan))

        eer_increase = bool(np.isfinite(attacker_eer) and np.isfinite(clean_eer) and attacker_eer > clean_eer)
        quality_ok = bool(
            np.isfinite(pesq_value)
            and np.isfinite(stoi_value)
            and pesq_value >= args.pesq_threshold
            and stoi_value >= args.stoi_threshold
        )

        if eer_increase:
            eer_improved_count += 1
        if quality_ok:
            quality_ok_count += 1

        attacker_eval[model_name] = {
            "attacked_eer": attacker_eer,
            "clean_eer": clean_eer,
            "eer_increase": eer_increase,
            "pesq": pesq_value,
            "stoi": stoi_value,
            "quality_ok": quality_ok,
        }

    summary = {
        "phase": "1_exit_checks",
        "target_defender": target,
        "clean_defender": defender_latest[target],
        "attackers": attacker_eval,
        "criteria": {
            "both_attackers_increase_eer": eer_improved_count >= 2,
            "both_attackers_meet_quality": quality_ok_count >= 2,
        },
        "pass": eer_improved_count >= 2 and quality_ok_count >= 2,
    }

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    if not summary["pass"]:
        raise SystemExit("Phase 1 exit criteria not met")


if __name__ == "__main__":
    main()
