#!/usr/bin/env python3
"""Run selected project phases in sequence with reproducible command logging."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Pipeline config JSON")
    parser.add_argument("--from-phase", type=int, default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--to-phase", type=int, default=4, choices=[0, 1, 2, 3, 4])
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    repo_root = Path(str(cfg["repo_root"])).resolve()
    scripts = cfg.get("phase_scripts", {})

    phases = list(range(args.from_phase, args.to_phase + 1))
    for phase in phases:
        key = f"phase{phase}"
        commands = scripts.get(key, [])
        if not commands:
            print(f"Skipping {key}: no commands configured")
            continue
        for command in commands:
            run_cmd(command, repo_root)

    print("Pipeline execution complete")


if __name__ == "__main__":
    main()
