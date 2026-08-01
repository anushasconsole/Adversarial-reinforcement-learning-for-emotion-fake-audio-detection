#!/usr/bin/env python3
"""Package project artifacts and generate completion summary for reproducibility."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return True


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    packaged = {
        "manifests": copy_if_exists(args.manifests_dir.resolve(), output_dir / "manifests"),
        "phase1_artifacts": copy_if_exists(args.artifacts_root.resolve() / "phase1", output_dir / "artifacts" / "phase1"),
        "phase2_artifacts": copy_if_exists(args.artifacts_root.resolve() / "phase2", output_dir / "artifacts" / "phase2"),
        "phase3_artifacts": copy_if_exists(args.artifacts_root.resolve() / "phase3", output_dir / "artifacts" / "phase3"),
        "phase4_artifacts": copy_if_exists(args.artifacts_root.resolve() / "phase4", output_dir / "artifacts" / "phase4"),
        "configs": copy_if_exists(repo_root / "scripts", output_dir / "scripts"),
        "environment": copy_if_exists(repo_root / "environment.phase0.yml", output_dir / "environment.phase0.yml"),
    }

    objectives = {
        "phase0_data_foundation": packaged["manifests"],
        "phase1_baselines": packaged["phase1_artifacts"],
        "phase2_stress_tests": packaged["phase2_artifacts"],
        "phase3_rl_loop": packaged["phase3_artifacts"],
        "phase4_final_evaluation": packaged["phase4_artifacts"],
    }

    summary = {
        "packaged": packaged,
        "objectives_met": objectives,
    }

    with (output_dir / "completion_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    archive_base = output_dir / "completion_package"
    shutil.make_archive(str(archive_base), "zip", root_dir=output_dir)

    print(json.dumps(summary, indent=2))
    print(f"Archive: {archive_base}.zip")


if __name__ == "__main__":
    main()
