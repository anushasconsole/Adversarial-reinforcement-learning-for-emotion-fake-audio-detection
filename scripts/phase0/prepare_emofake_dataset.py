#!/usr/bin/env python3
"""Prepare EmoFake dataset directory from a local source or downloadable archive."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Destination dataset root")
    parser.add_argument("--source-dir", type=Path, default=None, help="Existing local dataset directory")
    parser.add_argument("--archive-url", default=None, help="Optional HTTP(S) URL to archive (.zip/.tar/.tar.gz)")
    parser.add_argument("--archive-path", type=Path, default=None, help="Optional local archive path")
    parser.add_argument("--metadata-name", default="metadata.csv", help="Metadata CSV filename expected inside dataset root")
    parser.add_argument("--audio-dir-name", default="audio", help="Audio directory name expected inside dataset root")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlink when using --source-dir")
    return parser.parse_args()


def extract_archive(archive_path: Path, destination: Path) -> None:
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(destination)
        return

    with tarfile.open(archive_path, "r:*") as tf:
        tf.extractall(destination)


def resolve_dataset_root(base: Path, metadata_name: str) -> Path:
    if (base / metadata_name).exists():
        return base
    candidates = [p for p in base.glob("**/*") if p.is_dir() and (p / metadata_name).exists()]
    if not candidates:
        raise FileNotFoundError(f"Could not locate {metadata_name} under {base}")
    return sorted(candidates, key=lambda p: len(str(p)))[0]


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    dataset_root.parent.mkdir(parents=True, exist_ok=True)

    if args.source_dir is None and args.archive_url is None and args.archive_path is None:
        raise ValueError("Provide one of --source-dir, --archive-url, or --archive-path")

    if dataset_root.exists() and any(dataset_root.iterdir()):
        print(f"Dataset root already populated: {dataset_root}")
    else:
        if args.source_dir is not None:
            src = args.source_dir.resolve()
            if not src.exists():
                raise FileNotFoundError(f"Source directory not found: {src}")
            if args.copy:
                shutil.copytree(src, dataset_root, dirs_exist_ok=True)
            else:
                dataset_root.symlink_to(src, target_is_directory=True)
        else:
            archive_path = args.archive_path.resolve() if args.archive_path else dataset_root.parent / "emofake_download.tmp"
            if args.archive_url:
                print(f"Downloading archive: {args.archive_url}")
                urlretrieve(args.archive_url, archive_path)
            if not archive_path.exists():
                raise FileNotFoundError(f"Archive not found: {archive_path}")
            dataset_root.mkdir(parents=True, exist_ok=True)
            extract_archive(archive_path, dataset_root)

    materialized_root = dataset_root.resolve()
    located_root = resolve_dataset_root(materialized_root, args.metadata_name)

    metadata_path = located_root / args.metadata_name
    audio_dir = located_root / args.audio_dir_name
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Missing audio directory: {audio_dir}")

    print(f"Prepared dataset root: {located_root}")
    print(f"Metadata: {metadata_path}")
    print(f"Audio dir: {audio_dir}")


if __name__ == "__main__":
    main()
