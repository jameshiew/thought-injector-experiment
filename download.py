#!/usr/bin/env python3
"""Utility to materialize the local models/ directory used in experiments."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HF_BASE = "https://huggingface.co/microsoft/Phi-4-mini-instruct/resolve/main/{filename}?download=1"
CHUNK_SIZE = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class FileSpec:
    name: str
    sha256: str


FILE_SPECS = (
    FileSpec("config.json", "ac65d86061d3d0d704ee2511fd0eb8713ef19eb6eedba17c3080a4165d5b933b"),
    FileSpec("configuration_phi3.py", "ec2044b77d0b8111640ac134daa3af1f40dc552a739f8959626e7e58ea3352df"),
    FileSpec("generation_config.json", "3e3f48753753f92d2b958679151861d4fd7bf26e4dfc41fd47056116c4914dcd"),
    FileSpec("model.safetensors.index.json", "613a98d5e5716ca96fa75931abedc9c5a5d95f488ce4d62df71e639fe3ac6c59"),
    FileSpec("model-00001-of-00002.safetensors", "bc703090b63eda16f639fa4de7ac54635c23105ab1da2f6ec4d3403151d38ee6"),
    FileSpec("model-00002-of-00002.safetensors", "7ff79b9d2d31076bac2663393451f6530f4fc8ca49b09002116c92c373dba983"),
    FileSpec("modeling_phi3.py", "d5fd551a7fe759b1d25e5b16b5012aa98b9b8bd400d26c7ee69c0090a245d6cd"),
    FileSpec("special_tokens_map.json", "aff38493227d813e29fcf8406e8e90062f1f031aa47d589325e9c31d89ac7cc3"),
    FileSpec("tokenizer.json", "382cc235b56c725945e149cc25f191da667c836655efd0857b004320e90e91ea"),
    FileSpec("tokenizer_config.json", "9c9b6bc0c94d95f69f826c41069a3e8b387ac3ced89601d201886e99240ac9db"),
)


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()



def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".partial")
    start = time.time()
    print(f"[download] {dest.name} <- {url}")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded * 100 // total
                sys.stdout.write(f"\r    {downloaded / (1<<20):.2f} MiB / {total / (1<<20):.2f} MiB ({percent}%)")
                sys.stdout.flush()
    tmp_path.rename(dest)
    elapsed = time.time() - start
    sys.stdout.write(f"\r    done in {elapsed:.1f}s{' ' * 20}\n")
    sys.stdout.flush()


def ensure_file(spec: FileSpec, target_dir: Path, force: bool) -> None:
    dest = target_dir / spec.name
    if dest.exists() and not force:
        current = sha256sum(dest)
        if current == spec.sha256:
            print(f"[skip] {dest.name} already matches expected hash")
            return
        print(f"[warn] {dest.name} hash mismatch ({current}); redownloading")

    url = HF_BASE.format(filename=spec.name)
    download_file(url, dest)
    actual = sha256sum(dest)
    if actual != spec.sha256:
        raise RuntimeError(f"Hash mismatch for {dest.name}: expected {spec.sha256}, got {actual}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download the Phi-4 mini instruct weights + metadata locally.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Root directory where model folders will be stored (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if the existing hash already matches.",
    )
    args = parser.parse_args(argv)

    target_dir = (args.models_dir / "phi-4-mini-instruct").resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing files to {target_dir}")

    for spec in FILE_SPECS:
        ensure_file(spec, target_dir, args.force)

    print("All files present. models/ directory matches the expected layout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
