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
    FileSpec("modeling_phi3.py", "7c3c13e0af6fc3b75d6ce9d9564d5bc79772c6b6fcfaefb4f8351247120809e5"),
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


def maybe_patch_modeling(path: Path) -> None:
    """Apply the LossKwargs shim so the model runs on slightly older transformers."""

    text = path.read_text(encoding="utf-8")
    if "class LossKwargs(TypedDict" in text:
        return  # Already patched.

    original_typing = "from typing import Callable, List, Optional, Tuple, Union"
    patched_typing = "from typing import Callable, List, Optional, Tuple, TypedDict, Union"
    if original_typing in text:
        text = text.replace(original_typing, patched_typing, 1)

    loss_line = "    LossKwargs,\n"
    if loss_line in text:
        text = text.replace(loss_line, "", 1)

    shim = (
        "\ntry:  # Newer transformers expose LossKwargs; define a stub for older builds.\n"
        "    from transformers.utils import LossKwargs\n"
        "except ImportError:  # pragma: no cover - compatibility shim.\n"
        "    class LossKwargs(TypedDict, total=False):  # type: ignore\n"
        "        \"\"\"Minimal shim so Phi-3 modules can run on slightly older transformers.\"\"\"\n\n"
        "        pass\n"
    )

    anchor = "from transformers.utils.deprecation import deprecate_kwarg"
    if anchor not in text:
        raise RuntimeError("modeling_phi3.py format changed; please update the shim logic.")
    text = text.replace(anchor, shim + anchor, 1)
    path.write_text(text, encoding="utf-8")


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
    if spec.name == "modeling_phi3.py":
        maybe_patch_modeling(dest)
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
