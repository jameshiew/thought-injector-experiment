#!/usr/bin/env python3
"""Download supported Hugging Face model assets into the local models/ directory."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

CHUNK_SIZE = 1 << 20  # 1 MiB
SUFFIX_ALLOWLIST = (".json", ".py", ".safetensors", ".model", ".txt", ".pt", ".bin")


@dataclass(frozen=True)
class FileSpec:
    name: str
    sha256: str | None


class StaticSpec(TypedDict):
    repo_id: str
    files: tuple[FileSpec, ...] | None


STATIC_SPECS: dict[str, StaticSpec] = {
    "pharia-1-control": {
        "repo_id": "Aleph-Alpha/Pharia-1-LLM-7B-control-hf",
        "files": None,  # Enumerated dynamically; repo is fully public.
    },
    "llama-3.1-8b-instruct": {
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "files": None,  # Requires a gated token; enumerate dynamically once authenticated.
    },
    "phi-4-mini-instruct": {
        "repo_id": "microsoft/Phi-4-mini-instruct",
        "files": (
            FileSpec(
                "config.json", "ac65d86061d3d0d704ee2511fd0eb8713ef19eb6eedba17c3080a4165d5b933b"
            ),
            FileSpec(
                "configuration_phi3.py",
                "ec2044b77d0b8111640ac134daa3af1f40dc552a739f8959626e7e58ea3352df",
            ),
            FileSpec(
                "generation_config.json",
                "3e3f48753753f92d2b958679151861d4fd7bf26e4dfc41fd47056116c4914dcd",
            ),
            FileSpec(
                "model.safetensors.index.json",
                "613a98d5e5716ca96fa75931abedc9c5a5d95f488ce4d62df71e639fe3ac6c59",
            ),
            FileSpec(
                "model-00001-of-00002.safetensors",
                "bc703090b63eda16f639fa4de7ac54635c23105ab1da2f6ec4d3403151d38ee6",
            ),
            FileSpec(
                "model-00002-of-00002.safetensors",
                "7ff79b9d2d31076bac2663393451f6530f4fc8ca49b09002116c92c373dba983",
            ),
            FileSpec(
                "modeling_phi3.py",
                "d5fd551a7fe759b1d25e5b16b5012aa98b9b8bd400d26c7ee69c0090a245d6cd",
            ),
            FileSpec(
                "special_tokens_map.json",
                "aff38493227d813e29fcf8406e8e90062f1f031aa47d589325e9c31d89ac7cc3",
            ),
            FileSpec(
                "tokenizer.json", "382cc235b56c725945e149cc25f191da667c836655efd0857b004320e90e91ea"
            ),
            FileSpec(
                "tokenizer_config.json",
                "9c9b6bc0c94d95f69f826c41069a3e8b387ac3ced89601d201886e99240ac9db",
            ),
        ),
    },
    "phi-4": {
        "repo_id": "microsoft/phi-4",
        "files": None,  # Will be enumerated dynamically (requires token).
    },
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_specs(repo_key: str, token: str | None) -> tuple[FileSpec, ...]:
    """Return the list of FileSpec entries for a repo, fetching Hugging Face metadata if needed."""
    config: StaticSpec = STATIC_SPECS[repo_key]
    files = config["files"]
    if files is not None:
        return files

    api = HfApi(token=token)
    try:
        repo = api.repo_info(config["repo_id"], files_metadata=True)
    except HfHubHTTPError as err:
        if err.response is not None and err.response.status_code in (401, 403):
            raise SystemExit(
                f"Repository {config['repo_id']} requires authentication. "
                "Set HF_TOKEN or pass --token to authenticate."
            ) from err
        raise

    siblings = repo.siblings
    if siblings is None:
        raise SystemExit(
            "Hugging Face did not return file metadata; provide a token via HF_TOKEN or --token."
        )

    specs: list[FileSpec] = []
    for sibling in siblings:
        if not sibling.rfilename.endswith(SUFFIX_ALLOWLIST):
            continue
        sha = sibling.lfs.sha256 if sibling.lfs else None
        specs.append(FileSpec(sibling.rfilename, sha))
    if not specs:
        suffixes = ", ".join(SUFFIX_ALLOWLIST)
        raise SystemExit(
            f"No downloadable files with suffixes ({suffixes}) found for {config['repo_id']}."
        )
    return tuple(specs)


def download_file(repo_id: str, filename: str, dest: Path, token: str | None) -> None:
    """Stream a single file from Hugging Face and atomically move it into place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".partial")
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}?download=1"
    headers = {"User-Agent": "thought-injector/0.1 download.py"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    print(f"[download] {dest.name}")
    start = time.time()
    downloaded = 0
    total = 0
    try:
        with urllib.request.urlopen(request) as response, tmp_path.open("wb") as handle:
            total = int(response.headers.get("Content-Length", "0"))
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    sys.stdout.write(
                        f"\r    {downloaded / (1 << 20):.2f} MiB / {total / (1 << 20):.2f} MiB ({pct}%)"
                    )
                    sys.stdout.flush()
        tmp_path.rename(dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    elapsed = time.time() - start
    sys.stdout.write(
        f"\r[done] {dest.name} in {elapsed:.1f}s"
        f" ({downloaded / (1 << 20):.2f} MiB)" + " " * 8 + "\n"
    )


def ensure_file(
    repo_id: str, spec: FileSpec, target_dir: Path, force: bool, token: str | None
) -> None:
    """Skip downloads when possible, otherwise fetch and optionally verify SHA256."""
    dest = target_dir / spec.name
    if dest.exists() and not force:
        if spec.sha256:
            if sha256sum(dest) == spec.sha256:
                print(f"[skip] {spec.name} already verified")
                return
            print(f"[warn] {spec.name} hash mismatch; redownloading")
        else:
            print(f"[skip] {spec.name} already present")
            return

    download_file(repo_id, spec.name, dest, token)
    if spec.sha256:
        actual = sha256sum(dest)
        if actual != spec.sha256:
            raise RuntimeError(
                f"Hash mismatch for {spec.name}: expected {spec.sha256}, got {actual}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download model assets from Hugging Face")
    parser.add_argument("model", choices=tuple(STATIC_SPECS.keys()))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--token", default=None, help="HF access token (falls back to HF_TOKEN env var)"
    )
    args = parser.parse_args(argv)

    repo_key = args.model
    config: StaticSpec = STATIC_SPECS[repo_key]
    repo_id = config["repo_id"]
    token = args.token or os.getenv("HF_TOKEN")
    specs = resolve_specs(repo_key, token)

    target_dir = (args.models_dir / repo_key).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing files to {target_dir}")

    for spec in specs:
        ensure_file(repo_id, spec, target_dir, args.force, token)

    print("All files present. models/ directory matches the expected layout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
