from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import typer
from transformers import PreTrainedModel

from thought_injector.app import console


@dataclass
class VectorRecord:
    vector: torch.Tensor
    metadata: dict[str, Any]


def save_vector(path: Path, vector: torch.Tensor, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"vector": vector, "metadata": metadata}
    torch.save(payload, path)
    console.print(f"Saved vector -> {path}")


def load_vector(path: Path) -> VectorRecord:
    payload = torch.load(path, map_location="cpu")
    if "vector" not in payload:
        raise typer.BadParameter(f"Vector file {path} missing 'vector' key")
    metadata = payload.get("metadata", {})
    return VectorRecord(vector=payload["vector"], metadata=metadata)


def broadcast_vector(vector: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    return vector.to(dtype=hidden_states.dtype, device=hidden_states.device)


def normalize_vector(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(vector.to(torch.float32) ** 2))
    if torch.isnan(rms) or rms.item() < eps:
        raise typer.BadParameter("Vector has near-zero RMS; cannot normalize.")
    return vector / rms


def prepare_vector(vector: torch.Tensor, normalize: bool, scale_by: float) -> torch.Tensor:
    result = vector.to(torch.float32)
    if normalize:
        result = normalize_vector(result)
    if scale_by != 1.0:
        result = result * scale_by
    return result


def ensure_vector_matches_model(vector: torch.Tensor, model: PreTrainedModel) -> None:
    hidden_size = model.config.hidden_size
    if vector.shape[-1] != hidden_size:
        raise typer.BadParameter(
            f"Vector hidden size {vector.shape[-1]} != model hidden size {hidden_size}."
        )
