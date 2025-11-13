from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
import typer
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from safetensors import torch as safetensors_torch
from transformers import PreTrainedModel

from thought_injector.app import console


class VectorMetadata(BaseModel):
    """Structured metadata persisted alongside safetensors vectors."""

    model_config = ConfigDict(extra="allow")

    model_path: str | None = None
    layer_index: int | None = Field(default=None, ge=0)
    token_index: int | None = None
    word: str | None = None
    baseline_count: int | None = Field(default=None, ge=0)
    baseline_source: str | None = None
    prompts: dict[str, Any] | None = None


@dataclass
class VectorRecord:
    vector: torch.Tensor
    metadata: VectorMetadata


@dataclass
class PreparedVector:
    """Wrapper bundling a normalized vector with its original metadata."""

    tensor: torch.Tensor
    metadata: VectorMetadata


def save_vector(
    path: Path, vector: torch.Tensor, metadata: Mapping[str, Any] | VectorMetadata
) -> None:
    _require_safetensors_extension(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        metadata_model = (
            metadata
            if isinstance(metadata, VectorMetadata)
            else VectorMetadata.model_validate(metadata)
        )
    except ValidationError as err:  # pragma: no cover - defensive guard.
        raise typer.BadParameter(f"Invalid vector metadata: {err}") from err

    temp_tensor_path = path.with_suffix(path.suffix + ".tmp")
    metadata_path = path.with_suffix(".json")
    temp_metadata_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    tensor_to_save = vector.detach().cpu()
    try:
        save_file({"vector": tensor_to_save}, str(temp_tensor_path))
        json_payload = metadata_model.model_dump(mode="json")
        temp_metadata_path.write_text(json.dumps(json_payload, indent=2) + "\n", encoding="utf-8")
        temp_metadata_path.replace(metadata_path)
        temp_tensor_path.replace(path)
    finally:
        temp_tensor_path.unlink(missing_ok=True)
        temp_metadata_path.unlink(missing_ok=True)

    console.print(f"Saved vector -> {path} (+ metadata {metadata_path.name})")


def load_vector(path: Path) -> VectorRecord:
    _require_safetensors_extension(path)
    if not path.exists():
        raise typer.BadParameter(f"Vector file {path} not found.")
    tensors = load_file(str(path), device="cpu")
    if "vector" not in tensors:
        raise typer.BadParameter(f"Vector file {path} is missing the 'vector' tensor.")

    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise typer.BadParameter(f"Metadata file {metadata_path} not found for vector {path}.")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata_payload = json.load(handle)

    try:
        metadata_model = VectorMetadata.model_validate(metadata_payload)
    except ValidationError as err:
        raise typer.BadParameter(f"Vector metadata {metadata_path} is invalid: {err}") from err
    return VectorRecord(vector=tensors["vector"], metadata=metadata_model)


def load_prepared_vector(
    path: Path,
    model: PreTrainedModel,
    *,
    normalize: bool,
    scale_by: float,
) -> PreparedVector:
    """Load, validate, and scale a concept vector for a specific model."""

    record = load_vector(path)
    ensure_vector_matches_model(record.vector, model)
    tensor = prepare_vector(record.vector, normalize, scale_by)
    return PreparedVector(tensor=tensor, metadata=record.metadata)


def broadcast_vector(vector: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    """Return the vector on the same dtype/device as hidden_states without changing shape."""
    return vector.to(dtype=hidden_states.dtype, device=hidden_states.device)


def normalize_vector(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(vector.to(torch.float32) ** 2))
    if torch.isnan(rms) or rms.item() < eps:
        raise typer.BadParameter(f"Vector has near-zero RMS ({rms.item():.3e}); cannot normalize.")
    return vector / rms


def prepare_vector(vector: torch.Tensor, normalize: bool, scale_by: float) -> torch.Tensor:
    result = vector.to(torch.float32)
    if normalize:
        result = normalize_vector(result)
    if scale_by != 1.0:
        result = result * scale_by
    return result


def ensure_vector_matches_model(vector: torch.Tensor, model: PreTrainedModel) -> None:
    """Ensure a vector is 1-D with length model.config.hidden_size before injection."""
    hidden_size = model.config.hidden_size
    if vector.ndim != 1:
        raise typer.BadParameter(
            f"Vector must be 1-D with length equal to model.hidden_size; got shape {tuple(vector.shape)}."
        )
    if vector.shape[-1] != hidden_size:
        raise typer.BadParameter(
            f"Vector hidden size {vector.shape[-1]} != model hidden size {hidden_size}."
        )


def _require_safetensors_extension(path: Path) -> None:
    if path.suffix.lower() != ".safetensors":
        raise typer.BadParameter(f"Vectors must use the .safetensors extension; got {path.name}.")


class _LoadFileFn(Protocol):
    def __call__(
        self, filename: str | os.PathLike[str], device: str | int = "cpu"
    ) -> dict[str, torch.Tensor]: ...


class _SaveFileFn(Protocol):
    def __call__(
        self,
        tensors: Mapping[str, torch.Tensor],
        filename: str | os.PathLike[str],
        metadata: Mapping[str, str] | None = None,
    ) -> None: ...


load_file: _LoadFileFn = cast(_LoadFileFn, safetensors_torch.load_file)
save_file: _SaveFileFn = cast(_SaveFileFn, safetensors_torch.save_file)
