from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import typer
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from transformers import PreTrainedModel

from thought_injector.app import console


class VectorMetadata(BaseModel):
    """Structured metadata stored alongside vector tensors."""

    model_config = ConfigDict(extra="allow")

    model_path: str | None = None
    layer_index: int | None = Field(default=None, ge=0)
    token_index: int | None = None
    word: str | None = None
    baseline_count: int | None = Field(default=None, ge=0)
    baseline_source: str | None = None
    prompts: dict[str, Any] | None = None


class VectorPayload(BaseModel):
    """Serialized payload saved to disk via torch.save."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector: torch.Tensor
    metadata: VectorMetadata = Field(default_factory=VectorMetadata)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        metadata_model = (
            metadata
            if isinstance(metadata, VectorMetadata)
            else VectorMetadata.model_validate(metadata)
        )
    except ValidationError as err:  # pragma: no cover - defensive guard.
        raise typer.BadParameter(f"Invalid vector metadata: {err}") from err

    payload = VectorPayload(vector=vector, metadata=metadata_model)
    torch.save(
        {
            "vector": payload.vector,
            "metadata": payload.metadata.model_dump(mode="python"),
        },
        path,
    )
    console.print(f"Saved vector -> {path}")


def load_vector(path: Path) -> VectorRecord:
    payload = torch.load(path, map_location="cpu")
    try:
        validated = VectorPayload.model_validate(payload)
    except ValidationError as err:
        raise typer.BadParameter(f"Vector file {path} is invalid: {err}") from err
    return VectorRecord(vector=validated.vector, metadata=validated.metadata)


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
    if vector.ndim != 1:
        raise typer.BadParameter(
            f"Vector must be 1-D with length equal to model.hidden_size; got shape {tuple(vector.shape)}."
        )
    if vector.shape[-1] != hidden_size:
        raise typer.BadParameter(
            f"Vector hidden size {vector.shape[-1]} != model hidden size {hidden_size}."
        )
