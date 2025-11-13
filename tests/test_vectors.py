from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import typer

from thought_injector.vectors import (
    ensure_vector_matches_model,
    load_prepared_vector,
    load_vector,
    save_vector,
)


def test_save_and_load_vector_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "concept.pt"
    vector = torch.ones(2, 4)
    metadata = {
        "model_path": "./model",
        "layer_index": 1,
        "token_index": -1,
        "prompts": {"positive": "Tell me X"},
    }

    save_vector(path, vector, metadata)

    record = load_vector(path)
    assert torch.equal(record.vector, vector)
    assert record.metadata.layer_index == 1
    assert record.metadata.prompts == {"positive": "Tell me X"}


def test_save_vector_rejects_invalid_metadata(tmp_path: Path) -> None:
    path = tmp_path / "invalid.pt"
    vector = torch.zeros(1, 3)

    with pytest.raises(typer.BadParameter):
        save_vector(path, vector, {"layer_index": -4})


def test_ensure_vector_matches_model_rejects_non_1d_vectors() -> None:
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=4))
    vector = torch.zeros(2, 4)

    with pytest.raises(typer.BadParameter):
        ensure_vector_matches_model(vector, model)


def test_load_prepared_vector_normalizes_and_scales(tmp_path: Path) -> None:
    path = tmp_path / "prepared.pt"
    vector = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float32)
    save_vector(path, vector, {"layer_index": 0})

    model = SimpleNamespace(config=SimpleNamespace(hidden_size=3))
    prepared = load_prepared_vector(path, model, normalize=True, scale_by=2.0)

    rms = torch.sqrt(torch.mean(vector**2))
    expected = (vector / rms) * 2.0
    assert torch.allclose(prepared.tensor, expected)
    assert prepared.metadata.layer_index == 0


def test_load_prepared_vector_validates_hidden_size(tmp_path: Path) -> None:
    path = tmp_path / "mismatch.pt"
    vector = torch.tensor([1.0, 2.0], dtype=torch.float32)
    save_vector(path, vector, {"layer_index": 0})

    model = SimpleNamespace(config=SimpleNamespace(hidden_size=3))
    with pytest.raises(typer.BadParameter):
        load_prepared_vector(path, model, normalize=False, scale_by=1.0)
