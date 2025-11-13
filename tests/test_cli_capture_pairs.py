from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import typer

from thought_injector import cli


def _write_pairs_file(path: Path) -> None:
    payload = [
        {
            "positive": "A dog is chasing a ball.",
            "negative": "A person is chasing a ball.",
        },
        {
            "positive": "The dog curls up beside the fire.",
            "negative": "The person curls up beside the fire.",
        },
    ]
    path.write_text("\n".join(json.dumps(obj) for obj in payload) + "\n", encoding="utf-8")


def _fake_extract(prompt: str) -> torch.Tensor:
    lookup: dict[str, torch.Tensor] = {
        "A dog is chasing a ball.": torch.tensor([2.0, 1.0]),
        "A person is chasing a ball.": torch.tensor([1.0, 1.0]),
        "The dog curls up beside the fire.": torch.tensor([0.0, 3.0]),
        "The person curls up beside the fire.": torch.tensor([0.0, 1.0]),
    }
    try:
        return lookup[prompt]
    except KeyError:  # pragma: no cover - defensive guard for unexpected prompts.
        raise AssertionError(f"Unexpected prompt {prompt!r} in test harness")


def test_capture_pairs_averages_differences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    _write_pairs_file(pairs_path)

    saved: dict[str, Any] = {}

    def _fake_load_model_and_tokenizer(*args, **kwargs) -> tuple[object, object]:
        return object(), object()

    def _fake_extract_hidden_state(*args, **kwargs) -> torch.Tensor:
        prompt = kwargs.get("prompt")
        if prompt is None:
            prompt = args[2]
        return _fake_extract(prompt)

    def _fake_save_vector(path: Path, vector: torch.Tensor, metadata: Any) -> None:
        saved["path"] = path
        saved["vector"] = vector.clone()
        saved["metadata"] = metadata

    monkeypatch.setattr(cli, "load_model_and_tokenizer", _fake_load_model_and_tokenizer)
    monkeypatch.setattr(cli, "extract_hidden_state", _fake_extract_hidden_state)
    monkeypatch.setattr(cli, "save_vector", _fake_save_vector)

    cli.capture_pairs(
        model_path=tmp_path,
        pairs_path=pairs_path,
        layer_index=3,
        token_index=-1,
        max_pairs=None,
        output_path=tmp_path / "dog_vector.safetensors",
        dtype="auto",
        device="auto",
    )

    assert torch.allclose(saved["vector"], torch.tensor([0.5, 1.0]))
    metadata = saved["metadata"]
    assert metadata["layer_index"] == 3
    assert metadata["pair_count"] == 2
    assert metadata["pairs_source"].endswith("pairs.jsonl")
    assert metadata["prompts"]["pairs"][0]["positive"] == "A dog is chasing a ball."


def test_capture_pairs_respects_max_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    _write_pairs_file(pairs_path)

    saved: dict[str, Any] = {}

    monkeypatch.setattr(cli, "load_model_and_tokenizer", lambda *_, **__: (object(), object()))

    def _fake_extract_hidden_state(*args, **kwargs) -> torch.Tensor:
        prompt = kwargs.get("prompt")
        if prompt is None:
            prompt = args[2]
        return _fake_extract(prompt)

    def _fake_save_vector(path: Path, vector: torch.Tensor, metadata: Any) -> None:
        saved["vector"] = vector.clone()
        saved["metadata"] = metadata

    monkeypatch.setattr(cli, "extract_hidden_state", _fake_extract_hidden_state)
    monkeypatch.setattr(cli, "save_vector", _fake_save_vector)

    cli.capture_pairs(
        model_path=tmp_path,
        pairs_path=pairs_path,
        layer_index=1,
        token_index=-2,
        max_pairs=1,
        output_path=tmp_path / "dog_vector.safetensors",
        dtype="auto",
        device="auto",
    )

    # Only the first pair contributes diff == [1, 0].
    assert torch.allclose(saved["vector"], torch.tensor([1.0, 0.0]))
    assert saved["metadata"]["pair_count"] == 1


def test_capture_pairs_validates_max_pairs(tmp_path: Path) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    _write_pairs_file(pairs_path)

    with pytest.raises(typer.BadParameter, match="max-pairs must be positive"):
        cli.capture_pairs(
            model_path=tmp_path,
            pairs_path=pairs_path,
            layer_index=0,
            token_index=-1,
            max_pairs=0,
            output_path=tmp_path / "bad.safetensors",
            dtype="auto",
            device="auto",
        )
