from __future__ import annotations

from pathlib import Path

import pytest
import torch
import typer

from thought_injector import cli


def test_capture_word_raises_when_baseline_file_only_contains_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_path = tmp_path / "baseline.txt"
    baseline_path.write_text("Whales\n")

    def _fake_load_model_and_tokenizer(*args, **kwargs) -> tuple[object, object]:
        return object(), object()

    def _fake_extract_hidden_state(*args, **kwargs) -> torch.Tensor:
        return torch.zeros(4)

    monkeypatch.setattr(cli, "load_model_and_tokenizer", _fake_load_model_and_tokenizer)
    monkeypatch.setattr(cli, "extract_hidden_state", _fake_extract_hidden_state)

    with pytest.raises(typer.BadParameter, match="Baseline word list is empty"):
        cli.capture_word(
            model_path=tmp_path,
            word="Whales",
            layer_index=0,
            token_index=-1,
            baseline_path=baseline_path,
            baseline_count=1,
            output_path=tmp_path / "out.pt",
            dtype="auto",
            device="auto",
        )
