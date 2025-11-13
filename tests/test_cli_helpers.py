from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import typer

from thought_injector import cli
from thought_injector.injection import InjectionSchedule


def test_build_window_spec_enforces_validation() -> None:
    with pytest.raises(typer.BadParameter):
        cli._build_window_spec(
            start_index=None,
            end_index=None,
            start_match=None,
            end_match=None,
            start_occurrence=2,
            end_occurrence=1,
        )


def test_should_disable_cache_respects_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    schedule = InjectionSchedule(apply_all=True)
    dummy_model = SimpleNamespace()

    monkeypatch.setattr(cli, "requires_cache_disabled", lambda _: False)
    assert cli._should_disable_cache(dummy_model, schedule, has_injection=True) is True
    assert cli._should_disable_cache(dummy_model, schedule, has_injection=False) is False

    monkeypatch.setattr(cli, "requires_cache_disabled", lambda _: True)
    assert cli._should_disable_cache(dummy_model, schedule, has_injection=False) is True


def test_generate_text_with_schedule_toggles_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyTokenizer:
        def __init__(self) -> None:
            self.calls: list[tuple[list[int], bool]] = []

        def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool) -> str:
            self.calls.append((token_ids.tolist(), skip_special_tokens))
            return "decoded"

    class DummyModel:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.config = SimpleNamespace(model_type="llama")

        def generate(self, **kwargs: object) -> torch.Tensor:
            self.calls.append(kwargs)
            return torch.tensor([[0, 1]])

    injection_calls: list[tuple[int, float]] = []

    @contextmanager
    def fake_injection_context(
        model: object,
        layer_index: int,
        vector: torch.Tensor,
        strength: float,
        schedule: InjectionSchedule,
    ):
        injection_calls.append((layer_index, strength))
        yield

    monkeypatch.setattr(cli, "injection_context", fake_injection_context)
    monkeypatch.setattr(cli, "requires_cache_disabled", lambda _: False)

    model = DummyModel()
    tokenizer = DummyTokenizer()
    schedule = InjectionSchedule(apply_all=True)
    generation_config = SimpleNamespace()

    no_vector_text = cli._generate_text_with_schedule(
        model=model,
        tokenizer=tokenizer,
        inputs={"input_ids": torch.tensor([[1]])},
        generation_config=generation_config,
        schedule=schedule,
        vector=None,
        layer_index=None,
        strength=None,
        include_prompt=False,
    )
    assert no_vector_text == "decoded"
    assert model.calls[-1]["use_cache"] is True
    assert injection_calls == []

    injected_text = cli._generate_text_with_schedule(
        model=model,
        tokenizer=tokenizer,
        inputs={"input_ids": torch.tensor([[1]])},
        generation_config=generation_config,
        schedule=schedule,
        vector=torch.ones(2),
        layer_index=0,
        strength=0.5,
        include_prompt=False,
    )
    assert injected_text == "decoded"
    assert model.calls[-1]["use_cache"] is False
    assert injection_calls == [(0, 0.5)]
