from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import typer
from transformers.tokenization_utils_base import BatchEncoding

from thought_injector.models import extract_hidden_state, get_decoder_layers, tokenize


class FakeTokenizer:
    def __call__(self, prompt: str, return_tensors: str = "pt") -> BatchEncoding:
        data = {
            "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
            "token_type_ids": torch.zeros((1, 2), dtype=torch.long),
        }
        return BatchEncoding(data=data)


def test_tokenize_keeps_attention_mask_even_when_all_ones() -> None:
    tokenizer = FakeTokenizer()
    tensors = tokenize(tokenizer, "prompt", torch.device("cpu"))
    assert "attention_mask" in tensors
    assert torch.equal(tensors["attention_mask"], torch.ones((1, 2), dtype=torch.long))
    assert "token_type_ids" not in tensors


class FakeLayers(torch.nn.ModuleList):
    pass


def test_get_decoder_layers_handles_nested_model_attribute() -> None:
    layers = FakeLayers([torch.nn.Linear(1, 1)])
    nested = SimpleNamespace(model=SimpleNamespace(layers=layers))
    fake_model = SimpleNamespace(model=nested)

    resolved = get_decoder_layers(fake_model)  # type: ignore[arg-type]
    assert resolved is layers


class DummyModel:
    def __init__(self, num_layers: int = 2) -> None:
        self.config = SimpleNamespace(num_hidden_layers=num_layers)

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        input_ids = kwargs["input_ids"]
        batch, seq_len = input_ids.shape
        hidden_states = tuple(
            torch.zeros((batch, seq_len, 4), dtype=torch.float32)
            for _ in range(self.config.num_hidden_layers + 1)
        )
        return SimpleNamespace(hidden_states=hidden_states)


def test_extract_hidden_state_rejects_negative_layer_indices() -> None:
    model = DummyModel()
    tokenizer = FakeTokenizer()
    with pytest.raises(typer.BadParameter, match="layer-index must be in"):
        extract_hidden_state(
            model,
            tokenizer,
            "prompt",
            layer_index=-1,
            token_index=-1,
            device=torch.device("cpu"),
        )
