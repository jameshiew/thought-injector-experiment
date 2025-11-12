from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.tokenization_utils_base import BatchEncoding

from thought_injector.models import get_decoder_layers, tokenize


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
