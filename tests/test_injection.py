from __future__ import annotations

from types import SimpleNamespace

import torch

from thought_injector.injection import _remix_output


def _increment(tensor: torch.Tensor) -> torch.Tensor:
    return tensor + 1


def test_remix_output_handles_tensor() -> None:
    original = torch.zeros(2, 2)
    remixed = _remix_output(original, _increment)
    assert torch.equal(remixed, torch.ones_like(original))


def test_remix_output_handles_tuple_output() -> None:
    original = (torch.zeros(1, 1), torch.ones(1, 1))
    remixed = _remix_output(original, _increment)
    assert torch.equal(remixed[0], torch.ones_like(original[0]))
    assert torch.equal(remixed[1], original[1])


def test_remix_output_handles_last_hidden_state_attr() -> None:
    original = torch.zeros(1, 1, 1)
    payload = SimpleNamespace(last_hidden_state=original.clone())
    remixed = _remix_output(payload, _increment)
    assert torch.equal(remixed.last_hidden_state, torch.ones_like(original))


def test_remix_output_handles_hidden_states_attr_tuple() -> None:
    first = torch.zeros(1, 1, 1)
    second = torch.ones(1, 1, 1)
    payload = SimpleNamespace(hidden_states=(first.clone(), second))
    remixed = _remix_output(payload, _increment)
    assert torch.equal(remixed.hidden_states[0], torch.ones_like(first))
    assert torch.equal(remixed.hidden_states[1], second)


def test_remix_output_handles_dict_hidden_states() -> None:
    first = torch.zeros(1, 1, 1)
    payload = {"hidden_states": (first.clone(),)}
    remixed = _remix_output(payload, _increment)
    assert torch.equal(remixed["hidden_states"][0], torch.ones_like(first))


def test_remix_output_handles_dict_last_hidden_state() -> None:
    tensor = torch.zeros(1, 1, 2)
    payload = {"last_hidden_state": tensor.clone()}
    remixed = _remix_output(payload, _increment)
    assert torch.equal(remixed["last_hidden_state"], torch.ones_like(tensor))
