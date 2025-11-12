from __future__ import annotations

import torch

from thought_injector.injection import InjectionSchedule


def test_generated_only_without_single_index_targets_all_generated_tokens() -> None:
    schedule = InjectionSchedule(generated_only=True, prompt_length=5)
    mask = schedule.resolve_mask(seq_len=8, device=torch.device("cpu"))
    expected = torch.zeros(8, dtype=torch.bool)
    expected[5:] = True
    assert torch.equal(mask.cpu(), expected)


def test_generated_only_with_explicit_single_index_targets_last_token() -> None:
    schedule = InjectionSchedule(generated_only=True, prompt_length=5, single_index=-1)
    mask = schedule.resolve_mask(seq_len=8, device=torch.device("cpu"))
    expected = torch.zeros(8, dtype=torch.bool)
    expected[-1] = True
    assert torch.equal(mask.cpu(), expected)
