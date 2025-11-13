from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from thought_injector.injection import InjectionSchedule


def test_generated_only_requires_prompt_length() -> None:
    with pytest.raises(ValidationError):
        InjectionSchedule(generated_only=True)


def test_schedule_accepts_valid_configuration() -> None:
    schedule = InjectionSchedule(generated_only=True, prompt_length=4)
    assert schedule.has_window() is False
    # prompt_length propagates into resolved span computation helper
    assert schedule.resolved_span(6) == (4, 5)


def test_resolved_span_generated_only_with_window() -> None:
    schedule = InjectionSchedule(
        window_start=1,
        window_end=4,
        generated_only=True,
        prompt_length=3,
    )
    # window 1..4 intersect generated-only (>=3) => 3..4
    assert schedule.resolved_span(8) == (3, 4)


def test_resolved_span_generated_only_can_be_empty() -> None:
    schedule = InjectionSchedule(generated_only=True, prompt_length=5)
    # prompt fully covers sequence, so no span remains
    assert schedule.resolved_span(5) is None


def test_resolve_mask_handles_core_modes() -> None:
    device = torch.device("cpu")

    schedule_all = InjectionSchedule(apply_all=True)
    assert torch.equal(
        schedule_all.resolve_mask(4, device), torch.tensor([1, 1, 1, 1], dtype=torch.bool)
    )

    window_schedule = InjectionSchedule(window_start=1, window_end=3)
    expected_window = torch.tensor([0, 1, 1, 1, 0], dtype=torch.bool)
    assert torch.equal(window_schedule.resolve_mask(5, device), expected_window)

    single_schedule = InjectionSchedule(single_index=-1)
    expected_single = torch.tensor([0, 0, 0, 1], dtype=torch.bool)
    assert torch.equal(single_schedule.resolve_mask(4, device), expected_single)

    gen_only_schedule = InjectionSchedule(generated_only=True, prompt_length=2)
    expected_gen = torch.tensor([0, 0, 1, 1], dtype=torch.bool)
    assert torch.equal(gen_only_schedule.resolve_mask(4, device), expected_gen)

    gen_single_schedule = InjectionSchedule(generated_only=True, prompt_length=1, single_index=1)
    expected_gen_single = torch.tensor([0, 1, 0], dtype=torch.bool)
    assert torch.equal(gen_single_schedule.resolve_mask(3, device), expected_gen_single)

    empty_mask = schedule_all.resolve_mask(0, device)
    assert empty_mask.shape == (0,)
    assert empty_mask.dtype == torch.bool
