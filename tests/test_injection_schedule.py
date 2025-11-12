from __future__ import annotations

import pytest
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
