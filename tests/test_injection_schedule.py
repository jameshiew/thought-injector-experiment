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
    assert schedule.resolved_span(4) == (0, 3)
