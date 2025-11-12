from __future__ import annotations

import pytest

from thought_injector import spans

PROMPT = """System prompt\nTrial 1:\nAssistant: turn 1\nTrial 2:\nAssistant: turn 2\n"""


def test_locate_match_bounds_honors_occurrence() -> None:
    start, end = spans.locate_match_bounds(PROMPT, "Assistant:", 2, "--start-match")
    assert PROMPT[start:end] == "Assistant:"


def test_locate_match_bounds_raises_for_missing_occurrence() -> None:
    with pytest.raises(spans.AnchorError):
        spans.locate_match_bounds(PROMPT, "Assistant:", 3, "--start-match")


def test_locate_start_anchor_returns_newline_before_second_anchor() -> None:
    anchor = spans.locate_start_anchor(PROMPT, "Assistant:", 2)
    assert PROMPT[anchor] == "\n"
    assert PROMPT[anchor + 1 : anchor + 11] == "Assistant:"


def test_locate_end_anchor_without_newline_defaults_to_last_char() -> None:
    prompt = "Assistant: done"
    anchor = spans.locate_end_anchor(prompt, "Assistant: done", 1)
    assert anchor == len(prompt) - 1


def test_locate_end_anchor_targets_newline_after_match() -> None:
    match = "Assistant: turn 1"
    expected_char = PROMPT.find("\n", PROMPT.find(match) + len(match))
    anchor = spans.locate_end_anchor(PROMPT, match, 1)
    assert anchor == expected_char
