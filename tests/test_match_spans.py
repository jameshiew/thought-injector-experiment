from __future__ import annotations

import pytest

from thought_injector import spans
from thought_injector.text_utils import token_index_from_char


class FakeEncoding(dict[str, object]):
    def __init__(self, data: dict[str, object]):
        super().__init__(data)
        self.data = data


class CharTokenizer:
    def __call__(
        self,
        prompt: str,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ) -> FakeEncoding:
        token_ids = list(range(len(prompt)))
        data: dict[str, object] = {"input_ids": [token_ids]}
        if return_offsets_mapping:
            offsets = [(idx, idx + 1) for idx in range(len(prompt))]
            data["offset_mapping"] = [offsets]
        return FakeEncoding(data)


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


def test_locate_end_anchor_without_newline_returns_past_end_sentinel() -> None:
    prompt = "Assistant: done"
    anchor = spans.locate_end_anchor(prompt, "Assistant: done", 1)
    assert anchor == len(prompt)


def test_locate_end_anchor_targets_newline_after_match() -> None:
    match = "Assistant: turn 1"
    expected_char = PROMPT.find("\n", PROMPT.find(match) + len(match))
    anchor = spans.locate_end_anchor(PROMPT, match, 1)
    assert anchor == expected_char


def test_end_match_anchor_maps_to_last_token_without_trailing_newline() -> None:
    prompt = "Assistant: done"
    tokenizer = CharTokenizer()
    anchor = spans.locate_end_anchor(prompt, "Assistant: done", 1)
    token_index = token_index_from_char(tokenizer, prompt, anchor)
    assert token_index == len(prompt) - 1
