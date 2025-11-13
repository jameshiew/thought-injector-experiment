from __future__ import annotations

import pytest
import typer

from thought_injector import spans
from thought_injector.text_utils import WindowSpec, token_index_from_char


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


class ZeroWidthTokenizer:
    def __call__(
        self,
        prompt: str,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ) -> FakeEncoding:
        printable_offsets = [(idx, idx + 1) for idx in range(len(prompt))]
        offsets = [(0, 0), (0, 0), *printable_offsets]
        token_ids = list(range(len(offsets)))
        data: dict[str, object] = {"input_ids": [token_ids]}
        if return_offsets_mapping:
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


def test_token_index_from_char_skips_leading_zero_width_offsets() -> None:
    prompt = "Hello"
    tokenizer = ZeroWidthTokenizer()
    index = token_index_from_char(tokenizer, prompt, 0)
    assert index == 2  # two zero-width tokens precede the first printable token


def test_token_index_from_char_handles_end_anchor_with_zero_width_offsets() -> None:
    prompt = "abc"
    tokenizer = ZeroWidthTokenizer()
    anchor = len(prompt)
    index = token_index_from_char(tokenizer, prompt, anchor)
    assert index == len(prompt) + 1  # last printable token index (with two zero-width tokens)


def test_window_spec_requires_anchor_for_custom_occurrence() -> None:
    spec = WindowSpec(start_occurrence=2)
    with pytest.raises(typer.BadParameter):
        spec.validate()


def test_window_spec_resolves_matches_to_indices() -> None:
    tokenizer = CharTokenizer()
    spec = WindowSpec(start_match="Trial 1:", end_match="Trial 2:")
    spec.validate()

    start_idx, end_idx = spec.resolve(tokenizer, PROMPT)
    expected_start = PROMPT.rfind("\n", 0, PROMPT.find("Trial 1:"))
    trial2_end = PROMPT.find("Trial 2:") + len("Trial 2:")
    expected_end = PROMPT.find("\n", trial2_end)

    assert start_idx == expected_start
    assert end_idx == expected_end


def test_window_spec_defaults_end_to_negative_one_when_only_start_specified() -> None:
    tokenizer = CharTokenizer()
    spec = WindowSpec(start_index=5)
    spec.validate()
    start_idx, end_idx = spec.resolve(tokenizer, PROMPT)
    assert start_idx == 5
    assert end_idx == -1


def test_window_spec_leaves_indices_none_when_unset() -> None:
    tokenizer = CharTokenizer()
    spec = WindowSpec()
    spec.validate()
    start_idx, end_idx = spec.resolve(tokenizer, PROMPT)
    assert start_idx is None
    assert end_idx is None


def test_window_spec_build_schedule_resolves_window_and_prompt_length() -> None:
    tokenizer = CharTokenizer()
    spec = WindowSpec(start_match="Trial 1:", end_match="Trial 2:")
    spec.validate()

    prompt_length = len(PROMPT)
    schedule = spec.build_schedule(
        tokenizer=tokenizer,
        prompt=PROMPT,
        token_index=None,
        apply_all_tokens=False,
        generated_only=False,
        prompt_length=prompt_length,
    )

    expected_start = PROMPT.rfind("\n", 0, PROMPT.find("Trial 1:"))
    trial2_end = PROMPT.find("Trial 2:") + len("Trial 2:")
    expected_end = PROMPT.find("\n", trial2_end)

    assert schedule.window_start == expected_start
    assert schedule.window_end == expected_end
    assert schedule.prompt_length == prompt_length
