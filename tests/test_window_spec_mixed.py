from __future__ import annotations

from thought_injector.text_utils import WindowSpec


class CharTokenizer:
    def __call__(
        self, prompt: str, add_special_tokens: bool = True, return_offsets_mapping: bool = False
    ):
        token_ids = list(range(len(prompt)))
        data: dict[str, object] = {"input_ids": [token_ids]}
        if return_offsets_mapping:
            offsets = [(idx, idx + 1) for idx in range(len(prompt))]
            data["offset_mapping"] = [offsets]
        return data


PROMPT = """System prompt\nTrial 1:\nAssistant: turn 1\nTrial 2:\nAssistant: turn 2\n"""


def test_end_match_without_start_defaults_window_start_to_zero() -> None:
    tokenizer = CharTokenizer()
    spec = WindowSpec(end_match="Trial 2:")
    spec.validate()

    start_idx, end_idx, dynamic_match, _ = spec.resolve(tokenizer, PROMPT)
    assert start_idx is None
    assert dynamic_match is None

    schedule = spec.build_schedule(
        tokenizer=tokenizer,
        prompt=PROMPT,
        token_index=None,
        apply_all_tokens=False,
        generated_only=False,
        prompt_length=len(PROMPT),
    )
    span = schedule.resolved_span(len(PROMPT))
    assert span == (0, end_idx)
