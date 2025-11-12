from __future__ import annotations

import difflib
from typing import Any

import torch
import typer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from thought_injector.spans import AnchorError, locate_end_anchor, locate_start_anchor


def flatten_first_sequence(values: Any) -> Any:
    if isinstance(values, torch.Tensor):
        values = values.squeeze(0).tolist()
    if isinstance(values, list):
        if values and isinstance(values[0], list):
            return values[0]
        return values
    if isinstance(values, tuple):
        if values and isinstance(values[0], (list, tuple)):
            return list(values[0])
        return list(values)
    return values


def token_index_from_char(tokenizer: PreTrainedTokenizerBase, prompt: str, char_index: int) -> int:
    encoding = tokenizer(
        prompt,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = encoding.get("offset_mapping")
    offsets_seq = None
    if offsets is not None:
        offsets_seq = flatten_first_sequence(offsets)

    if offsets_seq is not None:
        for idx, (start, end) in enumerate(offsets_seq):
            if start <= char_index < end:
                return idx
        if char_index >= len(prompt):
            return len(offsets_seq) - 1
        raise typer.BadParameter(
            "Could not map character offset to a tokenizer index; check --start-match anchor."
        )

    prefix_end = min(char_index + 1, len(prompt))
    prefix = prompt[:prefix_end]
    prefix_tokens = tokenizer(prefix, add_special_tokens=True).get("input_ids")
    prefix_seq = flatten_first_sequence(prefix_tokens)
    if not isinstance(prefix_seq, list):
        raise typer.BadParameter(
            "Tokenizer must return list-based encodings for --start-match fallback."
        )
    return max(len(prefix_seq) - 1, 0)


def resolve_start_match_token_index(
    tokenizer: PreTrainedTokenizerBase, prompt: str, match: str, occurrence: int
) -> int:
    try:
        anchor_char = locate_start_anchor(prompt, match, occurrence)
    except AnchorError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return token_index_from_char(tokenizer, prompt, anchor_char)


def resolve_end_match_token_index(
    tokenizer: PreTrainedTokenizerBase, prompt: str, match: str, occurrence: int
) -> int:
    try:
        anchor_char = locate_end_anchor(prompt, match, occurrence)
    except AnchorError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return token_index_from_char(tokenizer, prompt, anchor_char)


def diff_length(reference: str, candidate: str) -> int:
    diff_total = 0
    for chunk in difflib.ndiff(reference, candidate):
        if not chunk:
            continue
        if chunk[0] in {"+", "-"}:
            diff_total += len(chunk[2:])
    return diff_total
