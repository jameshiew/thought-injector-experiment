from __future__ import annotations

import difflib
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import torch
import typer
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from thought_injector.spans import AnchorError, locate_end_anchor, locate_start_anchor


def flatten_first_sequence(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        squeezed = values.squeeze(0)
        tolist_fn = cast(Callable[[], list[Any]], squeezed.tolist)
        return tolist_fn()
    if isinstance(values, list):
        if values and isinstance(values[0], (list, tuple)):
            first_seq = cast(Sequence[Any], values[0])
            return list(first_seq)
        typed_values = cast(list[Any], values)
        return list(typed_values)
    if isinstance(values, tuple):
        if values and isinstance(values[0], (list, tuple)):
            first_seq = cast(Sequence[Any], values[0])
            return list(first_seq)
        typed_values = cast(tuple[Any, ...], values)
        return list(typed_values)
    if isinstance(values, Sequence):
        typed_values = cast(Sequence[Any], values)
        return list(typed_values)
    return [values]


def token_index_from_char(tokenizer: PreTrainedTokenizerBase, prompt: str, char_index: int) -> int:
    encoding: BatchEncoding = tokenizer(
        prompt,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    encoding_map = cast(Mapping[str, Any], encoding)
    offsets = cast(
        torch.Tensor | list[Any] | tuple[Any, ...] | None,
        encoding_map.get("offset_mapping"),
    )
    if offsets is None:
        encoding_data = cast(dict[str, Any], getattr(encoding, "data", dict(encoding)))
        offsets = cast(
            torch.Tensor | list[Any] | tuple[Any, ...] | None, encoding_data.get("offset_mapping")
        )
    offsets_seq: list[tuple[int, int]] | None = None
    if offsets is not None:
        flattened_offsets = flatten_first_sequence(offsets)
        candidate_offsets: list[tuple[int, int]] = []
        valid_offsets = True
        for pair in flattened_offsets:
            if not isinstance(pair, (list, tuple)):
                valid_offsets = False
                break
            pair_seq = cast(Sequence[Any], pair)
            if len(pair_seq) != 2:
                valid_offsets = False
                break
            start_val = int(pair_seq[0])
            end_val = int(pair_seq[1])
            candidate_offsets.append((start_val, end_val))
        if valid_offsets:
            offsets_seq = candidate_offsets

    if offsets_seq is not None:
        saw_non_zero = False
        last_non_zero_index: int | None = None
        for idx, (start, end) in enumerate(offsets_seq):
            if end <= start:
                continue
            saw_non_zero = True
            last_non_zero_index = idx
            if start <= char_index < end:
                return idx
        if saw_non_zero:
            if char_index >= len(prompt) and last_non_zero_index is not None:
                return last_non_zero_index
            raise typer.BadParameter(
                "Could not map character offset to a tokenizer index; check --start-match anchor."
            )
        offsets_seq = None

    prefix_end = min(char_index + 1, len(prompt))
    prefix = prompt[:prefix_end]
    prefix_encoding: BatchEncoding = tokenizer(prefix, add_special_tokens=True)
    prefix_data = cast(dict[str, Any], getattr(prefix_encoding, "data", dict(prefix_encoding)))
    prefix_tokens = cast(
        torch.Tensor | list[Any] | tuple[Any, ...] | None, prefix_data.get("input_ids")
    )
    if prefix_tokens is None:
        raise typer.BadParameter(
            "Tokenizer must return list-based encodings for --start-match fallback."
        )
    prefix_seq_raw = flatten_first_sequence(prefix_tokens)
    try:
        prefix_seq = [int(value) for value in prefix_seq_raw]
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(
            "Tokenizer must return integer token ids for --start-match fallback."
        ) from exc
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
