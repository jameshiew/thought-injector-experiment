from __future__ import annotations

import os
from collections.abc import Callable, Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any, Protocol, cast, runtime_checkable

import torch
import torch.nn as nn
import typer
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.hooks import RemovableHandle
from transformers import PreTrainedModel

from thought_injector.models import resolve_layer, resolve_token_index
from thought_injector.vectors import broadcast_vector


def _ti_debug_strict_enabled() -> bool:
    flag = os.getenv("TI_DEBUG_STRICT")
    if flag is None:
        return False
    normalized = flag.strip().lower()
    return normalized not in {"", "0", "false", "no"}


TI_DEBUG_STRICT = _ti_debug_strict_enabled()


@runtime_checkable
class LastHiddenStateOutput(Protocol):
    last_hidden_state: torch.Tensor | None


# Hugging Face outputs span tensors, tuples, and mapping-like containers; keep the union
# intentionally wide and let _remix_output_dict handle nested dict variants.
OutputLike = (
    LastHiddenStateOutput | torch.Tensor | tuple[torch.Tensor, ...] | MutableMapping[str, Any]
)


class InjectionSchedule(BaseModel):
    """Describe how injections target tokens.

    Priority order is apply_all -> explicit window -> single_index -> default last token.
    `window_start` / `window_end` are inclusive token indices, with `window_end=-1` treated
    as “open” until the final token. `generated_only` masks out prompt tokens using
    `prompt_length`, ensuring injections only touch tokens beyond the prompt regardless of
    the targeting mode.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    apply_all: bool = False
    single_index: int | None = Field(default=None)
    window_start: int | None = None
    window_end: int | None = None
    generated_only: bool = False
    prompt_length: int | None = Field(default=None, ge=0)
    generated_end_match: str | None = None
    generated_end_occurrence: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _generated_only_requires_prompt(self) -> InjectionSchedule:
        if self.generated_only and self.prompt_length is None:
            raise ValueError("--generated-only requires a known prompt token length.")
        return self

    def has_window(self) -> bool:
        """Return True when an explicit start/end window is configured."""
        return self.window_start is not None or self.window_end is not None

    def requires_full_sequence(self) -> bool:
        """Signal that the whole sequence must be recomputed (disables KV cache)."""
        return self.apply_all or self.has_window() or self.generated_only

    def _effective_apply_all(self) -> bool:
        """Determine whether the schedule should touch every token in the sequence."""
        if self.apply_all:
            return True
        if not self.generated_only:
            return False
        return self.single_index is None and not self.has_window()

    def _resolve_window_bounds(self, seq_len: int) -> tuple[int, int]:
        start_raw = 0 if self.window_start is None else self.window_start
        end_raw = -1 if self.window_end is None else self.window_end
        start_idx = resolve_token_index(start_raw, seq_len)
        end_idx = resolve_token_index(end_raw, seq_len)
        if end_idx < start_idx:
            raise typer.BadParameter(
                "Window end index must be greater than or equal to start index once resolved."
            )
        return start_idx, end_idx

    def resolve_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build a boolean mask highlighting the targeted token positions."""
        if seq_len <= 0:
            return torch.zeros(0, dtype=torch.bool, device=device)
        if self.generated_only:
            if self.prompt_length is None:
                raise typer.BadParameter("--generated-only requires a known prompt token length.")
            if self.prompt_length >= seq_len:
                return torch.zeros(seq_len, dtype=torch.bool, device=device)
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        has_window = self.has_window()
        effective_apply_all = self._effective_apply_all()

        if effective_apply_all:
            mask[:] = True
        elif has_window:
            start_idx, end_idx = self._resolve_window_bounds(seq_len)
            mask[start_idx : end_idx + 1] = True
        elif self.single_index is not None:
            idx = resolve_token_index(self.single_index, seq_len)
            mask[idx] = True
        else:
            idx = resolve_token_index(-1, seq_len)
            mask[idx] = True

        if self.generated_only:
            assert self.prompt_length is not None
            gen_start = min(self.prompt_length, seq_len)
            gen_mask = torch.zeros_like(mask)
            if gen_start < seq_len:
                gen_mask[gen_start:] = True
            mask &= gen_mask
        return mask

    def resolved_span(self, seq_len: int) -> tuple[int, int] | None:
        """Return the inclusive token span that will be modified, or None if the mask is empty."""
        if seq_len <= 0:
            return None

        has_window = self.has_window()
        effective_apply_all = self._effective_apply_all()

        span: tuple[int, int]
        if effective_apply_all:
            span = (0, seq_len - 1)
        elif has_window:
            span = self._resolve_window_bounds(seq_len)
        elif self.single_index is not None:
            idx = resolve_token_index(self.single_index, seq_len)
            span = (idx, idx)
        else:
            idx = resolve_token_index(-1, seq_len)
            span = (idx, idx)

        if not self.generated_only:
            return span
        if self.prompt_length is None:
            raise typer.BadParameter("--generated-only requires a known prompt token length.")
        gen_start = min(self.prompt_length, seq_len)
        if gen_start >= seq_len:
            return None
        start = max(span[0], gen_start)
        end = span[1]
        if start > end:
            return None
        return (start, end)


def apply_injection(
    hidden_states: torch.Tensor,
    vector: torch.Tensor,
    strength: float,
    schedule: InjectionSchedule,
) -> torch.Tensor:
    """Clone hidden_states, add the scaled vector where the mask is True, and return the result.

    Expects hidden_states shaped [B, T, H]; TI_DEBUG_STRICT enforces this invariant.
    """
    if TI_DEBUG_STRICT:
        _assert_residual_shape(hidden_states, vector)
    mask = schedule.resolve_mask(hidden_states.shape[1], hidden_states.device)
    if not torch.any(mask):
        return hidden_states
    vector = strength * broadcast_vector(vector, hidden_states)
    hidden_states = hidden_states.clone()
    hidden_states[:, mask, :] += vector
    return hidden_states


def _assert_residual_shape(hidden_states: torch.Tensor, vector: torch.Tensor) -> None:
    if hidden_states.ndim != 3:
        raise RuntimeError(
            "TI_DEBUG_STRICT=1: expected residual stream tensor with shape [B, T, H]; "
            f"got {tuple(hidden_states.shape)}."
        )
    if vector.ndim == 0:
        raise RuntimeError(
            f"TI_DEBUG_STRICT=1: vector must have at least one dimension; got {tuple(vector.shape)}."
        )
    expected_width = vector.shape[-1]
    if hidden_states.shape[-1] != expected_width:
        raise RuntimeError(
            "TI_DEBUG_STRICT=1: hidden state width "
            f"{hidden_states.shape[-1]} does not match vector length {expected_width}."
        )


def _remix_output(
    output: OutputLike,
    mutate_fn: Callable[[torch.Tensor], torch.Tensor],
) -> OutputLike:
    """Apply mutate_fn to the hidden-state-bearing portion of common HF outputs."""
    if isinstance(output, torch.Tensor):
        return mutate_fn(output)
    if isinstance(output, tuple):
        mutated = mutate_fn(output[0])
        return (mutated, *output[1:])
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is not None:
        cast(Any, output).last_hidden_state = mutate_fn(hidden)
        return output  # pragma: no cover
    hidden_states_attr = getattr(output, "hidden_states", None)
    if isinstance(hidden_states_attr, tuple):
        typed_hidden_states = cast(tuple[torch.Tensor, ...], hidden_states_attr)
        if typed_hidden_states:
            mutated_tuple = (mutate_fn(typed_hidden_states[0]), *typed_hidden_states[1:])
            cast(Any, output).hidden_states = mutated_tuple
        return output  # pragma: no cover
    if isinstance(hidden_states_attr, torch.Tensor):
        cast(Any, output).hidden_states = mutate_fn(hidden_states_attr)
        return output  # pragma: no cover
    if isinstance(output, dict):
        return _remix_output_dict(cast(MutableMapping[str, Any], output), mutate_fn)
    return output


def _remix_output_dict(
    mapping: MutableMapping[str, Any],
    mutate_fn: Callable[[torch.Tensor], torch.Tensor],
) -> MutableMapping[str, Any]:
    """Helper for _remix_output: mutate dict-based HF outputs in-place."""
    if "last_hidden_state" in mapping:
        lhs_value = mapping["last_hidden_state"]
        if isinstance(lhs_value, torch.Tensor):
            mapping["last_hidden_state"] = mutate_fn(lhs_value)
            return mapping
    hs_value = mapping.get("hidden_states")
    if hs_value is None:
        return mapping
    if isinstance(hs_value, tuple) and hs_value:
        hs_tuple = cast(tuple[torch.Tensor, ...], hs_value)
        mapping["hidden_states"] = (mutate_fn(hs_tuple[0]), *hs_tuple[1:])
    elif isinstance(hs_value, torch.Tensor):
        mapping["hidden_states"] = mutate_fn(hs_value)
    return mapping


def register_injection(
    model: PreTrainedModel,
    layer_index: int,
    vector: torch.Tensor,
    strength: float,
    schedule: InjectionSchedule,
) -> RemovableHandle:
    layer = resolve_layer(model, layer_index)

    def hook(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: OutputLike,
    ) -> OutputLike:
        return _remix_output(
            output,
            lambda hidden: apply_injection(hidden, vector, strength, schedule),
        )

    return layer.register_forward_hook(hook)


@contextmanager
def injection_context(
    model: PreTrainedModel,
    layer_index: int,
    vector: torch.Tensor,
    strength: float,
    schedule: InjectionSchedule,
) -> Iterator[None]:
    handle = register_injection(model, layer_index, vector, strength, schedule)
    try:
        yield
    finally:
        handle.remove()
