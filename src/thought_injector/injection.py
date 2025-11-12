from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
import typer
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.hooks import RemovableHandle
from transformers import PreTrainedModel

from thought_injector.models import resolve_layer, resolve_token_index
from thought_injector.vectors import broadcast_vector


@runtime_checkable
class LastHiddenStateOutput(Protocol):
    last_hidden_state: torch.Tensor | None


OutputLike = LastHiddenStateOutput | torch.Tensor | tuple[torch.Tensor, ...]


class InjectionSchedule(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    apply_all: bool = False
    single_index: int | None = Field(default=None)
    window_start: int | None = None
    window_end: int | None = None
    generated_only: bool = False
    prompt_length: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _generated_only_requires_prompt(self) -> InjectionSchedule:
        if self.generated_only and self.prompt_length is None:
            raise ValueError("--generated-only requires a known prompt token length.")
        return self

    def has_window(self) -> bool:
        return self.window_start is not None or self.window_end is not None

    def requires_full_sequence(self) -> bool:
        return self.apply_all or self.has_window() or self.generated_only

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
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        has_window = self.has_window()
        effective_apply_all = self.apply_all or (
            self.generated_only
            and not (self.apply_all or has_window or self.single_index is not None)
        )

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
            if self.prompt_length is None:
                raise typer.BadParameter("--generated-only requires a known prompt token length.")
            gen_start = min(self.prompt_length, seq_len)
            gen_mask = torch.zeros_like(mask)
            if gen_start < seq_len:
                gen_mask[gen_start:] = True
            mask &= gen_mask
        return mask

    def resolved_span(self, seq_len: int) -> tuple[int, int] | None:
        if seq_len <= 0:
            return None

        has_window = self.has_window()
        effective_apply_all = self.apply_all or (
            self.generated_only
            and not (self.apply_all or has_window or self.single_index is not None)
        )

        if effective_apply_all:
            return (0, seq_len - 1)
        if has_window:
            return self._resolve_window_bounds(seq_len)
        if self.single_index is not None:
            idx = resolve_token_index(self.single_index, seq_len)
            return (idx, idx)
        idx = resolve_token_index(-1, seq_len)
        return (idx, idx)


def apply_injection(
    hidden_states: torch.Tensor,
    vector: torch.Tensor,
    strength: float,
    schedule: InjectionSchedule,
) -> torch.Tensor:
    mask = schedule.resolve_mask(hidden_states.shape[1], hidden_states.device)
    if not torch.any(mask):
        return hidden_states
    vector = strength * broadcast_vector(vector, hidden_states)
    hidden_states = hidden_states.clone()
    hidden_states[:, mask, :] += vector
    return hidden_states


def _remix_output(
    output: OutputLike,
    mutate_fn: Callable[[torch.Tensor], torch.Tensor],
) -> OutputLike:
    if isinstance(output, torch.Tensor):
        return mutate_fn(output)
    if isinstance(output, tuple):
        mutated = mutate_fn(output[0])
        return (mutated, *output[1:])
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        return output
    output.last_hidden_state = mutate_fn(hidden)
    return output  # pragma: no cover


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
