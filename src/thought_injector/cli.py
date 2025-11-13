from __future__ import annotations

import csv
import json
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import torch
import typer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from thought_injector.app import app, console, typed_command
from thought_injector.baseline import load_baseline_words
from thought_injector.injection import InjectionSchedule, injection_context
from thought_injector.models import (
    clone_inputs,
    extract_hidden_state,
    get_decoder_layers,
    load_model_and_tokenizer,
    requires_cache_disabled,
    resolve_device,
    resolve_dtype,
    tokenize,
)
from thought_injector.pairs import PromptPair, load_prompt_pairs
from thought_injector.text_utils import WindowSpec, diff_length
from thought_injector.vectors import load_prepared_vector, load_vector, save_vector

if TYPE_CHECKING:

    class GenerationConfig:  # pragma: no cover - type stub
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    from transformers import PreTrainedModel, StoppingCriteria, StoppingCriteriaList

else:
    from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList


ManualSeedFn = Callable[[int], torch.Generator]
DecodeFn = Callable[..., str]


def _seed_rng(seed: int) -> None:
    """Seed torch's RNG so CLI commands remain reproducible."""
    seed_fn = cast(ManualSeedFn, torch.manual_seed)
    seed_fn(seed)


def _decode_output(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: torch.Tensor,
    *,
    include_prompt: bool,
) -> str:
    """Decode a tensor of token ids back into readable text."""
    decode_fn = cast(DecodeFn, tokenizer.decode)
    return decode_fn(token_ids, skip_special_tokens=not include_prompt)


class SubstringStoppingCriteria(StoppingCriteria):
    """Halts generation once the given substring appears in the decoded continuation."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        substring: str,
        occurrence: int,
        prompt_length: int,
    ) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._substring = substring
        self._occurrence = occurrence
        self._prompt_length = prompt_length
        self.triggered = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:  # type: ignore[override]
        if input_ids.shape[0] != 1:
            return False
        seq = input_ids[0]
        if seq.shape[0] <= self._prompt_length:
            return False
        decode_fn = cast(DecodeFn, self._tokenizer.decode)
        continuation = decode_fn(seq[self._prompt_length :], skip_special_tokens=False)
        if continuation.count(self._substring) >= self._occurrence:
            self.triggered = True
            return True
        return False


def _build_generation_config(
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> GenerationConfig:
    """Translate CLI sampling flags into a GenerationConfig."""
    pad_token_id = cast(int | None, getattr(tokenizer, "pad_token_id", None))
    eos_token_id = cast(int | None, getattr(tokenizer, "eos_token_id", None))
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=max(temperature, 1e-8),  # Clamp to avoid HF zero-temp errors.
        top_p=top_p,
        do_sample=temperature > 0,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )


SWEEP_LAYER_OPTION = typer.Option(
    ...,
    "--layer-index",
    help="One or more layers to evaluate (repeat flag).",
)
SWEEP_STRENGTH_OPTION = typer.Option(
    ...,
    "--strength",
    help="One or more strengths to evaluate (repeat flag).",
)
WINDOW_START_INDEX_OPTION = typer.Option(
    ...,
    "--start-index",
    help="Optional start index (inclusive) for windowed injection.",
)
WINDOW_END_INDEX_OPTION = typer.Option(
    ...,
    "--end-index",
    help="Optional end index (inclusive) for windowed injection.",
)
WINDOW_START_MATCH_OPTION = typer.Option(
    ...,
    "--start-match",
    help="Substring anchor; injection begins on the newline preceding this match.",
)
WINDOW_END_MATCH_OPTION = typer.Option(
    ...,
    "--end-match",
    help="Substring anchor; injection ends on the newline following this match.",
)
WINDOW_START_OCCURRENCE_OPTION = typer.Option(
    ...,
    "--start-occurrence",
    help="1-based occurrence index for --start-match.",
    min=1,
    show_default=True,
)
WINDOW_END_OCCURRENCE_OPTION = typer.Option(
    ...,
    "--end-occurrence",
    help="1-based occurrence index for --end-match.",
    min=1,
    show_default=True,
)
APPLY_ALL_TOKENS_OPTION = typer.Option(
    ...,
    "--apply-all-tokens",
    help="If set, injects into every token in the sequence.",
    show_default=True,
)
GENERATED_ONLY_OPTION = typer.Option(
    ...,
    "--generated-only",
    help="Restrict injection to newly generated tokens.",
    show_default=True,
)
NORMALIZE_OPTION = typer.Option(
    True,
    "--normalize/--no-normalize",
    help="Normalize the vector to unit RMS before scaling.",
    show_default=True,
)
SCALE_BY_OPTION = typer.Option(
    ...,
    "--scale-by",
    help="Extra multiplier applied after normalization.",
    show_default=True,
)


def _build_window_spec(
    *,
    start_index: int | None,
    end_index: int | None,
    start_match: str | None,
    end_match: str | None,
    start_occurrence: int,
    end_occurrence: int,
) -> WindowSpec:
    """Validate the mutual exclusivity rules for window-related CLI flags."""
    spec = WindowSpec(
        start_index=start_index,
        end_index=end_index,
        start_match=start_match,
        end_match=end_match,
        start_occurrence=start_occurrence,
        end_occurrence=end_occurrence,
    )
    spec.validate()
    return spec


def _print_resolved_span(schedule: InjectionSchedule, prompt_length: int) -> None:
    """Render the resolved span so users see the inclusive token indices."""
    span = schedule.resolved_span(prompt_length)
    if span is None:
        if schedule.generated_only:
            gen_origin = (
                schedule.prompt_length if schedule.prompt_length is not None else prompt_length
            )
            console.print(
                "[blue]Generated-only schedule:[/blue] prompt tokens remain untouched; "
                f"injection begins at index {gen_origin} once new tokens are produced."
            )
        else:
            console.print(
                f"[yellow]Resolved token span is empty for prompt length {prompt_length}. Check your flags."
            )
        return
    span_start, span_end = span
    notes: list[str] = []
    if schedule.has_window() and schedule.window_end == -1:
        notes.append("end follows the latest token (open-ended)")
    if schedule.generated_only:
        gen_origin = schedule.prompt_length if schedule.prompt_length is not None else prompt_length
        notes.append(f"generated-only (>= index {gen_origin})")
    note_suffix = f" [{' | '.join(notes)}]" if notes else ""
    console.print(
        f"[cyan]Resolved token span:[/cyan] [{span_start}..{span_end}] (inclusive){note_suffix}"
    )


def _should_disable_cache(
    model: PreTrainedModel | Any, schedule: InjectionSchedule, has_injection: bool
) -> bool:
    """Return True when we must recompute the full sequence (disabling KV cache)."""
    # Only disable KV cache when an injection is active and the schedule requires it;
    # baseline generations keep cache-enabled windows for speed.
    disable_cache = has_injection and schedule.requires_full_sequence()
    if requires_cache_disabled(model):
        return True
    return disable_cache


def _run_model_generate(
    *,
    model: PreTrainedModel | Any,
    inputs: dict[str, torch.Tensor],
    generation_config: GenerationConfig,
    schedule: InjectionSchedule,
    vector: torch.Tensor | None,
    layer_index: int | None,
    strength: float | None,
    stopping_criteria: StoppingCriteriaList | None = None,
) -> torch.Tensor:
    """Shared helper that runs `model.generate` with optional injection hooks."""
    use_injection = vector is not None and layer_index is not None and strength is not None
    disable_cache = _should_disable_cache(model, schedule, use_injection)
    sampling_model = cast(Any, model)

    if use_injection:
        assert layer_index is not None
        assert vector is not None
        assert strength is not None
        ctx = injection_context(model, layer_index, vector, strength, schedule)
    else:
        ctx = nullcontext()

    generate_kwargs: dict[str, Any] = {}
    if stopping_criteria is not None:
        generate_kwargs["stopping_criteria"] = stopping_criteria

    with torch.no_grad(), ctx:
        output_ids = cast(
            torch.Tensor,
            sampling_model.generate(
                **inputs,
                generation_config=generation_config,
                use_cache=not disable_cache,
                **generate_kwargs,
            ),
        )
    return output_ids


def _generate_text_with_schedule(
    *,
    model: PreTrainedModel | Any,
    tokenizer: PreTrainedTokenizerBase,
    inputs: dict[str, torch.Tensor],
    generation_config: GenerationConfig,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    schedule: InjectionSchedule,
    vector: torch.Tensor | None,
    layer_index: int | None,
    strength: float | None,
    include_prompt: bool,
) -> str:
    """Run generation while optionally installing an injection hook for the window."""
    prompt_length = inputs["input_ids"].shape[1]

    if schedule.generated_end_match:
        stop = SubstringStoppingCriteria(
            tokenizer=tokenizer,
            substring=schedule.generated_end_match,
            occurrence=schedule.generated_end_occurrence,
            prompt_length=prompt_length,
        )
        stop_list = StoppingCriteriaList([stop])
        output_phase1 = _run_model_generate(
            model=model,
            inputs=inputs,
            generation_config=generation_config,
            schedule=schedule,
            vector=vector,
            layer_index=layer_index,
            strength=strength,
            stopping_criteria=stop_list,
        )
        tokens_phase1 = output_phase1.shape[1] - prompt_length
        if not stop.triggered:
            console.print(
                f"[yellow]Warning:[/yellow] --end-match '{schedule.generated_end_match}' "
                "never appeared in the generated text; injection stayed active for the "
                "full run."
            )
            return _decode_output(tokenizer, output_phase1[0], include_prompt=include_prompt)

        max_new = getattr(generation_config, "max_new_tokens", None) or 0
        remaining = max(max_new - tokens_phase1, 0)
        if remaining <= 0:
            return _decode_output(tokenizer, output_phase1[0], include_prompt=include_prompt)

        prompt_text = _decode_output(tokenizer, output_phase1[0], include_prompt=True)
        device = inputs["input_ids"].device
        inputs_phase2 = tokenize(tokenizer, prompt_text, device)
        remaining_config = _build_generation_config(
            tokenizer,
            max_new_tokens=remaining,
            temperature=temperature,
            top_p=top_p,
        )

        output_phase2 = _run_model_generate(
            model=model,
            inputs=inputs_phase2,
            generation_config=remaining_config,
            schedule=schedule,
            vector=None,
            layer_index=None,
            strength=None,
        )
        phase2_prompt_len = inputs_phase2["input_ids"].shape[1]
        new_tokens = output_phase2[:, phase2_prompt_len:]
        combined = torch.cat([output_phase1, new_tokens], dim=1)
        return _decode_output(tokenizer, combined[0], include_prompt=include_prompt)

    output_ids = _run_model_generate(
        model=model,
        inputs=inputs,
        generation_config=generation_config,
        schedule=schedule,
        vector=vector,
        layer_index=layer_index,
        strength=strength,
    )
    return _decode_output(tokenizer, output_ids[0], include_prompt=include_prompt)


@typed_command()
def capture(
    model_path: Annotated[
        Path,
        typer.Option(
            ...,
            "--model-path",
            "-m",
            help="Path to the local HF-format model directory.",
        ),
    ],
    positive_prompt: Annotated[
        str, typer.Option(..., help="Prompt expected to activate the concept.")
    ],
    negative_prompt: Annotated[str, typer.Option(..., help="Prompt without the concept.")],
    layer_index: Annotated[
        int,
        typer.Option(help="Decoder layer to sample (0-based)."),
    ] = 0,
    token_index: Annotated[
        int,
        typer.Option(help="Token index (supports negatives for counting from the end)."),
    ] = -1,
    output_path: Annotated[
        Path,
        typer.Option(help="Where to store the concept vector."),
    ] = Path("vectors/concept.safetensors"),
    dtype: Annotated[
        str,
        typer.Option(
            help="Torch dtype for model weights; 'auto' picks bfloat16 when supported.",
        ),
    ] = "auto",
    device: Annotated[
        str,
        typer.Option(help="Device identifier or 'auto'."),
    ] = "auto",
) -> None:
    """Capture a concept vector by differencing two prompts."""

    torch_dtype = resolve_dtype(dtype)
    torch_device = resolve_device(device)
    model, tokenizer = load_model_and_tokenizer(model_path, torch_dtype, torch_device)

    pos_hidden = extract_hidden_state(
        model, tokenizer, positive_prompt, layer_index, token_index, torch_device
    )
    neg_hidden = extract_hidden_state(
        model, tokenizer, negative_prompt, layer_index, token_index, torch_device
    )
    vector = (pos_hidden - neg_hidden).to(torch.float32)

    metadata = {
        "model_path": str(model_path),
        "layer_index": layer_index,
        "token_index": token_index,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
    }
    save_vector(output_path, vector, metadata)


@typed_command()
def capture_word(
    model_path: Annotated[
        Path,
        typer.Option(
            ...,
            "--model-path",
            "-m",
            help="Path to the local HF-format model directory.",
        ),
    ],
    word: Annotated[str, typer.Option(..., help="Word to probe (e.g., 'aquariums').")],
    layer_index: Annotated[
        int,
        typer.Option(help="Decoder layer to sample (0-based)."),
    ] = 0,
    token_index: Annotated[
        int,
        typer.Option(help="Token index (supports negatives)."),
    ] = -1,
    baseline_path: Annotated[
        Path | None,
        typer.Option(help="Optional newline-delimited baseline word list."),
    ] = None,
    baseline_count: Annotated[
        int,
        typer.Option(help="Number of baseline words to average.", show_default=True),
    ] = 100,
    output_path: Annotated[
        Path,
        typer.Option(help="Where to store the concept vector."),
    ] = Path("vectors/concept.safetensors"),
    dtype: Annotated[
        str,
        typer.Option(help="Torch dtype for model weights; 'auto' picks bfloat16 when supported."),
    ] = "auto",
    device: Annotated[
        str,
        typer.Option(help="Device identifier or 'auto'."),
    ] = "auto",
) -> None:
    """Capture a concept vector by subtracting a baseline mean from a target word."""

    if baseline_count <= 0:
        raise typer.BadParameter("baseline-count must be positive.")

    torch_dtype = resolve_dtype(dtype)
    torch_device = resolve_device(device)
    model, tokenizer = load_model_and_tokenizer(model_path, torch_dtype, torch_device)

    positive_prompt = f"Tell me about {word}."
    target_hidden = extract_hidden_state(
        model, tokenizer, positive_prompt, layer_index, token_index, torch_device
    )

    baseline_words = load_baseline_words(baseline_path)
    baseline_words = [w for w in baseline_words if w.lower() != word.lower()]
    selected = baseline_words[:baseline_count]
    if not selected:
        raise typer.BadParameter("Baseline word list is empty after filtering the target word.")
    if len(selected) < baseline_count:
        console.print(
            f"[yellow]Warning:[/yellow] requested {baseline_count} baseline words but only found {len(selected)}."
        )

    baseline_vectors: list[torch.Tensor] = []
    for base_word in selected:
        base_prompt = f"Tell me about {base_word}."
        hidden = extract_hidden_state(
            model, tokenizer, base_prompt, layer_index, token_index, torch_device
        )
        baseline_vectors.append(hidden)
    if not baseline_vectors:
        raise typer.BadParameter(
            "Cannot compute baseline mean; no baseline vectors were collected."
        )
    baseline_mean = torch.stack(baseline_vectors).mean(dim=0)

    vector = (target_hidden - baseline_mean).to(torch.float32)
    metadata = {
        "model_path": str(model_path),
        "layer_index": layer_index,
        "token_index": token_index,
        "word": word,
        "baseline_count": len(selected),
        "baseline_source": str(baseline_path) if baseline_path else "default_list",
        "prompts": {
            "positive": positive_prompt,
            "baseline": [f"Tell me about {w}." for w in selected],
        },
    }
    save_vector(output_path, vector, metadata)
    rms = torch.sqrt(torch.mean(vector**2)).item()
    console.print(f"Vector RMS: {rms:.6f}")


@typed_command()
def capture_pairs(
    model_path: Annotated[
        Path,
        typer.Option(
            ...,
            "--model-path",
            "-m",
            help="Path to the local HF-format model directory.",
        ),
    ],
    pairs_path: Annotated[
        Path,
        typer.Option(
            ...,
            "--pairs-path",
            help="JSON/CSV file containing minimal prompt pairs.",
        ),
    ],
    layer_index: Annotated[
        int,
        typer.Option(help="Decoder layer to sample (0-based)."),
    ] = 0,
    token_index: Annotated[
        int,
        typer.Option(help="Token index to probe (supports negatives)."),
    ] = -1,
    max_pairs: Annotated[
        int | None,
        typer.Option(help="Optional cap; use only the first N pairs from the file."),
    ] = None,
    output_path: Annotated[
        Path,
        typer.Option(help="Where to store the concept vector."),
    ] = Path("vectors/concept.safetensors"),
    dtype: Annotated[
        str,
        typer.Option(help="Torch dtype for model weights; 'auto' prefers bf16 when supported."),
    ] = "auto",
    device: Annotated[
        str,
        typer.Option(help="Device identifier or 'auto'."),
    ] = "auto",
) -> None:
    """Capture a concept vector by averaging minimal prompt pairs."""

    if max_pairs is not None and max_pairs <= 0:
        raise typer.BadParameter("--max-pairs must be positive when provided.")

    pairs: list[PromptPair] = load_prompt_pairs(pairs_path)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
        if not pairs:
            raise typer.BadParameter(
                "Pairs list is empty after applying --max-pairs; increase the limit."
            )

    torch_dtype = resolve_dtype(dtype)
    torch_device = resolve_device(device)
    model, tokenizer = load_model_and_tokenizer(model_path, torch_dtype, torch_device)

    diffs: list[torch.Tensor] = []
    for pair in pairs:
        pos_hidden = extract_hidden_state(
            model, tokenizer, pair.positive, layer_index, token_index, torch_device
        )
        neg_hidden = extract_hidden_state(
            model, tokenizer, pair.negative, layer_index, token_index, torch_device
        )
        diffs.append(pos_hidden - neg_hidden)

    stacked = torch.stack(diffs)
    vector = stacked.mean(dim=0).to(torch.float32)

    metadata = {
        "model_path": str(model_path),
        "layer_index": layer_index,
        "token_index": token_index,
        "pairs_source": str(pairs_path),
        "pair_count": len(pairs),
        "prompts": {
            "pairs": [{"positive": pair.positive, "negative": pair.negative} for pair in pairs]
        },
    }
    save_vector(output_path, vector, metadata)
    rms = torch.sqrt(torch.mean(vector**2)).item()
    console.print(f"Vector RMS: {rms:.6f}")


@typed_command()
def run(
    model_path: Annotated[
        Path,
        typer.Option(
            ...,
            "--model-path",
            "-m",
            help="Path to the local HF-format model directory.",
        ),
    ],
    prompt: Annotated[str, typer.Option(..., help="Prompt to feed the model.")],
    vector_path: Annotated[
        Path | None,
        typer.Option(help="Optional concept vector to inject."),
    ] = None,
    layer_index: Annotated[
        int,
        typer.Option(help="Layer to inject into."),
    ] = 0,
    token_index: Annotated[
        int | None,
        typer.Option(help="Token index for injection (omit for default behavior)."),
    ] = None,
    start_index: Annotated[int | None, WINDOW_START_INDEX_OPTION] = None,
    end_index: Annotated[int | None, WINDOW_END_INDEX_OPTION] = None,
    start_match: Annotated[str | None, WINDOW_START_MATCH_OPTION] = None,
    end_match: Annotated[str | None, WINDOW_END_MATCH_OPTION] = None,
    start_occurrence: Annotated[int, WINDOW_START_OCCURRENCE_OPTION] = 1,
    end_occurrence: Annotated[int, WINDOW_END_OCCURRENCE_OPTION] = 1,
    strength: Annotated[
        float,
        typer.Option(help="Multiplier applied to the concept vector."),
    ] = 1.0,
    apply_all_tokens: Annotated[bool, APPLY_ALL_TOKENS_OPTION] = False,
    generated_only: Annotated[bool, GENERATED_ONLY_OPTION] = False,
    normalize: bool = NORMALIZE_OPTION,
    scale_by: Annotated[float, SCALE_BY_OPTION] = 1.0,
    max_new_tokens: Annotated[
        int,
        typer.Option(help="Number of new tokens to sample."),
    ] = 128,
    temperature: Annotated[
        float,
        typer.Option(help="Sampling temperature."),
    ] = 0.0,
    top_p: Annotated[
        float,
        typer.Option(help="Top-p nucleus sampling."),
    ] = 0.9,
    dtype: Annotated[
        str,
        typer.Option(
            help="Torch dtype for model weights; 'auto' picks bfloat16 when supported.",
        ),
    ] = "auto",
    device: Annotated[
        str,
        typer.Option(help="Device identifier or 'auto'."),
    ] = "auto",
    seed: Annotated[
        int | None,
        typer.Option(help="Optional RNG seed for reproducibility."),
    ] = None,
    include_prompt: Annotated[
        bool,
        typer.Option(help="Return the full decoded sequence (prompt + continuation)."),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(help="Print the resolved token span before running the model."),
    ] = False,
) -> None:
    """Run the prompt with optional activation injection."""

    window_spec = _build_window_spec(
        start_index=start_index,
        end_index=end_index,
        start_match=start_match,
        end_match=end_match,
        start_occurrence=start_occurrence,
        end_occurrence=end_occurrence,
    )

    if seed is not None:
        _seed_rng(seed)

    torch_dtype = resolve_dtype(dtype)
    torch_device = resolve_device(device)
    model, tokenizer = load_model_and_tokenizer(model_path, torch_dtype, torch_device)

    inputs = tokenize(tokenizer, prompt, torch_device)
    prompt_length = inputs["input_ids"].shape[1]

    schedule = window_spec.build_schedule(
        tokenizer=tokenizer,
        prompt=prompt,
        token_index=token_index,
        apply_all_tokens=apply_all_tokens,
        generated_only=generated_only,
        prompt_length=prompt_length,
    )

    if verbose:
        _print_resolved_span(schedule, prompt_length)
    if schedule.generated_end_match:
        console.print(
            "[blue]Window:[/blue] injection will stop once "
            f"'{schedule.generated_end_match}' (occurrence {schedule.generated_end_occurrence}) "
            "appears in the generated text."
        )

    generation_config = _build_generation_config(
        tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    vector_tensor: torch.Tensor | None = None
    if vector_path is None and strength != 0.0:
        console.print(
            "[yellow]Warning:[/yellow] --strength is ignored without --vector-path; running baseline."
        )
    if vector_path is not None:
        prepared_vector = load_prepared_vector(
            vector_path,
            model,
            normalize=normalize,
            scale_by=scale_by,
        )
        vector_tensor = prepared_vector.tensor
        metadata_layer = prepared_vector.metadata.layer_index
        if metadata_layer is not None and metadata_layer != layer_index:
            total_layers = len(get_decoder_layers(model))
            console.print(
                "[yellow]Warning:[/yellow] vector recorded from layer "
                f"{metadata_layer}, but we will inject at layer {layer_index} "
                f"(model has {total_layers} layers)."
            )

    text = _generate_text_with_schedule(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        generation_config=generation_config,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        schedule=schedule,
        vector=vector_tensor,
        layer_index=layer_index,
        strength=strength,
        include_prompt=include_prompt,
    )
    console.print("=== Model Output ===")
    console.print(text)


@typed_command()
def sweep(
    model_path: Annotated[
        Path,
        typer.Option(
            ...,
            "--model-path",
            "-m",
            help="Path to the local HF-format model directory.",
        ),
    ],
    prompt: Annotated[str, typer.Option(..., help="Prompt to feed the model.")],
    vector_path: Annotated[
        Path,
        typer.Option(..., help="Concept vector to inject during the sweep."),
    ],
    layer_indices: Annotated[list[int] | None, SWEEP_LAYER_OPTION] = None,
    strengths: Annotated[list[float] | None, SWEEP_STRENGTH_OPTION] = None,
    token_index: Annotated[
        int | None,
        typer.Option(help="Token index for single-token injection (omit for default behavior)."),
    ] = None,
    start_index: Annotated[int | None, WINDOW_START_INDEX_OPTION] = None,
    end_index: Annotated[int | None, WINDOW_END_INDEX_OPTION] = None,
    start_match: Annotated[str | None, WINDOW_START_MATCH_OPTION] = None,
    end_match: Annotated[str | None, WINDOW_END_MATCH_OPTION] = None,
    start_occurrence: Annotated[int, WINDOW_START_OCCURRENCE_OPTION] = 1,
    end_occurrence: Annotated[int, WINDOW_END_OCCURRENCE_OPTION] = 1,
    apply_all_tokens: Annotated[bool, APPLY_ALL_TOKENS_OPTION] = False,
    generated_only: Annotated[bool, GENERATED_ONLY_OPTION] = False,
    normalize: bool = NORMALIZE_OPTION,
    scale_by: Annotated[float, SCALE_BY_OPTION] = 1.0,
    diff_threshold: Annotated[
        int,
        typer.Option(help="Flag a row as changed when diff length exceeds this value."),
    ] = 40,
    max_new_tokens: Annotated[
        int,
        typer.Option(help="Number of new tokens to sample."),
    ] = 128,
    temperature: Annotated[
        float,
        typer.Option(help="Sampling temperature."),
    ] = 0.0,
    top_p: Annotated[
        float,
        typer.Option(help="Top-p nucleus sampling."),
    ] = 0.9,
    include_prompt: Annotated[
        bool,
        typer.Option(help="Return the full decoded sequence (prompt + continuation)."),
    ] = False,
    seed: Annotated[
        int | None,
        typer.Option(help="Optional RNG seed for reproducibility."),
    ] = None,
    dtype: Annotated[
        str,
        typer.Option(help="Torch dtype for model weights; 'auto' picks bfloat16 when supported."),
    ] = "auto",
    device: Annotated[
        str,
        typer.Option(help="Device identifier or 'auto'."),
    ] = "auto",
    output_path: Annotated[
        Path,
        typer.Option(help="CSV destination for sweep results."),
    ] = Path("sweeps/latest.csv"),
) -> None:
    """Sweep layer/strength combinations and log outputs to CSV.

    The first row always captures the baseline text (layer_index=None, strength=None) so callers
    can diff every trial against an injection-free run.
    """

    window_spec = _build_window_spec(
        start_index=start_index,
        end_index=end_index,
        start_match=start_match,
        end_match=end_match,
        start_occurrence=start_occurrence,
        end_occurrence=end_occurrence,
    )

    if seed is not None:
        _seed_rng(seed)
    layer_indices_list = list(layer_indices or [])
    strengths_list = list(strengths or [])
    if not layer_indices_list:
        raise typer.BadParameter("Provide at least one --layer-index for the sweep.")
    if not strengths_list:
        raise typer.BadParameter("Provide at least one --strength for the sweep.")

    torch_dtype = resolve_dtype(dtype)
    torch_device = resolve_device(device)
    model, tokenizer = load_model_and_tokenizer(model_path, torch_dtype, torch_device)

    base_inputs = tokenize(tokenizer, prompt, torch_device)
    prompt_length = base_inputs["input_ids"].shape[1]

    schedule = window_spec.build_schedule(
        tokenizer=tokenizer,
        prompt=prompt,
        token_index=token_index,
        apply_all_tokens=apply_all_tokens,
        generated_only=generated_only,
        prompt_length=prompt_length,
    )

    generation_config = _build_generation_config(
        tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    prepared_vector = load_prepared_vector(
        vector_path,
        model,
        normalize=normalize,
        scale_by=scale_by,
    )
    vector_tensor = prepared_vector.tensor

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _generate_text(layer_idx: int | None, strength_value: float | None) -> str:
        cloned_inputs = clone_inputs(base_inputs)
        vector_arg = vector_tensor if layer_idx is not None and strength_value is not None else None
        return _generate_text_with_schedule(
            model=model,
            tokenizer=tokenizer,
            inputs=cloned_inputs,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            schedule=schedule,
            vector=vector_arg,
            layer_index=layer_idx,
            strength=strength_value,
            include_prompt=include_prompt,
        )

    # Baseline text always omits injection regardless of CLI strengths for clarity.
    baseline_text = _generate_text(layer_idx=None, strength_value=None)

    rows: list[dict[str, object]] = [
        {
            "layer_index": "baseline",
            "strength": 0.0,
            "diff_len": 0,
            "changed": False,
            "text": baseline_text,
        }
    ]

    for layer_idx in layer_indices_list:
        for strength_value in strengths_list:
            trial_text = _generate_text(layer_idx=layer_idx, strength_value=strength_value)
            diff_len = diff_length(baseline_text, trial_text)
            rows.append(
                {
                    "layer_index": layer_idx,
                    "strength": strength_value,
                    "diff_len": diff_len,
                    "changed": diff_len > diff_threshold,
                    "text": trial_text,
                }
            )

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["layer_index", "strength", "diff_len", "changed", "text"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    console.print(f"Sweep complete -> {output_path}")


@typed_command()
def inspect_vector(
    vector_path: Annotated[Path, typer.Argument(..., help="Path to a saved vector.")],
) -> None:
    """Print metadata for a concept vector."""

    record = load_vector(vector_path)
    console.print(json.dumps(record.metadata.model_dump(mode="json"), indent=2))
    console.print(f"Vector shape: {tuple(record.vector.shape)}")


if __name__ == "__main__":
    app()
