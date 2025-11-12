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
    load_model_and_tokenizer,
    requires_cache_disabled,
    resolve_device,
    resolve_dtype,
    tokenize,
)
from thought_injector.text_utils import (
    diff_length,
    resolve_end_match_token_index,
    resolve_start_match_token_index,
)
from thought_injector.vectors import (
    ensure_vector_matches_model,
    load_vector,
    prepare_vector,
    save_vector,
)

if TYPE_CHECKING:

    class GenerationConfig:  # pragma: no cover - type stub
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

else:
    from transformers import GenerationConfig


ManualSeedFn = Callable[[int], torch.Generator]
DecodeFn = Callable[..., str]


def _seed_rng(seed: int) -> None:
    seed_fn = cast(ManualSeedFn, torch.manual_seed)
    seed_fn(seed)


def _decode_output(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: torch.Tensor,
    *,
    include_prompt: bool,
) -> str:
    decode_fn = cast(DecodeFn, tokenizer.decode)
    return decode_fn(token_ids, skip_special_tokens=not include_prompt)


SWEEP_LAYER_OPTION = typer.Option(
    None,
    "--layer-index",
    help="One or more layers to evaluate (repeat flag).",
)
SWEEP_STRENGTH_OPTION = typer.Option(
    None,
    "--strength",
    help="One or more strengths to evaluate (repeat flag).",
)


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
    ] = Path("vectors/concept.pt"),
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
    ] = Path("vectors/concept.pt"),
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
        int,
        typer.Option(help="Token index for injection."),
    ] = -1,
    start_index: Annotated[
        int | None,
        typer.Option(help="Optional start index (inclusive) for windowed injection."),
    ] = None,
    end_index: Annotated[
        int | None,
        typer.Option(help="Optional end index (inclusive) for windowed injection."),
    ] = None,
    start_match: Annotated[
        str | None,
        typer.Option(
            help="Substring anchor; injection begins on the newline preceding this match."
        ),
    ] = None,
    end_match: Annotated[
        str | None,
        typer.Option(help="Substring anchor; injection ends on the newline following this match."),
    ] = None,
    start_occurrence: Annotated[
        int,
        typer.Option(help="1-based occurrence index for --start-match.", min=1, show_default=True),
    ] = 1,
    end_occurrence: Annotated[
        int,
        typer.Option(help="1-based occurrence index for --end-match.", min=1, show_default=True),
    ] = 1,
    strength: Annotated[
        float,
        typer.Option(help="Multiplier applied to the concept vector."),
    ] = 1.0,
    apply_all_tokens: Annotated[
        bool,
        typer.Option(help="If set, injects into every token in the sequence."),
    ] = False,
    generated_only: Annotated[
        bool,
        typer.Option(help="Restrict injection to newly generated tokens."),
    ] = False,
    normalize: Annotated[
        bool,
        typer.Option(help="Normalize the vector to unit RMS before scaling.", show_default=True),
    ] = True,
    scale_by: Annotated[
        float,
        typer.Option(help="Extra multiplier applied after normalization.", show_default=True),
    ] = 1.0,
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

    if start_match is None and start_occurrence != 1:
        raise typer.BadParameter("--start-occurrence requires --start-match.")
    if end_match is None and end_occurrence != 1:
        raise typer.BadParameter("--end-occurrence requires --end-match.")

    if seed is not None:
        _seed_rng(seed)

    torch_dtype = resolve_dtype(dtype)
    torch_device = resolve_device(device)
    model, tokenizer = load_model_and_tokenizer(model_path, torch_dtype, torch_device)
    sampling_model = cast(Any, model)

    inputs = tokenize(tokenizer, prompt, torch_device)
    prompt_length = inputs["input_ids"].shape[1]

    resolved_start_index = start_index
    if start_match is not None:
        resolved_start_index = resolve_start_match_token_index(
            tokenizer, prompt, start_match, start_occurrence
        )
    resolved_end_index = end_index
    if end_match is not None:
        resolved_end_index = resolve_end_match_token_index(
            tokenizer, prompt, end_match, end_occurrence
        )
    if resolved_start_index is not None and resolved_end_index is None:
        resolved_end_index = -1

    schedule = InjectionSchedule(
        apply_all=apply_all_tokens,
        single_index=token_index,
        window_start=resolved_start_index,
        window_end=resolved_end_index,
        generated_only=generated_only,
        prompt_length=prompt_length,
    )

    if verbose:
        span = schedule.resolved_span(prompt_length)
        if span is None:
            console.print(
                f"[yellow]Resolved token span is empty for prompt length {prompt_length}. Check your flags."
            )
        else:
            span_start, span_end = span
            notes: list[str] = []
            if schedule.has_window() and schedule.window_end == -1:
                notes.append("end follows the latest token (open-ended)")
            if schedule.generated_only:
                gen_origin = (
                    schedule.prompt_length if schedule.prompt_length is not None else prompt_length
                )
                notes.append(f"generated-only (>= index {gen_origin})")
            note_suffix = f" [{' | '.join(notes)}]" if notes else ""
            console.print(
                f"[cyan]Resolved token span:[/cyan] {span_start}..{span_end}{note_suffix}"
            )

    pad_token_id = cast(int | None, getattr(tokenizer, "pad_token_id", None))
    eos_token_id = cast(int | None, getattr(tokenizer, "eos_token_id", None))

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=max(temperature, 1e-8),
        top_p=top_p,
        do_sample=temperature > 0,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )

    vector_tensor: torch.Tensor | None = None
    if vector_path is not None:
        record = load_vector(vector_path)
        ensure_vector_matches_model(record.vector, model)
        vector_tensor = prepare_vector(record.vector, normalize, scale_by)
        metadata_layer = record.metadata.layer_index
        if metadata_layer is not None and metadata_layer != layer_index:
            console.print(
                f"[yellow]Warning:[/yellow] vector recorded from layer {metadata_layer}, but we will inject at layer {layer_index}."
            )

    disable_cache = vector_tensor is not None and schedule.requires_full_sequence()
    if requires_cache_disabled(model):
        disable_cache = True

    ctx = (
        injection_context(model, layer_index, vector_tensor, strength, schedule)
        if vector_tensor is not None
        else nullcontext()
    )

    with torch.no_grad(), ctx:
        output_ids = sampling_model.generate(
            **inputs,
            generation_config=generation_config,
            use_cache=not disable_cache,
        )
    text = _decode_output(tokenizer, output_ids[0], include_prompt=include_prompt)
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
    layer_indices: list[int] = SWEEP_LAYER_OPTION,
    strengths: list[float] = SWEEP_STRENGTH_OPTION,
    token_index: Annotated[
        int,
        typer.Option(help="Token index for single-token injection."),
    ] = -1,
    start_index: Annotated[
        int | None,
        typer.Option(help="Optional start index (inclusive) for windowed injection."),
    ] = None,
    end_index: Annotated[
        int | None,
        typer.Option(help="Optional end index (inclusive) for windowed injection."),
    ] = None,
    start_match: Annotated[
        str | None,
        typer.Option(
            help="Substring anchor; injection begins on the newline preceding this match."
        ),
    ] = None,
    end_match: Annotated[
        str | None,
        typer.Option(help="Substring anchor; injection ends on the newline following this match."),
    ] = None,
    start_occurrence: Annotated[
        int,
        typer.Option(help="1-based occurrence index for --start-match.", min=1, show_default=True),
    ] = 1,
    end_occurrence: Annotated[
        int,
        typer.Option(help="1-based occurrence index for --end-match.", min=1, show_default=True),
    ] = 1,
    apply_all_tokens: Annotated[
        bool,
        typer.Option(help="If set, injects into every token in the sequence."),
    ] = False,
    generated_only: Annotated[
        bool,
        typer.Option(help="Restrict injection to newly generated tokens."),
    ] = False,
    normalize: Annotated[
        bool,
        typer.Option(help="Normalize the vector to unit RMS before scaling.", show_default=True),
    ] = True,
    scale_by: Annotated[
        float,
        typer.Option(help="Extra multiplier applied after normalization.", show_default=True),
    ] = 1.0,
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
    """Sweep layer/strength combinations and log outputs to CSV."""

    if start_match is None and start_occurrence != 1:
        raise typer.BadParameter("--start-occurrence requires --start-match.")
    if end_match is None and end_occurrence != 1:
        raise typer.BadParameter("--end-occurrence requires --end-match.")

    if seed is not None:
        _seed_rng(seed)
    if not layer_indices:
        raise typer.BadParameter("Provide at least one --layer-index for the sweep.")
    if not strengths:
        raise typer.BadParameter("Provide at least one --strength for the sweep.")

    torch_dtype = resolve_dtype(dtype)
    torch_device = resolve_device(device)
    model, tokenizer = load_model_and_tokenizer(model_path, torch_dtype, torch_device)
    sampling_model = cast(Any, model)

    base_inputs = tokenize(tokenizer, prompt, torch_device)
    prompt_length = base_inputs["input_ids"].shape[1]

    resolved_start_index = start_index
    if start_match is not None:
        resolved_start_index = resolve_start_match_token_index(
            tokenizer, prompt, start_match, start_occurrence
        )
    resolved_end_index = end_index
    if end_match is not None:
        resolved_end_index = resolve_end_match_token_index(
            tokenizer, prompt, end_match, end_occurrence
        )
    if resolved_start_index is not None and resolved_end_index is None:
        resolved_end_index = -1

    schedule = InjectionSchedule(
        apply_all=apply_all_tokens,
        single_index=token_index,
        window_start=resolved_start_index,
        window_end=resolved_end_index,
        generated_only=generated_only,
        prompt_length=prompt_length,
    )

    pad_token_id = cast(int | None, getattr(tokenizer, "pad_token_id", None))
    eos_token_id = cast(int | None, getattr(tokenizer, "eos_token_id", None))

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=max(temperature, 1e-8),
        top_p=top_p,
        do_sample=temperature > 0,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )

    record = load_vector(vector_path)
    ensure_vector_matches_model(record.vector, model)
    vector_tensor = prepare_vector(record.vector, normalize, scale_by)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _generate_text(layer_idx: int | None, strength_value: float | None) -> str:
        cloned_inputs = clone_inputs(base_inputs)
        disable_cache = strength_value is not None and schedule.requires_full_sequence()
        if requires_cache_disabled(model):
            disable_cache = True
        ctx = (
            injection_context(model, layer_idx, vector_tensor, strength_value, schedule)
            if layer_idx is not None and strength_value is not None
            else nullcontext()
        )
        with torch.no_grad(), ctx:
            output_ids = sampling_model.generate(
                **cloned_inputs,
                generation_config=generation_config,
                use_cache=not disable_cache,
            )
        decoded = _decode_output(tokenizer, output_ids[0], include_prompt=include_prompt)
        return decoded

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

    for layer_idx in layer_indices:
        for strength_value in strengths:
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
