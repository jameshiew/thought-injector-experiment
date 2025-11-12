from __future__ import annotations

import csv
import difflib
import json
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypedDict

import torch
import transformers.utils as _transformers_utils
import typer
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils import ModelOutput

if not hasattr(_transformers_utils, "LossKwargs"):

    class _LossKwargs(TypedDict, total=False):  # type: ignore[misc]
        """Shim for older transformers builds lacking LossKwargs."""

        pass

    _transformers_utils.LossKwargs = _LossKwargs  # type: ignore[attr-defined]

console = Console()
app = typer.Typer(help="Local concept-injection experiments for safetensors-based LMs.")

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

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

DEFAULT_BASELINE_WORDS: list[str] = [
    # Concrete nouns
    "hats",
    "radios",
    "shirts",
    "trains",
    "locks",
    "boxes",
    "pants",
    "papers",
    "windows",
    "rings",
    "houses",
    "chairs",
    "mirrors",
    "walls",
    "necklaces",
    "books",
    "batteries",
    "desks",
    "bracelets",
    "keys",
    "rocks",
    "computers",
    "trees",
    "bottles",
    "offices",
    "cameras",
    "gloves",
    "coins",
    "cars",
    "watches",
    "buildings",
    "lamps",
    "clocks",
    "bicycles",
    "speakers",
    "floors",
    "phones",
    "ceilings",
    "ships",
    "tables",
    "apartments",
    "bridges",
    "televisions",
    "shoes",
    "doors",
    "needles",
    "pens",
    "airplanes",
    "roads",
    "pencils",
    # Abstract nouns
    "duty",
    "evil",
    "progress",
    "creativity",
    "mastery",
    "competition",
    "change",
    "peace",
    "honor",
    "good",
    "unity",
    "diversity",
    "trust",
    "chaos",
    "liberty",
    "balance",
    "harmony",
    "equality",
    "conflict",
    "justice",
    "ugliness",
    "morality",
    "innovation",
    "power",
    "space",
    "tradition",
    "wisdom",
    "failure",
    "democracy",
    "time",
    "loyalty",
    "privilege",
    "order",
    "authority",
    "freedom",
    "ethics",
    "cooperation",
    "independence",
    "defeat",
    "truth",
    "betrayal",
    "dignity",
    "success",
    "courage",
    "victory",
    "faith",
    "knowledge",
    "rights",
    "intelligence",
    "beauty",
    # Verbs
    "thinking",
    "laughing",
    "drinking",
    "singing",
    "whispering",
    "reading",
    "dreaming",
    "catching",
    "pulling",
    "crying",
    "breathing",
    "studying",
    "writing",
    "screaming",
    "growing",
    "talking",
    "dancing",
    "falling",
    "cooking",
    "winning",
    "shouting",
    "learning",
    "creating",
    "eating",
    "pushing",
    "playing",
    "teaching",
    "swimming",
    "speaking",
    "destroying",
    "smiling",
    "shrinking",
    "sinking",
    "breaking",
    "rising",
    "floating",
    "racing",
    "sleeping",
    "working",
    "jumping",
    "driving",
    "walking",
    "flying",
    "sculpting",
    "building",
    "frowning",
    "striving",
    "running",
    "listening",
    "throwing",
]


@dataclass
class InjectionSchedule:
    apply_all: bool = False
    single_index: int | None = None
    window_start: int | None = None
    window_end: int | None = None
    generated_only: bool = False
    prompt_length: int | None = None

    def has_window(self) -> bool:
        return self.window_start is not None or self.window_end is not None

    def requires_full_sequence(self) -> bool:
        return self.apply_all or self.has_window() or self.generated_only

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
            start_raw = 0 if self.window_start is None else self.window_start
            end_raw = -1 if self.window_end is None else self.window_end
            start_idx = _resolve_token_index(start_raw, seq_len)
            end_idx = _resolve_token_index(end_raw, seq_len)
            if end_idx < start_idx:
                raise typer.BadParameter(
                    "Window end index must be greater than or equal to start index once resolved."
                )
            mask[start_idx : end_idx + 1] = True
        elif self.single_index is not None:
            idx = _resolve_token_index(self.single_index, seq_len)
            mask[idx] = True
        else:
            idx = _resolve_token_index(-1, seq_len)
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


def _gpu_supports_bfloat16() -> bool:
    """Return True when the primary CUDA device supports bfloat16."""

    if not torch.cuda.is_available():
        return False

    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(is_bf16_supported):
        try:
            if torch.cuda.is_bf16_supported():
                return True
        except RuntimeError:
            return False

    try:
        capability = torch.cuda.get_device_capability()
    except RuntimeError:
        return False
    # Ampere (sm80) or newer GPUs provide native bfloat16 support.
    return capability[0] >= 8


@dataclass
class VectorRecord:
    vector: torch.Tensor
    metadata: dict[str, object]


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "auto":
        name = "bfloat16" if _gpu_supports_bfloat16() else "float16"
        console.print(f"Auto-selecting torch dtype '{name}'.")
    try:
        return DTYPE_MAP[name]
    except KeyError as exc:  # pragma: no cover - defensive.
        raise typer.BadParameter(
            f"Unsupported dtype '{name}'. Choose from {list(DTYPE_MAP)}"
        ) from exc


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_model_and_tokenizer(model_path: Path, dtype: torch.dtype, device: torch.device):
    console.print(f"Loading model from {model_path} (dtype={dtype}, device={device}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _get_decoder_layers(model):
    candidates = ["model", "transformer", "base_model", "decoder"]
    for attr in candidates:
        sub = getattr(model, attr, None)
        if sub is None:
            continue
        for layer_attr in ("layers", "h"):
            layers = getattr(sub, layer_attr, None)
            if layers is not None:
                return layers
    raise RuntimeError(
        "Could not locate decoder layers. This implementation currently supports Llama-style models."
    )


def _resolve_layer(model, layer_index: int):
    layers = _get_decoder_layers(model)
    if layer_index < 0 or layer_index >= len(layers):
        raise typer.BadParameter(
            f"layer-index must be in [0, {len(layers) - 1}] but got {layer_index}"
        )
    return layers[layer_index]


def _tokenize(tokenizer, prompt: str, device: torch.device):
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded.pop("token_type_ids", None)  # Decoder-only models typically reject this field.
    return {k: v.to(device) for k, v in encoded.items()}


def _extract_hidden_state(
    model, tokenizer, prompt: str, layer_index: int, token_index: int, device: torch.device
):
    inputs = _tokenize(tokenizer, prompt, device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    if layer_index >= model.config.num_hidden_layers:
        raise typer.BadParameter(
            f"layer-index {layer_index} exceeds num_hidden_layers={model.config.num_hidden_layers}"
        )
    layer_hidden = hidden_states[layer_index + 1]  # +1 skips embedding output.
    resolved_index = _resolve_token_index(token_index, layer_hidden.shape[1])
    return layer_hidden[:, resolved_index, :].mean(dim=0).detach().cpu()


def _resolve_token_index(index: int, seq_len: int) -> int:
    if index < 0:
        index = seq_len + index
    if index < 0 or index >= seq_len:
        raise typer.BadParameter(f"token-index out of range for sequence length {seq_len}")
    return index


def _save_vector(path: Path, vector: torch.Tensor, metadata: dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"vector": vector, "metadata": metadata}
    torch.save(payload, path)
    console.print(f"Saved vector -> {path}")


def _load_vector(path: Path) -> VectorRecord:
    payload = torch.load(path, map_location="cpu")
    if "vector" not in payload:
        raise typer.BadParameter(f"Vector file {path} missing 'vector' key")
    metadata = payload.get("metadata", {})
    return VectorRecord(vector=payload["vector"], metadata=metadata)


def _broadcast_vector(vector: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    return vector.to(dtype=hidden_states.dtype, device=hidden_states.device)


def _load_baseline_words(baseline_path: Path | None) -> list[str]:
    if baseline_path is None:
        return list(DEFAULT_BASELINE_WORDS)
    if not baseline_path.exists():
        raise typer.BadParameter(f"Baseline word file '{baseline_path}' not found.")
    words = [line.strip() for line in baseline_path.read_text().splitlines() if line.strip()]
    if not words:
        raise typer.BadParameter("Baseline word list is empty.")
    return words


def _normalize_vector(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(vector.to(torch.float32) ** 2))
    if torch.isnan(rms) or rms.item() < eps:
        raise typer.BadParameter("Vector has near-zero RMS; cannot normalize.")
    return vector / rms


def _prepare_vector(vector: torch.Tensor, normalize: bool, scale_by: float) -> torch.Tensor:
    result = vector.to(torch.float32)
    if normalize:
        result = _normalize_vector(result)
    if scale_by != 1.0:
        result = result * scale_by
    return result


def _clone_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in inputs.items()}


def _locate_start_match(prompt: str, match: str) -> int:
    match_index = prompt.find(match)
    if match_index == -1:
        raise typer.BadParameter(f"start_match '{match}' not found inside prompt text.")
    newline_index = prompt.rfind("\n", 0, match_index)
    return newline_index if newline_index != -1 else match_index


def _token_index_from_char(tokenizer, prompt: str, char_index: int) -> int:
    encoding = tokenizer(
        prompt,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = encoding.get("offset_mapping")
    if offsets is None:
        raise typer.BadParameter("Tokenizer must provide offset_mapping for --start-match.")
    offsets_seq = offsets[0] if offsets and isinstance(offsets[0], list) else offsets

    for idx, (start, end) in enumerate(offsets_seq):
        if start <= char_index < end:
            return idx
    if char_index >= len(prompt):
        return len(offsets_seq) - 1
    raise typer.BadParameter(
        "Could not map character offset to a tokenizer index; check --start-match anchor."
    )


def _resolve_start_match_token_index(tokenizer, prompt: str, match: str) -> int:
    anchor_char = _locate_start_match(prompt, match)
    return _token_index_from_char(tokenizer, prompt, anchor_char)


def _diff_length(reference: str, candidate: str) -> int:
    diff_total = 0
    for chunk in difflib.ndiff(reference, candidate):
        if not chunk:
            continue
        if chunk[0] in {"+", "-"}:
            diff_total += len(chunk[2:])
    return diff_total


def _apply_injection(
    hidden_states: torch.Tensor,
    vector: torch.Tensor,
    strength: float,
    schedule: InjectionSchedule,
):
    mask = schedule.resolve_mask(hidden_states.shape[1], hidden_states.device)
    if not torch.any(mask):
        return hidden_states
    vector = strength * _broadcast_vector(vector, hidden_states)
    hidden_states = hidden_states.clone()
    hidden_states[:, mask, :] += vector
    return hidden_states


def _remix_output(output, mutate_fn):
    if isinstance(output, torch.Tensor):
        return mutate_fn(output)
    if isinstance(output, tuple):
        mutated = mutate_fn(output[0])
        return (mutated, *output[1:])
    if isinstance(output, ModelOutput):
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            return output
        output.last_hidden_state = mutate_fn(hidden)
        return output
    return output  # pragma: no cover - defensive fallback.


def _register_injection(
    model,
    layer_index: int,
    vector: torch.Tensor,
    strength: float,
    schedule: InjectionSchedule,
):
    layer = _resolve_layer(model, layer_index)

    def hook(module, inputs, output):  # pylint: disable=unused-argument
        return _remix_output(
            output,
            lambda hidden: _apply_injection(hidden, vector, strength, schedule),
        )

    return layer.register_forward_hook(hook)


@contextmanager
def injection_context(
    model,
    layer_index: int,
    vector: torch.Tensor,
    strength: float,
    schedule: InjectionSchedule,
):
    handle = _register_injection(model, layer_index, vector, strength, schedule)
    try:
        yield
    finally:
        handle.remove()


def _ensure_vector_matches_model(vector: torch.Tensor, model):
    hidden_size = model.config.hidden_size
    if vector.shape[-1] != hidden_size:
        raise typer.BadParameter(
            f"Vector hidden size {vector.shape[-1]} != model hidden size {hidden_size}."
        )


@app.command()
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
):
    """Capture a concept vector by differencing two prompts."""

    torch_dtype = _resolve_dtype(dtype)
    torch_device = _resolve_device(device)
    model, tokenizer = _load_model_and_tokenizer(model_path, torch_dtype, torch_device)

    pos_hidden = _extract_hidden_state(
        model, tokenizer, positive_prompt, layer_index, token_index, torch_device
    )
    neg_hidden = _extract_hidden_state(
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
    _save_vector(output_path, vector, metadata)


@app.command()
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
):
    """Capture a concept vector by subtracting a baseline mean from a target word."""

    if baseline_count <= 0:
        raise typer.BadParameter("baseline-count must be positive.")

    torch_dtype = _resolve_dtype(dtype)
    torch_device = _resolve_device(device)
    model, tokenizer = _load_model_and_tokenizer(model_path, torch_dtype, torch_device)

    positive_prompt = f"Tell me about {word}."
    target_hidden = _extract_hidden_state(
        model, tokenizer, positive_prompt, layer_index, token_index, torch_device
    )

    baseline_words = _load_baseline_words(baseline_path)
    baseline_words = [w for w in baseline_words if w.lower() != word.lower()]
    selected = baseline_words[:baseline_count]
    if not selected:
        raise typer.BadParameter("Baseline word list is empty after filtering the target word.")
    if len(selected) < baseline_count:
        console.print(
            f"[yellow]Warning:[/yellow] requested {baseline_count} baseline words but only found {len(selected)}."
        )

    baseline_vectors = []
    for base_word in selected:
        base_prompt = f"Tell me about {base_word}."
        hidden = _extract_hidden_state(
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
    _save_vector(output_path, vector, metadata)
    rms = torch.sqrt(torch.mean(vector**2)).item()
    console.print(f"Vector RMS: {rms:.6f}")


@app.command()
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
):
    """Run the prompt with optional activation injection."""

    if seed is not None:
        torch.manual_seed(seed)

    torch_dtype = _resolve_dtype(dtype)
    torch_device = _resolve_device(device)
    model, tokenizer = _load_model_and_tokenizer(model_path, torch_dtype, torch_device)

    inputs = _tokenize(tokenizer, prompt, torch_device)
    prompt_length = inputs["input_ids"].shape[1]

    resolved_start_index = start_index
    if start_match is not None:
        resolved_start_index = _resolve_start_match_token_index(tokenizer, prompt, start_match)
    resolved_end_index = end_index
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

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=max(temperature, 1e-8),
        top_p=top_p,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    vector_tensor: torch.Tensor | None = None
    handle = None
    if vector_path is not None:
        record = _load_vector(vector_path)
        _ensure_vector_matches_model(record.vector, model)
        vector_tensor = _prepare_vector(record.vector, normalize, scale_by)
        metadata_layer = record.metadata.get("layer_index")
        if metadata_layer is not None and metadata_layer != layer_index:
            console.print(
                f"[yellow]Warning:[/yellow] vector recorded from layer {metadata_layer}, but we will inject at layer {layer_index}."
            )
        handle = _register_injection(model, layer_index, vector_tensor, strength, schedule)

    disable_cache = vector_path is not None and schedule.requires_full_sequence()

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                generation_config=generation_config,
                use_cache=not disable_cache,
            )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=not include_prompt)
        console.print("=== Model Output ===")
        console.print(text)
    finally:
        if handle is not None:
            handle.remove()


@app.command()
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
):
    """Sweep layer/strength combinations and log outputs to CSV."""

    if seed is not None:
        torch.manual_seed(seed)
    if not layer_indices:
        raise typer.BadParameter("Provide at least one --layer-index for the sweep.")
    if not strengths:
        raise typer.BadParameter("Provide at least one --strength for the sweep.")

    torch_dtype = _resolve_dtype(dtype)
    torch_device = _resolve_device(device)
    model, tokenizer = _load_model_and_tokenizer(model_path, torch_dtype, torch_device)

    base_inputs = _tokenize(tokenizer, prompt, torch_device)
    prompt_length = base_inputs["input_ids"].shape[1]

    resolved_start_index = start_index
    if start_match is not None:
        resolved_start_index = _resolve_start_match_token_index(tokenizer, prompt, start_match)
    resolved_end_index = end_index
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

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=max(temperature, 1e-8),
        top_p=top_p,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    record = _load_vector(vector_path)
    _ensure_vector_matches_model(record.vector, model)
    vector_tensor = _prepare_vector(record.vector, normalize, scale_by)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _generate_text(layer_idx: int | None, strength_value: float | None) -> str:
        cloned_inputs = _clone_inputs(base_inputs)
        disable_cache = strength_value is not None and schedule.requires_full_sequence()
        ctx = (
            injection_context(model, layer_idx, vector_tensor, strength_value, schedule)
            if layer_idx is not None and strength_value is not None
            else nullcontext()
        )
        with torch.no_grad(), ctx:
            output_ids = model.generate(
                **cloned_inputs,
                generation_config=generation_config,
                use_cache=not disable_cache,
            )
        return tokenizer.decode(output_ids[0], skip_special_tokens=not include_prompt)

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
            diff_len = _diff_length(baseline_text, trial_text)
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


@app.command()
def inspect_vector(
    vector_path: Annotated[Path, typer.Argument(..., help="Path to a saved vector.")],
):
    """Print metadata for a concept vector."""

    record = _load_vector(vector_path)
    console.print(json.dumps(record.metadata, indent=2))
    console.print(f"Vector shape: {tuple(record.vector.shape)}")


if __name__ == "__main__":
    app()
