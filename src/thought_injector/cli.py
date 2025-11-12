import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import typer
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

console = Console()
app = typer.Typer(help="Local concept-injection experiments for safetensors-based LMs.")

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class VectorRecord:
    vector: torch.Tensor
    metadata: Dict[str, object]


def _resolve_dtype(name: str) -> torch.dtype:
    try:
        return DTYPE_MAP[name]
    except KeyError as exc:  # pragma: no cover - defensive.
        raise typer.BadParameter(f"Unsupported dtype '{name}'. Choose from {list(DTYPE_MAP)}") from exc


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_model_and_tokenizer(model_path: Path, dtype: torch.dtype, device: torch.device):
    console.print(f"Loading model from {model_path} (dtype={dtype}, device={device}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
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
    raise RuntimeError("Could not locate decoder layers. This implementation currently supports Llama-style models.")


def _resolve_layer(model, layer_index: int):
    layers = _get_decoder_layers(model)
    if layer_index < 0 or layer_index >= len(layers):
        raise typer.BadParameter(f"layer-index must be in [0, {len(layers) - 1}] but got {layer_index}")
    return layers[layer_index]


def _tokenize(tokenizer, prompt: str, device: torch.device):
    encoded = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in encoded.items()}


def _extract_hidden_state(model, tokenizer, prompt: str, layer_index: int, token_index: int, device: torch.device):
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


def _save_vector(path: Path, vector: torch.Tensor, metadata: Dict[str, object]):
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


def _apply_injection(hidden_states: torch.Tensor, vector: torch.Tensor, token_index: Optional[int], strength: float, apply_all: bool):
    vector = strength * _broadcast_vector(vector, hidden_states)
    if apply_all:
        return hidden_states + vector
    resolved = _resolve_token_index(token_index or -1, hidden_states.shape[1])
    hidden_states = hidden_states.clone()
    hidden_states[:, resolved, :] += vector
    return hidden_states


def _remix_output(output, mutate_fn):
    if isinstance(output, torch.Tensor):
        return mutate_fn(output)
    if isinstance(output, tuple):
        mutated = mutate_fn(output[0])
        return (mutated, *output[1:])
    return output  # pragma: no cover - defensive fallback.


def _register_injection(model, layer_index: int, vector: torch.Tensor, token_index: Optional[int], strength: float, apply_all: bool):
    layer = _resolve_layer(model, layer_index)

    def hook(module, inputs, output):  # pylint: disable=unused-argument
        return _remix_output(
            output,
            lambda hidden: _apply_injection(hidden, vector, token_index, strength, apply_all),
        )

    return layer.register_forward_hook(hook)


@contextmanager
def injection_context(model, layer_index: int, vector: torch.Tensor, token_index: Optional[int], strength: float, apply_all: bool):
    handle = _register_injection(model, layer_index, vector, token_index, strength, apply_all)
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
    model_path: Path = typer.Argument(..., help="Path to the local HF-format model directory."),
    positive_prompt: str = typer.Option(..., help="Prompt expected to activate the concept."),
    negative_prompt: str = typer.Option(..., help="Prompt without the concept."),
    layer_index: int = typer.Option(0, help="Decoder layer to sample (0-based)."),
    token_index: int = typer.Option(-1, help="Token index (supports negatives for counting from the end)."),
    output_path: Path = typer.Option(Path("vectors/concept.pt"), help="Where to store the concept vector."),
    dtype: str = typer.Option("float16", help="torch dtype for model weights."),
    device: str = typer.Option("auto", help="Device identifier or 'auto'."),
):
    """Capture a concept vector by differencing two prompts."""

    torch_dtype = _resolve_dtype(dtype)
    torch_device = _resolve_device(device)
    model, tokenizer = _load_model_and_tokenizer(model_path, torch_dtype, torch_device)

    pos_hidden = _extract_hidden_state(model, tokenizer, positive_prompt, layer_index, token_index, torch_device)
    neg_hidden = _extract_hidden_state(model, tokenizer, negative_prompt, layer_index, token_index, torch_device)
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
def run(
    model_path: Path = typer.Argument(..., help="Path to the local HF-format model directory."),
    prompt: str = typer.Option(..., help="Prompt to feed the model."),
    vector_path: Optional[Path] = typer.Option(None, help="Optional concept vector to inject."),
    layer_index: int = typer.Option(0, help="Layer to inject into."),
    token_index: int = typer.Option(-1, help="Token index for injection."),
    strength: float = typer.Option(1.0, help="Multiplier applied to the concept vector."),
    apply_all_tokens: bool = typer.Option(False, help="If set, injects into every token in the sequence."),
    max_new_tokens: int = typer.Option(128, help="Number of new tokens to sample."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    top_p: float = typer.Option(0.9, help="Top-p nucleus sampling."),
    dtype: str = typer.Option("float16", help="torch dtype for model weights."),
    device: str = typer.Option("auto", help="Device identifier or 'auto'."),
    seed: Optional[int] = typer.Option(None, help="Optional RNG seed for reproducibility."),
    include_prompt: bool = typer.Option(False, help="Return the full decoded sequence (prompt + continuation)."),
):
    """Run the prompt with optional activation injection."""

    if seed is not None:
        torch.manual_seed(seed)

    torch_dtype = _resolve_dtype(dtype)
    torch_device = _resolve_device(device)
    model, tokenizer = _load_model_and_tokenizer(model_path, torch_dtype, torch_device)

    handle = None
    if vector_path is not None:
        record = _load_vector(vector_path)
        _ensure_vector_matches_model(record.vector, model)
        metadata_layer = record.metadata.get("layer_index")
        if metadata_layer is not None and metadata_layer != layer_index:
            console.print(
                f"[yellow]Warning:[/yellow] vector recorded from layer {metadata_layer}, but we will inject at layer {layer_index}."
            )
        handle = _register_injection(model, layer_index, record.vector, token_index, strength, apply_all_tokens)

    try:
        inputs = _tokenize(tokenizer, prompt, torch_device)
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-8),
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        with torch.no_grad():
            output_ids = model.generate(**inputs, generation_config=generation_config)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=not include_prompt)
        console.print("=== Model Output ===")
        console.print(text)
    finally:
        if handle is not None:
            handle.remove()


@app.command()
def inspect_vector(vector_path: Path = typer.Argument(..., help="Path to a saved vector.")):
    """Print metadata for a concept vector."""

    record = _load_vector(vector_path)
    console.print(json.dumps(record.metadata, indent=2))
    console.print(f"Vector shape: {tuple(record.vector.shape)}")


if __name__ == "__main__":
    app()
