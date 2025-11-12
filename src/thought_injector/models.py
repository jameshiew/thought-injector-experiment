from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

import torch
import torch.nn as nn
import transformers.utils as _transformers_utils
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from thought_injector.app import console

if not hasattr(_transformers_utils, "LossKwargs"):

    class _LossKwargs(TypedDict, total=False):
        """Shim for older transformers builds lacking LossKwargs."""

        pass

    cast(Any, _transformers_utils).LossKwargs = _LossKwargs

if not hasattr(DynamicCache, "get_max_length"):

    def _compat_dynamic_cache_max_length(self: DynamicCache) -> int:
        return int(self.get_seq_length())

    cast(Any, DynamicCache).get_max_length = _compat_dynamic_cache_max_length


DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class LayerSequence(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> nn.Module: ...


class HiddenStateOutput(Protocol):
    hidden_states: tuple[torch.Tensor, ...] | None


def gpu_supports_bfloat16() -> bool:
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
        capability: tuple[int, int] = torch.cuda.get_device_capability()
    except RuntimeError:
        return False
    return capability[0] >= 8


def resolve_dtype(name: str) -> torch.dtype:
    if name == "auto":
        name = "bfloat16" if gpu_supports_bfloat16() else "float16"
        console.print(f"Auto-selecting torch dtype '{name}'.")
    try:
        return DTYPE_MAP[name]
    except KeyError as exc:  # pragma: no cover - defensive.
        raise typer.BadParameter(
            f"Unsupported dtype '{name}'. Choose from {list(DTYPE_MAP)}"
        ) from exc


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_model_and_tokenizer(
    model_path: Path, dtype: torch.dtype, device: torch.device
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    console.print(f"Loading model from {model_path} (dtype={dtype}, device={device}) ...")
    hf_model: Any = cast(Any, AutoModelForCausalLM).from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
    hf_model.to(device)
    hf_model.eval()
    model = cast(PreTrainedModel, hf_model)
    hf_tokenizer: Any = cast(Any, AutoTokenizer).from_pretrained(
        model_path, use_fast=True, local_files_only=True
    )
    tokenizer = cast(PreTrainedTokenizerBase, hf_tokenizer)
    pad_token = cast(str | list[str] | None, tokenizer.pad_token)
    if pad_token is None:
        eos_token = cast(str | list[str] | None, tokenizer.eos_token)
        tokenizer.pad_token = eos_token
    return model, tokenizer


def requires_cache_disabled(model: PreTrainedModel) -> bool:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return model_type in {"pharia", "pharia-v1", "pharia_v1"}


def get_decoder_layers(model: PreTrainedModel) -> LayerSequence:
    candidates = ["model", "transformer", "base_model", "decoder"]
    for attr in candidates:
        sub = getattr(model, attr, None)
        if sub is None:
            continue
        for layer_attr in ("layers", "h"):
            layers = getattr(sub, layer_attr, None)
            if layers is not None:
                return cast(LayerSequence, layers)
    raise RuntimeError(
        "Could not locate decoder layers. This implementation currently supports Llama-style models."
    )


def resolve_layer(model: PreTrainedModel, layer_index: int) -> nn.Module:
    layers = get_decoder_layers(model)
    if layer_index < 0 or layer_index >= len(layers):
        raise typer.BadParameter(
            f"layer-index must be in [0, {len(layers) - 1}] but got {layer_index}"
        )
    return layers[layer_index]


def tokenize(
    tokenizer: PreTrainedTokenizerBase, prompt: str, device: torch.device
) -> dict[str, torch.Tensor]:
    encoded: BatchEncoding = tokenizer(prompt, return_tensors="pt")
    encoded_data = cast(dict[str, torch.Tensor], encoded.data)
    tensors: dict[str, torch.Tensor] = dict(encoded_data)

    tensors.pop("token_type_ids", None)
    mask = tensors.get("attention_mask")
    if mask is not None and torch.count_nonzero(mask != 1) == 0:
        tensors.pop("attention_mask", None)
    return {k: v.to(device) for k, v in tensors.items()}


def resolve_token_index(index: int, seq_len: int) -> int:
    if index < 0:
        index = seq_len + index
    if index < 0 or index >= seq_len:
        raise typer.BadParameter(f"token-index out of range for sequence length {seq_len}")
    return index


def extract_hidden_state(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    layer_index: int,
    token_index: int,
    device: torch.device,
) -> torch.Tensor:
    inputs = tokenize(tokenizer, prompt, device)
    with torch.no_grad():
        outputs = cast(
            HiddenStateOutput,
            model(**inputs, output_hidden_states=True, use_cache=False),
        )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError(
            "Model did not return hidden states; enable output_hidden_states support."
        )
    config_layers = getattr(getattr(model, "config", object()), "num_hidden_layers", None)
    num_layers = int(config_layers) if config_layers is not None else len(get_decoder_layers(model))
    if layer_index >= num_layers:
        raise typer.BadParameter(
            f"layer-index {layer_index} exceeds available layers ({num_layers})."
        )
    layer_hidden = hidden_states[layer_index + 1]
    resolved_index = resolve_token_index(token_index, layer_hidden.shape[1])
    return layer_hidden[:, resolved_index, :].mean(dim=0).detach().cpu()


def clone_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in inputs.items()}
