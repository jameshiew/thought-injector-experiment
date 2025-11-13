from __future__ import annotations

"""Utilities for loading minimal-pair prompt datasets."""

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import typer

POSITIVE_KEYS: tuple[str, ...] = ("positive", "concept", "target")
NEGATIVE_KEYS: tuple[str, ...] = ("negative", "baseline", "control")


@dataclass(frozen=True, slots=True)
class PromptPair:
    positive: str
    negative: str


def load_prompt_pairs(path: Path) -> list[PromptPair]:
    """Load `PromptPair` definitions from json/jsonl/csv/tsv files."""

    if not path.exists():
        raise typer.BadParameter(f"Pairs file '{path}' not found.")

    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        pairs = _load_jsonl_pairs(path)
    elif suffix == ".json":
        pairs = _load_json_pairs(path)
    elif suffix == ".csv":
        pairs = _load_csv_pairs(path, delimiter=",")
    elif suffix == ".tsv":
        pairs = _load_csv_pairs(path, delimiter="\t")
    else:
        raise typer.BadParameter(
            f"Pairs files must be JSON (.json/.jsonl/.ndjson) or CSV/TSV; got '{path.name}'."
        )

    if not pairs:
        raise typer.BadParameter(f"Pairs file '{path}' did not contain any prompt pairs.")
    return pairs


def _load_jsonl_pairs(path: Path) -> list[PromptPair]:
    pairs: list[PromptPair] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive.
                raise typer.BadParameter(
                    f"Pairs file '{path}' has invalid JSON on line {line_no}: {exc.msg}"
                ) from exc
            mapping = _require_mapping(payload, path, f"line {line_no}")
            pairs.append(_pair_from_mapping(mapping, path, f"line {line_no}"))
    return pairs


def _load_json_pairs(path: Path) -> list[PromptPair]:
    try:
        payload_data: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive.
        raise typer.BadParameter(f"Pairs file '{path}' contains invalid JSON: {exc.msg}") from exc

    records: Any
    if isinstance(payload_data, Mapping):
        payload_mapping = cast(Mapping[str, Any], payload_data)
        records = payload_mapping.get("pairs")
        if records is None:
            raise typer.BadParameter(
                f"Pairs file '{path}' must contain a top-level list or a 'pairs' field."
            )
    elif isinstance(payload_data, list):
        records = cast(list[Any], payload_data)
    else:
        raise typer.BadParameter(
            f"Pairs file '{path}' must be a list of objects or include a top-level 'pairs' list."
        )

    records_iter = cast(Iterable[Any], records)
    return _pairs_from_iterable(records_iter, path, "entry")


def _load_csv_pairs(path: Path, *, delimiter: str) -> list[PromptPair]:
    pairs: list[PromptPair] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise typer.BadParameter(
                f"Pairs file '{path}' is missing a header row with 'positive'/'negative' columns."
            )
        for row_no, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            mapping = {key: value for key, value in row.items() if key is not None}
            pairs.append(_pair_from_mapping(mapping, path, f"row {row_no}"))
    return pairs


def _pairs_from_iterable(items: Iterable[Any], path: Path, label: str) -> list[PromptPair]:
    pairs: list[PromptPair] = []
    for index, item in enumerate(items, start=1):
        mapping = _require_mapping(item, path, f"{label} {index}")
        pairs.append(_pair_from_mapping(mapping, path, f"{label} {index}"))
    return pairs


def _require_mapping(obj: Any, path: Path, label: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise typer.BadParameter(
            f"Pairs file '{path}' {label} must be an object with positive/negative prompts."
        )
    return cast(Mapping[str, Any], obj)


def _pair_from_mapping(obj: Mapping[str, Any], path: Path, label: str) -> PromptPair:
    positive = _extract_prompt(obj, POSITIVE_KEYS, path, label, role="positive")
    negative = _extract_prompt(obj, NEGATIVE_KEYS, path, label, role="baseline/negative")
    return PromptPair(positive=positive, negative=negative)


def _extract_prompt(
    obj: Mapping[str, Any],
    keys: tuple[str, ...],
    path: Path,
    label: str,
    *,
    role: str,
) -> str:
    selected_key: str | None = None
    value: Any | None = None
    for key in keys:
        if key in obj:
            selected_key = key
            value = obj[key]
            break
    if selected_key is None:
        raise typer.BadParameter(
            f"Pairs file '{path}' {label} is missing the {role} field (expected one of {keys})."
        )
    if value is None:
        raise typer.BadParameter(
            f"Pairs file '{path}' {label} must supply text for field {selected_key!r}."
        )
    text = str(value).strip()
    if not text:
        raise typer.BadParameter(
            f"Pairs file '{path}' {label} has an empty {role} prompt (field {selected_key!r})."
        )
    return text
