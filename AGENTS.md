## Finishing a coding task

- Run `just lint` - fix all issues
- Run `just test` - fix all issues
- Finally, run `just fmt`
- Update `README.md` appropriately
- Add any useful learnings to this file for future agents.

## Learnings

- `torch_dtype` is deprecated - use `dtype`
- `run --verbose` now prints the resolved `--start-match`/`--end-match` window; use `--start-occurrence` / `--end-occurrence` for the Nth anchor when slicing transcripts.
- CLI window math now lives in `text_utils.WindowSpec`; import and reuse it whenever you need to resolve `--start-*`/`--end-*` flags so behavior stays consistent.
- `WindowSpec.build_schedule(...)` now resolves those anchors/indexes into a ready-to-use `InjectionSchedule`; stop instantiating the schedule manually in CLI commands.
- CLI internals now live in helper modules (`app.py`, `models.py`, `vectors.py`, `injection.py`, `text_utils.py`, `baseline.py`); import from those instead of re-implementing utilities in `cli.py`.
- Vector serialization + injection scheduling use Pydantic (`VectorMetadata`, `VectorPayload`, `InjectionSchedule`). Always go through these helpers (and `save_vector`) so validation stays consistent.
- When you need a normalized/validated concept vector for a model, call `load_prepared_vector(path, model, normalize=..., scale_by=...)` instead of repeating `load_vector` + `ensure_vector_matches_model` + `prepare_vector` inside the CLI.
- BasedPyright is strict about `Unknown` types; cast Hugging Face objects (e.g., `AutoModelForCausalLM`, `BatchEncoding.data`, tokenizer `.decode`/`.pad_token_id`) to concrete types before mutating them or passing to torch helpers.
- `--token-index` is optional now; leaving it unset (especially with `--generated-only`) targets the entire generated suffix, while explicitly passing `-1` limits injection to the most recent token.
- `just lint` is the gatekeeper; run it first and follow with `just test` before `just fmt` whenever you want the full CI bundle.
- Set `TI_DEBUG_STRICT=1` if you need the hook to assert residual tensors are `[batch, tokens, hidden]` with the expected width during debugging.
- `cli.py` now exposes reusable Typer option singletons plus `_build_window_spec(...)` and `_generate_text_with_schedule(...)`; lean on those helpers when adding commands so window math, cache toggles, and verbose span reporting stay consistent.
