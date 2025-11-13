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
- When you reuse those Typer option singletons inside `Annotated[...]` parameters, keep the first positional argument as `...` and assign the real default (`= False`, `= None`, etc.) on the parameter. Passing the default directly to `typer.Option` sneaks an extra `None` into the flag declarations and Typer/Click will crash before the CLI even renders.
- Keep Click pinned to 8.1.x for now. Click 8.3.0’s stricter boolean flag detection fights with Typer 0.12’s `--foo/--no-foo` syntax and raises `Secondary flag is not valid for non-boolean flag.` during CLI startup.
- Typer 0.20.0 still rides on Click 8.1.x—upgrading Click past 8.1.8 revives the `Secondary flag is not valid for non-boolean flag.` crash when using the shared `--normalize/--no-normalize` toggles, so keep the `<8.2` guard in `pyproject.toml`.
- `transformers==4.57.1` hard-pins `huggingface-hub<1.0`; the newest release that satisfies that cap is `huggingface-hub==0.36.0` (2025-10-23). Use that floor so you still pick up the latest fixes without tripping the constraint.
- README sanity check: you can replay the documented flow end-to-end today by running `capture-word`/`run` against `models/pharia-1-control`; saving a fresh vector (e.g., `vectors/aquariums_word_pharia_dep_bump.pt`) keeps both baseline and injected runs matching the write-up.
- `gpu_supports_bfloat16()` only opts you into `bfloat16` when CUDA is available *and* reports bf16 support; otherwise helpers automatically fall back to `float16`, so rely on `resolve_dtype('auto')` instead of guessing.
