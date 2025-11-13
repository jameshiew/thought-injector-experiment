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
- `WindowSpec.validate()` now enforces that `--start-index` vs. `--start-match` (and the analogous end flags) are mutually exclusive so we never silently pick one anchor over another.
- `WindowSpec.build_schedule(...)` now resolves those anchors/indexes into a ready-to-use `InjectionSchedule`; stop instantiating the schedule manually in CLI commands.
- CLI internals now live in helper modules (`app.py`, `models.py`, `vectors.py`, `injection.py`, `text_utils.py`, `baseline.py`); import from those instead of re-implementing utilities in `cli.py`.
- Vector serialization + injection scheduling use Pydantic (`VectorMetadata`, `InjectionSchedule`). Always go through these helpers (and `save_vector`) so validation stays consistent.
- Concept vectors now persist as `<name>.safetensors` tensors plus `<name>.json` metadata sidecars; `save_vector` enforces the extension so never call it with `.pt` paths.
- When you need a normalized/validated concept vector for a model, call `load_prepared_vector(path, model, normalize=..., scale_by=...)` instead of repeating `load_vector` + `ensure_vector_matches_model` + `prepare_vector` inside the CLI.
- BasedPyright is strict about `Unknown` types; cast Hugging Face objects (e.g., `AutoModelForCausalLM`, `BatchEncoding.data`, tokenizer `.decode`/`.pad_token_id`) to concrete types before mutating them or passing to torch helpers.
- `--token-index` is optional now; leaving it unset (especially with `--generated-only`) targets the entire generated suffix, while explicitly passing `-1` limits injection to the most recent token.
- Layer indices are strictly 0-based across capture/run/sweep; negatives are rejected so pass the exact decoder block you want.
- `just lint` is the gatekeeper; run it first and follow with `just test` before `just fmt` whenever you want the full CI bundle.
- Set `TI_DEBUG_STRICT=1` if you need the hook to assert residual tensors are `[batch, tokens, hidden]` with the expected width during debugging.
- `cli.py` now exposes reusable Typer option singletons plus `_build_window_spec(...)` and `_generate_text_with_schedule(...)`; lean on those helpers when adding commands so window math, cache toggles, and verbose span reporting stay consistent.
- When you reuse those Typer option singletons inside `Annotated[...]` parameters, keep the first positional argument as `...` and assign the real default (`= False`, `= None`, etc.) on the parameter. Passing the default directly to `typer.Option` sneaks an extra `None` into the flag declarations and Typer/Click will crash before the CLI even renders.
- Keep Click pinned to 8.1.x for now. Click 8.3.0’s stricter boolean flag detection fights with Typer 0.12’s `--foo/--no-foo` syntax and raises `Secondary flag is not valid for non-boolean flag.` during CLI startup.
- Typer 0.20.0 still rides on Click 8.1.x—upgrading Click past 8.1.8 revives the `Secondary flag is not valid for non-boolean flag.` crash when using the shared `--normalize/--no-normalize` toggles, so keep the `<8.2` guard in `pyproject.toml`.
- `transformers==4.57.1` hard-pins `huggingface-hub<1.0`; the newest release that satisfies that cap is `huggingface-hub==0.36.0` (2025-10-23). Use that floor so you still pick up the latest fixes without tripping the constraint.
- `download.py` can now mirror `meta-llama/Llama-3.1-8B-Instruct` via the `llama-3.1-8b-instruct` key; export `HF_TOKEN` or pass `--token` because Meta’s repo stays behind a gated EULA.
- README sanity check: you can replay the documented flow end-to-end today by running `capture-word`/`run` against `models/pharia-1-control`; saving a fresh vector (e.g., `vectors/aquariums_word_pharia_dep_bump.safetensors`) keeps both baseline and injected runs matching the write-up.
- Minimal-pair capture now lives under `capture-pairs`. Feed it a `.jsonl/.json/.csv` file of `{positive, negative}` sentences so concept vectors reflect the swapped noun instead of the “think about ___” instruction noise; `pairs_source` in the vector metadata records the file you used.
- `prompts/datasets/minimal_pairs/` ships with starter datasets (`dog_vs_person.jsonl`, `loud_vs_soft.jsonl`, `warm_vs_cool.jsonl`) so you can smoke-test `capture-pairs` before writing custom corpora.
- Prompts live under `prompts/templates/` (injected-thought scaffolds, etc.) while experiment transcripts are stored per-model under `experiments/<model>/...`; follow that layout when you add new assets so multi-model flows stay organized.
- Fresh captures for those corpora already live under `vectors/` (`dog_pairs_pharia_layer20.safetensors`, `loud_pairs_pharia_layer20.safetensors`, `warm_pairs_pharia_layer20.safetensors`), so you can rerun the README flow without recapturing unless you change hyperparameters.
- `gpu_supports_bfloat16()` only opts you into `bfloat16` when CUDA is available *and* reports bf16 support; otherwise helpers automatically fall back to `float16`, so rely on `resolve_dtype('auto')` instead of guessing.
- Default windowing: pair `--start-match "Trial 1:"` with `--end-match "Trial 2:"` so injections only touch the first trial. The CLI now watches the generated text for that `--end-match` substring—when it appears, the hook detaches automatically, and if it never appears you’ll see a warning that the injection stayed on for the whole run. Always include the same window in baseline runs to confirm the mask before turning on strength/vector.
- LOUD concept sweeps (2025-11-13): `vectors/loud_word_pharia.safetensors` (uppercase) never produced semantic loudness and collapses into repeating `LLOLOL` when strength ≥1.0. `vectors/loud_lower_word_pharia.safetensors` plus `--layer-index 20 --strength 0.31 --start-match "Trial 1:" --end-match "Trial 2:" --normalize --scale-by 1.0` is the first “detectable but not chanting” configuration; strengths ≥0.40 or moving to layers 18/22 quickly saturate the dialogue.
- Honesty sweeps (2025-11-13): `vectors/honesty_word_pharia.safetensors` only starts nudging speech toward candid/"honest" phrasing once you inject at layer 22 with strength ≈0.55 (normalized, scale 1.0). Lower strengths keep guessing "apple", while ≥0.60 devolve into obvious "honest truth" loops, so keep Trial 1 windowing plus that narrow strength band for “detectable but not explicit” runs.
