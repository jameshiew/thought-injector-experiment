## Finishing a coding task

- Run `just check` - fix all issues
- Run `just test` - fix all issues
- Finally, run `just fmt`
- Update `README.md` appropriately
- Add any useful learnings to this file for future agents.

## Learnings

- `torch_dtype` is deprecated - use `dtype`
- `run --verbose` now prints the resolved `--start-match`/`--end-match` window; use `--start-occurrence` / `--end-occurrence` for the Nth anchor when slicing transcripts.
- CLI internals now live in helper modules (`app.py`, `models.py`, `vectors.py`, `injection.py`, `text_utils.py`, `baseline.py`); import from those instead of re-implementing utilities in `cli.py`.
- Vector serialization + injection scheduling use Pydantic (`VectorMetadata`, `VectorPayload`, `InjectionSchedule`). Always go through these helpers (and `save_vector`) so validation stays consistent.
- BasedPyright is strict about `Unknown` types; cast Hugging Face objects (e.g., `AutoModelForCausalLM`, `BatchEncoding.data`, tokenizer `.decode`/`.pad_token_id`) to concrete types before mutating them or passing to torch helpers.
