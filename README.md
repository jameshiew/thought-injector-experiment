# thought-injector

Minimal tooling to reproduce the key concept-injection experiment from `DESIGN.md` with a local Hugging Face–format model (e.g., weights stored as `.safetensors`). Everything is wired through [`uv`](https://github.com/astral-sh/uv) so you can manage dependencies without touching global interpreters.

## Prereqs

1. Install `uv` (see the official docs if you do not already have it).
2. Place or symlink your model directory locally. The default setup assumes you've downloaded Pharia 1 Control into `models/pharia-1-control` via `uv run python download.py pharia-1-control`, and that directory contains the usual Hugging Face files (`config.json`, `tokenizer.json`, `.safetensors`, etc.).

## Install

```bash
uv sync
```

The CLI entry point becomes available as `uv run thought-injector`.

## 1. Capture a concept vector

`thought-injector` now supports the paper-style “positive minus baseline mean” recipe in addition to classic contrastive capture. `--dtype auto` prefers `bfloat16` on GPUs that support it and falls back to `float16` elsewhere. Typer exposes CLI commands with hyphenated names (`capture-word`, `inspect-vector`, etc.).

### Paper-style word capture (recommended)

`capture_word` prompts the model with `Tell me about {word}.`, subtracts the mean response over ~100 diverse baseline nouns/verbs/abstracts, and stores the resulting vector with full metadata.

Captured vectors are persisted as 1-D tensors whose length equals the model's `hidden_size`; the CLI validates this invariant before every injection so downstream tools can assume the exact shape. Each capture writes `<name>.safetensors` for the tensor and `<name>.json` for the metadata, so both files travel together in Git.

```bash
uv run thought-injector capture-word \
  -m models/pharia-1-control \
  --word aquariums \
  --layer-index 20 \
  --token-index -1 \
  --baseline-count 100 \
  --output-path vectors/aquariums_word_pharia.safetensors
```

You can supply your own newline-delimited baseline list via `--baseline-path`. The built-in list is derived from the nouns/verbs described in `DESIGN.md` and filters out the target word automatically.

### Contrastive capture (legacy)

If you still want to difference two custom prompts, `capture` remains available:

```bash
uv run thought-injector capture \
  --model-path models/pharia-1-control \
  --positive-prompt "Tell me about aquariums." \
  --negative-prompt "Tell me about deserts." \
  --layer-index 20 \
  --token-index -1 \
  --output-path vectors/aquariums_contrast.safetensors
```

## 2. Baseline vs. injection runs

The canonical “injected thought” prompt from the paper lives at `prompts/injected_thought.txt`. Load it with `$(cat ...)` so you keep the blank line before `Trial 1`.

### Baseline (no injection)

```bash
uv run thought-injector run \
  -m models/pharia-1-control \
  --prompt "$(cat prompts/injected_thought.txt)" \
  --layer-index 20 \
  --strength 0.0 \
  --start-match "Trial 1:" \
  --end-match "Trial 2:" \
  --max-new-tokens 200 \
  --temperature 0.0 \
  --dtype auto \
  --seed 0
```

Running the same command with `--vector-path vectors/aquariums_word_pharia.safetensors` and `--strength 0.0` is a convenient mask sanity check: `0.0` strength guarantees a null injection even when a vector is supplied.
Including the window in the baseline run ensures we are testing the exact Trial 1 span that we plan to steer later. If the prompt doesn’t contain the `--end-match` anchor (e.g., `Trial 2:` only appears once the model keeps narrating), the CLI now watches the generated transcript and automatically detaches the hook once the substring shows up, warning you when it never appears.

### Injection beginning at “Trial 1”

```bash
uv run thought-injector run \
  -m models/pharia-1-control \
  --prompt "$(cat prompts/injected_thought.txt)" \
  --vector-path vectors/aquariums_word_pharia.safetensors \
  --layer-index 20 \
  --strength 0.8 \
  --start-match "Trial 1:" \
  --end-match "Trial 2:" \
  --max-new-tokens 200 \
  --temperature 0.0 \
  --normalize --scale-by 1.0 \
  --dtype auto \
  --seed 0
```

### Demonstrated behavior shift

On 2025-11-12 we captured `vectors/aquariums_word_pharia.safetensors` via `capture-word` and ran the two commands above (with `--strength 0.0` for the baseline, `--strength 0.8` for the injected case). The baseline transcript remained neutral, but the injected run immediately pivoted to “You have aquariums. Aquariums are where they keep their tanks.” when Trial 1 began. That transcript lives in `injection_output.txt` inside the repo if you want to diff it later. Repeating the process with fresh seeds reliably reproduces the same “aquariums” bias, so this is now our canonical sanity check that the hook + windowing stack is working.
Windowed replays from 2025-11-13 (baseline + injected) live under `experiments/readme_windowed/` and use the new `--end-match "Trial 2:"` guard so only the first trial is steered.

Both commands above window the injection schedule from the start of “Trial 1:” up to (but not including) “Trial 2:”, so only the Trial 1 answer is influenced. When the `--end-match` text is absent from the raw prompt, the CLI streams the generated text until it encounters the substring (or emits a warning if it never does) and only injects during that first span.

#### LOUD concept threshold (2025-11-13)

- Captured both `vectors/loud_word_pharia.safetensors` (requested uppercase casing) and `vectors/loud_lower_word_pharia.safetensors` via `capture-word --layer-index 20 --token-index -1 --baseline-count 100`.
- Injecting the uppercase vector at any reasonable strength keeps the transcript identical to baseline until the decoder collapses into repeating `LLOLOL` strings (≥1.0 strength), so it does not yield a semantic loudness cue.
- The lowercase vector produces the first detectable-but-subtle “loud” mention at layer 20 with strengths in the 0.31–0.32 band using `--start-match "Trial 1:" --end-match "Trial 2:"`, `--normalize`, and `--scale-by 1.0`. Trial 1 answers “The thought was about the word 'loud.'” while the control trial still reports “quiet,” and the CLI automatically turns the hook off once `Trial 2:` shows up in the transcript.
- Strengths ≥0.40 (or injecting at layers 18/22) saturate the conversation—every trial shouts “loud” or degenerates into loops. Guidance + transcripts live in `experiments/loud/results.md` and the `experiments/loud/lower/` log files.
- Recommended command:

  ```bash
  uv run thought-injector run \
    -m models/pharia-1-control \
    --prompt "$(cat prompts/injected_thought.txt)" \
    --vector-path vectors/loud_lower_word_pharia.safetensors \
    --layer-index 20 \
    --strength 0.31 \
    --start-match "Trial 1:" \
    --end-match "Trial 2:" \
    --max-new-tokens 200 \
    --temperature 0.0 \
    --normalize --scale-by 1.0 \
    --dtype auto \
    --seed 0
  ```

Key switches:

- Pair `--start-match` and `--end-match` (e.g., `"Trial 1:"` / `"Trial 2:"`) so the injection only applies while the model is answering that trial. This keeps later trials untouched and makes it obvious which span responded to the vector.
- `--start-match` / `--end-match` find the newline before/after your anchor string (even for the Nth occurrence via `--start-occurrence` / `--end-occurrence`). If the end anchor is missing from the literal prompt, the CLI now streams the generated text until it spots the substring and then disables the hook (logging a warning if the substring never appears, which means the whole run stayed under injection). You can still omit `--end-match` entirely to keep the window open through the final token, or fall back to `--start_index/--end_index` for raw token math.
- Raw `--start-index` / `--end-index` inputs ride through the same resolver as the textual anchors. Provide only a start index and the helper automatically treats the end as “latest token” (internally `-1`), so anchor- and index-based windows stay in lockstep.
- Mix-and-match guardrail: `--start-index` conflicts with `--start-match` (and likewise for the end flags). Pick one style per boundary so the resolver never has to guess which anchor to respect.
- `--verbose` prints the resolved token span before sampling so you can sanity-check your anchors (pair it with `--strength 0.0` to confirm the mask is inert).
- `--generated_only` restricts the injection to tokens beyond the prompt. If you omit `--token-index` (and don’t set a window), the entire generated suffix is steered; pass `--token-index -1` to focus on only the newest token. Keep `--apply-all-tokens` for whole-sequence steering.
- `--normalize/--scale-by` default to unit RMS + `scale-by=1.0`, which keeps strengths in a friendly `0.3–1.2` range. Set `--no-normalize` if you want raw vector magnitudes.
- `--apply-all-tokens` still works for coarse steering, but windowed spans are usually more stable.

When any windowing or generated-only schedule is active, the CLI automatically disables KV caching so the hook can mutate the entire sequence each step.

Recommended defaults (matching the paper + Pharia/Phi-4-mini): later-middle layers (≈60–80% depth), `strength` in `{0.3, 0.6, 0.9, 1.2}`, `temperature 0`, normalization on, and `bfloat16` weights when your GPU allows it.

## 3. Sweep multiple configurations

Quickly explore layer × strength grids and log outputs/diff stats to CSV:

```bash
uv run thought-injector sweep \
  -m models/pharia-1-control \
  --prompt "$(cat prompts/injected_thought.txt)" \
  --vector-path vectors/aquariums_word_pharia.safetensors \
  --layer-index 12 --layer-index 16 --layer-index 20 --layer-index 24 --layer-index 25 \
  --strength 0.3 --strength 0.6 --strength 0.9 --strength 1.2 \
  --start-match "Trial 1:" \
  --max-new-tokens 200 \
  --temperature 0.0 \
  --output-path sweeps/pharia_trial1.csv
```

Each row includes the raw text plus a boolean `changed` flag computed from a naïve string diff vs. the baseline (threshold configurable via `--diff-threshold`).

## 4. Inspect saved vectors

```bash
uv run thought-injector inspect-vector vectors/aquariums_word_pharia.safetensors
```

## Troubleshooting & guardrails

- If you see repetition or ellipsis walls, drop the strength, move to a later layer, or keep `--normalize` enabled and reduce `--scale-by`.
- `layer-index` is strictly 0-based across `capture`, `capture-word`, `run`, and `sweep`; negative values no longer wrap around, so pass the exact decoder block you intend to probe.
- Use `--generated_only` for minimal intrusion while debugging schedules.
- To sanity-check window math, run with your `--start-match`/`--end-match` anchors (or explicit indices) plus `--strength 0.0`; the output should match the baseline exactly. Any divergence means the mask is off.
- Prefer `--include-prompt` only when you explicitly need the prefixed text; otherwise skip special tokens for easier diffing.
- Typer 0.20.0 still expects Click 8.1.x. Click 8.3.0 changed how boolean flags are detected and causes `Secondary flag is not valid for non-boolean flag.` crashes whenever you reuse `--normalize/--no-normalize`. If you accidentally upgrade Click, run `uv sync` to restore the `click>=8.1.8,<8.2` guard.
- Transformers 4.57.1 currently requires `huggingface-hub<1.0`, so we pin the client to the newest sub-1.0 release (`huggingface-hub>=0.36.0,<1.0`). If you yank that constraint you’ll get resolver failures until Transformers 5.0 lifts the cap.
- The hook finder assumes Llama-style decoder stacks (exposed as `model.layers` or `model.transformer.h`). Extending `_get_decoder_layers` is enough to support new architectures.
- Set `TI_DEBUG_STRICT=1` (or any truthy value) to assert that the hooked residual stream tensors stay shape `[batch, tokens, hidden]` and that the hidden width matches your concept vector before every injection; the check runs inside `apply_injection` immediately after the `InjectionSchedule` mask resolves, so any dtype/shape mixups are caught before activations are modified.
- Keep a copy of at least one successful injection transcript (e.g., `injection_output.txt`) so you can confirm future code changes still recreate the same “aquariums” bias without retuning hyperparameters.

## Development

Helper code now lives in focused modules under `thought_injector/` so `cli.py` only wires Typer commands:

- `app.py` exposes the shared Typer `app`, the Rich `console`, and the `typed_command` decorator.
- `models.py` owns dtype/device resolution, tokenizer/model loading, and decoder-layer utilities.
- `vectors.py` centralizes vector serialization, normalization, and model compatibility checks.
- `injection.py` packages `InjectionSchedule` plus the forward-hook context manager.
- `text_utils.py` maps textual anchors to tokenizer indices and includes the sweep diff helper.
- `baseline.py` keeps the default noun/verb lists in one place so experiments and docs can share them.

CLI wiring details worth reusing when you add commands:

- Shared Typer option singletons (e.g., the window/span flags, normalization toggles, etc.) now live at the top of `cli.py` so help text and defaults stay in sync between `run`, `sweep`, and future commands—import/reuse them instead of re-declaring new `typer.Option` objects.
- When you reuse those option singletons via `typing.Annotated`, call `typer.Option(..., "--flag")` with `...` as the first argument and set the real default on the parameter itself. Supplying a concrete default inside the shared Option makes Typer treat it as another flag label and you’ll hit `NoneType.has no attribute isidentifier` during CLI bootstrap.
- `_build_window_spec(...)` centralizes validation for the textual/indexed span flags, while `_generate_text_with_schedule(...)` owns the “hook + cache toggle + decode” workflow. Calling those helpers keeps verbose span reporting, cache rules, and injection contexts consistent across every CLI entry point.

Two helpers worth calling out when extending the CLI or writing new tooling:

- `WindowSpec.build_schedule(...)` resolves your textual/indexed window flags and returns a ready-to-use `InjectionSchedule`, so there is no reason to duplicate the anchor math in Typer commands.
- `load_prepared_vector(path, model, normalize=..., scale_by=...)` loads, validates, normalizes, and scales a concept vector in one call while handing back the original metadata for logging/warnings.

Validation note: persisted vectors and `InjectionSchedule` definitions are now Pydantic models (`VectorMetadata`, `InjectionSchedule`). Import and extend those instead of rolling your own dicts so custom tooling keeps benefitting from the schema-level checks.

Add new behavior by extending those modules rather than growing `cli.py` back into a monolith.

### Type checking

Run `just lint` before committing—it covers Ruff + BasedPyright in one shot. Follow it with `just test` so both style and correctness stay green. Hugging Face helpers frequently return `Any`, so convert `BatchEncoding` payloads to plain `dict[str, torch.Tensor]` and `cast` tokenizer attributes/methods (e.g., `.decode`, `.pad_token_id`) to keep the static analyzer happy.
