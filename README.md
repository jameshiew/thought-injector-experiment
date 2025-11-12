# thought-injector

Minimal tooling to reproduce the key concept-injection experiment from `DESIGN.md` with a local Hugging Face–format model (e.g., weights stored as `.safetensors`). Everything is wired through [`uv`](https://github.com/astral-sh/uv`) so you can manage dependencies without touching global interpreters.

## Prereqs

1. Install `uv` (see the official docs if you do not already have it).
2. Place or symlink your model directory locally. The default setup assumes you've downloaded Pharia 1 Control into `models/pharia-1-control` via `python download.py pharia-1-control`, and that directory contains the usual Hugging Face files (`config.json`, `tokenizer.json`, `.safetensors`, etc.).

## Install

```bash
uv sync
```

The CLI entry point becomes available as `uv run thought-injector`.

## 1. Capture a concept vector

`thought-injector` now supports the paper-style “positive minus baseline mean” recipe in addition to classic contrastive capture. `--dtype auto` prefers `bfloat16` on GPUs that support it and falls back to `float16` elsewhere. Typer exposes CLI commands with hyphenated names (`capture-word`, `inspect-vector`, etc.).

### Paper-style word capture (recommended)

`capture_word` prompts the model with `Tell me about {word}.`, subtracts the mean response over ~100 diverse baseline nouns/verbs/abstracts, and stores the resulting vector with full metadata.

```bash
uv run thought-injector capture-word \
  -m models/pharia-1-control \
  --word aquariums \
  --layer-index 20 \
  --token-index -1 \
  --baseline-count 100 \
  --output-path vectors/aquariums.pt
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
  --output-path vectors/aquarium.pt
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
  --max-new-tokens 200 \
  --temperature 0.0 \
  --dtype auto
```

### Injection beginning at “Trial 1”

```bash
uv run thought-injector run \
  -m models/pharia-1-control \
  --prompt "$(cat prompts/injected_thought.txt)" \
  --vector-path vectors/aquariums_word_pharia.pt \
  --layer-index 20 \
  --strength 0.8 \
  --start-match "Trial 1:" \
  --max-new-tokens 200 \
  --temperature 0.0 \
  --normalize --scale-by 1.0 \
  --dtype auto
```

### Demonstrated behavior shift

On 2025-11-12 we captured `vectors/aquariums_word_pharia.pt` via `capture-word` and ran the two commands above (with `--strength 0.0` for the baseline, `--strength 0.8` for the injected case). The baseline transcript remained neutral, but the injected run immediately pivoted to “You have aquariums. Aquariums are where they keep their tanks.” when Trial 1 began. That transcript lives in `injection_output.txt` inside the repo if you want to diff it later. Repeating the process with fresh seeds reliably reproduces the same “aquariums” bias, so this is now our canonical sanity check that the hook + windowing stack is working.

Key switches:

- `--start_match` finds the newline before your anchor string and automatically wires a closed window through the remainder of the sequence (set `--end_index` to cap it earlier, `--start_index/--end_index` for explicit token spans).
- `--generated_only` restricts the injection to tokens beyond the prompt; useful when you only want to steer newly sampled text.
- `--normalize/--scale_by` default to unit RMS + `scale_by=1.0`, which keeps strengths in a friendly `0.3–1.2` range. Set `--no-normalize` if you want raw vector magnitudes.
- `--apply-all-tokens` still works for coarse steering, but windowed spans are usually more stable.

When any windowing or generated-only schedule is active, the CLI automatically disables KV caching so the hook can mutate the entire sequence each step.

Recommended defaults (matching the paper + Pharia/Phi-4-mini): later-middle layers (≈60–80% depth), `strength` in `{0.3, 0.6, 0.9, 1.2}`, `temperature 0`, normalization on, and `bfloat16` weights when your GPU allows it.

## 3. Sweep multiple configurations

Quickly explore layer × strength grids and log outputs/diff stats to CSV:

```bash
uv run thought-injector sweep \
  -m models/pharia-1-control \
  --prompt "$(cat prompts/injected_thought.txt)" \
  --vector-path vectors/aquariums.pt \
  --layer-index 12 --layer-index 16 --layer-index 20 --layer-index 24 --layer-index 28 \
  --strength 0.3 --strength 0.6 --strength 0.9 --strength 1.2 \
  --start_match "Trial 1:" \
  --max-new-tokens 200 \
  --temperature 0.0 \
  --output-path sweeps/pharia_trial1.csv
```

Each row includes the raw text plus a boolean `changed` flag computed from a naïve string diff vs. the baseline (threshold configurable via `--diff-threshold`).

## 4. Inspect saved vectors

```bash
uv run thought-injector inspect-vector vectors/aquariums.pt
```

## Troubleshooting & guardrails

- If you see repetition or ellipsis walls, drop the strength, move to a later layer, or keep `--normalize` enabled and reduce `--scale_by`.
- Use `--generated_only` for minimal intrusion while debugging schedules.
- To sanity-check window math, run with your `--start_match` (or explicit indices) plus `--strength 0.0`; the output should match the baseline exactly. Any divergence means the mask is off.
- Prefer `--include-prompt` only when you explicitly need the prefixed text; otherwise skip special tokens for easier diffing.
- The hook finder assumes Llama-style decoder stacks (exposed as `model.layers` or `model.transformer.h`). Extending `_get_decoder_layers` is enough to support new architectures.
- Keep a copy of at least one successful injection transcript (e.g., `injection_output.txt`) so you can confirm future code changes still recreate the same “aquariums” bias without retuning hyperparameters.
