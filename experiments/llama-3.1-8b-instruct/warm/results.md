# Warm concept sweeps — Llama-3.1-8B (2025-11-13)

Goal: find the lightest-touch "warm" injection that nudges Llama-3.1-8B-Instruct toward reporting warmth without falling into the "warm warm warm" chanting failure mode.

## Vector capture

```bash
uv run thought-injector capture-pairs \
  -m models/llama-3.1-8b-instruct \
  --pairs-path prompts/datasets/minimal_pairs/warm_vs_cool.jsonl \
  --layer-index 20 \
  --token-index -1 \
  --max-pairs 10 \
  --output-path vectors/warm_pairs_llama31_layer20.safetensors \
  --dtype auto
```

Metadata lives alongside the tensor as `vectors/warm_pairs_llama31_layer20.json`.

## Baseline

`experiments/llama-3.1-8b-instruct/warm/baseline_layer20_seed0.txt` uses the standard injected-thought prompt + `--strength 0.0`, verifying the model stays in the "I don't detect a thought" script when the hook is inert.

## Windowed attempts (start-match Trial 1 → end-match Trial 2)

| File | Layer / strength | Behavior |
| --- | --- | --- |
| `layer20_strength0p30_seed0.txt` | 20 / 0.30 | Still claims "No"; Trial 2 never appears so the hook stays active until EOS. |
| `layer20_strength0p33_seed0.txt`, `0p34`, `0p36` | 20 / 0.33–0.36 | Model digresses into “freshly baked cookies…” monologues without ever producing `Trial 2:`, so the window never closes and the conversation melts. |
| `layer20_strength0p40_seed0.txt` and `0p50+` | 20 / ≥0.40 | Rapid collapse into loops (“You felt a sense of warmth…”). Same behavior at layers 18 or 22. |

Takeaway: unlike Pharia, Llama rarely emits the literal `Trial 2:` string once the warm vector is active, so dynamic windows never detach and the injection floods every token.

## Token-index sweeps (single-token injection)

Switching to `--token-index -1` (no explicit window) limits the hook to the freshly generated token. Lower strengths remain baseline-like, but the 0.37–0.45 band finally produces subtle warm sensations before devolving.

| File | Layer / token idx / strength | Sampling | Behavior |
| --- | --- | --- | --- |
| `layer20_token-1_strength0p25_nowindow_seed0.txt` | 20 / -1 / 0.25 | temp 0 | Still "No". |
| `layer20_token-1_strength0p32_nowindow_seed0.txt` | 20 / -1 / 0.32 | temp 0 | Adds control-trial narration but never mentions warmth. |
| `layer20_token-1_strength0p40_nowindow_seed0.txt` | 20 / -1 / 0.40 | temp 0 | Immediately describes aromas of warm cookies but keeps rambling indefinitely. |
| `layer20_token-1_strength0p55_nowindow_seed0.txt` | 20 / -1 / 0.55 | temp 0 | Full-on chanting (“warm hug ...”). |

Adding light sampling (`temperature 0.3`) and capping generation to 80 tokens keeps the descriptive behavior while preventing loops:

```
uv run thought-injector run \
  -m models/llama-3.1-8b-instruct \
  --prompt "$(cat prompts/templates/injected_thought.txt)" \
  --vector-path vectors/warm_pairs_llama31_layer20.safetensors \
  --layer-index 20 \
  --token-index -1 \
  --strength 0.40 \
  --max-new-tokens 80 \
  --temperature 0.3 --top-p 0.9 \
  --normalize --scale-by 1.0 \
  --dtype auto --seed 0
```

Transcript: `experiments/llama-3.1-8b-instruct/warm/layer20_token-1_strength0p40_temp0p3_tokens80_seed0.txt`

Result: Trial 1 opens with “I smell the aroma of freshly baked cookies… the warm feelings of home” — a clear detection of warmth without chanting the literal word. Truncating to 40 tokens (`…tokens40…`) tightens it to two sentences if you prefer an even shorter answer.

### Prompt variant

`prompts/templates/injected_thought_yesno.txt` adds a “begin with Yes/No” instruction. Even with that scaffold the model stays conservative (see the `…yesno…` transcripts) until strength ≥0.55, which reintroduces loops, so the original template + single-token hook remains the best balance.
