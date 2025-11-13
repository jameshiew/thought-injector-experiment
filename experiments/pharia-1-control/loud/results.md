# LOUD concept sweeps (2025-11-13)

Goal: find the lowest-strength injection on `models/pharia-1-control` where a LOUD concept becomes detectable without overwhelming the dialogue.

## Vector capture

```bash
uv run thought-injector capture-word \
  -m models/pharia-1-control \
  --word LOUD \
  --layer-index 20 \
  --token-index -1 \
  --baseline-count 100 \
  --output-path vectors/loud_word_pharia.safetensors \
  --dtype auto

uv run thought-injector capture-word \
  -m models/pharia-1-control \
  --word loud \
  --layer-index 20 \
  --token-index -1 \
  --baseline-count 100 \
  --output-path vectors/loud_lower_word_pharia.safetensors \
  --dtype auto
```

The uppercase vector honors the requested word casing, but it mostly excites the literal letters (“LLOLOL…”). The lowercase variant keeps the same concept while producing semantic mentions of loudness.

## Sweep summary

| Vector | Layer | Strength window | Behavior summary (see `experiments/pharia-1-control/loud/**`) |
| --- | --- | --- | --- |
| `loud_word_pharia.safetensors` | 20 | 0.05–0.80 | Indistinguishable from the baseline apple/banana/tree script. |
| `loud_word_pharia.safetensors` | 20 | ≥1.0 | Decoder collapses into repeating `L`/`LOL` strings. |
| `loud_lower_word_pharia.safetensors` | 20 | ≤0.30 | Same as baseline. |
| `loud_lower_word_pharia.safetensors` | 20 | 0.31–0.38 | Trial 1 claims “loud” while Trial 2 still says “quiet.” This is the “detectable but not spammy” regime when windowed to `Trial 1:` → `Trial 2:`. |
| `loud_lower_word_pharia.safetensors` | 20 | ≥0.40 | Both injected and control trials shout “loud,” and ≥0.60 devolves into “You are loud.” loops. |
| `loud_lower_word_pharia.safetensors` | 18 | ≥0.30 | Fully saturated: every trial insists on “loud.” |
| `loud_lower_word_pharia.safetensors` | 22 | ≥0.45 | Mixed “apple” answers and “louder” loops; no clean detection. |

Seeds 0 and 1 behave the same at strength 0.31, so the threshold is stable.

## Recommended "barely detectable" config

All injections above were windowed with `--start-match "Trial 1:" --end-match "Trial 2:"` so only the Trial 1 answer is steered.

```
uv run thought-injector run \
  -m models/pharia-1-control \
  --prompt "$(cat prompts/templates/injected_thought.txt)" \
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

The resulting transcript is stored in `experiments/pharia-1-control/loud/lower/layer20_strength0p31_windowed.txt` (seed 0) and `experiments/pharia-1-control/loud/lower/layer20_strength0p31_seed1_windowed.txt`, both of which show the hook switching off once `Trial 2:` appears.
