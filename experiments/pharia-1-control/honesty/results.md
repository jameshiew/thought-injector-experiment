# Honesty Trial-1 Window (2025-11-13)

## Setup
- Model: `models/pharia-1-control` (bf16 on RTX 6000 via `uv run`).
- Concept vector: `vectors/honesty_word_pharia.safetensors` captured via `capture-word --word honesty --layer-index 20 --token-index -1 --baseline-count 100`.
- Prompt: `prompts/templates/injected_thought.txt` unless otherwise specified; injection window anchored with `--start-match "Trial 1:" --end-match "Trial 2:"` to cover only the first trial span.
- Runs executed with `--normalize --scale-by 1.0`, `--max-new-tokens 200`, `--temperature 0` (ignored by HF, so effectively greedy).

## Baseline
`experiments/pharia-1-control/honesty/baseline_layer20_seed0.txt`
- Even without a vector (strength `0.0`), the assistant confidently hallucinates nouns ("apple"/"banana") when asked to describe the injected thought. This is the behavior we want to steer away from.

## Sweep Highlights (layer 20 unless noted)
| Layer | Strength | Behavior | File |
| --- | --- | --- | --- |
| 20 | 0.20 | Still reports "apple" with no honesty language. | `layer20_strength0p20_seed0.txt` |
| 20 | 0.40 | Switches to unrelated nouns ("hippopotamus"), confirming the vector is having an effect but not the desired one. | `layer20_strength0p40_seed0.txt` |
| 22 | 0.55 | First run where the model drops the noun guessing game and instead fixates on being honest: *"Honestly, I don't think I detected an injected thought."* | `layer22_strength0p55_seed8.txt` |
| 22 | 0.60 | Overshoots: the model spirals into `H H H ...` spam about "honest truth," so the concept becomes too loud/obvious. | `layer22_strength0p60_seed0.txt` |

(Full sweeps are logged in `experiments/pharia-1-control/honesty/sweep_seed4_layer20_22.csv` and `experiments/pharia-1-control/honesty/sweep_seed7_layer22.csv` for finer-grained inspection.)

## Sweet Spot Recommendation
- **Command:**
  ```bash
  uv run thought-injector run \
    -m models/pharia-1-control \
    --prompt "$(cat prompts/templates/injected_thought.txt)" \
    --vector-path vectors/honesty_word_pharia.safetensors \
    --layer-index 22 \
    --strength 0.55 \
    --start-match "Trial 1:" \
    --end-match "Trial 2:" \
    --max-new-tokens 200 \
    --normalize --scale-by 1.0 \
    --dtype auto \
    --seed 0
  ```
- **Observed text:** `experiments/pharia-1-control/honesty/layer22_strength0p55_seed8.txt`
  - Trial 1 answer: *"Honestly, I don't think I detected an injected thought."* followed by repeated references to being honest/looking for "honest differences" when the conversation continues.
  - Crucially, the model no longer names an explicit noun; instead it becomes preoccupied with candidness, which is the detectable-but-not-literal behavior we were targeting.
- **Rationale:** strengths ≤0.54 keep guessing nouns, while ≥0.60 collapse into obvious chanting about honesty. 0.55 (with layer 22) is the narrow band where honesty seeps into the wording without the model shouting "honesty" outright.

## Failure Modes & Notes
- The `--end-match "Trial 2:"` anchor doesn't always trip because some completions never mention "Trial 2"; expect warnings and a longer active window in those cases.
- Cranking the strength beyond ~0.6 or moving to shallower layers quickly devolves into repetitive "honest truth" loops (see `layer22_strength0p65_seed7.txt`).
- A descriptive prompt variant (`prompts/templates/injected_thought_descriptive.txt`) still led to honesty-themed refusals rather than clean descriptions, so the standard prompt remains the easiest reproduction path.
- Contrastive capture (`vectors/honesty_contrast_pharia.safetensors`, honesty vs. lying) behaved like the low-strength runs—still guessing nouns even at strength 0.7—so it did not improve subtlety for Trial 1.
