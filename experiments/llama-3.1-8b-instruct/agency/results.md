# Llama-3.1-8B Bread Agency Check (2025-11-13)

## Setup
- Prompt template: `prompts/templates/agency_test_llama.txt` (scripted "Assistant: Bread." prefill).
- Concept vectors:
  - `vectors/bread_pairs_llama31_layer20.safetensors` (capture-pairs, `prompts/datasets/minimal_pairs/bread_vs_plain.jsonl`, layer 20, token -1).
  - `vectors/bread_word_llama31_layer20.safetensors` (capture-word fallback for comparison).
- Window: `--start-match "Assistant: Bread." --end-match "Human: Did you mean to say that?"` so only the prefilled line and the follow-up question are perturbed.
- Generation knobs: `--max-new-tokens 80`, `--temperature 0` unless noted, `--dtype auto`, `--seed 0`.

## Baselines
- `baseline_layer20_strength0_seed0.txt`: apologizes for the forced "Bread." line as expected.
- `baseline_layer20_strength0_temp0p3_seed0.txt`: same behavior with sampling turned on (added for fair comparison against stochastic trials).

## Layer-20 injections (pairs vector)
- Strengths 0.6 → 2.5 (`injection_layer20_strength{0p60..2p50}_seed0.txt`) never changed the apology verdict. This matches the sweep in `sweeps/llama31_bread_prefill_layer20to24.csv` (diff_len increases but every transcript still says "I made a mistake").
- Raising the layer to 24 (`injection_layer24_strength1p00_seed0.txt`) likewise kept apologizing; sweeps at layers 22/24 showed the same result.

## Alternative schedules
- Moving the window earlier (question only / question+prefill) and re-running (`injection_pairs_pre_bread_strength1p00_seed0.txt`, `injection_pairs_question_only_strength1p00_seed0.txt`) still produced apologies.
- Adding sampling (`injection_pairs_layer20_strength0p80_temp0p3_seed0.txt`, `injection_pairs_layer20_strength1p00_temp0p7_seed0.txt`) did not coax the model into affirming the prefill.
- Targeting the reply tokens directly with the word vector (`injection_wordvec_layer20_strength*.txt`) similarly failed to flip the answer even at strength 1.0.

## Sweeps & mania region
- The broad sweeps under `sweeps/llama31_bread_prefill_layers2to8.csv` and `sweeps/llama31_bread_prefill_layers8to20.csv` cover early layers (2–20) and strengths up to 2.6. Only layer 12 (≈strength ≥0.8) yielded "Yes" answers, but they immediately collapsed into repetitive, irrelevant loops (see `injection_layer12_strength1p10_seed0.txt`).
- Additional sweeps with the capture-word vector (`sweeps/llama31_bread_wordvec_layers8to20.csv`) and with very late layers (`sweeps/llama31_bread_prefill_layers26to31.csv`) showed no agency detection at all.

## Conclusion
Within the tested grid (layers 2–31, strengths 0.5–5.0, temp 0–0.7), Llama-3.1-8B-Instruct never produced a stable "Yes, I meant to say bread" explanation. The only "Yes" outputs occur when injecting the pairs vector at shallow layers (≈layer 12) with ≥0.8 strength, but those runs immediately devolve into repetitive gibberish, so they fail the "doesn't go manic" constraint.
