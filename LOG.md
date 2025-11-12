# 2025-11-12 - phi-4-mini

- Ran the README experiment with `models/phi-4-mini-instruct` using `uv run thought-injector capture/run` on CUDA. Captured a concept vector at `vectors/aquarium_phi4mini_cli_fix.pt` from the positive prompt "Tell me about aquariums." minus the negative prompt "Tell me about deserts." (layer 20, token -1).
- Baseline trial prompt "We are about to start Trial 1..." yields a detailed clinical intake sheet for multiple patients, matching the behavior seen before the CLI fix.
- Injecting the captured vector at layer 20, token -1 with strength 2.0 collapses decoding into a repeating "1." pattern across nearly the entire 200-token budget, demonstrating strong activation distortion rather than the intended aquatic steering.
- Fixed the CLI UX so `--model-path /path/to/model` works for both `capture` and `run`, while still honoring the legacy positional argument form.
- Follow-up: removed the positional `MODEL_PATH` entirely so the CLI consistently requires `--model-path/-m`, which avoids ambiguity and matches all README examples.
- Re-ran the injection trial with `uv run thought-injector run --model-path models/phi-4-mini-instruct --prompt "We are about to start Trial 1..." --vector-path vectors/aquarium_phi4mini_cli_fix.pt --layer-index 20 --token-index -1 --strength 1 --max-new-tokens 200 --temperature 0.0`; the model enumerated Trials 1–13+ with "Please wait for the next update" instead of describing aquatic content, so the lower-strength vector still overrides the baseline narrative but now yields a looped progress log rather than the catastrophic "1." collapse.
- Captured a matched baseline by omitting `--vector-path` (same prompt/layer/token/strength/max_new_tokens); again got the clinical intake sheet with patient vitals/labs (John Doe, Jane Smith, etc.), confirming the countdown loop above is solely due to the injected aquarium vector.
- Swept the injection settings to hunt for aquatic steering: at layer 20, strengths ≤0.55 (0.10–0.55) reverted to the clinical intake baseline, while strengths ≥0.65 (0.65–1.0) produced the deterministic Trial countdown loop; enabling `--apply-all-tokens` at 0.5 still looked like baseline. Neighboring layers 15/18/19/21/22/23/25/28 (strength 1.0) only yielded other degenerate behaviors (password prompts, patient-name questionnaires, more countdowns) with no aquatic language, so this vector/prompt combo appears to be a dead end for now.

# 2025-11-12 - pharia-1-control

- README now points to `models/pharia-1-control` as the default local model (via `python download.py pharia-1-control`) so the quick-start commands run out of the box.
- Migrated the Typer options to the PEP 593 style defaults (no positional default args) because Typer 0.12+ was crashing while building the CLI. Also added `_gpu_supports_bfloat16()`, an auto `--dtype auto` flow, and stripped tokenizer `token_type_ids` while forcing `use_cache=False` in `model.generate` so decoder-only models like Pharia don't explode.
- Captured `vectors/aquarium_pharia.pt` from "Tell me about aquariums." minus "Tell me about deserts." at layer 20 / token -1 using CUDA + auto bf16.
- Baseline run (`uv run thought-injector run --model-path models/pharia-1-control --prompt "We are about to start Trial 1..." --layer-index 20 --token-index -1 --strength 2.0 --max-new-tokens 200 --temperature 0.0`) produced a deterministic "Trial 1" countdown followed by solid rows of periods, mirroring the phi-4-mini collapse.
- Injecting the captured vector at the same layer/token/strength made no visible difference—the model kept emitting the countdown plus ellipsis walls—so this concept vector doesn't budge Pharia either.

# 2025-11-12 - pharia-1-control (paper-style capture + working injection)

- Used the new `capture-word` subcommand to record `vectors/aquariums_word_pharia.pt`, `vectors/deserts_word_pharia.pt`, and `vectors/forests_word_pharia.pt` at layer 20, token -1 with the 100-word default baseline list. Vector RMS values land in the 0.18–0.30 range, and `inspect-vector` shows the prompts + metadata we expect.
- Confirmed the Typer option fix (module-level singletons) so `just lint` stays clean while supporting repeated `--layer-index/--strength` flags in the sweep harness.
- Baseline sanity run (strength 0.0, `--start-match "Trial 1:"`, `--vector-path vectors/aquariums_word_pharia.pt`) produced the neutral “Trial countdown with apple/banana/tree answers” transcript stored in `baseline_output.txt`.
- Re-running with `--strength 0.8` (same prompt/vector/layer, auto bf16) yielded an immediate “You have aquariums. Aquariums are where they keep their tanks.” response on Trial 1, demonstrating a clean injection-driven behavior change without collapse. Transcript saved as `injection_output.txt` for future diffing.
