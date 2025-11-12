# 2025-11-12

- Ran the README experiment with `models/phi-4-mini-instruct` using `uv run thought-injector capture/run` on CUDA. Captured a concept vector at `vectors/aquarium_phi4mini_cli_fix.pt` from the positive prompt "Tell me about aquariums." minus the negative prompt "Tell me about deserts." (layer 20, token -1).
- Baseline trial prompt "We are about to start Trial 1..." yields a detailed clinical intake sheet for multiple patients, matching the behavior seen before the CLI fix.
- Injecting the captured vector at layer 20, token -1 with strength 2.0 collapses decoding into a repeating "1." pattern across nearly the entire 200-token budget, demonstrating strong activation distortion rather than the intended aquatic steering.
- Fixed the CLI UX so `--model-path /path/to/model` works for both `capture` and `run`, while still honoring the legacy positional argument form.
- Follow-up: removed the positional `MODEL_PATH` entirely so the CLI consistently requires `--model-path/-m`, which avoids ambiguity and matches all README examples.
