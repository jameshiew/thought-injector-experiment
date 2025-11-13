# Experiment Logs

Experiments are organized by model so it stays obvious which checkpoint produced each transcript or sweep:

- `experiments/<model>/readme_windowed/` — canonical baseline vs. injected transcripts that mirror the README sanity check. Treat this as a quick regression harness when touching the CLI.
- `experiments/<model>/<concept>/` — concept-specific folders (e.g., `loud/`, `honesty/`) that contain transcripts, CSV sweeps, and local notes for that model.
- Empty directories for other checkpoints (e.g., `phi-4-mini-instruct/`, `llama-3.1-8b-instruct/`) keep the hierarchy ready for future runs; drop a `.gitkeep` inside until the first log lands.

When adding a new model or concept:

1. Create `experiments/<model>/<concept>/` and store raw transcripts plus `results.md`/CSV summaries there.
2. Reference files via their full `experiments/<model>/...` path inside docs so people can jump straight to the right model.
3. If a run mirrors the README flow, copy the transcripts into that model’s `readme_windowed/` folder so regressions stay easy to diff.

This layout keeps Pharia, Phi-4, Llama, etc. from mixing outputs and makes it trivial to answer “which model did this behavior come from?” months later.
