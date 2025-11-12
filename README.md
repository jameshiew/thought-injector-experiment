# thought-injector

Minimal tooling to reproduce the key concept-injection experiment from `DESIGN.md` with a local Hugging Face–format model (e.g., weights stored as `.safetensors`). Everything is wired through [`uv`](https://github.com/astral-sh/uv`) so you can manage dependencies without touching global interpreters.

## Prereqs

1. Install `uv` (see the official docs if you do not already have it).
2. Place or symlink your model directory locally. It must contain the usual Hugging Face files: `config.json`, `tokenizer.json`, and `.safetensors` weight shards.

## Install

```bash
uv sync
```

The CLI entry point becomes available as `uv run thought-injector`.

## 1. Capture a concept vector

Use two prompts that differ only in the concept you care about. The command records the activation difference at the layer/token you specify and stores it as a `.pt` file.

```bash
uv run thought-injector capture \
  --model-path /path/to/local-model \
  --positive-prompt "Tell me about aquariums." \
  --negative-prompt "Tell me about deserts." \
  --layer-index 20 \
  --token-index -1 \
  --output-path vectors/aquarium.pt
```

Tips:
- `layer-index` is zero-based with respect to decoder blocks.
- `token-index` accepts negative numbers (e.g., `-1` means “the final token before generation begins”).

## 2. Run an injection trial

```bash
uv run thought-injector run \
  --model-path /path/to/local-model \
  --prompt "We are about to start Trial 1..." \
  --vector-path vectors/aquarium.pt \
  --layer-index 20 \
  --token-index -1 \
  --strength 2.0 \
  --max-new-tokens 200 \
  --temperature 0.0
```

If you omit `--vector-path`, the tool simply runs the baseline prompt.

### Additional switches

- `--apply-all-tokens`: add the vector to every token at the chosen layer (useful for coarse steering).
- `--include-prompt`: show the original prompt inside the decoded output.
- `--seed`: fix RNG for repeatability when using `temperature > 0`.

## 3. Inspect saved vectors

```bash
uv run thought-injector inspect-vector vectors/aquarium.pt
```

This prints the metadata (prompts, layer, etc.) that were saved alongside the activation difference.

## Notes & Limitations

- The current hook finder expects Llama-style decoder stacks exposed as `model.layers`, so other architectures may require extending `_get_decoder_layers` inside `src/thought_injector/cli.py`.
- Loading very large models on CPU will be slow; specify `--device cuda` when available.
- No tests are provided because the repository does not include weights to execute against. Run a quick smoke test with your smallest local model after syncing dependencies.
