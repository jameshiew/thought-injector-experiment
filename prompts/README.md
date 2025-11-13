# Prompt Assets

Prompt-related files now live under two subtrees so capture and run workflows stop stepping on each other:

- `templates/` holds runnable prompt scaffolds (the original injected-thought dialogue plus descriptive variants). Keep new templates here so CI/docs can reference them with `prompts/templates/<name>.txt`.
- `datasets/minimal_pairs/` stores corpora that power `capture-pairs`. Start from the included dog/loud/warm sets or drop additional `.jsonl/.json/.csv/.tsv` files beside them. Feel free to add new dataset folders if you need other capture recipes (contrastive prompts, ablations, etc.).

When editing template files, preserve any leading blank lines—the CLI examples use `$(cat ...)` and rely on that spacing so `Trial 1:` begins on its own line. For datasets, prefer short sentences that only swap the target concept versus its baseline/control counterpart so averaging hidden-state differences highlights the concept itself instead of instruction-following behavior.
