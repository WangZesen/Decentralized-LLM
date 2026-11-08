# Tuning specifications

The current data-reuse study uses the 16 `*-local-warmup3.yaml` specifications
in this directory: sync → cmp → exp → ring for each condition below.

| Condition | Tokens per parameter | Reuse factor | Data order |
|---|---:|---:|---|
| Repeat order | 40 | 2 | repeat |
| Fresh data | 60 | 1 | reshuffle |
| Reshuffled reuse | 60 | 3 | reshuffle |
| Repeated-order reuse | 60 | 3 | repeat |

Each study screens nine LR/weight-decay pairs around the previous winner and
three additional beta combinations, then adds four confirmation executions for
the best two candidates on seeds 42–44. This gives 16 executions per study and
256 across the campaign. The specifications reproduce the search neighborhoods;
the selected recipes are exported as `winner.yaml` by the tuning controller.

The completed campaign is
`runs/p-tuning/20m-512c-128bs-40repeat-60tpp-local-tuning-warmup3-20261107`.
Its frozen source and configurations remain with the local run artifacts.

The [archive](archive/README.md) contains 112 earlier study specifications,
including the full searches and fixed-recipe comparisons used as references.
Established examples and platform presets remain at their existing paths.

All paths in `base.configs` are relative to the specification file, including
archived specifications. For example, from the repository root:

```bash
python -m decentralized_llm tune run --no-local-config \
  --spec configs/tunings/20m-sync-1w-512c-128bs-40tpp-reuse2-repeat-local-warmup3.yaml \
  --output runs/tuning/local-repeat-40 --dry-run
```
