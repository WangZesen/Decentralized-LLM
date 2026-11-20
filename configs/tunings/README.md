# Tuning specifications

The 27 specifications for 20M tune AdamW with global batch size 128 (65,536
tokens), context length 512, gradient clipping 1.0, and evaluation batch size 256.
Each of the nine scheme/topology/worker combinations has 20-, 40-, and 60-TPP
variants, all including 4 TPP of warmup. Sync and AWC use broad searches at 20/40
TPP and reduced grids at 60 TPP based on both completed rounds. ATC on complete
topology uses the original broad ranges at all three budgets, matching the 10M
ATC settings with only the model configuration changed. All studies use cosine
decay, fresh data (reuse factor 1), no error feedback, and no checkpoints.

| Method | Scheme | Logical workers | 20 TPP | 40 TPP | 60 TPP |
|---|---|---:|---|---|---|
| sync | — | 1 | [Broad search](20m-sync-1w-512c-128bs-20tpp-lr-wd-betas-warmup4.yaml) | [Broad search](20m-sync-1w-512c-128bs-40tpp-lr-wd-betas-warmup4.yaml) | [Refinement](20m-sync-1w-512c-128bs-60tpp-lr-wd-betas-warmup4.yaml) |
| cmp | AWC | 4 | [Broad search](20m-awc-4w-cmp-512c-128bs-20tpp-lr-wd-betas-warmup4.yaml) | [Broad search](20m-awc-4w-cmp-512c-128bs-40tpp-lr-wd-betas-warmup4.yaml) | [Refinement](20m-awc-4w-cmp-512c-128bs-60tpp-lr-wd-betas-warmup4.yaml) |
| cmp | AWC | 8 | [Broad search](20m-awc-8w-cmp-512c-128bs-20tpp-lr-wd-betas-warmup4.yaml) | [Broad search](20m-awc-8w-cmp-512c-128bs-40tpp-lr-wd-betas-warmup4.yaml) | [Refinement](20m-awc-8w-cmp-512c-128bs-60tpp-lr-wd-betas-warmup4.yaml) |
| cmp | ATC | 4 | [Broad search](20m-atc-4w-cmp-512c-128bs-20tpp-lr-wd-betas-warmup4.yaml) | [Broad search](20m-atc-4w-cmp-512c-128bs-40tpp-lr-wd-betas-warmup4.yaml) | [Broad search](20m-atc-4w-cmp-512c-128bs-60tpp-lr-wd-betas-warmup4.yaml) |
| cmp | ATC | 8 | [Broad search](20m-atc-8w-cmp-512c-128bs-20tpp-lr-wd-betas-warmup4.yaml) | [Broad search](20m-atc-8w-cmp-512c-128bs-40tpp-lr-wd-betas-warmup4.yaml) | [Broad search](20m-atc-8w-cmp-512c-128bs-60tpp-lr-wd-betas-warmup4.yaml) |
| exp | AWC | 4 | [Broad search](20m-awc-4w-exp-512c-128bs-20tpp-lr-wd-betas-warmup4.yaml) | [Broad search](20m-awc-4w-exp-512c-128bs-40tpp-lr-wd-betas-warmup4.yaml) | [Refinement](20m-awc-4w-exp-512c-128bs-60tpp-lr-wd-betas-warmup4.yaml) |
| exp | AWC | 8 | [Broad search](20m-awc-8w-exp-512c-128bs-20tpp-lr-wd-betas-warmup4.yaml) | [Broad search](20m-awc-8w-exp-512c-128bs-40tpp-lr-wd-betas-warmup4.yaml) | [Refinement](20m-awc-8w-exp-512c-128bs-60tpp-lr-wd-betas-warmup4.yaml) |
| ring | AWC | 4 | [Broad search](20m-awc-4w-ring-512c-128bs-20tpp-lr-wd-betas-warmup4.yaml) | [Broad search](20m-awc-4w-ring-512c-128bs-40tpp-lr-wd-betas-warmup4.yaml) | [Refinement](20m-awc-4w-ring-512c-128bs-60tpp-lr-wd-betas-warmup4.yaml) |
| ring | AWC | 8 | [Broad search](20m-awc-8w-ring-512c-128bs-20tpp-lr-wd-betas-warmup4.yaml) | [Broad search](20m-awc-8w-ring-512c-128bs-40tpp-lr-wd-betas-warmup4.yaml) | [Refinement](20m-awc-8w-ring-512c-128bs-60tpp-lr-wd-betas-warmup4.yaml) |

Each trial uses one physical GH200. Decentralized AWC and ATC pack four or eight
logical workers onto that GPU; synchronous training uses one distributed worker.

## 10M searches

The [10M model](../models/10m.yaml) uses eight layers, width 192, three attention
heads, and FFN width 544: 9,833,664 parameters with the default vocabulary.
Its 27 specifications cover sync, AWC on complete/exponential/ring topologies,
and ATC on the complete topology, with 4 or 8 logical decentralized workers.
All three 10M budgets use the original broad 20/40-TPP LR/decay and beta grids,
screening betas, seeds, and allocation settings. Within each scheme/topology/worker
combination, the specifications differ only in total token budget; warmup remains
4 TPP. Each ATC configuration differs from its AWC counterpart only in
`distributed.scheme: atc`.

| Method | Scheme | Logical workers | 20 TPP | 40 TPP | 60 TPP |
|---|---|---:|---|---|---|
| sync | — | 1 | [Broad search](10m-sync-1w-512c-128bs-20tpp-lr-wd-betas-warmup4.yaml) | [Broad search](10m-sync-1w-512c-128bs-40tpp-lr-wd-betas-warmup4.yaml) | [Broad search](10m-sync-1w-512c-128bs-60tpp-lr-wd-betas-warmup4.yaml) |
| cmp | AWC | 4 | [Broad search](10m-awc-4w-cmp-512c-128bs-20tpp-lr-wd-betas-warmup4.yaml) | [Broad search](10m-awc-4w-cmp-512c-128bs-40tpp-lr-wd-betas-warmup4.yaml) | [Broad search](10m-awc-4w-cmp-512c-128bs-60tpp-lr-wd-betas-warmup4.yaml) |
| cmp | AWC | 8 | [Broad search](10m-awc-8w-cmp-512c-128bs-20tpp-lr-wd-betas-warmup4.yaml) | [Broad search](10m-awc-8w-cmp-512c-128bs-40tpp-lr-wd-betas-warmup4.yaml) | [Broad search](10m-awc-8w-cmp-512c-128bs-60tpp-lr-wd-betas-warmup4.yaml) |
| cmp | ATC | 4 | [Broad search](10m-atc-4w-cmp-512c-128bs-20tpp-lr-wd-betas-warmup4.yaml) | [Broad search](10m-atc-4w-cmp-512c-128bs-40tpp-lr-wd-betas-warmup4.yaml) | [Broad search](10m-atc-4w-cmp-512c-128bs-60tpp-lr-wd-betas-warmup4.yaml) |
| cmp | ATC | 8 | [Broad search](10m-atc-8w-cmp-512c-128bs-20tpp-lr-wd-betas-warmup4.yaml) | [Broad search](10m-atc-8w-cmp-512c-128bs-40tpp-lr-wd-betas-warmup4.yaml) | [Broad search](10m-atc-8w-cmp-512c-128bs-60tpp-lr-wd-betas-warmup4.yaml) |
| exp | AWC | 4 | [Broad search](10m-awc-4w-exp-512c-128bs-20tpp-lr-wd-betas-warmup4.yaml) | [Broad search](10m-awc-4w-exp-512c-128bs-40tpp-lr-wd-betas-warmup4.yaml) | [Broad search](10m-awc-4w-exp-512c-128bs-60tpp-lr-wd-betas-warmup4.yaml) |
| exp | AWC | 8 | [Broad search](10m-awc-8w-exp-512c-128bs-20tpp-lr-wd-betas-warmup4.yaml) | [Broad search](10m-awc-8w-exp-512c-128bs-40tpp-lr-wd-betas-warmup4.yaml) | [Broad search](10m-awc-8w-exp-512c-128bs-60tpp-lr-wd-betas-warmup4.yaml) |
| ring | AWC | 4 | [Broad search](10m-awc-4w-ring-512c-128bs-20tpp-lr-wd-betas-warmup4.yaml) | [Broad search](10m-awc-4w-ring-512c-128bs-40tpp-lr-wd-betas-warmup4.yaml) | [Broad search](10m-awc-4w-ring-512c-128bs-60tpp-lr-wd-betas-warmup4.yaml) |
| ring | AWC | 8 | [Broad search](10m-awc-8w-ring-512c-128bs-20tpp-lr-wd-betas-warmup4.yaml) | [Broad search](10m-awc-8w-ring-512c-128bs-40tpp-lr-wd-betas-warmup4.yaml) | [Broad search](10m-awc-8w-ring-512c-128bs-60tpp-lr-wd-betas-warmup4.yaml) |

Every 10M study uses the full broad search described below: 97 executions and
49 Slurm allocations before retries. The nine studies at each budget need
873 executions and 441 allocations. Each study trains the 10M model from scratch.

Each trial uses one physical GH200 with fused execution. Decentralized AWC and ATC pack
four or eight logical workers onto that GPU; sync uses one distributed worker.
The retained `runtime.compile: auto` enables compilation, and automatic fused RoPE
supports its three attention heads. A100 execution is not enabled for this model.

## Broad search ranges

The broad searches cover all 20/40-TPP studies, all 10M 60-TPP studies, and the
20M ATC 60-TPP studies. They follow the archived
[20M complete-topology specification](archive/20m-awc-8w-cmp-512c-128bs-20tpp-lr-wd-betas-warmup3.yaml),
with LR `2^-9` and weight decay `0.2` removed:

1. Screen 16 LR/weight-decay pairs on seed 42 and retain the best three.
   LR values are `2^-8`, `2^-7`, `2^-6`, and `2^-5`; weight-decay values are
   `0.0125`, `0.025`, `0.05`, and `0.1`. Initial betas remain `0.9` and
   `0.996712898272925`.
2. Search the original 4-by-6 beta grid for each retained pair and keep the best
   six candidates. Beta1 is `0.9` raised to `1/8`, `1/4`, `1/2`, or `1`;
   beta2 is `0.9` raised to `1/2048`, `1/512`, `1/128`, `1/32`, `1/8`, or `1/2`.
   Three existing screening results are reused, leaving 69 new executions.
3. Confirm the best six candidates on seeds 42–44, adding 12 executions.

Assuming finite results and no retries, each study needs 97 executions and
49 Slurm allocations. At 20 or 40 TPP, the nine studies for each model size need
873 executions and 441 allocations. At 60 TPP, the nine 10M studies have the same
counts; the two 20M ATC studies add 194 executions and 98 allocations to the
seven 20M refinements described below.

## 20M refinement at 60 TPP

All 14 source studies completed their 97 executions without retries. The starting
recipe is each method's 40-TPP winner selected by mean validation loss on seeds
42–44. The table records the confirmed winners from both rounds; each tuple is
`(LR, weight decay, beta1 exponent, beta2 exponent)`, with `beta = 0.9^exponent`.
The linked local reports contain the full candidate grids and individual seed
losses, so comparisons can use matching seeds rather than just the winning mean.

| Method | Workers | 20-TPP winner | 40-TPP starting recipe | Source reports |
|---|---:|---|---|---|
| sync | 1 | `(2^-7, .1, 1/2, 1/8)` | `(2^-7, .05, 1/2, 1/32)` | [20 TPP](../../runs/r-tuning/20m-sync-1w-512c-128bs-20tpp-lr-wd-betas-warmup4/report.json), [40 TPP](../../runs/r-tuning/20m-sync-1w-512c-128bs-40tpp-lr-wd-betas-warmup4/report.json) |
| cmp | 4 | `(2^-6, .05, 1/2, 1/32)` | `(2^-5, .025, 1/2, 1/32)` | [20 TPP](../../runs/r-tuning/20m-awc-4w-cmp-512c-128bs-20tpp-lr-wd-betas-warmup4/report.json), [40 TPP](../../runs/r-tuning/20m-awc-4w-cmp-512c-128bs-40tpp-lr-wd-betas-warmup4/report.json) |
| cmp | 8 | `(2^-5, .025, 1/2, 1/32)` | `(2^-5, .025, 1/2, 1/128)` | [20 TPP](../../runs/r-tuning/20m-awc-8w-cmp-512c-128bs-20tpp-lr-wd-betas-warmup4/report.json), [40 TPP](../../runs/r-tuning/20m-awc-8w-cmp-512c-128bs-40tpp-lr-wd-betas-warmup4/report.json) |
| exp | 4 | `(2^-6, .05, 1/2, 1/32)` | `(2^-6, .05, 1/4, 1/32)` | [20 TPP](../../runs/r-tuning/20m-awc-4w-exp-512c-128bs-20tpp-lr-wd-betas-warmup4/report.json), [40 TPP](../../runs/r-tuning/20m-awc-4w-exp-512c-128bs-40tpp-lr-wd-betas-warmup4/report.json) |
| exp | 8 | `(2^-5, .025, 1/2, 1/128)` | `(2^-6, .05, 1/4, 1/128)` | [20 TPP](../../runs/r-tuning/20m-awc-8w-exp-512c-128bs-20tpp-lr-wd-betas-warmup4/report.json), [40 TPP](../../runs/r-tuning/20m-awc-8w-exp-512c-128bs-40tpp-lr-wd-betas-warmup4/report.json) |
| ring | 4 | `(2^-6, .05, 1/2, 1/32)` | `(2^-6, .05, 1/4, 1/128)` | [20 TPP](../../runs/r-tuning/20m-awc-4w-ring-512c-128bs-20tpp-lr-wd-betas-warmup4/report.json), [40 TPP](../../runs/r-tuning/20m-awc-4w-ring-512c-128bs-40tpp-lr-wd-betas-warmup4/report.json) |
| ring | 8 | `(2^-6, .05, 1/4, 1/512)` | `(2^-6, .05, 1/4, 1/128)` | [20 TPP](../../runs/r-tuning/20m-awc-8w-ring-512c-128bs-20tpp-lr-wd-betas-warmup4/report.json), [40 TPP](../../runs/r-tuning/20m-awc-8w-ring-512c-128bs-40tpp-lr-wd-betas-warmup4/report.json) |

The reduced grids retain both rounds' winners. Every study searches beta1
exponents `{1/4, 1/2}`. Sync searches beta2 exponents `{1/128, 1/32, 1/8}`;
all AWC studies search `{1/2048, 1/512, 1/128, 1/32}`.

| Method | Workers | LR values | Weight-decay values | Executions | Slurm allocations |
|---|---:|---|---|---:|---:|
| sync | 1 | `2^-8, 2^-7, 2^-6` | `.025, .05, .1` | 36 | 19 |
| cmp | 4 | `2^-6, 2^-5` | `.025, .05` | 37 | 19 |
| cmp | 8 | `2^-6, 2^-5` | `.0125, .025, .05` | 39 | 20 |
| exp | 4 | `2^-7, 2^-6, 2^-5` | `.025, .05, .1` | 42 | 22 |
| exp | 8 | `2^-6, 2^-5` | `.0125, .025, .05` | 39 | 20 |
| ring | 4 | `2^-6, 2^-5` | `.025, .05` | 37 | 19 |
| ring | 8 | `2^-6, 2^-5` | `.0125, .025, .05` | 39 | 20 |
| **Total** | | | | **269** | **139** |

The evidence behind the reduction is:

- Every confirmed winner in both rounds uses one of the two retained beta1
  values. Sync retains its 20-TPP beta2 winner, while AWC retains all four
  near-one beta2 values: the best three 40-TPP exp-8w and ring-8w means differ
  by less than `0.0009`. With LR, decay, and beta1 fixed, each alternative
  beats the winning beta2 on one of the three matching seeds.
- Sync retains weight decay `.025` because its 40-TPP screen at LR `2^-7`
  was only `0.000101` worse than the retained `.05` candidate. Exp-4w keeps
  LR `2^-7` and decay `.1` because that pair's 20-TPP confirmed loss was only
  `0.002113` above the winner. Eight-worker studies retain decay `.0125`,
  which remained competitive in the 40-TPP screens.
- The old LR/decay screens fixed beta1 at `.9`; the new screens use the
  method-specific confirmed beta pair. Previous rankings are therefore
  conditional evidence, not proof that excluded combinations are inferior.
- Complete-topology winners reach the old upper LR limit `2^-5`. This
  refinement retains that limit; performance above it remains untested.

The phases still retain the best three LR/decay pairs, then the best six beta
candidates, and confirm on seeds 42–44. The starting LR/decay pair is included
in the first grid, and its beta pair is used in `initial_values`; it must still
earn its place in the next phase. Exact beta values match the later grid so each
retained pair reuses its seed-42 screening execution within the 60-TPP study.
Historical 20/40-TPP losses are not reused as 60-TPP results.

For `N` LR/decay pairs, the execution count is `N + 3*(B-1) + 12`, where
`B=6` for sync and `B=8` for AWC. Counts assume finite results and no retries;
allocation counts round up separately within each phase. This is **60.4% fewer
executions** than a broad seven-study campaign. Accounting for the longer
training horizon, `269*60 / (679*40)` is approximately `0.594`: **40.6% less
nominal training-token expenditure than the completed 40-TPP campaign**,
excluding runtime overhead and retries.

## Execution

`jobs_per_allocation: 2` groups up to two trials sequentially, with a 30-minute
budget per trial and a 60-minute budget per paired allocation. Each controller
allows up to 25 outstanding allocations and two retries.

From the repository root, with the project environment activated, the shared
cache prepared, and Slurm/account settings available:

```bash
python -m decentralized_llm tune run --no-local-config \
  --spec configs/tunings/20m-sync-1w-512c-128bs-60tpp-lr-wd-betas-warmup4.yaml \
  --dry-run
```

Omitting `--output` uses `runs/r-tuning/<specification-name-without-extension>`.
Pass `--output PATH` to choose another study directory. Remove `--dry-run` to
start a study; use `tune resume PATH` for an existing study.

All 143 earlier specifications, including examples and platform presets, are in
the [archive](archive/README.md). Their relative `base.configs` paths are adjusted
for that directory; their numerical recipes and controller settings are preserved.
Completed campaign artifacts and original reports remain in their run directories.
