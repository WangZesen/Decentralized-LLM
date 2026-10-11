# decentralized-llm

Reproducible Llama pretraining with one process per GPU, DDP or decentralized AdamW,
and one immutable uint16 token cache on shared storage. Python **3.14** and dependencies
are pinned in `uv.lock`.

The portable model, optimizer grouping, learning-rate schedules, virtual-epoch rounding,
and C4 preparation conventions come from
[tiny-llm dd03e5037b46961b1bbcbd014c445b6efebb4841](https://github.com/WangZesen/tiny-llm/tree/dd03e5037b46961b1bbcbd014c445b6efebb4841).
The distributed runtime, checkpoint format, global sampler, and network reader are new.
Adaptive consensus, GH200-specific kernels, old training-checkpoint imports, and WSD
branching workflows are outside this implementation.

## Install and check

Run commands from this repository. On a single architecture:

```bash
uv sync --locked --python 3.14
uv run decentralized-llm --help
bash scripts/check.sh
```

`check.sh` runs Ruff lint and formatting, `pyright --warnings`, and all CPU tests,
including two- and four-process Gloo comparisons. It hides GPUs for CPU tests because
PyTorch's AdamW accelerator health check can otherwise access a visible but prohibited GPU.
The targeted typing exceptions document PyTorch's `compile`/multiprocessing stubs and
Transformers' dynamically exposed tokenizer fields.
Ruff explicitly enables E501 at 120 columns; formatting alone does not enforce line length.

For an x86-64 login node and ARM compute nodes, install a **native uv** on each
architecture, either on PATH or at
`${DECENTRALIZED_ENV_ROOT:-$HOME/.local/share/decentralized-llm}/<architecture>/bin/uv`.
Then source:

```bash
source scripts/environment.sh
```

This uses `.venv-x86_64` or `.venv-aarch64`, separate Python installations and UV caches,
Python 3.14, and `uv sync --locked`. A shared lock serializes environment synchronization.
The SLURM launcher sources this script on every node; a custom
`launcher.environment_script` may instead activate a site environment and export
`DECENTRALIZED_PYTHON`. Install/synchronize ARM dependencies on an ARM node before a
network-restricted compute job. Project, cache, output, and environment paths must be
shared, and each allocation must have homogeneous GPUs.

## Configuration and preparation

Configuration files merge from left to right; dotted `--set key=value` overrides apply
last. Values use YAML syntax. Unknown fields and nonfinite numbers are rejected.
Changing `lr_schedule.name` or `launcher.kind` clears the previous variant's fields.
Five dimension presets are in `configs/{20m,50m,90m,250m,500m}.yaml`.

```bash
uv run decentralized-llm prepare -c configs/base.yaml -c configs/20m.yaml \
  --set data.cache_dir=/shared/tokens/c4
uv run decentralized-llm validate -c configs/base.yaml -c configs/20m.yaml \
  --set data.cache_dir=/shared/tokens/c4
```

Preparation resolves dataset and tokenizer revisions to commit SHAs **before reading**,
records the SHAs and tokenizer fingerprint, appends EOS to every document, writes little-endian
uint16 shards, and atomically publishes a checksummed version-1 manifest. Set
`data.revision` and `data.tokenizer_revision` to those resolved SHAs when creating the
same cache elsewhere. The source default `main` resolves once during preparation; an
existing immutable cache is reused and verified. Tokenizer worker results are consumed
in document order, so worker completion order does not affect the bytes. A `.partial`
directory is never a published cache; an exclusive preparation lock protects publication.

Existing valid tiny-llm version-1 token caches remain readable, regardless of their shard
sizes. Cache identity and vocabulary/preprocessing settings are checked. There is no
local staging or cache copy per training seed, rank, node count, or reader count.
`data.shuffle_seed` governs preparation's document order; `runtime.seed` governs training
sample order. Changing the latter does not require preparation.

By default preparation includes a full validation split and enough training tokens for
the configured global budget. A smaller already-prepared training cache may be reused
for multiple passes. `prepare_train_tokens` and `prepare_validation_tokens` provide
bounded smoke fixtures; truncated validation is marked in all evaluation results.

## Local GPU smoke runs

Prepare the small C4 cache once:

```bash
uv run decentralized-llm prepare -c configs/base.yaml -c configs/smoke.yaml
uv run decentralized-llm train -c configs/base.yaml -c configs/smoke.yaml -c configs/local-ddp.yaml
uv run decentralized-llm train -c configs/base.yaml -c configs/smoke.yaml -c configs/local-decentralized.yaml
```

On an allocated GPU, run the hardware tests with `uv run pytest -q -m cuda` to exercise
pinned-buffer reuse, BF16/SDPA parity, fused AdamW, and CUDA compilation.

The decentralized local example requests two GPUs. Set `distributed.gpus_per_node=1`
for a single-worker no-op mixing smoke. Set it to the actual GPU count for DDP.
Local multi-node configurations are rejected. The launcher checks GPU visibility and
executes the **current interpreter** with standalone `torchrun`, with no elastic restarts.
`--dry-run` validates and prints the launch command without writing a run.

For a CPU smoke, additionally set `runtime.device=cpu`, `distributed.backend=gloo`,
`runtime.amp=false`, and `runtime.fused_optimizer=false`, and run with
`CUDA_VISIBLE_DEVICES=''`. In CPU tests `gpus_per_node` is the number of processes.

## Two-node SLURM smoke runs

Set the account **only** through the lowercase environment variable `account`, using your
site's account value. Missing or empty values fail preflight, including dry runs.
There is deliberately no account configuration field. Its value is passed only as an
`sbatch` argument and is excluded from generated scripts, configuration, and receipts.

```bash
uv run decentralized-llm submit -c configs/base.yaml -c configs/smoke.yaml -c configs/slurm-ddp.yaml \
  --set launcher.partition=YOUR_GPU_PARTITION --dry-run
uv run decentralized-llm submit -c configs/base.yaml -c configs/smoke.yaml -c configs/slurm-ddp.yaml \
  --set launcher.partition=YOUR_GPU_PARTITION
uv run decentralized-llm submit -c configs/base.yaml -c configs/smoke.yaml -c configs/slurm-decentralized.yaml \
  --set launcher.partition=YOUR_GPU_PARTITION
```

Both examples allocate two nodes with one GPU each. Optional fields include `gpu_type`,
`constraint`, `qos`, `time`, `cpus_per_task`, `memory`, and `environment_script`.
One `srun` launcher task per node spawns `gpus_per_node` workers; `SLURM_PROCID` supplies
the node rank. The first allocated hostname and port **29500** supply rendezvous.
Override `distributed.rendezvous_port` for concurrent jobs sharing nodes. GPU visibility
and SLURM GPU/node allocation are checked on compute nodes. Communication and rendezvous
have configured timeouts; rank failures propagate through torchrun and `srun`.

Preflight checks configuration, cache metadata/checksums, sampler memory, token budgets,
checkpoint compatibility, output writability, environment scripts, and launch prerequisites.
It does not submit a job during a dry run. Submission artifacts are retained in a unique
`output_dir/submission-*` directory, including the resolved configuration and job receipt.

## Training and recovery semantics

`batch_tokens` is **global**; `micro_batch_size` is **per rank**. Accumulation supports a
smaller last microbatch and global sequence counts not divisible by world size. DDP
weights summed losses by their contribution to the global batch, uses `no_sync()` for
intermediate backwards, clips synchronized gradients, then steps AdamW. Empty final
rank microbatches perform a zero-loss backward so every rank participates.

Decentralized workers initialize independently and retain independent AdamW moments.
Each normalizes its local loss, clips local gradients, and communicates parameters once
per optimizer update. AWC mixes before applying the already-computed gradients; ATC
updates before mixing. `complete` uses all-reduce. `one_peer_ring` alternates reciprocal
neighbor pairings and requires an even world size; `one_peer_exponential` flips one rank
bit per update and requires a power of two. All support a single-worker no-op. Evaluation
and checkpoint coordination add their own collectives outside optimizer mixing.

Models retain tied embeddings, RoPE, RMSNorm, SwiGLU, depth-scaled initialization,
reference/SDPA attention, BF16 autocast, optional compilation, and fused CUDA AdamW.
Weight decay applies to matrices, not normalization vectors. The token budget uses one
model's parameter count; cumulative virtual-epoch boundaries round to the nearest global
batch. Cosine and WSD schedules retain upstream formulas. Virtual epochs do not reset
sampling. Each epoch evaluates one fixed validation subset; completion evaluates all
validation targets, including a partial final block. Full validation reads sequentially
and rank partitions do not duplicate samples. Decentralized evaluation uses separate
FP32-averaged weights without changing training replicas or optimizer state.

Checkpoint policies `interval`, `explicit`, `final`, and `none` retain upstream behavior:
interval/explicit select epoch snapshots; all except `none` also save final training state.
With `save_epoch_training_state=false`, selected epochs contain weights and rank metadata
but cannot resume training. A checkpoint is a directory whose `manifest.json` appears
only after every rank file succeeds. Each rank saves RNG and committed sampler state;
DDP saves one shared model/optimizer, while decentralized workers each save their own.

```bash
uv run decentralized-llm train -c runs/local-ddp/config.yaml \
  --resume runs/local-ddp/epoch-001 --set runtime.output_dir=runs/resumed
uv run decentralized-llm evaluate -c runs/local-ddp/config.yaml \
  --checkpoint runs/local-ddp/final
uv run decentralized-llm evaluate -c runs/local-ddp/config.yaml \
  --checkpoint runs/local-ddp/final --subset
uv run decentralized-llm submit -c runs/slurm-ddp/config.yaml \
  --resume runs/slurm-ddp/epoch-001 --set runtime.output_dir=runs/slurm-resumed
uv run decentralized-llm submit -c runs/slurm-ddp/config.yaml \
  --action evaluate --resume runs/slurm-ddp/final
```

Resume requires the same world size, training recipe, cache, and sampler version/NumPy
version. Output paths, reader threads, prefetch depth, host budget, and homogeneous physical
node placement may change. Only **optimizer-committed global sample positions** are saved;
prefetched or transferred unused samples are discarded and deterministically regenerated.
Sampling is exact across resume; floating-point collective reduction order can produce
small numerical differences after process recreation. A completed budget should be evaluated,
not resumed. Checkpoints use PyTorch pickle for RNG/optimizer state; load trusted files only.

## Network loader and benchmarks

Every complete context-length block with one lookahead token is eligible. Each dataset
pass uses a seeded NumPy PCG64 permutation of **all** eligible blocks without replacement;
pass seeds derive from cache identity, context length, training seed, and sampler version.
Rank `r` receives global stream positions congruent to `r` modulo world size, continuously
across passes and virtual epochs.

The permutation stays in host memory. Persistent reader threads reuse bounded file
descriptor caches and use `pread`, splitting blocks across shards as needed. Within a
read-ahead window, reads are sorted by shard/offset and coalesced subject to
`read_range_bytes`, `coalesce_gap_bytes`, and `max_read_amplification`. Batches are delivered
in prescribed order even when reads finish out of order. Defaults are **4 reader threads,
8 prefetched microbatches, and 512 MiB loader host memory per rank**. Preflight estimates
permutation, Python request metadata, compact windows, read scratch, validation and transfer
buffers. It rejects an estimated over-budget configuration before launch. Model/optimizer
memory, interpreter overhead, and the OS page cache are separate from this loader budget.

Only upcoming batches convert to int64. Double-buffered pinned storage, nonblocking H2D
copies, a dedicated CUDA stream, and completion events protect reuse. The initial read
queue fills before the training timer starts. Pending work is canceled on shutdown;
a running POSIX read must return before its descriptor is closed. Python cannot interrupt
a kernel-blocked network read. Finite prefetch cannot guarantee zero stalls under arbitrary
storage slowdowns.

Train diagnostics retain requested/read bytes, amplification, read calls, queue occupancy,
starvation count, p50/p95/p99/max batch wait, estimated buffer bytes, actual sampler bytes,
and process peak RSS. Starvation counts ready-queue-empty requests; wait time measures
CPU blocking, not direct GPU idle time. Read bytes count bytes returned by positional reads,
not storage-server traffic or filesystem readahead. Wait quantiles cover the most recent
8,192 delivered batches.
`metrics-rank-*.jsonl`, `loader-rank-*.json`, `rank-*.json`, failure tracebacks, resolved
configuration, source digest, library versions, and evaluation metrics remain in the run.

The benchmark compares complete forward/backward/clip/AdamW/mixing pipelines after warmup,
using either the network loader or a host-resident batch with the same conversion/transfer
path and microbatch shapes. It reports a **95% baseline throughput target** and marks misses;
it does not infer acceptance from read bandwidth alone. Global benchmark batches must be
divisible by world size so the resident baseline has identical rank shapes. Distributed
elapsed time is the maximum over ranks; each rank writes separate loader diagnostics.

```bash
uv run decentralized-llm benchmark-data -c configs/base.yaml -c configs/20m.yaml \
  --set data.cache_dir=/shared/tokens/c4 --warmup 20 --steps 100 \
  --cache-state cold --output runs/benchmark/cold.json
uv run decentralized-llm benchmark-data -c configs/base.yaml -c configs/20m.yaml \
  --set data.cache_dir=/shared/tokens/c4 --warmup 20 --steps 100 \
  --cache-state warm --output runs/benchmark/warm.json
```

Warm mode primes the exact stream before measuring. Cold mode requests
`POSIX_FADV_DONTNEED` on training shards; this is best-effort client eviction, **not proof
of cold storage-server caches**. Record site cache-control procedures separately when
certified cold-cache measurements are required. Repeat runs to estimate variability;
use the same model, precision, world size, topology, and global batch for comparisons.

For multiple local GPUs, launch the benchmark explicitly with torchrun:

```bash
uv run python -m torch.distributed.run --standalone --nproc-per-node=2 --max-restarts=0 \
  --module decentralized_llm benchmark-data -c configs/base.yaml -c configs/20m.yaml \
  --set distributed.gpus_per_node=2 --steps 100 --warmup 20 --output runs/benchmark/multi.json
```

Inside a matching two-node SLURM allocation, use one task per node (source the environment
inside each task to select its architecture):

```bash
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
srun --ntasks=2 --ntasks-per-node=1 --gpus-per-task=1 --gpu-bind=none --kill-on-bad-exit=1 \
  bash -c 'source scripts/environment.sh
    exec "$DECENTRALIZED_PYTHON" -m torch.distributed.run --nnodes=2 --nproc-per-node=1 \
      --node-rank="$SLURM_PROCID" --master-addr="$MASTER_ADDR" --master-port=29500 \
      --rdzv-backend=static --rdzv-conf=timeout=300 --max-restarts=0 \
      --module decentralized_llm benchmark-data -c configs/base.yaml -c configs/20m.yaml \
      -c configs/slurm-ddp.yaml --steps 100 --warmup 20 --output runs/benchmark/slurm.json'
```

A reproducible synthetic CPU fixture is available without downloading C4:

```bash
uv run python scripts/make_synthetic_cache.py -c configs/benchmark-cpu.yaml
CUDA_VISIBLE_DEVICES='' uv run decentralized-llm benchmark-data -c configs/benchmark-cpu.yaml \
  --steps 100 --warmup 20 --cache-state cold --output runs/benchmark-cpu/cold.json
CUDA_VISIBLE_DEVICES='' uv run decentralized-llm benchmark-data -c configs/benchmark-cpu.yaml \
  --steps 100 --warmup 20 --cache-state warm --output runs/benchmark-cpu/warm.json
```

The fixture is explicitly labeled synthetic and must not be used to claim language-model
quality. See [validation evidence](docs/validation.md) for what was actually tested and
which hardware checks still require an allocated GPU job.
