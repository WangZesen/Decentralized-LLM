# decentralized-llm

Reproducible Llama pretraining with one process per GPU, DDP or decentralized AdamW,
and one immutable uint16 token cache on shared storage. Python **3.14** and dependencies
are pinned in `uv.lock`.

The portable model, optimizer grouping, learning-rate schedules, virtual-epoch rounding,
and C4 preparation conventions come from
[tiny-llm dd03e5037b46961b1bbcbd014c445b6efebb4841](https://github.com/WangZesen/tiny-llm/tree/dd03e5037b46961b1bbcbd014c445b6efebb4841).
The distributed runtime, checkpoint format, global sampler, and network reader are new.
The optional GH200 execution path also adapts upstream fused normalization, SwiGLU,
RoPE, and cross-entropy kernels, preserving canonical parameters and distributed AdamW.
Adaptive consensus, old training-checkpoint imports, and WSD branching workflows are
outside this implementation.

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
After successful synchronization, a lockfile/project fingerprint and installed-version
check let subsequent jobs activate the environment without invoking uv or the network.
The SLURM launcher sources this script on every node; a custom
`launcher.environment_script` may instead activate a site environment and export
`DECENTRALIZED_PYTHON`. Install/synchronize ARM dependencies on an ARM node before a
network-restricted compute job. Project, cache, output, and environment paths must be
shared, and each allocation must have homogeneous GPUs.

For compiled execution, configure the host C/C++ compilers using the
[toolchain setup guide](docs/toolchain-setup.md). It covers per-architecture local
settings, compiler modules, verification, and retrying a failed run.

## Configuration and preparation

Configuration starts with built-in defaults, then local execution defaults, then
configuration files from left to right; dotted `--set key=value` overrides apply last.
Values use YAML syntax. Unknown fields and nonfinite numbers are rejected.
Changing `lr_schedule.name` or `launcher.kind` clears the previous variant's fields.
Five dimension presets are in `configs/{20m,50m,90m,250m,500m}.yaml`.

Run commands from the repository root. User-facing configuration commands, including
training, submission, benchmarking, and `tune run`/`extend`, automatically read the
Git-ignored `configs/local/defaults.yaml`. If it is missing, a normal command creates
the following compatible baseline atomically, preserving any existing file:

```yaml
runtime:
  execution_backend: native
  compile: auto
  cpu_affinity: none
```

This baseline uses eager native execution without exclusive CPU binding. Set
`execution_backend: gh200` or `execution_backend: a100` in this one local file to
select fused execution for new runs and tuning studies without adding a hardware
overlay to each experiment. `compile: auto` enables compilation for either fused
backend. Set `cpu_affinity: exclusive` locally if your GPU nodes support that placement.
The [example](configs/local-defaults.example.yaml) also documents these settings.

The file accepts partial `runtime` settings except `seed` and `output_dir`; all other
sections are rejected. An empty existing file supplies no overrides. Explicit experiment
settings (including `auto`), command-line overrides, and tuning `base.overrides` win over
local defaults. Scheduler, network, model, and training settings belong in experiment
configurations. Backend compatibility checks remain strict; incompatible hardware or
workloads do not silently fall back to another backend.

Use `--no-local-config` to bypass both creation and loading, for example for a CPU or
small smoke run. Dry runs use the baseline in memory and report a missing defaults file
on stderr without creating it. Normal creation notices also go to stderr, preserving
JSON command output. Invalid or unreadable existing files fail with their path in the
diagnostic and are never replaced.

Workers and `tune resume`/`status`/`report` use saved configurations and do not read or
create local defaults. `tune extend` resolves the current defaults into a new revision.
The Python `load_config` and `read_spec` APIs remain explicit by default; opt in with
`use_local_defaults=True`. These loaders never write files. The tuning controller
initializes defaults for new runs/revisions unless `use_local_defaults=False` is passed.

Other local experiments can remain in `configs/local/`. When copying a tuning spec
there, change its `base.configs` references to `../base.yaml`, `../20m.yaml`, and the
appropriate `../` launcher filename. No reference to `defaults.yaml` is needed.

Tracked `tuning-*.yaml` specifications reference only the shared base, model, and
launcher configurations. They do not require an ignored config or a GH200/A100 preset,
and leave the execution backend to local defaults. The generic multi-node specifications
also leave scheduler network selection and transport activation unset. Put site-specific
`launcher.network` and `launcher.environment_overlays` in a local experiment or tuning
copy, after checking its interconnect setup; these fields do not belong in `defaults.yaml`.
The standard environment script supports both host architectures and does not require
local toolchain files. Smoke and CPU benchmark configs explicitly choose eager native
execution and no CPU binding, so they also work with fused local defaults.

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

## Training across GPU platforms

Select the backend once in `configs/local/defaults.yaml`, then use shared experiment
and launcher files. For example, this four-GPU submission uses the local backend:

```bash
uv run decentralized-llm submit -c configs/base.yaml -c configs/90m.yaml \
  -c configs/slurm-ddp.yaml --set distributed.nodes=1 --set distributed.gpus_per_node=4 \
  --set data.cache_dir=/shared/tokens/c4 --set launcher.time=02:00:00 \
  --set runtime.output_dir=runs/90m-4gpu --dry-run
```

Set the lowercase `account` environment variable and the site's GPU partition before
submission. Changing the local backend does not change GPU counts, batching, search
grids, or scheduler settings. Hardware and workload compatibility checks still apply.

## Optional four- and eight-GPU GH200 presets

After preparing the shared token cache, choose one hardware preset. Each is layered
**directly after the base configuration and model preset**. These explicit recipes
override local execution defaults and are intended for reproducing the measured
hardware setup. Generic training and tuning do not depend on them:

```bash
uv run decentralized-llm submit -c configs/base.yaml -c configs/90m.yaml \
  -c configs/gh200-4gpu.yaml --set data.cache_dir=/shared/tokens/c4 \
  --set launcher.time=02:00:00 --set runtime.output_dir=runs/90m-4gpu --dry-run
uv run decentralized-llm submit -c configs/base.yaml -c configs/90m.yaml \
  -c configs/gh200-8gpu.yaml --set data.cache_dir=/shared/tokens/c4 \
  --set launcher.time=02:00:00 --set runtime.output_dir=runs/90m-8gpu --dry-run
```

Set the lowercase `account` environment variable to your scheduler account, inspect
the dry run, then remove `--dry-run` to submit. Choose walltime for your model and token
budget; the presets retain a short **15-minute** limit unless overridden. Both allocate
four GPUs per node, use a global batch of 32,768 tokens, and launch one process per GPU.
Per-rank microbatches are eight on one node and four on two nodes.
DDP is the default; add `--set distributed.mode=decentralized` for complete-topology
AWC, or select another supported topology with `distributed.topology`.

The presets explicitly enable **compiled GH200 kernels and exclusive GPU-local CPU
binding**. They use BF16, fused AdamW, fused projections, automatic SDPA selection,
eight PyTorch threads per worker, and persistent compilation caches. Slurm determines
the CPU allocation: the measured site supplies 72 CPUs per GPU, or 288 per full node.
Torchrun's `--numa-binding=exclusive` binds each worker before Python starts, and worker
threads inherit its GPU-local CPU mask. Install `numactl` on every GPU node.
This is CPU placement, not an explicit NUMA memory policy.

`runtime.fused_rope` defaults to `auto`, enabling the fused RoPE kernel for GH200
20M/90M execution with projection fusion. Other execution paths use ordinary RoPE.
Set `--set runtime.fused_rope=false` to disable it for comparisons. Explicit `true`
still requires a supported GH200 model and projection fusion.

The eight-GPU preset also requests `--network=job_vni` and sources
[`scripts/activate_cxi.sh`](scripts/activate_cxi.sh) after the Python environment and
before torchrun. This selects the verified Slingshot/CXI RDMA stack while preserving
PyTorch's NCCL. **Its library paths and network settings are installation-specific**;
inspect them before use on another site. `launcher.environment_script` can select
another validated Python environment. For a different interconnect, replace the
activation through `launcher.environment_overlays` and adjust `launcher.network`.

These are the selected throughput defaults, not a claim of numerical equivalence or
full-training convergence. Larger-model and eight-GPU short trajectories did not pass
all numerical gates. The [performance report](docs/gh200-performance.md) records those
failures, the measured improvements, and the limits of the affinity comparisons.
For comparisons, override the recipe explicitly:

| Execution | Overrides |
| --- | --- |
| Compiled native | `runtime.execution_backend=native`, `runtime.compile=true` |
| Eager native | `runtime.execution_backend=native`, `runtime.compile=false` |
| Strict qualified selection | `runtime.execution_backend=auto`, `runtime.compile=auto` |

Pass each override with `--set`. Generic configurations use the local backend, initially
`native`. Explicit `auto` selection retains strict qualification; unmatched workloads
use eager native execution. The sole historical
qualified profile is four-GPU 20M AWC with FlashAttention and 64 allowed CPUs per node,
without exclusive binding. It does not qualify the new full-CPU presets. Compiled
native also showed trajectory drift in the short comparisons. The GH200 backend
requires supported model geometry on GH200, BF16, SDPA, and nondeterministic execution.

Compiled processes share `.cache/compile` by default. Set `runtime.compile_cache_dir`
to one persistent shared directory when tuning from different checkouts. Namespaces
include architecture, Python/PyTorch/CUDA/Triton versions, GPU capability, and executable
model/compiler code. Seeds, optimizer hyperparameters, and output directories do not
create separate namespaces; different tensor shapes or compiler guards can still
require compilation. Keep the cache between trials. Hits reduce compilation work,
while Python imports, NCCL setup, and input loading still take time.

Use the run's saved `config.yaml` when resuming; checkpoint manifests pin the resolved
execution policy. See [training and recovery semantics](#training-and-recovery-semantics)
for checkpoint and evaluation commands. To measure the recipe before a larger run,
replace `submit` with `benchmark-training`; bounded measurement and reproduction
commands are in the [performance report](docs/gh200-performance.md#reproducing-the-measurements).

## A100 fused execution

The explicit A100 backend reuses the fused residual/RMSNorm, SwiGLU, projection,
RoPE, and cross-entropy implementations. Set `execution_backend: a100` in local defaults
for shared training and tuning configurations. To reproduce the measured hardware
recipe, optionally layer `configs/a100-1gpu.yaml` after the base configuration and
model preset, then set your site's partition and walltime. This preset overrides
local execution defaults.
The [A100 performance report](docs/a100-performance.md) records measured throughput,
numerical checks, and reproduction commands. Automatic backend selection retains
its qualification requirements; an explicit A100 selection requires supported model
geometry, BF16 SDPA, compilation, and nondeterministic execution on an A100.

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
`cpus_per_task` defaults to `null`: no CPU-count flag is sent, so Slurm applies the
site's GPU-to-CPU policy. The measured GH200 partition has `DefCpuPerGPU=72`, giving
288 CPUs per four-GPU node. Set `launcher.cpus_per_task` only for an explicit override;
`--set launcher.cpus_per_task=null` removes a limit inherited from another config.
`runtime.cpu_threads` independently controls each worker's PyTorch/OMP thread count.
`launcher.memory` also defaults to `null`, and the presets leave it unset, so Slurm
allocates memory according to the GPU count. Set it only for an explicit override;
`--set launcher.memory=null` clears a memory request inherited from another config.
One `srun` launcher task per node spawns `gpus_per_node` workers; `SLURM_PROCID` supplies
the node rank. The first allocated hostname and port **29500** supply rendezvous.
Single-node jobs automatically select an available rendezvous port. Override
`distributed.rendezvous_port` for concurrent multi-node jobs sharing nodes. GPU visibility
and SLURM GPU/node allocation are checked on compute nodes. Communication and rendezvous
have configured timeouts; rank failures propagate through torchrun and `srun`.

Preflight checks configuration, cache metadata/checksums, sampler memory, token budgets,
checkpoint compatibility, output writability, environment scripts, and launch prerequisites.
It does not submit a job during a dry run. Submission artifacts are retained in a unique
`output_dir/submission-*` directory, including the resolved configuration and job receipt.

## Phased hyperparameter tuning

`decentralized-llm tune` runs a persistent controller over local training processes or
SLURM jobs. Prepare the shared cache first. Run the controller from this repository in
an enduring shell such as tmux, or in your own CPU allocation. SLURM uses the same
lowercase `account` environment variable as ordinary submissions; it stays out of
configuration, receipts, and reports.

Start with [`configs/tuning-example.yaml`](configs/tuning-example.yaml), updating the
referenced training configuration and SLURM partition for your environment:

```bash
uv run decentralized-llm tune run --spec configs/tuning-example.yaml \
  --output runs/tuning/example --dry-run
uv run decentralized-llm tune run --spec configs/tuning-example.yaml \
  --output runs/tuning/example
uv run decentralized-llm tune status runs/tuning/example
uv run decentralized-llm tune resume runs/tuning/example
uv run decentralized-llm tune extend runs/tuning/example --spec configs/tuning-example.yaml --dry-run
uv run decentralized-llm tune extend runs/tuning/example --spec configs/tuning-example.yaml
uv run decentralized-llm tune report runs/tuning/example
```

The tuning specification has a `base.configs` list and optional dotted `base.overrides`
mapping. Referenced YAML filenames resolve relative to the tuning specification;
paths inside training configurations retain the normal repository-working-directory
semantics. Local execution defaults apply before these files and `base.overrides`;
resolved configurations are frozen in the study state. The same specification can
use GH200 or A100 by editing only `configs/local/defaults.yaml` when it does not
explicitly select a backend. `--no-local-config` is available on `tune run` and `extend`.

The shared SLURM presets use the example partition `gpu`. Find your site's partition
with `sinfo --noheader --format=%P`, then set `launcher.partition` under the tuning
specification's `base.overrides`, or append a site launcher config to `base.configs`.
The automatically loaded `configs/local/defaults.yaml` accepts runtime settings only;
it does not select a scheduler partition. Tuning dry runs query the configured
partition's walltime limit, so they also require a valid partition and access to the
scheduler.

For a four-GPU study using the local backend, copy the example and replace its `base`
block with:

```yaml
base:
  configs: [base.yaml, 90m.yaml, slurm-ddp.yaml]
  overrides:
    data.cache_dir: /shared/tokens/c4
    training.checkpoint_policy: none
    distributed.nodes: 1
    distributed.gpus_per_node: 4
    launcher.time: '02:00:00'
```

This example assumes the copied specification remains in `configs/`; use `../` prefixes
for the references in a copy under `configs/local/`. Set `distributed.nodes: 2` for
eight GPUs and configure any site-specific transport in that local copy.
Keep the example's phases and seed settings, set the cache path and intended walltime,
and run `tune run --dry-run` before starting the study. The local execution defaults
are resolved once for the study and apply to every trial, with compatible compilation
artifacts reused. Set affinity and compilation-cache preferences in `defaults.yaml`.

Each phase's `search` maps **one or more dotted configuration fields** to search
specifications. The controller jointly evaluates their Cartesian product, averages
final full-validation loss over the required training seeds, and keeps the **global
top K complete combinations** specified by `top_k`. A later phase expands each
surviving combination with its new grid. Earlier winning coordinates remain paired.
It can also revisit previous search fields, replacing those coordinates and
deduplicating resulting configurations.

For every parameter first searched in a later phase, the current phase must explicitly
set its value in `initial_values`. Earlier winning values are inherited. Every other
setting comes from the base configuration, including the common training-budget
settings. Each candidate must pass the existing training configuration validation.
Training seeds, output directories, the cache directory, and launcher settings are
controller/base settings rather than searchable fields.

Search specifications can be freely mixed within a phase:

```yaml
search:
  optimizer.lr:
    scale: power
    base: 2
    exponent_min: -12
    exponent_max: -8
    points: 5
  optimizer.weight_decay: [0.0, 0.1, 0.2]
  optimizer.beta1: {scale: linear, min: 0.8, max: 0.95, points: 4}
  optimizer.eps: {scale: log, min: 0.00000001, max: 0.000001, points: 3}
```

Linear and logarithmic intervals include both endpoints. Power intervals use a fixed
positive base and default to `exponent_scale: linear`, with evenly spaced exponents:
the example generates `2^-12`, `2^-11`, `2^-10`, `2^-9`, and `2^-8`.
Set `exponent_scale: log` to space exponent magnitudes geometrically:

```yaml
optimizer.lr:
  scale: power
  base: 2
  exponent_scale: log
  exponent_min: -8
  exponent_max: -2
  points: 3
```

This generates exponents `[-8, -4, -2]` and values `[2^-8, 2^-4, 2^-2]`.
Positive exponent bounds work too: `2` to `8` with three points gives `[2, 4, 8]`.
Logarithmic exponent bounds must be nonzero and have the same sign; zero and
mixed-sign bounds are rejected. Ordinary `scale: log` intervals require positive
value bounds. Both exponent spacing modes include the exact endpoints and support
fractional exponents. One-point intervals require equal bounds; otherwise, the
minimum must be less than the maximum. Integer configuration fields require integral values.
Power expressions remain visible in dry runs and reports, alongside numeric values;
generated training configurations use numbers. Numerically identical configurations
share a trial even when their search notation differs.

`screening_seeds` is a required positive count. Every screening phase uses the same
seed sequence starting at `seed_start` (default **42**). `confirmation.seeds` is the
**total** confirmation count, which must be at least the screening count. Confirmation
adds only missing seeds for the final surviving combinations. Every selection uses
exactly its phase's seed set, even if more results already exist. The lowest mean
final validation loss wins; exact ties use a stable candidate ID. Reports include
sample standard deviation, or null for a single seed.

Dry runs show a prominent total-job estimate and per-phase required, reused, and new
executions. One job means one complete configuration/seed training execution, regardless
of its node or GPU count. The example specification searches 15 learning-rate/decay
pairs on two seeds, then six momentum combinations, then confirms two survivors on
five total seeds: **44 new jobs**, or **at most 132 training attempts** with two
retries per trial. Dry runs for resume and extension also count existing results,
outstanding jobs, and consumed retries. When unknown winners affect reuse or deduplication,
the output explicitly labels a conservative upper estimate. Dry runs create no study
artifacts and submit no jobs; normal cache and launcher prerequisites still apply.

The controller settings default to `max_jobs: 50`, `jobs_per_allocation: 1`,
`max_retries: 2`, `poll_seconds: 30`, and `result_grace_seconds: 120`. The SLURM cap
counts queued and running **allocations**, each of which can contain multiple trials.
Local training is sequential and ignores `jobs_per_allocation`. Scheduler queries are batched.

To amortize queue waits over several short trials, for example:

```yaml
base:
  configs: [base.yaml, 20m.yaml, gh200-8gpu.yaml]
  overrides:
    launcher.time: '00:15:00'  # Individual trial limit.
controller:
  jobs_per_allocation: 4
  max_jobs: 2                # At most two queued/running sbatch allocations.
```

Keep the other required tuning fields (phases and seed counts) in your specification.
`jobs_per_allocation` is a maximum count of configuration/seed trials, not candidates
or nodes. Within each phase, including confirmation, compatible pending trials are
balanced across the fewest allocations: ten trials with a limit of four become
**4, 3, 3**, with walltimes **60, 45, 45 minutes** for the example above. Grouping happens
before the concurrency cap is applied. Trials needing different node/GPU resources are
grouped separately; retries are grouped separately from first attempts. Completed and
already submitted trials are excluded. Group membership survives controller restarts.

Each allocation submits one `sbatch` script and executes its trials sequentially as
`srun` steps, retaining its resources between trials. Each step retains its individual
time limit, environment, configuration, seed, and training output directory. Allocation
walltime is the sum of the individual limits, rounded to [Slurm's minute resolution](https://slurm.schedmd.com/sbatch.html#OPT_time);
there is no extra startup/cleanup allowance. Finite positive trial limits are required.
The controller queries the partition's `MaxTime` before submitting a phase and rejects
an oversized group with a suggested smaller `jobs_per_allocation`, without automatically
splitting it. Dry runs also check exact allocation estimates against the partition limit.
Unknown future winners/resource combinations are labeled as conservative estimates;
their actual groups are checked before that phase is submitted.

The allocation continues through individual trial failures. Each step publishes a
durable start/exit record, so successful trials remain successful even if a later step
or the parent allocation fails. When the allocation is ending, or the remaining time
rounded up to minutes is smaller than the next trial's limit, remaining trials are left
unstarted and rescheduled. Sub-minute launcher overhead is tolerated, but the parent
allocation's deadline always applies. Even an allocation with all trials finished keeps
its concurrency slot until Slurm reports the parent job has ended.

Retries start from scratch with the same seed in a fresh attempt directory, preserving
previous logs. Two retries means at most three started training attempts. Unstarted
trials do not consume training retries; unsuccessful allocation attempts before a trial
starts have a separate budget, also bounded by `max_retries` (at most three with the
default). Submission rejection counts against that allocation-failure budget. Legacy
individual jobs and local processes retain their original submission-based retry counting.
Exhausting either budget stops new submissions and phase advancement, exits unsuccessfully,
and writes a partial report with a reason identifying the budget. A candidate requires
every seed to succeed. Already submitted
jobs can finish and their results are collected on resume.

Dry-run JSON preserves trial counts such as `new_jobs` and `additional_attempts_upper_bound`
(training attempts), and adds `new_allocations`, `already_submitted_allocations`,
`allocations_exact`, and per-phase allocation sizes/durations. `additional_allocations_upper_bound`
also allows for retries and failures before trial startup; it is conservative because
retry groups need not fill an allocation. Status and reports list allocation membership,
duration, scheduler state, and log locations separately from trial results.

Ctrl-C stops the controller while retaining submitted work. `resume` reconciles persisted
jobs and continues the frozen specification and retry counters. Active local processes
can be reattached on their original host. Missing scheduler records or scheduler outages
do not trigger duplicate submissions. If an attempt remains unresolved, inspect its
receipt, scheduler history, and original host; **after verifying it has no live job**,
`tune resume DIR --retry-unresolved ATTEMPT_NAME` marks that attempt failed and permits
a retry within the existing limit. This explicit recovery covers interruption between
submission intent and a durable launch receipt.
For grouped jobs, use the allocation name shown in the warning (a member attempt name
also resolves to its allocation). Verify that the **whole allocation** has no live job:
recovery reconciles all its members together, preserves completed results, and releases
unfinished members for retry. Missing scheduler records never release individual members
of a potentially live allocation.

`extend` creates a new revision and replays phases using matching successful results.
It supports revised grids, changed phase sequences, initial values, and seed counts.
Increasing `max_retries` can reopen exhausted trials. Previous reports and state are
archived under `revisions/`. Active jobs remain tracked even if a revised search no
longer needs them. Reuse requires matching effective configuration, seed, cache content,
and source/dependency fingerprints. Changed software or cache identity requires an
extension and produces new trials. Every accepted result is checked against its
configuration, completed budget, and recorded provenance.

The study directory contains atomically updated `state.json`, isolated
`trials/<phase-name>/<id>/attempt-*/` artifacts, and `report.md`/`report.json`. A trial's
physical directory belongs to the phase that first creates it; later phases reuse it
through relative symlinks directly to that directory. Retries stay in the original
directory. Phase directories persist across revisions, while reports describe the current
revision's candidates. Existing flat trial directories remain in place and can be linked
from new phases. Reports identify their
revision, show complete candidate combinations, power notation, seed losses, selections,
reuse, retries, and failures. A successfully confirmed winner also produces `winner.yaml`
with a fresh output destination. `tune status` reads the last persisted snapshot;
`tune report` regenerates reports from that snapshot. A controller lock prevents
concurrent writers to the same study.

New SLURM tuning submissions, including groups of one, place shared batch artifacts in
`slurm/<phase-name>/<allocation-id>/`: `submit.sh`, `manifest.json`, `receipt.json`,
`slurm-<jobid>.out`, and `slurm-<jobid>.err`. Its `steps/` directory contains separate
stdout, stderr, and execution JSON records for each trial attempt. Trial configurations,
worker/step scripts, and training results remain in the trial attempt directories;
attempt records and reports link to the allocation and step logs. Existing logs are
never relocated. State version 1 studies are read compatibly and saved as version 2.
Existing active individual jobs remain tracked alongside new grouped allocations.

As with other code changes, updating the controller changes the source fingerprint:
`resume` requires the same provenance, while `extend` creates a new revision and new
trials for changed provenance. Use an isolated checkout while queued jobs still refer
to an older checkout, so those jobs continue to load their intended code.

Reports include Seaborn heatmaps of mean final validation loss over exactly the phase's
requested seeds. Screening plots use the first two search fields as horizontal and
vertical axes; one-field searches use a heatmap strip. Additional searched fields and
carried-over parameter combinations produce separate panels with a shared color scale.
Confirmation plots show each candidate's seed losses and a mean column. Power notation
is retained, selected cells are outlined, and pending, failed, and unsearched cells are
labelled separately. A candidate's mean appears only after all its required seeds succeed.
PNG figures are embedded in `report.md`, SVG copies are linked for zooming/export, and
`report.json` lists the relative paths in each phase's `heatmaps` array. Figures live in
`plots/` and are copied with archived revision reports. Regenerating an older study's
report adds plots without moving trials or modifying training logs.

## Training and recovery semantics

Training metrics are saved every **50 optimizer iterations** by default; override
`training.log_every` for a different cadence. A final partial interval is also logged
when training finishes or stops at a requested step. Each per-rank JSONL record contains
the step, committed global cursor, learning rate, local loss, gradient norm, and throughput.
Loss and gradient norm describe the logged update, not an average across the interval.
Full loader diagnostics are written once to `loader-rank-*.json` at shutdown, including
failure cleanup, instead of being repeated in every training record.

`global_tokens_per_second` measures global tokens committed since the previous log
against active training time in that interval. CUDA work completes before timing ends,
and elapsed time is reduced with a maximum across ranks. Validation, checkpointing, and
log-writing pauses are excluded. Intervals span epoch boundaries, account for partial
batches, and restart on resume. Compilation can affect the first interval; later records
reflect recent training speed rather than a cumulative average. Benchmark throughput
retains its separate warmup and measurement windows.

All training modes use one contiguous parameter allocation with 256-byte-aligned
parameter starts. Existing parameter objects, tied weights, optimizer groups, and
checkpoint tensor names/shapes are preserved. Storage is prepared after device/dtype
placement and before optimizer or execution setup; evaluation copies are prepared
independently. Decentralized mixing copies between this live allocation and a separate
snapshot in bulk. `distributed.reuse_mix_buffers` controls snapshot allocation reuse,
not the parameter layout. DDP continues to synchronize gradients through its normal
buckets and does not allocate a parameter-mixing snapshot.

`training.batch_tokens` is **global**. `training.num_accumulation` controls the number
of forward/backward micro-batches per optimizer update and defaults to **1** (no
accumulation). The per-rank micro-batch size is inferred internally:

```text
workers = distributed.nodes * distributed.gpus_per_node
micro_batch_size = batch_tokens / (context_length * workers * num_accumulation)
```

This division must produce a positive integer; invalid configurations fail before
launch. With `batch_tokens=32768`, context length `1024`, and `num_accumulation=1`,
one, four, and eight workers use micro-batches of `32`, `8`, and `4` sequences per
rank. Increasing `num_accumulation` reduces the micro-batch size while preserving
the global token budget. Preflight output and `run.json` report both values.

Remove `training.micro_batch_size` from older configs and set
`training.num_accumulation: 1` for no accumulation. To preserve an older recipe,
use `num_accumulation = batch_tokens / (context_length * workers * old_micro_batch_size)`;
this must also be a positive integer. Equivalent recipes retain checkpoint
compatibility; recipes with uneven micro-batches require a divisible batch configuration.

DDP weights summed losses by their contribution to the global batch, uses `no_sync()`
for intermediate backwards, clips synchronized gradients, then steps AdamW. Loader-only
truncated ranges may still produce smaller or empty final rank micro-batches; empty
ones perform a zero-loss backward so every rank participates.

Decentralized workers initialize independently and retain independent AdamW moments.
Each normalizes its local loss, clips local gradients, and communicates parameters once
per optimizer update. AWC starts an asynchronous exchange of a parameter snapshot before
forward, overlapping communication with forward and all accumulated backwards. It waits
after local gradient clipping, installs the mixed weights, and then steps AdamW; the live
weights used by autograd stay unchanged until backward finishes. The snapshot is temporary
and each update joins its exchange before checkpointing. ATC updates before mixing.
`complete` uses all-reduce over the world group. `one_peer_ring` alternates reciprocal
neighbor pairings and requires an even world size; `one_peer_exponential` flips one rank
bit per update and requires a power of two. Both peer topologies use asynchronous
all-reduce over two-rank process groups, followed by division by two. The groups are
created in a consistent order at startup and reused as peers rotate; at world size two,
the existing world group is reused. Every exchange is joined before switching groups.
All support a single-worker no-op. Evaluation
and checkpoint coordination add their own collectives outside optimizer mixing.

Set `--set distributed.error_feedback=true` in decentralized mode to apply
`2 * current_update - previous_update` at each local optimizer step. An update is
the full signed AdamW parameter displacement, including weight decay, after local
gradient clipping. History starts at zero, so the first update is doubled. AdamW
moments advance once per step, and the fused CUDA optimizer remains available.
This adapts the update correction in
[Algorithm 2 of Zmushko et al.](https://arxiv.org/pdf/2606.30634#page=7), with the
zero-history, doubled-first-update initialization used here.

AWC applies the correction after installing mixed parameters; ATC applies it before
mixing. The saved update excludes mixing and stays local to each rank. The option
also works with one decentralized worker and is rejected in DDP mode. It defaults
to `false`.

Error feedback uses two additional contiguous buffers per worker, each matching
the padded parameter allocation's dtype and length: history and scratch. Bulk
copy/subtraction computes the update, and the buffers swap roles after each step
without another history copy. They are separate from communication snapshots and
are reused regardless of `distributed.reuse_mix_buffers`. For FP32 parameters the
additional memory is approximately 8 bytes per parameter, plus alignment padding.
Training checkpoints save each rank's previous update alongside AdamW state;
scratch is not saved. Resume requires matching error-feedback configuration and
valid history. Training benchmarks report `error_feedback.history_bytes` and
`error_feedback.scratch_bytes` in addition to overall CUDA peak memory.

To isolate the local optimizer overhead, run
`uv run python scripts/benchmark_error_feedback.py -c configs/base.yaml -c configs/20m.yaml --output runs/feedback-benchmark.json`.
It compares ordinary AdamW and error feedback with identical fixed synthetic gradients
and reports per-step time and buffer bytes. This excludes forward/backward and
communication; use the training benchmark for end-to-end throughput.

Models retain tied embeddings, RoPE, RMSNorm, SwiGLU, depth-scaled initialization,
reference/SDPA attention, BF16 autocast, optional compilation, and fused CUDA AdamW.
Weight decay applies to matrices, not normalization vectors. The token budget uses one
model's parameter count; cumulative virtual-epoch boundaries round to the nearest global
batch. Cosine and WSD schedules retain upstream formulas. Virtual epochs do not reset
sampling. Each epoch evaluates one fixed validation subset; completion evaluates all
validation targets, including a partial final block. Full validation reads sequentially
and rank partitions do not duplicate samples. Decentralized evaluation uses separate
FP32-averaged weights without changing training replicas or optimizer state.

Both cosine and WSD use `lr_schedule.warmup_tokens_per_parameters`, defaulting to **1.0**
tokens per parameter (TPP). Multiply TPP by `model.parameter_count`, divide by the global
`training.batch_tokens`, and round to the nearest
optimizer update, with half-update ties rounded up. Positive budgets always receive at
least one update; **0** disables warmup. Values must be finite and nonnegative. Increasing
the global batch size reduces the warmup update count, while increasing the model size
increases it. Microbatch size, world size, and data reuse do not independently scale warmup.
The learning rate rises linearly to the base rate at the rounded warmup boundary, then
follows the selected decay schedule. Warmup is not capped to the training budget, so a
short run can finish during warmup. Benchmark timing `--warmup` remains measured in updates.

`lr_schedule.warmup_steps` has been removed. Migrate YAML files and CLI overrides to the
TPP field and start new runs; checkpoints with the previous schedule recipe cannot resume
under this configuration. To preserve an old update count for a particular model and batch,
use `old_warmup_steps * training.batch_tokens / model.parameter_count` as the TPP value.

Set `training.data_reuse_factor=K` to reuse a smaller fixed pool while retaining the
same global token budget, optimizer steps, learning-rate schedule, and virtual epochs:

```bash
uv run decentralized-llm train -c configs/base.yaml -c configs/20m.yaml \
  --set training.data_reuse_factor=4 --set runtime.output_dir=runs/reuse-4
```

`K` must be a positive integer and defaults to **1**, which preserves the existing sampler
and checkpoint compatibility, including cycling an already-small cache. For `K > 1`,
let `B` be the full training budget after the existing global-batch rounding, measured in
context-length blocks. The sampler selects `U = ceil(B / K)` unique blocks and reshuffles
that same pool on each pass. Every block is used once per complete pass before the next
pass starts in the global stream. Training stops after exactly `B` blocks: every selected
block is used exactly `K` times when `B` divides evenly by `K`; otherwise the last pass
may be partial. For example, `B=42, K=5` uses 9 unique blocks, with four complete passes
and six blocks from a fifth pass. If `K > B`, the pool contains one block used `B` times.

Pool selection takes the first `U` entries of the normal sampler's first-pass permutation.
It is reproducible for a given cache and training seed; larger factors select nested pools.
The unique training-token count is `U * context_length`, counting token positions rather
than distinct vocabulary IDs and excluding the extra lookahead token. Preflight and
`run.json` report the factor, pool size, unique training tokens, and full/partial pass counts
under `sampling`. Resume requires the same reuse settings and regenerates the selected pool
and pass order from the committed global cursor. Changing reader threads or prefetch depth
does not change sampling.

Preparation is unchanged: prepare the normal cache once to compare different K values.
For `K > 1`, a cache with fewer than `U` complete blocks is rejected before loading.
Sampling passes span ranks and may cross microbatch, optimizer-update, and virtual-epoch
boundaries; they introduce no extra distributed synchronization.

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
version. Output paths, loader mode, reader threads, prefetch depth, host budget, and homogeneous physical
node placement may change. Only **optimizer-committed global sample positions** are saved;
prefetched or transferred unused samples are discarded and deterministically regenerated.
Sampling is exact across resume; floating-point collective reduction order can produce
small numerical differences after process recreation. A completed budget should be evaluated,
not resumed. Checkpoints use PyTorch pickle for RNG/optimizer state; load trusted files only.

## Network loader and benchmarks

Every complete context-length block with one lookahead token is eligible. With
`data_reuse_factor=1`, each dataset pass uses a seeded NumPy PCG64 permutation of **all**
eligible blocks without replacement. With a larger factor, passes permute the selected
fixed pool. Pass seeds derive from cache identity, context length, training seed, sampler
version, and (for reuse) the factor and full training budget.
Rank `r` receives global stream positions congruent to `r` modulo world size, continuously
across passes and virtual epochs.

`data.loader_mode=auto` keeps the run's unique training blocks in RAM on **each rank** when the
estimated loader footprint fits `data.host_memory_mib` (default **2048 MiB per rank**).
Otherwise it uses continuous streaming. `resident` and `streaming` force either mode;
preflight rejects an estimated over-budget configuration before launch. The two modes
produce identical samples and support resuming each other's checkpoints.

For K=1, a run shorter than one cache pass loads only the prefix of the existing shuffled
stream that its full token budget will consume. It does not need the entire on-disk cache
to fit. Longer runs retain all eligible blocks; K>1 retains the entire selected reuse pool.
The sampler identity, sample order, token budget, and resume behavior stay the same.

Resident mode starts training as soon as the initial batch queue is ready. Upcoming
batches get read priority while background reads fill the remaining working set in
storage order, into compact uint16 rows including each block's lookahead token.
With multiple reader threads, background work leaves one thread available for incoming
demand reads. Rows already being read are shared with demand requests, so no row is
loaded twice. Background reads coalesce nearby blocks using
`read_range_bytes`, `coalesce_gap_bytes`, and `preload_max_read_amplification` (default 16).
This permits reading intervening bytes to reduce small network requests; only selected
blocks are retained. Once the cache is complete, the reader threads and file handles
close; each reshuffle then gathers from RAM, with **no further training-data filesystem
reads**. Before completion, training may still wait for a requested row. Every rank
holds the whole working set, so reshuffling can assign it blocks
that another rank consumed in the previous pass. At a 0.4B-token budget with K=8, the
token buffer is about 100 MB per rank (about 800 MB for K=1), plus indices and loader buffers.
Validation and checkpoint I/O remain separate.

Streaming uses persistent reader threads, bounded descriptor caches, and `pread`, splitting
blocks across shards as needed. Reads within each microbatch are sorted by shard/offset
and coalesced subject to `read_range_bytes`, `coalesce_gap_bytes`, and
`max_read_amplification` (default 2). Each batch becomes ready as soon as its own reads
finish; consumption immediately opens another prefetch slot. Delivery remains in sampler
order even when reads finish out of order. Defaults are **4 reader threads and 8 prefetched
microbatches**.

The permutation stays in host memory. Reuse additionally retains the selected pool;
selection temporarily builds the full-cache permutation. The memory estimate includes
this peak, resident tokens, indices and row readiness when enabled, Python request metadata, compact
windows, read scratch, validation and transfer buffers. Model/optimizer memory, interpreter
overhead, and the OS page cache are separate from this loader budget.

Only upcoming batches convert to int64. Double-buffered pinned storage, nonblocking H2D
copies, a dedicated CUDA stream, and completion events protect reuse. The initial read
queue fills before the training timer starts; filling the rest of RAM overlaps training.
Loader metrics distinguish `startup_seconds` from `preload_seconds` (time until complete
residency), report `resident_loaded_blocks` / `resident_total_blocks` and `resident_complete`,
and count I/O after queue warmup in `training_read_bytes` / `training_read_calls`.
Pending work is canceled on shutdown;
a running POSIX read must return before its descriptor is closed. Python cannot interrupt
a kernel-blocked network read. Finite prefetch cannot guarantee zero stalls under arbitrary
storage slowdowns.

Train diagnostics report the resolved loader mode, startup/preload time, resident buffer
bytes, and preload/training read bytes and calls (these overlap during background cache
fill). They also retain total
requested/read bytes, amplification, read calls, queue occupancy,
starvation count, p50/p95/p99/max batch wait, estimated buffer bytes, actual sampler bytes,
and process peak RSS. Starvation counts ready-queue-empty requests; wait time measures
CPU blocking, not direct GPU idle time. Read bytes count bytes returned by positional reads,
not storage-server traffic or filesystem readahead. Wait quantiles cover the most recent
8,192 delivered batches. Lifetime total/max CPU wait and the eight largest waits survive
window expiration. Each retained event records its global sample positions, queue occupancy,
time until the producer made the batch ready, time after readiness, and the batch's slowest
read group (shard, byte offset, sample, descriptor-lookup/open time, and read-loop time).
The lifetime slowest read group is also retained. These wall-clock timings can include OS
scheduling; they are not pure disk latency or a measure of GPU idle time.
`metrics-rank-*.jsonl`, `loader-rank-*.json`, `rank-*.json`, failure tracebacks, resolved
configuration, source digest, library versions, and evaluation metrics remain in the run.
Validation entries in `metrics.jsonl` include a UTC `timestamp` with second precision
(for example, `2026-10-17T14:30:00+00:00`), recorded when the result is written.
They report validation duration as `val_seconds` and wall time in the last virtual
epoch's training loop as `train_seconds`, both rounded to two decimal places.
Training time excludes validation and checkpoint saving. On a mid-epoch resume,
it covers only the training since resuming.

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
