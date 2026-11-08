# decentralized-llm

Reproducible Llama pretraining with DDP or decentralized AdamW/Muon + AdamW, using real distributed
workers or packed workers on one GPU, and one immutable uint16 token cache on shared storage. Python **3.14** and dependencies
are pinned in `uv.lock`.

The portable model, optimizer grouping, learning-rate schedules, virtual-epoch rounding,
and C4 preparation conventions come from
[tiny-llm dd03e5037b46961b1bbcbd014c445b6efebb4841](https://github.com/WangZesen/tiny-llm/tree/dd03e5037b46961b1bbcbd014c445b6efebb4841).
The distributed runtime, checkpoint format, global sampler, and network reader are new.
The optional GH200 execution path also adapts upstream fused normalization, SwiGLU,
RoPE, and cross-entropy kernels, preserving canonical parameters and distributed AdamW.
Packed execution adapts upstream batched models, parameter/moment arenas, and complete-update CUDA graphs.
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
Five dimension presets are in `configs/models/{20m,50m,90m,250m,500m}.yaml`.

| Directory | Contents |
| --- | --- |
| `configs/models/` | Model dimensions |
| `configs/setups/` | Local/SLURM launchers, GPU allocations, hardware execution presets, CPU benchmark |
| `configs/tunings/` | Phased tuning specifications |
| `configs/` | Shared base, training defaults, WSD, smoke, and example configs |
| `configs/local/` | Git-ignored execution defaults and local experiments |

The shared `training-512c-{32,64}bs-20tpp.yaml` overlays set context length 512,
global batch tokens 16,384 or 32,768, and 20 training tokens per model parameter.
Batch tokens are fixed: changing context length changes the global sequence count.

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

Other local experiments can remain in `configs/local/`. Specs copied there from
`configs/tunings/` keep the same relative references, such as `../base.yaml`,
`../models/20m.yaml`, and `../setups/slurm-ddp.yaml`. No reference to `defaults.yaml`
is needed. Specs in deeper directories must adjust their relative references.

Tracked `tuning-*.yaml` specifications reference shared base, model, training, and
setup configurations. They do not require an ignored config or a GH200/A100 preset,
and leave the execution backend to local defaults. The generic multi-node specifications
also leave scheduler network selection and transport activation unset. Put site-specific
`launcher.network` and `launcher.environment_overlays` in a local experiment or tuning
copy, after checking its interconnect setup; these fields do not belong in `defaults.yaml`.
The standard environment script supports both host architectures and does not require
local toolchain files. Smoke and CPU benchmark configs explicitly choose eager native
execution and no CPU binding, so they also work with fused local defaults.

```bash
uv run decentralized-llm prepare -c configs/base.yaml -c configs/models/20m.yaml \
  --set data.cache_dir=/shared/tokens/c4
uv run decentralized-llm validate -c configs/base.yaml -c configs/models/20m.yaml \
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
uv run decentralized-llm submit -c configs/base.yaml -c configs/models/90m.yaml \
  -c configs/setups/slurm-ddp.yaml --set distributed.nodes=1 --set distributed.gpus_per_node=4 \
  --set data.cache_dir=/shared/tokens/c4 --set launcher.time=02:00:00 \
  --set runtime.output_dir=runs/90m-4gpu --dry-run
```

Set the lowercase `account` environment variable and the site's GPU partition before
submission. Changing the local backend does not change GPU counts, batching, search
grids, or scheduler settings. Hardware and workload compatibility checks still apply.

## Simulating multiple workers on one GPU

Set `runtime.training_backend=packed` or layer `configs/setups/packed.yaml` after
an existing recipe. `distributed.nodes * distributed.gpus_per_node` remains the
**logical world size**, controlling data partitioning and decentralized topology.
Local and Slurm launches use **one process on one node with one GPU**. Packed runs
create no process group and require no NCCL communication.

```bash
# Eight logical decentralized workers on one GH200 GPU.
uv run decentralized-llm submit -c configs/base.yaml -c configs/models/20m.yaml \
  -c configs/setups/gh200-1gpu.yaml -c configs/setups/packed.yaml \
  --set distributed.nodes=2 --set distributed.gpus_per_node=4 \
  --set distributed.mode=decentralized --set distributed.topology=one_peer_exponential \
  --set data.cache_dir=/shared/tokens/c4 --set runtime.output_dir=runs/packed-8 \
  --set launcher.time=01:00:00 --dry-run
```

Inspect the dry run, then remove `--dry-run` to submit. The ordinary hardware presets
still choose the execution kernels; the packed overlay chooses training placement.
For a local GPU, use `train` with `launcher.kind=local`. Native eager/compiled execution
works through `runtime.execution_backend=native`; the GH200 and A100 presets enable
their fused paths on the corresponding hardware. Automatic selection still requires
matching qualification evidence and otherwise uses native execution.

`distributed.mode=decentralized` keeps separate parameters and AdamW moments for every
logical worker. Complete, reciprocal ring, and reciprocal exponential topologies,
AWC/ATC ordering, local gradient clipping, accumulation, and error feedback retain their
ordinary semantics. Batched operations process worker-specific samples in parallel.
`distributed.mode=ddp` simulates synchronous training with one shared model and optimizer,
processing the entire global batch. In both cases the sampler visits exactly the same
global positions as a real distributed run.

`training.batch_tokens` and the token budget remain global. The configured microbatch
size is per logical worker; packed staging holds `world_size * micro_batch_size` sequences.
Decentralized parameter and optimizer memory scale with the logical world size, so a
large simulated run must still fit on one GPU. Host loader estimates include the larger
staging buffers while sharing the resident token cache and sampler permutation.

With fused GH200/A100 execution and `runtime.fused_optimizer=true`, packed mode captures
the complete accumulated forward/backward, clipping, AdamW, and mixing update in a CUDA
graph. Warmup and capture restore parameters, moments, feedback history, counters, and
RNG before training begins. `runtime.decentralized_optimizer=reference` disables this
captured optimizer path. A nonfinite update prevents every worker from committing and
raises before the sampler cursor advances. `compile_scope=blocks` is unsupported for
packed execution; use the default `model` scope.

Packed checkpoints store all worker state in `packed.pt` and publish `manifest.json`
last. Resume requires the same training backend and logical world size; conversion
between packed and distributed checkpoints is unsupported. Validation evaluates the
averaged decentralized model without modifying local workers. Packed training metrics
are in `metrics-rank-00000.jsonl`, with `local_losses` and `local_gradient_norms` arrays
in logical-rank order (one entry for the shared synchronous model). Preflight, run metadata,
and benchmarks report both logical world size and physical resource counts.

## Comparing TF32 and FP32 training

Set `runtime.amp=false` and `runtime.execution_backend=native` before choosing
`runtime.fp32_precision=tf32` or `ieee`. Both keep FP32 tensors; TF32 permits
reduced input precision in CUDA matrix multiplication. `runtime.sdpa_backend=math`
selects the same math attention implementation for both groups. Optional
`evaluation.fp32_precision=ieee` evaluates both groups in full FP32 and restores
the training setting afterward. Unset precision fields retain existing behavior.
Explicit precision rejects AMP and conflicting TF32 environment overrides.

The paired campaign helper derives six runs from a tuning winner, using seeds
42–44, one GH200 per run, native eager execution, and common FP32 evaluation:

```bash
python scripts/precision_campaign.py prepare
python scripts/precision_campaign.py smoke-submit
# After the smoke job has produced smoke/result.json:
python scripts/precision_campaign.py submit
python scripts/precision_campaign.py report
```

The default destination is
`runs/precision/20m-awc-ef-8w-ring-512c-128bs-20tpp`. Override it with `--root`,
and select another winner using `prepare --winner PATH`. Preparation freezes
source files, dependencies' version requirements, and resolved configurations;
queued workers import the frozen source. Submission requires matching successful
GPU smoke evidence and records scheduler receipts without duplicating known jobs.
An uncertain submission must be reconciled with Slurm before retrying.
The report writes `report.json`, `results.csv`, and `validation-loss.png`;
three-seed aggregate statistics remain unavailable until all three seeds finish.

## Optional A100 and GH200 setup presets

After preparing the shared token cache, choose one hardware preset. Each is layered
**directly after the base configuration and model preset**. These explicit recipes
override local execution defaults. Generic training and tuning do not depend on them.
Use `configs/setups/a100-{1,2,4,8}gpu.yaml` or
`configs/setups/gh200-{1,2,4,8}gpu.yaml`:

| Total GPUs | A100 nodes × GPUs per node | GH200 nodes × GPUs per node |
| --- | --- | --- |
| 1 | 1 × 1 | 1 × 1 |
| 2 | 1 × 2 | 1 × 2 |
| 4 | 1 × 4 | 1 × 4 |
| 8 | 1 × 8 | 2 × 4 |

Hardware setups set execution and allocation settings; batching comes from the base
or training configs. The resource-only `configs/setups/1gpu.yaml` and
`configs/setups/8gpu-2nodes.yaml` select 1 × 1 and 2 × 4 without choosing an execution
backend or launcher. Single-node hardware presets clear cross-node network settings
and environment overlays when layered after another setup.

For the existing four- and eight-GPU GH200 recipes:

```bash
uv run decentralized-llm submit -c configs/base.yaml -c configs/models/90m.yaml \
  -c configs/setups/gh200-4gpu.yaml --set data.cache_dir=/shared/tokens/c4 \
  --set launcher.time=02:00:00 --set runtime.output_dir=runs/90m-4gpu --dry-run
uv run decentralized-llm submit -c configs/base.yaml -c configs/models/90m.yaml \
  -c configs/setups/gh200-8gpu.yaml --set data.cache_dir=/shared/tokens/c4 \
  --set launcher.time=02:00:00 --set runtime.output_dir=runs/90m-8gpu --dry-run
```

Set the lowercase `account` environment variable to your scheduler account, inspect
the dry run, then remove `--dry-run` to submit. Choose walltime for your model and token
budget; the presets retain a short **15-minute** limit unless overridden. These two
examples allocate four GPUs per node and launch one process per GPU. With `base.yaml`'s
32,768-token batch and 1,024-token context, per-rank microbatches are eight on one node
and four on two nodes.
DDP is the default; add `--set distributed.mode=decentralized` for complete-topology
AWC, or select another supported topology with `distributed.topology`.

The GH200 presets explicitly enable **compiled GH200 kernels and exclusive GPU-local CPU
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
model/compiler code, training backend, and logical world size. Seeds, optimizer hyperparameters, and output directories do not
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
recipe, optionally layer `configs/setups/a100-1gpu.yaml` after the base configuration and
model preset, then set your site's partition and walltime. This preset overrides
local execution defaults.
The [A100 performance report](docs/a100-performance.md) records measured throughput,
numerical checks, and reproduction commands. Automatic backend selection retains
its qualification requirements; an explicit A100 selection requires supported model
geometry, BF16 SDPA, compilation, and nondeterministic execution on an A100.

The 20M preset can also explicitly test these kernels on Ada GPUs (SM89), using
`configs/setups/ada-20m-experimental.yaml`. This is recorded as experimental in execution
metadata and does not change automatic backend selection. Use the local launcher and
retain the desired packed worker count; the A100 SLURM setup files also change placement.
The [Ada benchmark report](docs/ada-performance.md) compares eager native, compiled native,
and compiled A100 execution on the local RTX 5000 Ada.

## Local GPU smoke runs

For eight logical decentralized workers on one local GPU, use the bounded 20M example:

```bash
uv run decentralized-llm prepare -c configs/base.yaml -c configs/models/20m.yaml \
  -c configs/local-packed-8-example.yaml
CUDA_VISIBLE_DEVICES=0 uv run decentralized-llm train \
  -c configs/base.yaml -c configs/models/20m.yaml -c configs/local-packed-8-example.yaml --dry-run
CUDA_VISIBLE_DEVICES=0 uv run decentralized-llm train \
  -c configs/base.yaml -c configs/models/20m.yaml -c configs/local-packed-8-example.yaml
```

This uses native BF16 execution, packed workers, reciprocal ring mixing, and AWC AdamW,
with no SLURM allocation. The 100 updates process 409,600 global tokens at context length
128, with four sequences per worker per update. Preparation streams a small real C4
sample into `data/c4-local-packed-example`; its reduced document shuffle buffer keeps
startup bounded. Validation is intentionally truncated, so this is an execution example,
not a model-quality benchmark. Logs, per-worker metrics, and the final packed checkpoint
are written under `runs/local-packed-8-example`. To repeat training, add
`--set runtime.output_dir=runs/local-packed-8-example-2` to use a fresh output directory.

For a full 20-TPP run, replace `configs/local-packed-8-example.yaml` in those commands
with `configs/local-packed-8-20tpp.yaml`. This keeps context length 128 and uses a global
batch of 128 sequences (16,384 tokens), for 24,907 updates and 408,076,288 training tokens
after whole-batch rounding. It prepares a fresh cache at `data/c4-20m-20tpp` with complete
validation, restores the one-TPP warmup, and saves resumable checkpoints every ten virtual
epochs plus the final checkpoint under `runs/local-packed-8-20tpp-bs128`.

Prepare the small C4 cache once:

```bash
uv run decentralized-llm prepare -c configs/base.yaml -c configs/smoke.yaml
uv run decentralized-llm train -c configs/base.yaml -c configs/smoke.yaml -c configs/setups/local-ddp.yaml
uv run decentralized-llm train -c configs/base.yaml -c configs/smoke.yaml -c configs/setups/local-decentralized.yaml
```

On an allocated GPU, run the hardware tests with `uv run pytest -q -m cuda` to exercise
pinned-buffer reuse, BF16/SDPA parity, fused AdamW, and CUDA compilation.

The distributed smoke example in `local-decentralized.yaml` requests two GPUs. Set `distributed.gpus_per_node=1`
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
uv run decentralized-llm submit -c configs/base.yaml -c configs/smoke.yaml -c configs/setups/slurm-ddp.yaml \
  --set launcher.partition=YOUR_GPU_PARTITION --dry-run
uv run decentralized-llm submit -c configs/base.yaml -c configs/smoke.yaml -c configs/setups/slurm-ddp.yaml \
  --set launcher.partition=YOUR_GPU_PARTITION
uv run decentralized-llm submit -c configs/base.yaml -c configs/smoke.yaml -c configs/setups/slurm-decentralized.yaml \
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

### Bounded 50M campaign

The eight `configs/tunings/50m-*-1024c-64bs-20tpp.yaml` specifications compare
synchronous training and eight-worker AWC on ring, exponential, and complete graphs,
each with feedback gamma 0 or 1. All use packed execution on one physical GH200,
context 1024, 65,536 global batch tokens, and 20 TPP (970,129,408 rounded training
tokens). Their shared overlay pins the execution recipe independently of local defaults.

Each core study runs five learning rates at transferred 20M beta values, keeps two
rates for a joint 3-by-3 beta search, and confirms the best three combinations on
seeds 42, 43, and 44. Exact reuse gives 27 training executions: 5 + 16 + 6.
Every specification records its source winner report and SHA256. The controller
allows 30 outstanding jobs per study, one trial per allocation, a 45-minute walltime,
and one infrastructure retry. Numerical divergence is eliminated without retrying.

Use `scripts/tune_50m.py` to run the core study followed by one bounded boundary
check. For each coordinate where the confirmed winner is on the edge of its original
grid, the wrapper evaluates one outward neighbor while fixing the other coordinates.
Each neighbor uses all three seeds; there are at most nine additional executions.
The boundary plan is fixed from the core winner and is not expanded recursively.
Rerunning the same command resumes existing work. Separate wrapper instances may
run the eight studies concurrently.

```bash
python scripts/tune_50m.py run \
  --spec configs/tunings/50m-awc-8w-ring-1024c-64bs-20tpp.yaml \
  --output runs/p-tuning/50m-campaign/50m-awc-8w-ring-1024c-64bs-20tpp
python scripts/tune_50m.py report --root runs/p-tuning/50m-campaign --watch
```

The report command reads a `campaign.json` manifest containing a `studies` list,
with an absolute `directory` for each study. It refreshes every minute, produces
`report.json` and `report.md`, and exports each completed study's combined winner
as `final-winner.yaml`. The controller's original `winner.yaml` remains the core
search winner. Reports include three-seed statistics, consensus error, winner curves,
observed runtime projections, and Slurm allocation GPU-hours including retries.
Missing accounting is explicitly marked incomplete. Final winners at the expanded
search boundary remain flagged. These are tuning results, not independent test scores.

For a campaign spanning queued jobs, use a frozen source/configuration snapshot,
absolute shared cache and compilation-cache paths, and an environment script that
checks the pinned dependency versions without updating them. Run all dry runs from
that snapshot before submission. The wrapper also dry-runs each new boundary study.
The core budget is 216 executions across eight studies, or at most 288 with boundary
checks, excluding infrastructure retries. The existing ring timing probe estimates
about 91 or 121 GPU-hours respectively; actual costs vary across conditions.

### Joint 50M extension

The `extend` command imports authenticated results from the completed eight-study
campaign, its boundary checks, and the three-seed synchronous follow-up. It keeps
the original artifacts and training snapshot intact, freezes the new orchestration,
and saves separate specifications and controller states. Historical reuse requires
an exact configuration, seed, and provenance match, including full validation.

```bash
source runs/p-tuning/50m-1024c-64bs-20tpp/environment.sh
"$DECENTRALIZED_PYTHON" scripts/tune_50m.py extend \
  --source-root runs/p-tuning/50m-1024c-64bs-20tpp \
  --output runs/p-tuning/50m-1024c-64bs-20tpp-extension-20261022 \
  --dry-run
```

The dry run prepares and validates all eight saved studies without submitting jobs.
Repeat without `--dry-run` to start resumable controllers and a report watcher in
tmux. Repeating the command reuses existing states and does not duplicate active
controllers. Run it from the repository with the original frozen environment above.

Each study screens a full joint 3-by-3-by-3 LR/beta grid on seed 42, then confirms
the best three candidates on seeds 43 and 44. The grids include each topology's
EF/no-EF winners and reopen learning rates at the improved betas. Historical reuse
limits this stage to 196 new runs across the eight studies. Each study's
`budget.json` caps the extension at 30 new configuration/seed executions, including
one boundary expansion round, with infrastructure retries counted separately.
Boundary probes reserve all missing seeds before admission, in LR, beta1, beta2
order, and skipped probes are reported. The 30-job concurrency limit and 45-minute
single-GPU allocations remain in force.

The extension report compares all completed three-seed candidates with the best
historical result, including the improved synchronous baseline. It includes
per-seed loss changes, sample standard deviations, consensus error, winner curves,
and incremental GPU-hours; historical allocations are excluded. Boundary and retry
spending are reported separately (retry spending overlaps its search phase).
Final configurations are exported when each study completes. The measured original
allocation times project approximately 75 GPU-hours for the joint search and
confirmation, or 92 if all eight studies use their full 30-run budget, before retries.

### Controller usage

`decentralized-llm tune` runs a persistent controller over local training processes or
SLURM jobs. Prepare the shared cache first. Run the controller from this repository in
an enduring shell such as tmux, or in your own CPU allocation. SLURM uses the same
lowercase `account` environment variable as ordinary submissions; it stays out of
configuration, receipts, and reports.

Start with [`configs/tunings/tuning-example.yaml`](configs/tunings/tuning-example.yaml), updating the
referenced training configuration and SLURM partition for your environment:

```bash
uv run decentralized-llm tune run --spec configs/tunings/tuning-example.yaml \
  --output runs/tuning/example --dry-run
uv run decentralized-llm tune run --spec configs/tunings/tuning-example.yaml \
  --output runs/tuning/example
uv run decentralized-llm tune status runs/tuning/example
uv run decentralized-llm tune resume runs/tuning/example
uv run decentralized-llm tune extend runs/tuning/example --spec configs/tunings/tuning-example.yaml --dry-run
uv run decentralized-llm tune extend runs/tuning/example --spec configs/tunings/tuning-example.yaml
uv run decentralized-llm tune report runs/tuning/example
```

The tuning specification has a `base.configs` list and optional dotted `base.overrides`
mapping. Both `tune run` and `tune extend` also accept repeatable `--config/-c FILE`
and `--set dotted.key=value`. Precedence, from lowest to highest, is:

1. Built-in defaults and optional local execution defaults.
2. The spec's `base.configs`, in order.
3. CLI config files, in order.
4. The spec's `base.overrides`.
5. CLI `--set` values, in order.

Spec filenames resolve relative to the tuning specification; CLI filenames resolve
relative to the working directory. Paths inside training configs retain normal
working-directory semantics. Phase search and initial values apply after this base
configuration. The Python `read_spec` and controller `execute` APIs accept equivalent
`configs=` and `overrides=` keyword arguments.

Reuse a spec for a different model, hardware setup, batch, and context:

```bash
uv run decentralized-llm tune run \
  --spec configs/tunings/20m-sync-512c-64bs-20tpp.yaml \
  -c configs/models/90m.yaml \
  -c configs/setups/a100-8gpu.yaml \
  -c configs/training-512c-32bs-20tpp.yaml \
  --set model.context_length=1024 \
  --output runs/tuning/90m-a100-8gpu --dry-run
```

Each command configures one study; use separate output directories for different
combinations. Shared recipes keep allocation, batch, context, and training-budget
defaults in referenced config files, so CLI config files can replace them. Explicit
`base.overrides` still win over those files; use `--set` to replace a pinned value.
The example above uses 16,384 batch tokens: 16 sequences at context length 1,024.

Resolved configurations, including CLI overrides, are frozen in study state.
`resume`, `status`, and `report` use saved settings and accept no base overrides.
`extend` resolves its supplied spec and CLI arguments afresh; repeat any CLI inputs
that should carry into the new revision. The same generic specification can also use
GH200 or A100 by editing only `configs/local/defaults.yaml` when it does not explicitly
select a backend. `--no-local-config` is available on `tune run` and `extend`.

The shared SLURM presets use the example partition `gpu`. Find your site's partition
with `sinfo --noheader --format=%P`, then set `launcher.partition` under the tuning
specification's `base.overrides`, pass `--set launcher.partition=NAME`, or add a site
launcher config through `base.configs` or `-c`.
The automatically loaded `configs/local/defaults.yaml` accepts runtime settings only;
it does not select a scheduler partition. Tuning dry runs query the configured
partition's walltime limit, so they also require a valid partition and access to the
scheduler.

For a four-GPU SLURM study using local execution defaults, copy the example and replace its `base`
block with:

```yaml
base:
  configs: [../base.yaml, ../models/90m.yaml, ../setups/slurm-ddp.yaml]
  overrides:
    data.cache_dir: /shared/tokens/c4
    training.checkpoint_policy: none
    distributed.nodes: 1
    distributed.gpus_per_node: 4
    launcher.time: '02:00:00'
```

This example assumes the copied specification is in `configs/tunings/` or
`configs/local/`. Set `distributed.nodes: 2` for eight GH200 GPUs; for eight A100 GPUs,
keep one node and set `distributed.gpus_per_node: 8`. Configure any site-specific
transport in that local copy.
Keep the example's phases and seed settings, set the cache path and intended walltime,
and run `tune run --dry-run` before starting the study. The local execution defaults
are resolved once for the study and apply to every trial, with compatible compilation
artifacts reused. Set affinity and compilation-cache preferences in `defaults.yaml`.

Each phase's `search` maps **one or more dotted configuration fields** to search
specifications. The controller jointly evaluates their Cartesian product, averages
final full-validation loss over the required training seeds, and keeps up to the **global
top K finite-loss combinations** specified by `top_k`. A later phase expands each
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
For local CUDA sweeps, `max_jobs` caps concurrent trials, also limited by available GPUs
in the study's pool. Local CPU trials remain sequential. Local execution ignores
`jobs_per_allocation`. Scheduler queries are batched.

To dispatch local trials across GPUs, set `launcher.kind: local` in the base training
configuration and add the following to the tuning specification:

```yaml
controller:
  max_jobs: 2
  local_gpus: [0, 1]
  poll_seconds: 5
```

`local_gpus` selects indices from the controller's CUDA-visible devices; omitting it
uses all visible GPUs. For example, with `CUDA_VISIBLE_DEVICES=2,3`, `[0, 1]` selects
physical GPUs 2 and 3. Each packed trial reserves one GPU regardless of its logical
worker count. A distributed trial reserves `distributed.gpus_per_node` GPUs and waits
until that many are free. Each child receives only its assigned GPU UUIDs through
`CUDA_VISIBLE_DEVICES`, so concurrent trials within the study do not share GPUs.

The controller records its host, GPU pool, and each trial's assignment before launch.
`resume` keeps the saved UUIDs even if CUDA device ordering changes, and requires the
same host with all pool devices still visible. Active or unresolved trials retain
their reservations across restarts; an older active trial without an assignment blocks
new local launches until it finishes. `extend` can select a different pool while still
accounting for active trials. Dry runs report `local_gpu_pool` and `local_max_jobs`;
`tune status` also lists `local_gpu_assignments`.

GPU reservations apply within one study. Select disjoint pools for separate studies
or unrelated training jobs; the controller does not detect external GPU usage. For
example, while another run uses physical GPU 0, start a sweep with
`CUDA_VISIBLE_DEVICES=1` and omit `local_gpus` (or set it to `[0]`).

The bounded [local packed example](configs/tunings/local-packed-example.yaml) runs
three learning-rate candidates using the small C4 cache prepared above:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run decentralized-llm tune run \
  --spec configs/tunings/local-packed-example.yaml --output runs/tuning/local-packed --dry-run
# Remove --dry-run to start the sweep on the selected GPUs.
```

To amortize queue waits over several short trials, for example:

```yaml
base:
  configs: [../base.yaml, ../models/20m.yaml, ../setups/gh200-8gpu.yaml]
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
and writes a partial report with a reason identifying the budget. Already submitted
jobs can finish and their results are collected on resume.

Nonfinite training losses, gradient norms, and rejected fused updates produce a
terminal `diverged` trial with objective loss +∞. These trials are not retried and
do not halt tuning. The worker still exits unsuccessfully and preserves its traceback,
rank, and last committed progress; the controller requires a structured numerical
failure record with matching configuration and provenance. Other errors, including
OOM, cancellation, and timeouts, retain their failure and retry behavior.

Any diverged seed makes the candidate's mean objective loss +∞, with no sample
standard deviation. Only candidates with finite means advance, up to `top_k`; later
phases can therefore have fewer candidates. If none remain, the study finishes
`complete` with a reason, `winner: null`, and no `winner.yaml`. JSON artifacts encode
infinite losses and means as the string `"inf"` and their standard deviations as
`null`; reports display ∞. Existing failed attempts are not reclassified.

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

`extend` creates a new revision and replays phases using matching successful and diverged results.
It supports revised grids, changed phase sequences, initial values, and seed counts.
Increasing `max_retries` can reopen exhausted trials. Previous reports and state are
archived under `revisions/`. Active jobs remain tracked even if a revised search no
longer needs them. Reuse requires matching effective configuration, seed, cache content,
and source/dependency fingerprints. Changed software or cache identity requires an
extension and produces new trials. Every accepted result is checked against its
configuration and recorded provenance. Successful results must also complete the
training budget and full validation; divergence records preserve the incomplete budget.

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
never relocated. State versions 1 and 2 are read compatibly and saved as version 3.
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
is retained, selected cells are outlined, and pending, failed, diverged (∞), and unsearched
cells are labelled separately. Finite means appear after all required seeds succeed;
divergence is labelled as soon as any seed diverges and is excluded from the color scale.
PNG figures are embedded in `report.md`, SVG copies are linked for zooming/export, and
`report.json` lists the relative paths in each phase's `heatmaps` array. Figures live in
`plots/` and are copied with archived revision reports. Regenerating an older study's
report adds plots without moving trials or modifying training logs.

Across reports, present training methods in communication order: **synchronous →
complete → exponential → ring** (`sync`, `cmp`, `exp`, `ring`). Apply the same order
to plot panels, legends, categorical axes, correlation heatmaps, and interactive
selectors; method colors stay fixed. Shared definitions are in `scripts/report_style.py`.

Regenerate the joint-tuning, warmup, sensitivity, and batch/TPP comparison reports with:

```bash
.venv-x86_64/bin/python scripts/render_tuning_reports.py
# Use a fresh suffix when repeating the render:
.venv-x86_64/bin/python scripts/render_tuning_reports.py --suffix communication-order-repeat
```

This writes presentation revisions, verifies plot ordering and unchanged numerical
tables, and updates report indexes. The audited source reports remain available.
The combined index is `runs/analysis/reports-communication-order.json`.

## Muon + AdamW

AdamW remains the default optimizer. Add `configs/muon.yaml` after your base/model/setup
configs to select `optimizer.name=muon_adamw`:

```bash
uv run decentralized-llm train --config configs/base.yaml --config configs/muon.yaml \
  --set optimizer.lr=0.0003 --set optimizer.muon_lr_ratio=2.0
```

`optimizer.lr` is the AdamW base learning rate (default `0.001`). Muon's base learning
rate is `optimizer.lr * optimizer.muon_lr_ratio`, with the ratio defaulting to `1.0`.
For example, the command above uses AdamW at `0.0003` and Muon at `0.0006`. Both follow
the same token-based warmup and cosine/WSD decay, preserving the configured ratio.
Hybrid training metrics still report the scheduled `muon_lr` and `adamw_lr`; `lr`
remains an alias for the AdamW rate. These are base rates before Muon's matrix-shape
adjustment. Tune the base rate and ratio independently, for example:

```yaml
search:
  optimizer.lr: [0.0001, 0.0003, 0.001]
  optimizer.muon_lr_ratio: [0.5, 1.0, 2.0]
```

The ratio must be positive and finite. Replace any earlier `optimizer.muon_lr` setting
with `optimizer.muon_lr_ratio = old_muon_lr / optimizer.lr`; the absolute configuration
field is no longer accepted.

Muon updates each block's Q/K/V/output attention projections and gate/up/down MLP
matrices. AdamW updates the shared embedding/LM-head weight once, all RMSNorm scales,
and any biases. The shared `optimizer.weight_decay` applies to matrix weights, including
the embedding, while scales and biases have zero decay. `optimizer.beta1`, `beta2`, and
`eps` configure AdamW; `optimizer.muon_momentum` defaults to `0.95`.

The implementation uses [PyTorch Muon](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)
with `match_rms_adamw` scaling, Nesterov momentum, five Newton–Schulz iterations, and
epsilon `1e-7`. Its orthogonalization runs in BF16, including when model autocast is off.
The scaling makes AdamW-scale learning rates a useful starting point; optimal rates
still depend on the workload.

Both distributed and packed training support the hybrid optimizer, with independent
worker state, AWC/ATC mixing, gradient accumulation, and checkpoint resume. Error feedback
corrects the combined Muon/AdamW displacement, including weight decay, once per update.
Packed Muon operates on each worker's individual matrices, never across the worker axis.

Compiled and fused model execution remains available. Hybrid runs bypass the custom
AdamW-only decentralized Triton updates and packed full-update CUDA graph capture;
`runtime.fused_optimizer` still controls PyTorch's fused auxiliary AdamW where supported.
Reports identify the selected Muon implementation. Existing AdamW checkpoints retain
their recipe identities; changing optimizer kind, base learning rate, or active Muon ratio requires
a new training run.

`runtime.muon_optimizer` defaults to `auto`: CUDA uses `cudagraph`, and CPU uses
the reference PyTorch implementation. Explicit choices are `reference`,
`batched` (group equal oriented matrix shapes), `compiled` (compile those batched
updates), and `cudagraph` (also capture the Muon update).
Batched modes require FP32 or FP64 parameters; compiled and
captured modes also require CUDA. The auxiliary AdamW
still uses its existing fused setting. Each matrix retains independent momentum
and normalization; all modes use the same five Newton–Schulz iterations and
RMS learning-rate adjustment. Compiled updates preserve explicit precision casts.
Capture stages fresh gradients and the scheduled learning rate on every step,
restores warmup state, and is rebuilt after checkpoint loading or a change in
which parameters have gradients. This captures the Muon optimizer only.

The canonical optimizer checkpoint format is shared across modes. Changing the
mode may change floating-point rounding; tuning identities distinguish active
optimized modes, while inactive AdamW settings and explicit reference mode
retain their previous identities. Automatic and explicit selections of the same
implementation share tuning and qualification identities. Benchmark the modes on
the target hardware:

```bash
python scripts/benchmark_muon.py optimizer -c configs/base.yaml -c configs/models/20m.yaml \
  -c configs/muon.yaml --output runs/benchmarks/muon-optimizer
python scripts/benchmark_muon.py training -c configs/base.yaml -c configs/models/20m.yaml \
  -c configs/setups/gh200-1gpu.yaml -c configs/training-512c-64bs-20tpp.yaml \
  -c configs/muon.yaml --output runs/benchmarks/muon-training
```

Run these inside a single-GPU allocation with the repository environment active.
The optimizer benchmark checks scheduled updates and restored state against
PyTorch Muon, exports CUDA profiles, and compares all four hybrid modes with
fused AdamW. Both benchmarks rotate execution order over three trials and exclude
warmup/compilation from timing. Training measurements use the existing resident
loader and reject compiler activity during timed windows.

GH200 measurements on 2026-10-23 (20M, context 512, one worker, momentum 0.95,
ratio 1.0; medians of three trials) were:

| Implementation | Optimizer step (ms) | Training, batch 64 (ms) | Training, batch 128 (ms) |
|---|---:|---:|---:|
| Fused AdamW | 0.38 | 18.99 | 34.79 |
| Reference Muon-AdamW | 36.91 | 53.42 | 64.23 |
| Batched Muon-AdamW | 3.77 | 20.65 | 37.54 |
| Compiled Muon-AdamW | 2.82 | 20.00 | 36.89 |
| CUDA-graph Muon-AdamW | 1.15 | 19.52 | 35.32 |

CUDA training uses the fastest measured mode by default. Use
`--set runtime.muon_optimizer=reference` to select the previous implementation.
It improved steady-state training throughput by **2.74×** at batch 64 and
**1.82×** at batch 128. These measurements exclude compilation, evaluation and
checkpoint I/O; AdamW was benchmarked in the same jobs on the same revision.

The compiled modes are not bitwise equivalent to reference Muon. With identical
gradients, the maximum relative L2 update difference was 0.52%, with identical
momentum buffers. Two matched 1,000-step batch-64 runs had validation-subset loss
increases of 0.00129 and 0.00527 versus reference. Parameters diverged during
training, so these short checks do not establish equal full-run convergence.
Keep the execution mode fixed within a tuning study. The completed 2026-10-23
tuning campaign used reference Muon and retains its frozen source and results;
reproducing that implementation with current code requires the explicit
`reference` override. See the [benchmark summary](docs/performance/muon-summary.json)
for per-trial timings, numerical checks, source hashes, job IDs and artifact paths.

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

Real distributed training uses one contiguous parameter allocation with 256-byte-aligned
parameter starts. Existing parameter objects, tied weights, optimizer groups, and
checkpoint tensor names/shapes are preserved. Packed workers use aligned rows in a shared arena. Storage is prepared after device/dtype
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

All new runs construct one canonical CPU FP32 initialization from `runtime.seed`.
Every decentralized worker starts with an exact copy, and synchronous training with the
same model and seed starts with exactly the same parameter values. This holds across
packed/distributed placement, world sizes, and native/fused/compiled execution. Rank
numbers never offset the model seed. Workers retain independent parameter storage and
AdamW moments. The initialization contract is recorded as `shared_seed_v1` in run and
checkpoint metadata and participates in tuning identities. Existing checkpoints restore
their saved parameters without reinitialization and retain a legacy initialization label.

In real decentralized training, each worker normalizes its local loss, clips local gradients,
and communicates parameters once per optimizer update. AWC starts an asynchronous exchange of a parameter snapshot before
forward, overlapping communication with forward and all accumulated backwards. It waits
after local gradient clipping and uses the mixed weights as the AdamW base; the live
weights used by autograd stay unchanged until backward finishes. The fused optimizer reads
the reduced snapshot directly, normalizing and updating in one pass. The reference path
installs mixed weights before stepping. Each update joins its exchange before checkpointing.
ATC updates before mixing; the fused path pre-normalizes its output and sums live weights
in place, avoiding a communication snapshot and a separate normalization/copy.
`complete` uses all-reduce over the world group. `one_peer_ring` alternates reciprocal
neighbor pairings and requires an even world size; `one_peer_exponential` flips one rank
bit per update and requires a power of two. Both peer topologies use asynchronous
all-reduce over two-rank process groups, followed by division by two. The groups are
created in a consistent order at startup and reused as peers rotate; at world size two,
the existing world group is reused. Every exchange is joined before switching groups.
All support a single-worker no-op. Evaluation
and checkpoint coordination add their own collectives outside optimizer mixing.

Set `--set distributed.error_feedback_gamma=0.5` to apply
`(1 + gamma) * current_update - gamma * previous_update` at each optimizer step.
The coefficient accepts any finite number, including negative values. It defaults
to `0`, which uses ordinary AdamW without feedback buffers; `1` reproduces the
original error feedback. An update is the full signed AdamW parameter displacement,
including weight decay, after gradient clipping. History stores the uncorrected
displacement and starts at zero, so the first update is scaled by `1 + gamma`.
AdamW moments advance once per step, and the fused CUDA optimizer remains available.
The `gamma=1` case adapts the update correction in
[Algorithm 2 of Zmushko et al.](https://arxiv.org/pdf/2606.30634#page=7), with the
zero-history, doubled-first-update initialization used here.

Both DDP and decentralized training support feedback, including single-worker and
packed execution. DDP applies correction after gradient averaging, accumulation,
and clipping; packed synchronous training corrects its shared model. AWC computes
the correction from mixed base weights; ATC applies it before mixing. The saved
update excludes mixing and stays local to each rank. Feedback gamma is independent
of the packed model's mixing coefficient and remains fixed for the run.
The former `distributed.error_feedback` flag is rejected, and boolean values are
also rejected for `distributed.error_feedback_gamma`.

`runtime.decentralized_optimizer: auto` selects grouped Triton updates for FP32 CUDA
decentralized training when `runtime.fused_optimizer=true` and deterministic mode is off.
Selection is independent of `runtime.execution_backend`. Set it to `reference` to use
PyTorch AdamW with the original correction and mixing passes. CPU, other dtypes, and
unsupported configurations retain the reference implementation. Small floating-point
differences are expected, including ATC normalization before reduction.

The fused path calculates the ordinary AdamW displacement and applies feedback in
registers, keeping one padded history buffer and no feedback scratch: approximately
4 bytes per FP32 parameter. The reference correction keeps history and scratch,
approximately 8 bytes per FP32 parameter, and swaps their roles after each update.
AWC additionally retains its isolated communication snapshot; fused ATC needs none.
History excludes mixing and includes weight decay in both paths. Missing gradients
leave moments and counters unchanged but still subtract `gamma * previous_update`
and clear previous history.

Checkpoints retain the existing rank-local `error_feedback.previous_update` tensor and
AdamW state. Numeric gamma participates in checkpoint, tuning, and qualification
identities; resuming with a different coefficient is rejected. Configurations and
checkpoint identities using the former boolean setting are not migrated.
The optimizer selector can change on resume; a resumed trajectory across implementations
need not be bitwise identical. Training benchmarks
report the resolved optimizer, source revision, history/scratch bytes, and communication
snapshot bytes alongside overall CUDA peak memory.

To isolate the local optimizer overhead, run
`uv run python scripts/benchmark_error_feedback.py -c configs/base.yaml -c configs/models/20m.yaml --output runs/feedback-benchmark.json`.
It compares reference and fused execution at gamma `0` and `1`,
using identical fixed synthetic gradients. It reports per-step time, speedups, and buffer
bytes; `--profile` also exports traces. This excludes forward/backward and communication.
Use `scripts/benchmark_decentralized_optimizer.py` under torchrun for three paired trials
of the 20M/90M, AWC/ATC, complete/ring, feedback on/off matrix. Use
`scripts/qualify_decentralized_optimizer.py` for the corresponding 100-update trajectories
with seeds 42 and 43, or `scripts/qualify_gh200.py --optimizer-only` for a single case.
Both comparisons keep the forward backend fixed. The matrix runners support `--resume`;
only complete timing pairs from the same allocation are reused. These scripts accept the
same layered configs and overrides. Numerical acceptance retains the existing first-five-update
state probe, adds history to that probe, and checks validation loss after 100 updates.
All-update diagnostics are also recorded, with exact step counters required throughout.
See [measurements and reproduction](docs/decentralized-optimizer-performance.md), including
the distinction between the accepted gates and later trajectory drift.

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
uv run decentralized-llm train -c configs/base.yaml -c configs/models/20m.yaml \
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

Resume requires the same training backend, world size, training recipe, cache, and sampler version/NumPy
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

Validation also reports `consensus_error` in `metrics.jsonl`, the final
`result.json` validation object, and checkpoint evaluation output:

$$
\mathrm{consensus\_error} = \frac{1}{W}\sum_{i=1}^{W}\|\theta_i-\bar{\theta}\|_2,
\qquad \bar{\theta} = \frac{1}{W}\sum_{i=1}^{W}\theta_i.
$$

This is the mean L2 distance across workers, with no parameter-count normalization.
Before validation, each local model is compared with the existing FP32 global average,
using FP64 accumulation over unique parameters; tied weights count once, and buffers
and storage padding are excluded. Evaluation leaves local training weights unchanged.
Single-worker runs and synchronized DDP report `0.0`. Multi-worker decentralized
evaluation with `average=False` reports `null`, since no global reference is computed.

The benchmark compares complete forward/backward/clip/AdamW/mixing pipelines after warmup,
using either the network loader or a host-resident batch with the same conversion/transfer
path and microbatch shapes. It reports a **95% baseline throughput target** and marks misses;
it does not infer acceptance from read bandwidth alone. Global benchmark batches must be
divisible by world size so the resident baseline has identical rank shapes. Distributed
elapsed time is the maximum over ranks; each rank writes separate loader diagnostics.

```bash
uv run decentralized-llm benchmark-data -c configs/base.yaml -c configs/models/20m.yaml \
  --set data.cache_dir=/shared/tokens/c4 --warmup 20 --steps 100 \
  --cache-state cold --output runs/benchmark/cold.json
uv run decentralized-llm benchmark-data -c configs/base.yaml -c configs/models/20m.yaml \
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
  --module decentralized_llm benchmark-data -c configs/base.yaml -c configs/models/20m.yaml \
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
      --module decentralized_llm benchmark-data -c configs/base.yaml -c configs/models/20m.yaml \
      -c configs/setups/slurm-ddp.yaml --steps 100 --warmup 20 --output runs/benchmark/slurm.json'
```

A reproducible synthetic CPU fixture is available without downloading C4:

```bash
uv run python scripts/make_synthetic_cache.py -c configs/setups/benchmark-cpu.yaml
CUDA_VISIBLE_DEVICES='' uv run decentralized-llm benchmark-data -c configs/setups/benchmark-cpu.yaml \
  --steps 100 --warmup 20 --cache-state cold --output runs/benchmark-cpu/cold.json
CUDA_VISIBLE_DEVICES='' uv run decentralized-llm benchmark-data -c configs/setups/benchmark-cpu.yaml \
  --steps 100 --warmup 20 --cache-state warm --output runs/benchmark-cpu/warm.json
```

The fixture is explicitly labeled synthetic and must not be used to claim language-model
quality. See [validation evidence](docs/validation.md) for what was actually tested and
which hardware checks still require an allocated GPU job.

## Steady-state eight-GPU campaign

The campaign helper benchmarks 216 configurations on real distributed GH200 workers:
20M/50M/90M models, contexts 512/1024/2048, global sequence batches 32/64/128, and
synchronous DDP or decentralized AWC with complete/ring/exponential topology, each
with EF gamma 0 or 1. Each global batch is split across eight GPUs without gradient
accumulation. The compiled GH200 BF16 recipe uses the site's CXI transport setup.

After activating the architecture-specific environment, run:

```bash
python scripts/benchmark_training_campaign.py prepare --dry-run
python scripts/benchmark_training_campaign.py prepare
python scripts/benchmark_training_campaign.py run --detach
python scripts/benchmark_training_campaign.py status
python scripts/benchmark_training_campaign.py report
```

The default destination is `runs/benchmarks/steady-8gpu-20261022`; every command accepts
`--root PATH`, and preparation accepts `--cache PATH` (default `data/c4`). A dry run
checks the matrix and bounded loader memory without writing files or submitting jobs.
Preparation verifies the immutable token cache once and freezes source, dependency
metadata, resolved configurations, and environment/transport scripts. Workers import
that frozen source using the existing architecture-specific environment.

The 54 allocations each request two nodes and four GPUs per node for at most 30 minutes.
Each allocation fixes model/context/batch and EF, then runs synchronous, complete,
ring, and exponential variants sequentially in fresh processes, with a seven-minute
limit per variant. The four pilot allocations cover both EF settings at
20M/context 512/batch 32 and 90M/context 2048/batch 128. The remaining 50 allocations
start only after all pilot measurements pass. At most eight allocations are outstanding.
The detached controller writes `controller.log` and can be inspected with `status`;
rerunning `run` reconciles persisted scheduler receipts instead of duplicating live or
completed work. Uncertain submissions remain unresolved until scheduler evidence exists.

After a controller has stopped, explicitly retry terminal failed cases without replacing
their evidence:

```bash
python scripts/benchmark_training_campaign.py retry --case 90m-c2048-b128-ef1-sync --dry-run
python scripts/benchmark_training_campaign.py retry --case 90m-c2048-b128-ef1-sync
python scripts/benchmark_training_campaign.py run --detach
```

Repeat `--case` to retry several failures. Successful cases are retained, while each retry
gets separate configuration, logs, reports, and scheduler receipts; only its output path
changes in the training recipe. The report preserves prior attempt histories, and retries
add allocation attempts to the original 54 logical groups. Active and uncertain allocations
cannot be retried. If the supervisor itself was updated after preparation, launch it with
`run --detach --controller-script scripts/benchmark_training_campaign.py`; this records a
separate hashed controller revision while preserving the frozen training source.

For a duration-limit failure, add `--max-chunks-factor 2` to `retry` to double the selected
cases' iteration allowance. The command preflights the larger resident working set before
creating any retry artifacts and records the changed measurement bounds. Training settings,
the 15-second minimum, three windows, and original attempt evidence are preserved. The same
option works with `--dry-run` to inspect the new bounds and per-rank host-memory estimate.

To append global batch 256 to a finished campaign, use:

```bash
python scripts/benchmark_training_campaign.py extend --batch-size 256 --dry-run
python scripts/benchmark_training_campaign.py extend --batch-size 256
python scripts/benchmark_training_campaign.py run --detach --controller-script scripts/benchmark_training_campaign.py
```

This adds 72 cases in 18 allocations, with 32 sequences per GPU and no accumulation.
It copies the existing frozen recipes, changing only batch tokens, output paths and the
measurement iteration ceiling. The training source, transport and prior evidence remain
unchanged. New windows allow at least 256 Mi tokens of update capacity, with resident
memory checked before any new configurations are written. Four new pilot allocations
cover both EF settings for 20M/context 512 and 90M/context 2048 at the added batch size;
the remaining allocations wait for their results. The eight-allocation concurrency cap
still applies. Repeating `extend` does not duplicate already registered cases. Reports
and plots include every registered batch size, including the new 256 column.

Every case preloads its bounded real C4 working set into host RAM, performs 30 warmup
updates, then measures three windows of at least 60 updates and 15 seconds each.
The timer includes host batch preparation, H2D transfers, forward/backward, clipping,
optimizer updates, EF, and communication. It excludes startup, compilation, preload,
evaluation, checkpoints, profiling, and metric-file writes. Elapsed time is the maximum
across ranks. Measurements with timed compilation or further resident-data filesystem
reads are invalid. Window variation is reported; three windows are not independent jobs.

The generic benchmark's positive `benchmark.max_chunks` setting (CLI `--max-chunks`)
defaults to eight. The campaign scales it with global batch tokens to provide at least
128 Mi tokens of update capacity per timing window. Its shared preflight/runtime horizon
bounds resident loading without changing ordinary training/resume working sets or reuse
pools. A case that cannot reach its minimum window duration within that bound fails.

CUDA peaks reset after warmup and are captured before profiling and result collection.
The main memory result is the maximum per-GPU **allocated** peak across all eight ranks,
including persistent model, gradient, optimizer, EF, and communication storage. Reserved
memory is a separate allocator diagnostic and can include retained startup cache. These
PyTorch allocator measurements exclude external CUDA/NCCL allocations and do not report
whole-process lifetime RSS as training memory.

`report.json` and `report.csv` retain all registered cases, including missing, failed, OOM, and
invalid measurements. Successful rows include window statistics and cross-rank peaks;
each case retains its eight rank reports, configuration, and logs. Throughput and memory
plots are exported as PNG and PDF. Failures never silently change batching or execution.

## Source layout

Python imports use responsibility-based packages under `src/decentralized_llm`:

| Package | Responsibility |
| --- | --- |
| `models` | Canonical Llama initialization, model representations, parameter storage |
| `algorithms` | AdamW, error feedback, schedules, topology rules |
| `backends` | Training/runtime contracts, factory, distributed and packed implementations |
| `execution` | Hardware policy, compilation, qualification, native and CUDA kernels |
| `data` | Token cache, global sampling, host/device loading |
| `training` | Shared lifecycle, evaluation, checkpoint validation and publication |
| `launch` | Preflight, local processes, Slurm allocations and scripts |
| `benchmarks` | Data-path and complete-training benchmarks |
| `tuning` | Trial planning, allocation grouping, execution, and results |

`config.py`, `runtime.py`, and `state.py` hold common configuration, utilities, and types.
The CLI is unchanged. Python callers must migrate the former flat imports; for example,
`decentralized_llm.train` becomes `decentralized_llm.training.loop`, `model` becomes
`models.llama`, `loader` becomes `data.loader`, and `launch` helpers move to `launch.local`
or `launch.slurm`. There are no forwarding modules. CUDA kernels load only when selected.
