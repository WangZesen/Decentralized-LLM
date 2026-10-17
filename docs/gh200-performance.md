# GH200 training: optimization decisions and measurements

The standard GH200 recipe uses compiled kernels, persistent compilation caches,
exclusive GPU-local CPU binding, and the scheduler's full CPU allowance. The two-node
recipe also activates OFI/CXI RDMA. These choices target training and hyperparameter
searches on one node with four GPUs or two nodes with eight GPUs.

The strongest findings were that compilation artifacts can be reused across fresh
training processes, socket fallback severely constrained cross-node communication,
and early worker binding reliably placed helper threads near their GPUs. Throughput
improved in several measured workloads, but the experiments do not establish numerical
equivalence, universal speedups, or full-training convergence. The selected presets
explicitly enable GH200 execution; automatic numerical qualification remains separate.

## Selected recipe and implementation

For ordinary training and tuning, set `runtime.execution_backend: gh200` in the ignored
`configs/local/defaults.yaml`; see [local configuration](../README.md#configuration-and-preparation).
The presets below are optional reproduction recipes that override local defaults.
The generic training and tuning configurations do not depend on them or activate CXI.

Layer either `configs/gh200-4gpu.yaml` or `configs/gh200-8gpu.yaml` after the base and
model configurations. Each preset is self-contained. Both use a 32,768-token global
batch, BF16, model compilation, automatic SDPA selection, fused AdamW, eight PyTorch
threads per worker, and exclusive CPU affinity. Microbatches are eight per rank on
four GPUs and four on eight GPUs. DDP is the default; complete-topology AWC is selected
with `distributed.mode=decentralized`. The presets leave CPU allocation to Slurm.

The kernel implementation is adapted from
[tiny-llm revision dd03e5037b46961b1bbcbd014c445b6efebb4841](https://github.com/WangZesen/tiny-llm/tree/dd03e5037b46961b1bbcbd014c445b6efebb4841/src/tiny_llm/gh200).
Original source hashes are retained in `src/decentralized_llm/gh200/provenance.json`.
The port combines Q/K/V and gate/up projections during execution while retaining
independent canonical parameters. It adds fused residual/RMSNorm, paired SwiGLU, and
masked per-token cross entropy that avoids an FP32 vocabulary-sized copy. Parameter
names, tied embeddings, optimizer groups, and checkpoint tensor layouts are preserved.
Fused RoPE now defaults to `auto`, enabling it on the GH200 20M/90M paths when
projection fusion is enabled. Other models and native execution use ordinary RoPE.
Set `runtime.fused_rope=false` for an explicit opt-out; explicit `true` still requires
a supported GH200 path and projection fusion.

A follow-up on 2026-10-15 compared two short 90M jobs with one GH200 each, context
512, batch 64 sequences (32,768 tokens), seed 42, 50 warmup updates, and five timing
windows of at least five seconds. Median throughput was 552,908 tokens/s with fused
RoPE off (job 3854530, n164) and 563,887 with it on (job 3854531, n196): a **1.99%**
increase. Both jobs completed with finite logged losses and gradients. Compilation
and resident preloading were excluded from timing. One process per setting on
different nodes establishes only a preliminary gain; it does not establish a
universal speedup or convergence equivalence. Raw reports and the source snapshot
are retained locally under `runs/fused-rope-90m-20261015-001618/`.
The historical measurements below used `fused_rope=false`.

The distributed runtime keeps ordinary fused AdamW, real DDP gradient synchronization,
independent decentralized optimizer moments, and NCCL parameter exchanges. It does not
adopt the upstream packed multi-worker engine or whole-update CUDA graphs. Compilation
wraps DDP while preserving `no_sync()` during accumulation. CUDA graph capture is disabled
for this path. Incomplete microbatches and validation tails use eager execution to limit
extra Inductor graphs; Triton kernels can still specialize on previously unseen shapes.

Shared software changes reduce overhead in both native and GH200 execution:

- Reusable decentralized snapshot buffers avoid repeated allocation. Their ownership
  lasts through communication, and AWC installs mixed weights only after backward.
- All training modes now bind canonical parameters to a shared allocation with
  256-byte-aligned starts. Decentralized snapshots and installation use whole-buffer
  copies. This later storage change is not covered by the historical measurements
  below; see the [storage validation and reproduction commands](validation.md#contiguous-parameter-storage).
- GPU scalar metrics are read only at logging steps. Nonfinite gradient checks remain
  enabled during clipping.
- Validation uses pinned host tensors and asynchronous copies, with float64 loss
  aggregation.
- FX graph, AOTAutograd, and Triton caches persist across processes. Cache identity
  includes architecture, Python/PyTorch/CUDA/Triton versions, GPU capability, and model/
  compiler source. Seed, optimizer hyperparameters, and output paths do not split it.
- Environment activation checks a project/lockfile fingerprint and installed versions,
  avoiding repeated dependency synchronization when the environment is already ready.
- Checkpoints record the resolved execution policy. Resume pins it; legacy checkpoints
  retain native execution.

The contiguous-storage change invalidates the historical automatic qualification's
compiler revision. Its measured hashes remain unchanged, so automatic selection uses
the unqualified native policy until new evidence is recorded. Explicit GH200 presets
remain available for validation; their performance must be remeasured with the new layout.

DDP uses 25 MiB buckets without gradient bucket views or batched copies by default.
The larger-bucket combination remains configurable because its gains were measured
in single-run screens. Native model/block compilation and alternative attention or
binding modes also remain available for controlled comparisons.

## Experimental setup and interpretation

Measurements used four GH200 120 GB GPUs per node, 288 physical CPU cores per node,
shared token storage, and Slingshot/CXI networking between nodes. The measured software
was ARM64 Python 3.14.5, PyTorch 2.14.0+cu130, CUDA 13.0, Triton 3.8.0, and PyTorch's
NCCL 2.30.7. The transport activation additionally used aws-ofi-nccl 1.21.1,
libfabric 2.6.0 with CXI 1.8.0, the GCC 14.3 C++ runtime, and GDRCopy 2.5.

The main models were 20M and 90M with context 1,024 and global batch 32,768. A 500M
model was used only for capacity/performance probes. Throughput includes real data
loading, forward/backward, gradient clipping, optimizer updates, and communication.
Each process performs 20 warmup updates and three timing windows of at least two
seconds. Scoring uses maximum elapsed time across ranks and complete rotation cycles
for peer topologies. Resident loaders must finish loading before measurement;
streaming configurations continue streaming. The CPU follow-up used resident loading
for 20M and streaming for 90M.

A process's windows are not independent repetitions. Tables state the number of fresh
processes or matched process pairs. Earlier repeated screens varied seed and learning
rate and alternated backend order; the later FlashAttention comparison kept learning
rate fixed. CPU comparisons pair the same seed and configuration within one allocation.
Cross-allocation comparisons also include node, input-cache, and run-order variation.
An initial screen that compared different resident-preload states was excluded.

The campaign consumed **19.84 of 30 allocated GPU-hours**, including setup, failed
allocations, and timed-out steps; queue waiting was excluded. It audited 114 benchmark
cases with reports, 42 numerical cases, and three collective transport cases across
all participating ranks. Two additional CPU controls produced no timing reports.
Raw reports, logs, source snapshots, and generated summaries remain in the local,
ignored experiment archive. This document is the committed measurement record.

## From socket fallback to RDMA

The initial eight-GPU jobs found neither a usable OFI plugin nor an InfiniBand device
and selected sockets across ordinary Ethernet and high-speed interfaces. In one
profiled 20M DDP update, three NCCL reductions occupied about 18 ms of CUDA ranges,
compared with 1.3 ms for GEMMs. These overlapping ranges are diagnostic, not an
additive wall-time breakdown.

A previous training project supplied the useful transport recipe: a CXI-capable OFI
plugin, CUDA DMA-BUF registration, GDRCopy, and a Slurm job VNI. Its old installation
was unavailable, so the installed modules and libraries were inspected instead. The
ordinary NCCL module exposed build paths without activating OFI. A newer installed
plugin required `GLIBCXX_3.4.32`; selecting the newer C++ runtime and libfabric before
Python imported torch was essential to loading the intended stack.

`scripts/activate_cxi.sh` now selects that stack before torchrun, preserves PyTorch's
NCCL, and sets `NCCL_NET='AWS Libfabric'`, `FI_PROVIDER=cxi`,
`OFI_NCCL_PROTOCOL=RDMA`, `FI_HMEM_CUDA_USE_DMABUF=1`,
`FI_CXI_ENABLE_WRITEDATA=1`, and `NCCL_LAUNCH_ORDER_IMPLICIT=1`.
The eight-GPU preset requests `--network=job_vni`. Installation paths in the script
must be adapted on another site; neutral naming does not make this transport universal.
Synthetic NCCL topology overrides were not adopted because the prior project's
matched training comparison found no benefit.

A same-allocation eight-rank collective probe compared three transport choices:

| All-reduce payload | Mixed-interface TCP | High-speed-only TCP | CXI RDMA |
| ---: | ---: | ---: | ---: |
| 25 MiB | 4.16 GB/s | 3.60 GB/s | 38.06 GB/s |
| 80 MiB | 8.09 GB/s | 6.04 GB/s | 71.01 GB/s |
| 360 MiB | 8.48 GB/s | 6.43 GB/s | 84.61 GB/s |
| 2,000 MiB | 8.66 GB/s | 7.06 GB/s | 89.88 GB/s |

These are algorithm bandwidths, not training speedups. Loop timing includes control
and synchronization overhead, with up to 200 collectives per size. All eight ranks
passed value checks. Logs confirmed RDMA, DMA-BUF, and GDRDMA connections; traffic was
balanced over four CXI interfaces per node. Optional GIN-plugin warnings remained,
but the tested ordinary collectives passed. Subsequent RDMA training and CPU follow-up
launches selected OFI on all eight ranks without socket data transport.

Training throughput with RDMA, using the earlier restricted 64-CPU-per-node layout:

| Workload | Eager native tokens/s | Compiled native | GH200 | GH200 over compiled | Processes: eager / compiled / GH200 |
| --- | ---: | ---: | ---: | ---: | --- |
| 20M DDP | 797,302 | 1,752,217 | 2,045,123 | +16.7% | 1 / 1 / 1 |
| 20M AWC | 927,228 | 2,017,468 | 2,260,899 | +12.1% | 1 / 3 / 3 |
| 90M DDP | 474,454 | 877,572 | 1,186,193 | +35.2% | 1 / 1 / 1 |
| 90M AWC | 552,080 | 1,246,527 | 1,402,368 | +12.5% | 1 / 1 / 1 |

Earlier TCP GH200 medians were 990,348, 2,100,649, 298,448, and 515,987 tokens/s for
these rows, respectively. The resulting 2.07x, 1.08x, 3.97x, and 2.72x differences
span allocations and are not isolated transport effects. The collective probe supplies
the direct transport comparison. Eight GPUs do not double throughput at this small
fixed global batch; increasing the batch would also change the optimization problem.

## CPU allocation and GPU-local binding

Earlier launchers imposed 64 CPUs per node, shared by four GPU workers. On two-node
jobs the allocation sometimes reserved all 576 CPUs while the training steps still
received only 64 per node. Exclusive binding then gave each worker just 16 local CPUs.
The site policy was already `DefCpuPerGPU=72`: omitting CPU-count flags supplies
288 CPUs per full node. `launcher.cpus_per_task` now defaults to `null`; the PyTorch
thread count remains independently set to eight.

Late `cpu_affinity=local` binding affected the calling thread but left some existing
helper threads on the wider allocation mask. Torchrun's `--numa-binding=exclusive`
starts workers through `numactl --physcpubind`, so their threads inherit an exclusive
share of the allowed CPUs near each GPU. This is CPU affinity, not an explicit NUMA
memory policy. With the full allocation, every sampled thread in the exclusive runs
stayed within its worker's 72 GPU-local CPUs.

Two approximately ten-minute jobs compared 64 versus 288 CPUs per node within the
same allocation, reversing order between two requested repeats. All 20 launches
passed CPU/NUMA allocation checks. Eighteen exited cleanly with complete reports.
The first 20M 64-CPU control in each job exceeded its 150-second limit before producing
reports; later controls completed. The precise stalled stage is unknown. The table
therefore uses only complete same-seed pairs:

| GPUs / workload | Pairs | 64 CPUs/node tokens/s | 288 CPUs/node tokens/s | Change |
| --- | ---: | ---: | ---: | ---: |
| 4 / 20M AWC, FlashAttention | 1 | 2,584,045 | 2,533,414 | -1.96% |
| 4 / 90M AWC, auto SDPA | 2 | 1,488,271 | 1,465,158 | -1.55% |
| 8 / 20M DDP, auto SDPA | 1 | 1,925,095 | 2,093,883 | +8.77% |
| 8 / 90M AWC, auto SDPA | 2 | 1,410,295 | 1,486,620 | +5.41% |

The full CPU allowance helped the completed eight-GPU comparisons. Four-GPU results
were slightly lower: the individual 90M changes were +0.51% and -3.43%, so a median
within 2% does not establish that every trial avoids regression. Eight-GPU 90M changes
were +2.96% and +8.00%. These short comparisons are not a statistical noninferiority test.

Each workload then had one full-CPU exclusive-binding screen:

| GPUs / workload | Exclusive tokens/s | Change from same-seed full-CPU unbound run |
| --- | ---: | ---: |
| 4 / 20M AWC, FlashAttention | 2,783,247 | +47.00% |
| 4 / 90M AWC, auto SDPA | 1,551,831 | +9.01% |
| 8 / 20M DDP, auto SDPA | 2,548,102 | +21.98% |
| 8 / 90M AWC, auto SDPA | 1,527,956 | +2.51% |

Exclusive cases ran last, so order effects were not controlled. Four-GPU 20M unbound
throughput varied from 1.893M to 2.533M across repeats; its +47% result is particularly
unsuitable as a precise affinity effect. Earlier restricted-CPU screens were mixed:
exclusive binding reduced four-GPU 90M throughput by 2.4%, while separate eight-GPU
RDMA screens improved 20M DDP by 25.6% and 90M AWC by 22.0%. One early four-GPU 20M
unbound reference produced reports but timed out during shutdown; its later clean
repeat ran in a different allocation and cannot repair that matched comparison.

Exclusive binding is adopted in the hardware presets because it gives predictable
GPU-local placement and improved the full-CPU 90M screens and the sampled eight-GPU
workloads. The evidence does not establish that it wins for every workload. The
`none` and late `local` options remain available for comparisons.

## Compilation reuse, startup, and other throughput results

Fresh processes reused compatible compiler artifacts even when seed and learning
rate changed. First-update time includes work beyond compilation, so counters are
needed alongside timing to establish reuse:

| Four-GPU workload | Cold first update | Warm first update | Warm cache evidence |
| --- | ---: | ---: | --- |
| 20M DDP GH200 | 60.78 s | 17.72 s | 4 FX hits; AOT cache bypassed |
| 20M AWC GH200 | 26.49 s | 12.82 s | 1 AOTAutograd hit, 2 FX hits |

Repeated eight-GPU AWC processes also recorded AOT hits, with first updates of 11.02
and 10.24 seconds. The four-GPU FlashAttention compiled-native sequence fell from
51.35 seconds without FX hits to 15.63 and 13.17 seconds with two FX hits each. Its
GH200 comparison was already warm: 18.38, 14.29, and 16.73 seconds, each with two FX hits.
DDP's AOT bypass does not prevent FX/Triton reuse.

Input preloading remained significant: initial four-GPU 20M resident pools took about
26-94 seconds. Throughput timing waits for residency, while startup retains that cost.
Node and input-cache differences prevent attributing all first-update differences to
compiler caching. Cache namespaces include executable source; an earlier automatic-
selection correction created a fresh namespace even though kernel arithmetic was
unchanged. This configuration/documentation cleanup leaves compiler-source identity
unchanged. Subsequent compatible trials share artifacts, without deleting old caches.

A single four-GPU 20M end-to-end comparison trained 311 updates, wrote checkpoints,
and evaluated all 197,411,295 validation targets. Native eager took 53.46 seconds from
training entry to completion; GH200 took 44.19 seconds. Full validation took 25.80
versus 9.41 seconds, while first updates took 9.46 versus 22.01 seconds. These were
partially warm runs, not a controlled cold/warm study or convergence experiment.

Earlier four-GPU screens with automatic SDPA and 64 CPUs per node produced:

| Workload | Eager native tokens/s | Compiled native | GH200 | Processes: eager / compiled / GH200 |
| --- | ---: | ---: | ---: | --- |
| 20M DDP | 886,197 | 1,806,594 | 2,011,213 | 1 / 3 / 3 |
| 20M AWC | 901,267 | 2,011,478 | 2,281,543 | 1 / 3 / 3 |
| 90M DDP | 521,731 | 978,560 | 1,158,738 | 1 / 3 / 3 |
| 90M AWC | 544,397 | 1,238,841 | 1,294,864 | 1 / 3 / 3 |

These imply GH200 gains of 11.3%, 13.4%, 18.4%, and 4.5% over compiled native in
those samples. Cross-allocation variation was substantial; later CPU experiments
must not be merged into these medians.

With RDMA and the restricted CPU layout, single-run DDP screens combining 64 MiB
buckets, gradient views, and batched copies improved 20M throughput from 2,045,123
to 2,157,179 tokens/s (+5.5%) and 90M from 1,186,193 to 1,348,401 (+13.7%). Bucket
and affinity changes were tested separately; their gains cannot be multiplied.

The 500M capacity probe fit both layouts. Four-GPU eager/GH200 throughput was
231,618/249,513 tokens/s, with peak allocated memory 25.28/19.68 GiB. Eight-GPU
throughput was 52,421/45,628 tokens/s and memory 17.07/14.26 GiB, using the old socket
transport. Both five-update numerical probes failed the gradient-error limit
(relative L2 0.0990 and 0.0517). These observations do not qualify 500M training.

## Numerical behavior and automatic qualification

The qualification procedure compares five matched BF16 updates: loss uses absolute/
relative tolerance 0.002; gradient and both Adam moment relative L2 errors must be
below 0.05, parameters below 0.02, and actual updates below 0.10. Two 100-update
trajectories with different seeds must keep candidate-minus-native fixed-subset
validation loss within 0.02 nats in absolute magnitude. Performance promotion also
requires at least 5% median gain over compiled native and no representative regression
above 2%. A negative loss delta exceeding the absolute threshold still fails.

Earlier auto-SDPA trajectories gave the following validation deltas; eight-GPU rows
used TCP:

| GPUs / workload | Seed 42 delta (nats) | Seed 43 delta (nats) | Both seeds pass all gates |
| --- | ---: | ---: | --- |
| 4 / 20M DDP | +0.017838 | -0.034515 | No |
| 4 / 20M AWC | -0.036745 | +0.032617 | No |
| 4 / 90M DDP | +0.177911 | -0.180478 | No |
| 4 / 90M AWC | -0.072471 | -0.019266 | No |
| 8 / 20M DDP | -0.034230 | -0.037634 | No |
| 8 / 20M AWC | -0.008433 | +0.000731 | Yes |
| 8 / 90M DDP | -0.087979 | +0.010105 | No |
| 8 / 90M AWC | -0.034116 | +0.016559 | No |

The eight-GPU 90M AWC seed-43 case also exceeded the first-five-update gradient and
second-moment limits (0.0640 and 0.1051); the other primary first-five-update checks
passed. The passing TCP trajectory did not transfer to RDMA:

| Eight-GPU 20M AWC with RDMA | Seed 42 delta | Seed 43 delta | Both pass |
| --- | ---: | ---: | --- |
| Auto SDPA | +0.004123 | -0.051463 | No |
| FlashAttention | -0.007185 | +0.031009 | No |

All four passed the first-five-update checks. Both seed-43 trajectories exceeded the
unchanged validation-drift threshold. Transport changes can affect reduction arithmetic;
these results do not demonstrate transport corruption, and collective value tests passed.

Four-GPU FlashAttention trajectories tested whether attention selection resolved drift:

| Workload | Seed 42 delta | Seed 43 delta | Both pass every gate |
| --- | ---: | ---: | --- |
| 20M DDP | -0.041867 | -0.025842 | No |
| 20M AWC | +0.012264 | -0.014256 | Yes |
| 90M DDP | +0.196477 | +0.300846 | No |
| 90M AWC | -0.004158 | +0.067578 | No |

All eight first-five-update checks passed. For 20M AWC, three matched fresh processes
per compiled backend, with seeds 42/43/44 and unchanged learning rate 0.001, gave
median throughput 2,153,965 tokens/s for compiled native and 2,269,587 for GH200
(+5.37%). All three paired repetitions improved. The single eager reference was
907,183 tokens/s. This is the sole profile admitted by strict automatic selection:
four GPUs, 20M complete-topology AWC, FlashAttention, 64 allowed CPUs per node, and
no affinity binding, with exact workload/software/kernel/compiler guards.

Native controls were not identical under nondeterministic execution. Native eager
20M DDP seed 43 and AWC seed 42 drifted +0.003950 and +0.017395 nats; native compilation
failed those two trajectory checks at -0.032025 and -0.037833. Projection-fusion and
precision-emulation ablations did not consistently resolve the differences. Thus the
experiments do not isolate all drift to the ported kernels.

The full-CPU/affinity follow-ups measured throughput only. They do not extend the
historical profile to the new presets. The presets explicitly choose compiled GH200
for the intended throughput regime; generic `auto` selection retains its strict
guards and falls back to eager native for unmatched settings. No thresholds were
relaxed and failed checks were not relabeled. Long-training quality and arbitrary
hyperparameter equivalence remain unestablished.

## Reproducing the measurements

Prepare the token cache and native ARM environment as described in the README.
Select one hardware preset, an output directory, and a persistent cache directory:

```bash
uv run decentralized-llm benchmark-training -c configs/base.yaml -c configs/90m.yaml \
  -c configs/gh200-4gpu.yaml --set data.cache_dir=/shared/tokens/c4 \
  --set runtime.output_dir=runs/benchmark-4gpu --dry-run
uv run decentralized-llm benchmark-training -c configs/base.yaml -c configs/90m.yaml \
  -c configs/gh200-8gpu.yaml --set data.cache_dir=/shared/tokens/c4 \
  --set runtime.output_dir=runs/benchmark-8gpu --dry-run
```

Remove `--dry-run` to submit. Both require the lowercase `account` environment variable.
`--warmup`, `--steps`, `--windows`, and `--min-seconds` set measurement bounds;
`--profile` adds per-rank CPU/CUDA traces. Use `runtime.execution_backend=native` with
`runtime.compile=false` or `true` for eager or compiled controls. To measure cache
reuse, run fresh processes against the same persistent compilation directory and
inspect both first-update time and compiler counters. An unused directory provides
a cold namespace without deleting existing artifacts.

For CPU or affinity comparisons, keep backend, seed, thread count, model, and batch
fixed, alternate case order, and compare steps within the same allocation. Set
`runtime.cpu_affinity=none` for the unbound control. Restrict only control steps to
64 CPUs per node; retain the full allocation for the other steps. Report complete
same-seed pairs, timeouts, and run counts. Separate CLI submissions use different
allocations and are not equivalent to that paired design.

The reusable `scripts/benchmark_collectives.py` and `scripts/qualify_gh200.py` run
inside allocations. For example, on an already allocated two-node, four-GPU-per-node
job with a VNI and enough time, the numerical check can be launched as follows:

```bash
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
srun --ntasks=2 --ntasks-per-node=1 --gpus-per-task=4 --gpu-bind=none \
  --cpu-bind=none --kill-on-bad-exit=1 bash -c '
    source scripts/environment.sh
    source scripts/activate_cxi.sh
    export OMP_NUM_THREADS=8
    exec "$DECENTRALIZED_PYTHON" -m torch.distributed.run \
      --nnodes=2 --nproc-per-node=4 --node-rank="$SLURM_PROCID" \
      --master-addr="$MASTER_ADDR" --master-port=29500 --rdzv-backend=static \
      --numa-binding=exclusive --max-restarts=0 scripts/qualify_gh200.py \
      -c configs/base.yaml -c configs/90m.yaml -c configs/gh200-8gpu.yaml \
      --set data.cache_dir=/shared/tokens/c4 --set lr_schedule.warmup_tokens_per_parameters=0.007 \
      --set runtime.seed=42 --steps 100 --output runs/qualification-90m-s42'
```

Repeat with another seed and output directory. The script replays identical real
batches through native and candidate executions and records each numerical gate;
it exits unsuccessfully when a gate fails. For a transport probe, use the same launcher
with `scripts/benchmark_collectives.py --world-size 8 --output runs/collectives` as
the worker script. Check loaded libraries and NCCL transport logs alongside bandwidth.
Library or model/compiler changes require new measurements; reproducing the method
does not guarantee the same timing or trajectory.

## Validation and failure accounting

Final cleanup validation passed **135 CPU/Gloo tests**, Ruff lint and formatting,
Pyright with zero errors/warnings, and shell syntax checks. Training and benchmark
CLI dry runs passed for DDP and AWC on both presets; tuning dry runs passed for both
layouts. These used a small synthetic token fixture to exercise configuration and
launch validation without GPU execution, scheduler submissions, or study creation.
The eight CUDA tests were excluded from this CPU run; their earlier allocated results
are recorded below. Compilation-source identity and numerical-selection guards were
preserved during cleanup.

The earlier implementation passed 123 CPU/Gloo tests and eight allocated CUDA tests.
The GPU checks covered BF16, fused AdamW, Inductor, fused-kernel gradients and ignored
labels, reusable NCCL snapshots on two/four GPUs, and pinned transfer reuse. Subsequent
launcher tests covered torchrun argument parsing, early affinity, environment activation,
CPU allocation defaults, and transport-overlay ordering. CPU compiled-DDP tests use
the eager compiler backend; the allocated jobs exercised Inductor and NCCL.

One CUDA-suite allocation failed before tests started because its manually prepared
ARM environment lacked pytest's `py.py` compatibility shim. Restoring the locked file
and verifying dependency hashes allowed the retry to pass. An earlier numerical
allocation timed out; only unfinished cases were continued. Completed numerical
failures stayed failures, including the RDMA case where a worker's deliberate failure
caused Slurm to terminate its peer task after all reports had been written.

The CPU audit initially treated every loader as resident; it was corrected to accept
intentional 90M streaming while still requiring 20M residency before scoring. No
training reports were changed. The two missing initial CPU controls remain timeouts,
and no replacement GPU jobs were submitted for this cleanup. The final CPU allocation
accounting uses the longer allocation/extern elapsed times, 602 seconds on four GPUs
and 593 seconds on eight GPUs, within the 19.84 GPU-hour total.
