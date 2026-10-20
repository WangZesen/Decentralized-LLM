# Fused decentralized AdamW and mixing

The CUDA optimizer fuses AdamW, local error feedback, and the arithmetic around
parameter mixing. It is independent of the model execution backend. Select
`runtime.decentralized_optimizer=reference` to reproduce the previous optimizer
and mixing passes; the default `auto` selects Triton for FP32 CUDA decentralized
training with `fused_optimizer=true` and deterministic mode disabled.

## Implementation and compatibility

The grouped kernels read existing gradient and moment allocations through device
pointer tables, following the [Triton grouped-pointer pattern](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html).
They do not pack gradients or moments into full-model buffers. Small per-parameter
metadata records offsets, sizes, pointers, and bias corrections. Learning rate and
step-dependent corrections are runtime inputs. The selected schedule uses 2,048
FP32 elements per tile and four warps, with vectorized accesses when aligned.

For each parameter, AdamW computes ordinary updated weights `q` from `base`, then
computes `d = q - base`, stores `q + d - previous_update`, and overwrites history
with `d`. History includes weight decay, starts at zero, and stays rank-local.
The first displacement is therefore doubled. Missing gradients leave moments and
step counters unchanged while subtracting and clearing previous history.

AWC still exchanges a snapshot before forward/backward. After clipping and waiting,
the optimizer reads the reduced snapshot and divides by group size inside the
kernel. It writes final live weights once. ATC writes corrected local weights
already divided by group size, then sums live storage in place. Its history is
saved before division. All communication completes before the next forward,
evaluation, or checkpoint. Failure paths drain outstanding exchanges.

Parameter identities, tied weights, zero padding, and autograd version notifications
are preserved. The `error_feedback.previous_update` checkpoint layout and ordinary
AdamW state remain compatible with reference checkpoints. Loading rebuilds runtime
pointer metadata. The execution selector is excluded from recipe identity and
included in qualification metadata; optimizer, kernel, mixing, and training source
hashes identify measurements. The public snapshot mixing helpers remain available.

## Local GH200 measurements

Job 4128359 ran on GH200 node n75 on 2026-10-20, using ARM64 Python 3.14.5,
PyTorch 2.14.0+cu130, CUDA 13.0, and Triton 3.8.0. Each variant used the same fixed
synthetic gradients, 20 warmup updates, and five windows of 100 updates. These
measurements exclude forward/backward and communication. The context setting has
no effect on this local optimizer workload.

| Model | Feedback | Reference ms/update | Fused ms/update | Speedup |
| --- | --- | ---: | ---: | ---: |
| 20M | Off | 0.300 | 0.174 | 1.721x |
| 20M | On | 0.450 | 0.376 | 1.196x |
| 90M | Off | 0.752 | 0.719 | 1.046x |
| 90M | On | 1.851 | 1.623 | 1.140x |

The preceding bounded schedule sweep also improved feedback steps for both model
sizes (1.195x for 20M and 1.142x for 90M). The isolated local measurements are not
end-to-end speedup claims.

| Model | Padded FP32 parameter bytes | Fused history bytes | Fused scratch bytes | Reference history + scratch bytes |
| --- | ---: | ---: | ---: | ---: |
| 20M | 81,614,080 | 81,614,080 | 0 | 163,228,160 |
| 90M | 366,420,480 | 366,420,480 | 0 | 732,840,960 |

With feedback disabled, history and scratch are both zero. AWC retains one
communication snapshot of the padded parameter size. Fused ATC allocates no
communication snapshot. Pointer metadata is small and separate from these buffers.
Unsupported per-step inputs can fall back to PyTorch; feedback fallback lazily
allocates scratch and reports its actual allocation.

Local profiler traces with input shapes confirm that the reference feedback path's
full-model `copy_`, `sub`, `add_`, and `sub_` passes disappear from the fused path.
The fused kernel handles history in the parameter update pass.

The pilot distributed traces (20M, complete topology, one profiled update per
variant) confirm the mixing changes too. Without feedback, reference AWC/ATC
each performed two full-model copies and one division. Fused AWC retained only
the required exchange snapshot copy; fused ATC had none of these separate passes.
With feedback, the reference added another copy plus subtraction/addition passes;
the fused counts remained one copy for AWC and zero for ATC. The pilot timing
results are excluded from performance acceptance.

## Distributed acceptance

The throughput matrix passed in job 4129125 on 2026-10-20: eight GH200s across
nodes n40 and n110, context 512, and global batch 32,768 tokens. All 96 timing
cases completed (48 paired comparisons, three trials per combination). All 768
rank reports were checked for matching forward policies/configurations, source
revision, allocation, timing windows, and buffer sizes. Both variants used the
same compiled GH200 BF16 forward backend.

| Aggregate across eight combinations | 20M | 90M |
| --- | ---: | ---: |
| Median paired throughput gain | +4.29% | +3.51% |
| Geometric mean paired throughput gain | +4.60% | +3.59% |
| Worst combination | +1.19% | -0.01% |

The per-combination results below are medians of three paired fused/reference
throughput ratios. They are not ratios of separately computed throughput medians.

| Scheme | Topology | Feedback | 20M gain | 90M gain |
| --- | --- | --- | ---: | ---: |
| AWC | Complete | Off | +6.58% | +3.65% |
| AWC | Complete | On | +4.19% | +0.52% |
| ATC | Complete | Off | +4.38% | -0.01% |
| ATC | Complete | On | +4.15% | +3.38% |
| AWC | Ring | Off | +4.78% | +2.67% |
| AWC | Ring | On | +1.19% | +11.30% |
| ATC | Ring | Off | +3.96% | +3.71% |
| ATC | Ring | On | +7.67% | +3.88% |

Every combination met the 0.98 ratio floor. Local feedback steps also improved
for both sizes. Startup is excluded from these timing windows. On rank zero,
the initial reference cases spent 32.60 s (20M) and 19.14 s (90M) in PyTorch
compilation. Later cases shared compilation caches. Their corresponding first
updates took 36.10 s and 19.97 s; the first fused cases took 1.25 s and 0.52 s,
including optimizer initialization/Triton startup. These order-dependent startup
measurements are reported separately and are not optimizer speedup claims.

Numerical acceptance passes under the repository's existing procedure: the first
five matched updates must meet the state-error limits, and validation loss after
100 updates must remain within 0.02 nats. This window is specified in the original
[`qualify_gh200.py`](../scripts/qualify_gh200.py) and
[GH200 qualification documentation](gh200-performance.md#numerical-behavior-and-automatic-qualification).
History is included in the five-update state probe with the same 0.10 relative L2
limit as updates. Step counters must match at every update, and all 100 recorded
measurements must be finite. No numerical tolerance was changed.

The initial optimizer runner incorrectly applied the state-probe thresholds to all
100 updates. That expanded the acceptance procedure beyond the existing gates and caused
the reported job failures. The harness now separates the original gates from the
stricter all-update diagnostics. The optimizer kernels and the measured source
revision are unchanged.

Job 4133550 completed all 32 cases in each precision on eight GH200s across nodes
n132 and n189. Each case ran 100 updates with matched batches and forward policies;
the matrix covered both model sizes, schemes, topologies, feedback settings, and
seeds 42/43. Reassessment checked all 512 rank reports. The raw reports and their
original failed status remain unchanged; separate assessments record their SHA256
hashes, the gate protocol/revision, and both outcomes.

| Matched forward policy | Existing gates | Worst absolute validation delta | Stricter all-update checks |
| --- | ---: | ---: | ---: |
| Native eager FP32 | 32/32 pass | 0.003754 | 17/32 pass |
| Compiled GH200 BF16 | 32/32 pass | 0.005691 | 16/32 pass |

Across the first five updates, the largest BF16 gradient/moment relative L2 errors
were 0.003898/0.003801 (limit 0.05), update error 0.014236 (limit 0.10), history
error 0.011597 (limit 0.10), and parameter error 0.000001700 (limit 0.02).
All step counters agreed for all 100 updates. Fused feedback scratch remained zero.

The stronger diagnostics remain failures: 15/16 FP32 and 16/16 BF16 20M trajectories
exceeded later gradient, update, moment, or history limits. All 90M cases passed
the stronger checks in both precisions. Diagnostic job 4133159 also found late
failures in reference/reference controls (maximum gradient relative L2 of 0.3988
and 0.1109 for compiled GH200 seeds 42/43, and 0.5478 for native eager seed 42).
This evidence does not isolate the late drift to fusion, nor establish equivalence
of every independently evolving gradient or history vector over 100 updates.
The first-five-update and final-loss acceptance result should not be read as that
stronger claim.

## Reproduction and evidence

Run local timings inside a GH200 allocation:

```bash
source scripts/environment.sh
"$DECENTRALIZED_PYTHON" scripts/benchmark_error_feedback.py \
  -c configs/base.yaml -c configs/20m.yaml --profile \
  --output runs/fused-optimizer/local-20m.json
```

Repeat with `configs/90m.yaml`. For the distributed matrix, use a two-node allocation
with four GH200s per node and the site's CXI environment. Launch one torchrun agent
per node with `srun`; set `MASTER_ADDR` to the first allocated host and use
`--nnodes=2 --nproc-per-node=4 --node-rank="$SLURM_PROCID"`, the same rendezvous
address/port, and `--numa-binding=exclusive`. The worker commands are:

```bash
source scripts/environment.sh
source scripts/activate_cxi.sh
export OMP_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=1
# Run this worker script through srun, once on each allocated node.
"$DECENTRALIZED_PYTHON" -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=4 --node-rank="$SLURM_PROCID" \
  --master-addr="$MASTER_ADDR" --master-port=29533 --rdzv-backend=static \
  --numa-binding=exclusive scripts/benchmark_decentralized_optimizer.py \
  -c configs/base.yaml -c configs/gh200-8gpu.yaml \
  --set model.context_length=512 --set distributed.mode=decentralized \
  --set training.tokens_per_parameters=0.1 --set training.epoch_tokens_per_parameters=0.1 \
  --set benchmark.warmup=20 --set benchmark.steps=40 \
  --set benchmark.windows=3 --set benchmark.min_seconds=1 \
  --set benchmark.profile=false --trials=3 --resume \
  --output runs/fused-optimizer/matrix-bounded
"$DECENTRALIZED_PYTHON" -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=4 --node-rank="$SLURM_PROCID" \
  --master-addr="$MASTER_ADDR" --master-port=29535 --rdzv-backend=static \
  --numa-binding=exclusive scripts/qualify_decentralized_optimizer.py \
  -c configs/base.yaml -c configs/gh200-8gpu.yaml \
  --set model.context_length=512 --set distributed.mode=decentralized \
  --set training.tokens_per_parameters=0.1 --set training.epoch_tokens_per_parameters=0.1 \
  --steps=100 --resume --output runs/fused-optimizer/numerical-bounded
```

For the FP32 trajectory comparison, add `--set runtime.execution_backend=native`,
`--set runtime.compile=false`, and `--set runtime.amp=false` to the numerical worker
command, and use `--output runs/fused-optimizer/numerical-fp32`.

The small recipe token budget bounds resident preloading to the benchmark horizon.
Set both the total and epoch token budgets: the total must be an integer multiple
of the epoch budget for configuration validation.
With `data_reuse_factor=1`, the seeded batch order is unchanged; the benchmark supplies
its own update horizon. An initial full-recipe preload pilot is archived separately
under `runs/fused-optimizer/matrix/` and excluded from acceptance. Its traces and
the local optimizer traces provide memory-pass evidence. Timed matrix trials disable
profiling; collect additional traces in a separate launch with `benchmark.profile=true`.
Resident loading finishes before compute warmup, avoiding idle time between warmup
and measurement.

Both runners reuse process groups and compilation caches across cases. Timing order
alternates between trials. Each timing result is the median of three synchronized
windows after warmup and resident-loader preparation; compilation/startup times are
reported separately, including per-rank PyTorch compiler phase times (overlapping
phases must not be summed). First-update time also includes optimizer initialization
and Triton JIT startup. Matrix comparisons use median paired trial ratios. `--resume`
reuses timing pairs only when both variants match source/configuration and were
measured in the same SLURM allocation on every rank. Partial pairs are remeasured.
Numerical cases resume only when all rank reports match and pass. Source changes
during a run invalidate its reports.

The archived job used the stricter all-update acceptance by mistake. Reassess its
unchanged measurements with the restored original protocol using:

```bash
"$DECENTRALIZED_PYTHON" scripts/assess_optimizer_trajectories.py \
  --matrix runs/fused-optimizer/numerical-bounded/matrix.json \
  --output runs/fused-optimizer/numerical-bounded/assessment.json
"$DECENTRALIZED_PYTHON" scripts/assess_optimizer_trajectories.py \
  --matrix runs/fused-optimizer/numerical-fp32/matrix.json \
  --output runs/fused-optimizer/numerical-fp32/assessment.json
```

The assessment never rewrites input measurements and checks every rank. Future
qualification runs record both outcomes directly and invalidate resume metadata
when the gate implementation changes.

The local raw reports, traces, schedule sweep, and GPU test log are in
`runs/fused-optimizer/`. Optimizer source revision:
`a06b8feb499846df9ebbd39a560ed80ff4871393ed4f7885431889ad04006857`.

Sixteen CUDA tests passed on four GH200s in job 4128359. They include FP32 comparisons
at `rtol=1e-5, atol=1e-6`, missing gradients, changing/zero learning rates, grouped
weight decay, tied weights, padding, closures, checkpoint restoration, eager/compiled
forwards, accumulation, snapshot reuse, all supported topologies, two/three/four-rank
mixing, and failure cleanup. All 484 CPU/Gloo tests passed with GPUs hidden (27 CUDA
tests deselected), including the restored gate-window and audit tests; Ruff and
Pyright also passed. The machine-readable record is
[decentralized-optimizer-summary.json](performance/decentralized-optimizer-summary.json).

GPU measurements cover GH200 only; there are no performance claims for other CUDA
architectures. Qualification covers the original five-update state probes and
100-update validation-loss gates; full-training convergence has not been assessed. The existing GH200 forward-backend qualification and its
limitations remain separate from this optimizer comparison.
