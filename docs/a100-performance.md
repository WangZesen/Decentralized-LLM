# A100 execution and measurements

The explicit `runtime.execution_backend=a100` path shares the fused implementation
with GH200: combined Q/K/V and gate/up projections, residual/RMSNorm, paired SwiGLU,
packed RoPE for 20M/90M, and masked cross-entropy. Parameters and checkpoint tensors
retain their canonical names and shapes. The backend requires an A100, a supported
model preset, nondeterministic BF16 SDPA, and compilation. `auto` retains the existing
qualification policy and does not automatically select this experimental backend.

Use `configs/a100-1gpu.yaml` after the base and model files. It selects one GPU,
32,768 tokens per update, GPU-local exclusive CPU binding, eight PyTorch threads,
and persistent compilation caches. Set the scheduler partition and walltime for your
site. The compiler setup is described in [toolchain setup](toolchain-setup.md).
The shared kernels remain in the historical `gh200` package; the GH200 backend and
its presets keep their existing behavior.

## Measurement scope

The October 2026 screening uses the 90M preset (91,605,120 parameters), context 1,024,
32 sequences per update, one A100-SXM4-80GB, BF16, fused AdamW, and the existing C4
cache. Complete training updates include forward, backward, clipping, optimizer,
and input loading. Each throughput trial warms up for ten updates and measures
three windows of at least twenty updates and two seconds. Compilation is excluded
from throughput and reported separately in the raw benchmark files. Startup and
GPU memory are also recorded. Auto SDPA selected FlashAttention in the profiles.

The numerical check replays identical batches and initialization against eager
native execution for 100 updates. It checks the first five updates' loss, gradients,
parameters, Adam moments and actual parameter updates, then compares validation on
a fixed subset. The existing limits are unchanged: gradient/moment relative L2
below 0.05, parameter error below 0.02, update error below 0.10, loss tolerance
`0.002 + 0.002 * abs(reference_loss)`, and absolute validation delta at most 0.02 nats.
The production warmup setting is retained, so these 100-update checks cover early
warmup. They are not evidence of full-training convergence or later high-learning-rate
trajectory equivalence.

## Final measured recipe

On one A100-SXM4-80GB, the final code measured **256,646 tokens/s**, a **6.36%**
gain over compiled native at **241,292 tokens/s**. Each number is the median of two
trial medians, with three timing windows per trial, using the same allocation.
Final verification increased the windows to at least thirty updates and three seconds.
The stack was Python 3.14.8, PyTorch 2.14.0+cu130, CUDA 13.0 and Triton 3.8.0 on x86-64.

| Final-code variant | Trial medians, tokens/s | Peak allocated GiB |
| --- | ---: | ---: |
| Compiled native | 241,296 / 241,287 | 13.92 |
| Fused A100, default compilation | 256,816 / 256,475 | 16.97 |
| Fused A100, max-autotune | 253,772 | 16.97 |

The preset selects **default compilation**, all supported fusions, auto SDPA, and the
measured RMSNorm schedule. Max-autotune was slower on the same GPU and its first update,
including compilation, took 120.7 seconds versus 33.9 seconds for default fused
execution. These startup figures depend on the existing cache state. Compilation
caches remain enabled for subsequent runs.

This is a throughput improvement with a memory tradeoff: peak allocated memory
increases by about 3.05 GiB. No throughput claim is made for other model sizes,
A100 memory variants, multi-GPU configurations, or different batch shapes.

The final source passed both 100-update numerical comparisons:

| Seed | Validation loss delta, candidate − eager native | Max gradient relative L2 | Max update relative L2 | All gates |
| --- | ---: | ---: | ---: | --- |
| 42 | +0.00004663 | 0.004091 | 0.053871 | Pass |
| 43 | −0.00002204 | 0.003993 | 0.052674 | Pass |

The [machine-readable summary](performance/a100-summary.json) retains timing windows,
peak memory, numerical maxima, compiler identities, job IDs, and relative paths to
the raw reports under the ignored `runs/a100-optimization/` directory. The final
verification job was `17813443`. Earlier screens used jobs `17813067` (kernels),
`17813070` (ablation), `17813071` (numerical controls), `17813400` (kernel schedules),
and `17813406` (autotuning). Two earlier submissions failed at scheduler step creation
before Python ran and contribute no measurements.

Validation also passed all 365 CPU tests, all four CUDA execution/kernel tests,
Ruff lint and formatting, and Pyright with no warnings. After the final GPU schedule
change, the 36 focused CPU execution/startup tests passed again. Launcher dry-run
validation confirmed one GPU, microbatch 32, and exclusive CPU binding.

## Screening decisions

The initial ablation job measured the following medians on the same A100. These
screens precede the final RMSNorm schedule change; final-code results are reported
separately. The last fused repeat hit the job's ten-minute limit before producing
a result, so it is excluded.

| Variant | Tokens/s | Peak allocated GiB |
| --- | ---: | ---: |
| Compiled native | 243,793 | 13.92 |
| Fused A100 | 256,687 | 16.97 |
| Fused, ordinary RoPE | 254,214 | 16.97 |
| Fused, separate projections | 248,529 | 15.87 |
| Fused, forced FlashAttention | 256,712 | 16.97 |
| Fused, forced cuDNN attention | 250,009 | 16.97 |
| Fused, two accumulation steps | 245,653 | 9.46 |
| Compiled native repeat | 243,023 | 13.92 |

Projection fusion and fused RoPE remain enabled. Auto SDPA is retained: forcing
FlashAttention adds no benefit, and forcing cuDNN is slower. Two accumulation steps
remain an explicit memory-saving option rather than the throughput default.

A separate same-GPU kernel-scheduling comparison measured 251,928 and 251,730 tokens/s
for the original fused schedule, versus 254,152 with the A100 RMSNorm schedule.
The selected policy uses 1,024 independent groups, four rows, four warps, one stage,
and prefetching. Dispatch is restricted to width 640 and 32,768 token rows per replica
(the measured 90M microbatch); other shapes and GH200 retain their previous policies.
The forward result is unchanged and the standalone gradient relative L2 difference
was below 0.000001.

Reducing cross-entropy from 32 to eight warps improved its isolated kernel timing,
but complete updates reached only 252,161 tokens/s, too close to the control to
justify a new default. The existing 1,024-element SwiGLU block was best in its
isolated screen. Both retain their existing settings. Matrix-multiplication
autotuning is tested separately because it can add minutes to startup.

## Reproduce

Inside a one-GPU allocation, activate the environment and compare these commands.
Use distinct output directories for each trial. The local launcher starts one
worker and applies GPU-local CPU binding; `numactl` must be installed.

```bash
source scripts/environment.sh
"$DECENTRALIZED_PYTHON" -m decentralized_llm benchmark-training \
  -c configs/base.yaml -c configs/90m.yaml -c configs/a100-1gpu.yaml \
  --set launcher.kind=local --set data.cache_dir=/shared/tokens/c4 \
  --set runtime.execution_backend=native --set runtime.output_dir=runs/a100-native \
  --warmup 10 --steps 30 --windows 3 --min-seconds 3
"$DECENTRALIZED_PYTHON" -m decentralized_llm benchmark-training \
  -c configs/base.yaml -c configs/90m.yaml -c configs/a100-1gpu.yaml \
  --set launcher.kind=local --set data.cache_dir=/shared/tokens/c4 \
  --set runtime.output_dir=runs/a100-fused \
  --warmup 10 --steps 30 --windows 3 --min-seconds 3
```

For scheduler submission, use `submit --action benchmark-training`, retain the
SLURM launcher, set `launcher.partition`, and set the equivalent `benchmark.*`
configuration fields. Set the lowercase `account` environment variable as described
in the README.

The numerical script's historical name is retained; it also accepts A100 configs:

```bash
"$DECENTRALIZED_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=1 \
  --numa-binding=exclusive scripts/qualify_gh200.py \
  -c configs/base.yaml -c configs/90m.yaml -c configs/a100-1gpu.yaml \
  --set launcher.kind=local --set data.cache_dir=/shared/tokens/c4 \
  --set runtime.seed=42 --steps 100 --output runs/a100-numerical-42
```

Repeat with seed 43 and a fresh output directory. `--reference-compile` optionally
compares against compiled native instead of eager native; reports identify the
reference execution policy. Do not interchange those two comparisons when deciding
whether a numerical gate passed.

The selected backend is saved in checkpoints. Changing from native to A100 execution
requires a new run; do not relabel or overwrite an existing native checkpoint.
Compilation caches are separated by GPU capability and executable source revision,
so incompatible GH200 artifacts are not reused on A100.
