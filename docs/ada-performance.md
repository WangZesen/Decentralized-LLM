# Local Ada 20M packed benchmark

Measured on 2026-11-05 using one NVIDIA RTX 5000 Ada Generation (SM89), driver
595.91.07, Python 3.14.7, PyTorch 2.14.0+cu130, and Triton 3.8.0.

The explicit A100 kernel path works for this bounded Ada experiment. Most of the
throughput improvement comes from compilation; the fused A100 path adds a smaller
gain and reduces peak allocated memory further.

| Execution | Global tokens/s | Time/update | Peak allocated memory | Speedup over eager |
| --- | ---: | ---: | ---: | ---: |
| Native eager | 109,000 | 150.31 ms | 10.95 GiB | 1.00x |
| Native compiled | 180,377 | 90.83 ms | 5.44 GiB | 1.65x |
| A100 compiled on Ada | 185,390 | 88.38 ms | 3.67 GiB | 1.70x |

The A100 path is approximately 2.8% faster than compiled native. It enables
projection/RoPE fusions, Triton AdamW, and complete-update CUDA graph capture.
Memory numbers are PyTorch peak allocated memory during training updates; reserved
memory and allocations outside PyTorch are separate metrics in the raw results.

## Workload and measurement

- 20,403,520 parameters per logical worker; eight packed decentralized workers on GPU 0.
- Reciprocal ring topology, AWC AdamW, no error feedback, BF16, SDPA.
- Context length 128; global batch 128 sequences / 16,384 tokens; accumulation 1.
- Same prepared C4 cache, seed, model/optimizer settings, and optimizer/kernel source fingerprint.
- Ten warmup updates, then three windows of 40 updates each per process.
- Native eager measured once; compiled native and A100 each measured twice, with their
  order reversed for the repeat. Table throughput is the median over all timed windows.
- Every window was valid: no compilation or resident-cache reads during measurement.

Observed startup, including warmup, was 4.17 seconds for eager, 16.53 then 5.65 seconds
for compiled native, and 18.10 then 8.18 seconds for A100. These are observed startup
times with the local compilation cache, not isolated compiler timing. Startup is
excluded from the throughput table.

The measured native-compiled run medians were 180,379 and 180,360 tokens/s; A100 run
medians were 185,531 and 185,248 tokens/s.

## Reproduce

Prepare the full-run cache as described in the local training instructions, then run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run decentralized-llm benchmark-training --no-local-config \
  -c configs/base.yaml -c configs/models/20m.yaml -c configs/local-packed-8-20tpp.yaml \
  -c configs/setups/ada-20m-experimental.yaml \
  --set runtime.output_dir=runs/ada-a100-benchmark \
  --warmup 10 --steps 40 --windows 3 --min-seconds 2 --max-chunks 2
```

For compiled native, replace the Ada overlay with
`--set runtime.execution_backend=native --set runtime.compile=true` and select a fresh
output directory. For eager native, set `runtime.compile=false`.

The same Ada overlay can be layered after a local training recipe to train with
compilation and the A100 kernels. It changes execution only; the recipe determines
worker count, batching, data, and training budget.

## Scope and checks

Ada selection is explicit and restricted to SM89 with the 20M preset, nondeterministic
BF16 SDPA, and compilation. Execution metadata identifies the experimental Ada path.
Automatic selection and the other hardware requirements retain their existing rules.
This benchmark does not qualify other model sizes or establish full-training convergence.

Validation passed: 22 CUDA fused-forward/gradient and captured-update parity cases,
one production compiled checkpoint/resume/evaluation test, and 100 CPU policy/startup/
preset tests. Ruff and Pyright checks also passed.

Local raw artifacts are under `runs/ada-packed-benchmark/` (Git-ignored), including
`summary.json`, each variant's `benchmark.json`, and the test logs. The summary verifies
the same GPU, cache, model, distributed configuration, training configuration, optimizer
settings, and optimizer/kernel source fingerprint across the five measurements.
