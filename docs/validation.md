# Validation record

The earlier GH200 cleanup passed 135 CPU/Gloo tests, Ruff lint/format, Pyright, and
shell syntax checks. Both hardware presets pass training, benchmark, and tuning dry
runs. For the allocated multi-GPU measurements and numerical limitations, see the
[GH200 execution and benchmark record](gh200-performance.md).

The remainder of this page preserves the original CPU implementation record.

Implementation environment: Python 3.14.5, PyTorch 2.14.0+cu130, NumPy 2.5.3,
x86-64 Linux, shared Lustre filesystem. Final CPU suite: **27 passed**; CUDA tests
were excluded. Ruff lint/format and `pyright --warnings` pass. The repository's locked development tools are
Ruff 0.16.6, Pyright 1.1.414, and pytest 9.1.1.

CPU tests exercise two- and four-process Gloo DDP against a serial global-batch model,
and both AWC and ATC against sequential worker references for complete, reciprocal ring,
and reciprocal exponential mixing. They compare parameters and optimizer moments,
including accumulation and unequal rank sample counts. Distributed evaluation checks
FP32 averaging and exactly 38 validation targets without replica mutation. Resume tests
recreate process groups, change reader count/prefetch depth, compare model state with
FP32 tolerances, and require exact sampler and RNG state. Single-process resume is bitwise.

Loader/preparation tests cover all-block pass coverage, global rank striding, deterministic
pass changes, delayed/out-of-order reads, cross-shard lookahead, short reads, corruption,
I/O failure propagation, cancellation, memory rejection, preparation ordering/EOS, and
legacy version-1 cache metadata. Launcher tests cover lowercase account enforcement,
redaction from artifacts, quoted paths, typed GPU requests, node/rank mapping, dry-run
side effects, real local torchrun, injected rank failure, and preflight failure before launch.
Checkpoint tests check weights-only rejection, recipe/world-size compatibility, and
manifest-last failure handling. Compilation is exercised with PyTorch's eager compiler
backend; production Inductor/CUDA compilation remains a hardware check.

The synthetic benchmark cache contains 33,554,433 training tokens (~64 MiB) on Lustre,
with uint16 shards of 1,048,576 tokens and a final one-token shard. Its model has two
layers, width 64, 1,024 vocabulary entries, context 128, a global 4,096-token batch, and
per-rank microbatch size 8. Measurements use one CPU process, four CPU threads,
four reader threads, eight prefetched microbatches, 20 warmup updates and 100 measured
updates. These short synthetic runs test the benchmark mechanism and filesystem path;
they do not establish GPU or full-C4 acceptance.

| Cache request | Resident baseline tokens/s | Network tokens/s | Ratio | Queue starvations | p99 wait |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold client eviction request | 47,977 | 45,763 | 95.4% | 2 | 0.0171 ms |
| Warm primed stream | 50,212 | 51,529 | 102.6% | 0 | 0.0368 ms |

Both measured ratios exceed 95%. Values above 100% reflect short-run variation and are
not a speedup claim. Cold eviction used `POSIX_FADV_DONTNEED`; storage-server cache state
was not controlled. Raw reports retain the exact source digest/configuration and machine
provenance: [cold](validation/cpu-lustre-cold.json) and [warm](validation/cpu-lustre-warm.json).
Reports were generated against the final implementation source digest. The cold request
had two empty-queue events and a maximum CPU wait of 60.3 ms; the warm run had no empty-queue
events and a maximum wait of 0.111 ms. Neither run proves zero GPU starvation. The loader
budget is distinct from total process RSS.

The visible NVIDIA L40 was in **Prohibited** compute mode; allocation failed with
`CUDA_ERROR_DEVICE_UNAVAILABLE`. CUDA pinned-buffer, BF16/fused optimizer and compilation tests are provided but were not run,
and no GPU throughput, BF16, fused CUDA AdamW, NCCL, or CUDA compilation acceptance is
claimed. No SLURM jobs were submitted. Two-node ARM/NCCL smoke runs, actual C4/tokenizer
network preparation, and cold/warm GPU-versus-resident throughput still require appropriate
compute allocations. Commands and supplied smoke configurations are in the README.

## Contiguous parameter storage

Parameters now share one live allocation in every training mode. The base and each
parameter start are aligned to 256 bytes; gaps are zero-initialized. Preparation
preserves parameter identities, actual weight ties, optimizer groups, and checkpoint
names/shapes. It runs after placement and before optimizer/DDP/compiler setup. Model
copies and device/dtype conversions need preparation again before execution. Different
parameter objects with overlapping storage, mixed devices/dtypes, sparse tensors,
and noncontiguous tensors are rejected rather than silently changing their meaning.

Mixing retains a distinct snapshot and performs one bulk copy in each direction.
Autograd version counters are updated in a batched metadata call after installation.
AWC waits until backward has finished before installing the result; ATC mixes after
the optimizer step. DDP retains its existing gradient buckets. The preparation and
copy work does not change parameter names or checkpoint versions, and default
`load_state_dict` copies preserve the prepared layout. Padding is not an optimizer
parameter. A conversion temporarily needs the new allocation alongside existing
storage; steady-state live parameter memory is one allocation plus small alignment
overhead. Only decentralized mixing creates a communication snapshot.

The historical GH200 qualification hashes have not been rewritten. Fresh automatic
selection treats that evidence as stale because the storage/compiler source changed;
explicit GH200 execution and checkpoint-pinned policies retain their existing behavior.

### Checks and CPU measurements (2026-10-14)

The broad CPU/Gloo run passed 152 tests. Its one tuning-cache failure observed a source
edit during the run and passed in isolation with fixed sources. The final 30-case
storage/distributed rerun passed, including the added compiled/nonreusable-buffer case,
all mixing topologies, optimizer moments, snapshot isolation, and checkpoint resume.
Ruff lint/format and Pyright pass. CUDA tests were excluded on this host.
An isolated CPU profile of a warmed-up exchange records zero concatenate, split,
or foreach-copy operations, versus one of each in the parent implementation.

The comparison used x86-64 Python 3.14.5 / PyTorch 2.14.0+cu130 on the shared CPU host
with GPUs hidden. The training workload was `configs/benchmark-cpu.yaml`: two Gloo
ranks, one PyTorch thread per rank, 166,208 parameters, global batch 4,096 tokens,
microbatch eight, and fully resident synthetic data. Each process warmed up for 20
updates and measured three windows of at least 20 updates and one second. Four fresh
process pairs per mode used matching seeds 42–45 and balanced revision order. Timing
uses the maximum elapsed time across ranks; windows are not independent repetitions.

| Mode | Median paired throughput change | Paired range |
| --- | ---: | ---: |
| DDP | +0.00% | -2.86% to +3.07% |
| AWC | +0.47% | -0.95% to +1.51% |
| ATC | -3.57% | -5.04% to +2.78% |

These results do not establish a universal training speedup or nonregression. ATC's
median was slower in this bounded comparison. DDP was approximately unchanged and
AWC's gain was small. Median startup times were 1.82–1.96 seconds across mode/layout
combinations; per-process peak RSS ranged from 638 to 654 MiB. This workload has no
transmitted padding and adds 252 bytes to the live allocation for base alignment.

The copy probe used the actual 20M and 90M parameter shapes, one CPU thread, three
fresh-process pairs, 20 warmups, and three windows of 20 repetitions per operation.
Entries below are medians of each process's median window time. The cycle measures
snapshot plus installation together; separate operation timings are not additive
because their memory-cache conditions differ.

| Model | Snapshot, legacy/new (ms) | Installation, legacy/new (ms) | Combined cycle, legacy/new (ms) |
| --- | ---: | ---: | ---: |
| 20M | 3.012 / 2.982 | 4.859 / 4.751 | 10.445 / 9.167 |
| 90M | 15.891 / 19.200 | 26.083 / 23.280 | 41.015 / 39.240 |

Combined-cycle medians were 12.2% and 4.3% lower, respectively, but individual paired
cycle changes ranged from -27.8% to +16.6% for 20M and -6.1% to +15.5% for 90M. The
90M snapshot alone was slower. Both models have zero transmitted padding and 252
extra live-allocation bytes. Preparation took 21.2–22.8 ms for 20M and 93.1–99.6 ms
for 90M. Median process peak RSS rose from 598.1 to 637.5 MiB and from 1,143.8 to
1,230.9 MiB, respectively; these peaks include initialization and allocator retention,
not just steady-state tensors.

Local raw reports, source snapshots, benchmark drivers, and the
[machine-readable summary](../runs/parameter-storage/summary.json) are retained under
the ignored `runs/parameter-storage/` directory. An aborted helper-import attempt
produced no timing report and is excluded; its failure log is retained. The successful
training reports identify parent source digest `cbf383bb26f9c1aecb84da4483da5ea760e28376d9028e7a271d638994b7975a`
and contiguous source digest `f113ffaa95a4feb985eab2f42aa168b333960865f998cc7e453dc6db62415efc`.

### Reproduce copy measurements

The microbenchmark includes the former concatenate/foreach implementation only as a
measurement reference; production has one storage implementation. Each invocation
runs in a fresh process and records individual timing windows, setup cost, padding,
process peak RSS, CUDA allocation peaks when available, and environment/source identity.
It measures reusable buffers without collectives. Division by one preserves values
over repeated installation timings while executing normalization on both paths.

```bash
source scripts/environment.sh
for layout in legacy contiguous; do
  CUDA_VISIBLE_DEVICES='' "$DECENTRALIZED_PYTHON" scripts/benchmark_parameter_storage.py \
    -c configs/20m.yaml --set runtime.device=cpu --set runtime.amp=false \
    --set runtime.cpu_threads=1 --set distributed.backend=gloo --layout "$layout" \
    --output "runs/storage-copy-20m-$layout.json"
done
```

Repeat with `configs/90m.yaml`. For GPU measurements, run inside an allocated GPU
node, omit `CUDA_VISIBLE_DEVICES=''`, and set `runtime.device=cuda` and
`distributed.backend=nccl`. Repeat in alternating
layout order; windows within one process are not independent repetitions.

### Reproduce training comparisons

Run the following from each revision's checkout with its environment activated and
the same prepared cache. Use distinct output directories, pair the same seeds, and
alternate revision order across at least three fresh-process pairs. The parent
reference for this change is `e202dff4ca86c2ed90356c265fc46b3306ca94a8`.
These commands submit bounded jobs through the existing GH200 preset.

```bash
source scripts/environment.sh
for recipe in ddp awc atc; do
  mode=decentralized
  scheme="$recipe"
  if [ "$recipe" = ddp ]; then mode=ddp; scheme=awc; fi
  "$DECENTRALIZED_PYTHON" -m decentralized_llm benchmark-training \
    -c configs/base.yaml -c configs/20m.yaml -c configs/gh200-4gpu.yaml \
    --set data.cache_dir=/shared/tokens/c4 \
    --set distributed.mode="$mode" --set distributed.scheme="$scheme" \
    --set runtime.output_dir="runs/storage-training-$recipe" \
    --warmup 20 --steps 40 --windows 3 --min-seconds 2
done
```

Repeat with the 90M preset and, for cross-node validation, the eight-GPU preset.
Compare the maximum-rank window times, throughput, startup time, CUDA memory peaks,
and parameter-storage bytes. Profile with `--profile` in a separate diagnostic run.
The local L40 is in prohibited compute mode; GPU performance and nonregression remain
unverified until these allocation-based measurements and CUDA tests are run.
