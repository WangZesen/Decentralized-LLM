# Validation record

The current GH200 cleanup passes 135 CPU/Gloo tests, Ruff lint/format, Pyright, and
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
