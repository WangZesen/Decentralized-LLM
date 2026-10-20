# Host compiler setup

Compiled execution uses PyTorch Inductor and Triton, which can build host-side C/C++
code as well as GPU kernels. A compiler command on PATH may be a module-system wrapper;
its presence alone does not mean it works. The activation helper checks each selected
compiler with `--version`. Actual compilation is verified by a compiled benchmark.

## Configure each compute architecture

Run `uname -m` on the compute node, not just the login node. From the repository root:

```bash
cp configs/toolchain.example.sh ".toolchain-$(uname -m).sh"
```

Edit that file. The default environment supports `x86_64` and `aarch64` and reads only
its matching root settings file. These files are ignored by Git; do not commit local
installation paths. Without a matching file, startup behavior is unchanged and no
compiler checks or module loads are added.

| Setting | Meaning | Default |
| --- | --- | --- |
| `TOOLCHAIN_MODULE_INIT` | Shell file initializing the module command | Empty |
| `TOOLCHAIN_MODULE_PATHS` | Bash array of extra module search directories | `()` |
| `TOOLCHAIN_MODULES` | Bash array of compiler modules, loaded in order | `()` |
| `TOOLCHAIN_CC` | C compiler executable name or path, without arguments | Discover GCC, then Clang |
| `TOOLCHAIN_CXX` | C++ compiler executable name or path, without arguments | Discover GCC, then Clang |

All settings are optional. Nonempty existing `CC` and `CXX` independently override
file settings. Empty selections discover a working `gcc`/`clang` or `g++`/`clang++`.
Explicit selections are retained even if initially unavailable, then checked again
after module loading. Use executable paths without embedded flags; paths containing
spaces are supported when quoted in the settings file.

`scripts/environment.sh` calls `scripts/toolchain.sh` before Python synchronization
or activation, including cached activation. If both compilers work, it skips modules.
Otherwise it initializes the module command if needed, adds search paths in order,
loads modules in order, and checks compilers again (discovering any still-unselected
compiler). Failures stop environment startup with a diagnostic. Successful activation
exports `CC` and `CXX`. Repeated sourcing checks them again and skips module loading
when they still work.

Custom `launcher.environment_script` files retain control of their own setup. To opt
one in, source the helper and call `activate_toolchain /path/to/repository "$(uname -m)"`
before activating Python. Launcher environment overlays still run after the environment
script, including existing GH200 network overlays. No launcher or YAML changes are needed.

Source `scripts/environment.sh` in the login shell before running compiled commands
directly; `uv run` alone does not activate these compiler settings. The standard SLURM
launcher sources the environment separately on each compute node, selecting that node's
architecture. Existing valid Python environments can be reused without reinstalling
dependencies when only compiler settings change.

Activation preserves the caller's Bash options, including `errexit`, `nounset`, and
`pipefail`, so interactive tab completion keeps working. Setup failures return a
nonzero status; batch launchers already use strict mode to stop on these failures.
If a shell sourced an older version that enabled strict mode, run
`set +e +u; set +o pipefail` or start a fresh shell before activating again.

SLURM normally inherits exported `CC` and `CXX`. When the login and compute
architectures differ, clear login-node selections for submission so the compute node
can select its own settings, for example:

```bash
env -u CC -u CXX uv run decentralized-llm submit -c path/to/training.yaml
```

Retain those variables when intentionally overriding the compute-node compilers.

## Examples

Module-based settings (replace placeholders using your site's documentation):

```bash
TOOLCHAIN_MODULE_INIT='/path/to/modules/init/bash'
TOOLCHAIN_MODULE_PATHS=('/path/to/compiler/modulefiles')
TOOLCHAIN_MODULES=('compiler-family/version' 'compiler-tools/version')
TOOLCHAIN_CC='gcc'
TOOLCHAIN_CXX='g++'
```

Explicit compilers with no module setup:

```bash
TOOLCHAIN_CC='/path/to/compiler/bin/clang'
TOOLCHAIN_CXX='/path/to/compiler/bin/clang++'
```

Use `type module`, `module avail`, `module list`, and (where supported) `module spider`
to inspect the compute-node environment. `module show NAME` describes a candidate
module. Consult site documentation or support for the initialization script and extra
module search paths; login and compute nodes may differ. Do not assume another site's
module names or copy an entire login environment into the settings file.

## Verify and troubleshoot

On an allocated GPU node, from the repository root:

```bash
source scripts/environment.sh
"$CC" --version
"$CXX" --version
"$DECENTRALIZED_PYTHON" -c 'import platform, torch; print(platform.machine(), torch.cuda.is_available())'
```

For a short single-GPU compiled benchmark, first prepare the shared cache as described
in the README, then run:

```bash
"$DECENTRALIZED_PYTHON" -m decentralized_llm benchmark-training \
  -c configs/base.yaml -c configs/models/90m.yaml \
  --set data.cache_dir=/shared/tokens/c4 \
  --set distributed.gpus_per_node=1 --set runtime.execution_backend=native \
  --set runtime.compile=true --set training.num_accumulation=4 \
  --set runtime.output_dir=runs/toolchain-check \
  --warmup 2 --steps 3 --windows 1 --min-seconds 0.1
```

Compilation may make the first call slow. Use a scheduler allocation appropriate to
your site's GPU policy; the benchmark runs on the current node. Reduce the microbatch
by increasing `training.num_accumulation` to a divisor of the batch's sequence count
if memory is insufficient. A passing `--version` check does not prove the compiler's
headers, libraries, or compatibility with the installed CUDA/PyTorch stack; inspect
the benchmark traceback if compilation fails.

If the module command is unavailable, set its initialization file. If a module cannot
be found, verify its name and search paths on the compute node. If a compiler remains
unusable, inspect `CC`/`CXX`: stale environment overrides take precedence. Unset them
to use the local settings. Module scripts must also work with caller-enabled Bash
strict mode, as used by SLURM launchers. A working preexisting compiler intentionally skips all
configured module loads; use explicit compiler paths if a particular version is needed.

Retry a failed training run in a **fresh output directory**, preserving the failed
run's diagnostics and configuration. For example, use its saved configuration and
change the destination before submitting:

```bash
"$DECENTRALIZED_PYTHON" -m decentralized_llm submit -c runs/failed/config.yaml \
  --set runtime.output_dir=runs/90m-1gpu-retry
```

Set the lowercase `account` environment variable for SLURM as described in the README.
A compiler-startup failure is not a resumable checkpoint; use `--resume` only with an
actual committed training checkpoint.
