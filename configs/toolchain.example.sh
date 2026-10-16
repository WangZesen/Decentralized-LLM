# Copy to .toolchain-x86_64.sh or .toolchain-aarch64.sh at the repository root.
# Optional settings only; scripts/toolchain.sh performs activation.
TOOLCHAIN_MODULE_INIT=''
TOOLCHAIN_MODULE_PATHS=()
TOOLCHAIN_MODULES=()
# Executable names or paths, without arguments. Nonempty CC/CXX take precedence.
# Empty values discover GCC first, then Clang.
TOOLCHAIN_CC=''
TOOLCHAIN_CXX=''
