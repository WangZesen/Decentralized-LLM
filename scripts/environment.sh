#!/usr/bin/env bash
# Source this from both login shells and each SLURM launcher task.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
architecture="$(uname -m)"
case "$architecture" in
    x86_64|aarch64) ;;
    *) echo "Unsupported architecture: $architecture" >&2; return 1 2>/dev/null || exit 1 ;;
esac
architecture_root="${DECENTRALIZED_ENV_ROOT:-$HOME/.local/share/decentralized-llm}/$architecture"
uv_executable="$architecture_root/bin/uv"
if [[ ! -x "$uv_executable" ]]; then
    uv_executable="$(command -v uv || true)"
fi
if [[ -z "$uv_executable" ]] || ! "$uv_executable" --version >/dev/null 2>&1; then
    echo "Install a native $architecture uv binary in $architecture_root/bin, then source this file again." >&2
    return 1 2>/dev/null || exit 1
fi
export UV_PROJECT_ENVIRONMENT="$project_root/.venv-$architecture"
export UV_PYTHON_INSTALL_DIR="$architecture_root/python"
export UV_CACHE_DIR="$architecture_root/cache"
export UV_LINK_MODE=copy
mkdir -p "$architecture_root"
# Shared filesystem: only one node synchronizes a given architecture's venv at a time.
(
    flock -x 9
    cd "$project_root"
    "$uv_executable" sync --locked --python 3.14
) 9>"$project_root/.uv-sync-$architecture.lock"
export DECENTRALIZED_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
