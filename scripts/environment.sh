#!/usr/bin/env bash
# Source this from both login shells and each SLURM launcher task.
# Preserve the caller's shell options: strict mode breaks interactive completion.
# Check failures explicitly so activation is also safe without caller-side errexit.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || { return 1 2>/dev/null || exit 1; }
architecture="$(uname -m)" || { return 1 2>/dev/null || exit 1; }
case "$architecture" in
    x86_64|aarch64) ;;
    *) echo "Unsupported architecture: $architecture" >&2; return 1 2>/dev/null || exit 1 ;;
esac
source "$project_root/scripts/toolchain.sh" || { return 1 2>/dev/null || exit 1; }
activate_toolchain "$project_root" "$architecture" || { return 1 2>/dev/null || exit 1; }
architecture_root="${DECENTRALIZED_ENV_ROOT:-$HOME/.local/share/decentralized-llm}/$architecture"
export UV_PROJECT_ENVIRONMENT="$project_root/.venv-$architecture"
export UV_PYTHON_INSTALL_DIR="$architecture_root/python"
export UV_CACHE_DIR="$architecture_root/cache"
export UV_LINK_MODE=copy
export DECENTRALIZED_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
environment_stamp="$project_root/.uv-sync-$architecture.fingerprint"
environment_fingerprint="$(
    set -o pipefail
    sha256sum "$project_root/uv.lock" "$project_root/pyproject.toml" | sha256sum | cut -d ' ' -f 1
)" || { return 1 2>/dev/null || exit 1; }
# Fast activation needs neither network access nor a uv executable.
environment_ready() {
    [[ -x "$DECENTRALIZED_PYTHON" && -f "$environment_stamp" ]] &&
    [[ "$(cat "$environment_stamp")" == "$environment_fingerprint" ]] &&
    "$DECENTRALIZED_PYTHON" - "$project_root/pyproject.toml" "$architecture" <<'PYCHECK'
import importlib.metadata
import platform
import sys
import tomllib
from pathlib import Path
assert sys.version_info[:2] == (3, 14) and platform.machine() == sys.argv[2]
project = tomllib.loads(Path(sys.argv[1]).read_text())
for requirement in project["project"]["dependencies"] + project["dependency-groups"]["dev"]:
    name, expected = requirement.split("==")
    assert importlib.metadata.version(name).split("+")[0] == expected
PYCHECK
}
if ! environment_ready 2>/dev/null; then
    mkdir -p "$architecture_root" || { return 1 2>/dev/null || exit 1; }
    (
        flock -x 9 || exit 1
        if ! environment_ready 2>/dev/null; then
            uv_executable="$architecture_root/bin/uv"
            if [[ ! -x "$uv_executable" ]]; then
                uv_executable="$(command -v uv || true)"
            fi
            if [[ -z "$uv_executable" ]] || ! "$uv_executable" --version >/dev/null 2>&1; then
                echo "Install native $architecture uv in $architecture_root/bin before environment synchronization." >&2
                exit 1
            fi
            cd "$project_root" || exit 1
            "$uv_executable" sync --locked --python 3.14 || exit 1
            printf '%s\n' "$environment_fingerprint" > "$environment_stamp.tmp" || exit 1
            mv "$environment_stamp.tmp" "$environment_stamp" || exit 1
            environment_ready || exit 1
        fi
    ) 9>"$project_root/.uv-sync-$architecture.lock" || { return 1 2>/dev/null || exit 1; }
fi
export DECENTRALIZED_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
