#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
acados_root="${repo_root}/libs/acados"

export ACADOS_SOURCE_DIR="${acados_root}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${repo_root}/.cache/uv}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${repo_root}/.cache/pycache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${repo_root}/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${repo_root}/.cache/matplotlib}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "$(uname -s)" == "Darwin" ]]; then
    export DYLD_LIBRARY_PATH="${acados_root}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
else
    export LD_LIBRARY_PATH="${acados_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec uv run "$@"
