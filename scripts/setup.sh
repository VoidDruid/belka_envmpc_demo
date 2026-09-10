#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
acados_root="${repo_root}/libs/acados"
build_dir="${acados_root}/build"

git -C "${repo_root}" submodule update --init --recursive
cmake -S "${acados_root}" -B "${build_dir}" \
    -DCMAKE_INSTALL_PREFIX="${acados_root}"
cmake --build "${build_dir}" --target install --parallel "${JOBS:-4}"
CARGO_HOME="${CARGO_HOME:-${repo_root}/.cache/cargo}" cargo build --manifest-path \
    "${acados_root}/interfaces/acados_template/tera_renderer/Cargo.toml" --release
cmake -E make_directory "${acados_root}/bin"
cmake -E copy \
    "${acados_root}/interfaces/acados_template/tera_renderer/target/release/t_renderer" \
    "${acados_root}/bin/t_renderer"
cd "${repo_root}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${repo_root}/.cache/uv}" uv sync --frozen

echo "Setup complete. Run ./scripts/run_reference.sh"
