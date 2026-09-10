#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${repo_root}/scripts/with_env.sh" \
    python -m experiments.research.paper_1_external_wrench.reference_case "$@"
