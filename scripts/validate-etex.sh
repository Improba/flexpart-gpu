#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GPU_COMPOSE_FILE="${PROJECT_ROOT}/docker/docker-compose.yml"
GPU_NVIDIA_COMPOSE_FILE="${PROJECT_ROOT}/docker/docker-compose.nvidia.yml"

cd "${PROJECT_ROOT}"

# ETEX validation wrapper.
# Usage:
#   scripts/validate-etex.sh local [-- etex-validation args...]
#   scripts/validate-etex.sh compose [-- etex-validation args...]
#   scripts/validate-etex.sh nvidia [-- etex-validation args...]
#
# If no mode is provided, defaults to compose.

MODE="${1:-compose}"
if [[ $# -gt 0 ]]; then
  shift
fi

DEFAULT_ARGS=(
  --fixture "fixtures/etex/scaffold/validation_fixture.json"
  --mode fixture-only
  --output-dir "target/validation/etex-fixture-only"
  --print-json
)

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ $# -gt 0 ]]; then
  USER_ARGS=("$@")
else
  USER_ARGS=("${DEFAULT_ARGS[@]}")
fi

case "${MODE}" in
  local)
    cargo run --bin etex-validation -- "${USER_ARGS[@]}"
    ;;
  compose)
    docker compose -f "${GPU_COMPOSE_FILE}" run --rm flexpart-gpu cargo run --bin etex-validation -- "${USER_ARGS[@]}"
    ;;
  nvidia)
    docker compose -f "${GPU_COMPOSE_FILE}" -f "${GPU_NVIDIA_COMPOSE_FILE}" run --rm flexpart-gpu cargo run --bin etex-validation -- "${USER_ARGS[@]}"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Usage: scripts/validate-etex.sh [local|compose|nvidia] [-- etex-validation args...]" >&2
    exit 2
    ;;
esac
