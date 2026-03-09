#!/usr/bin/env bash
set -euo pipefail

# GPU preflight wrapper.
# Usage:
#   scripts/gpu-preflight.sh local [--backend <value>] [--no-smoke]
#   scripts/gpu-preflight.sh compose [--backend <value>] [--no-smoke]
#   scripts/gpu-preflight.sh nvidia [--backend <value>] [--no-smoke]
#
# Modes:
#   local   -> runs directly in the current environment
#   compose -> runs in default docker compose setup
#   nvidia  -> runs with NVIDIA overlay compose file

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_COMPOSE_FILE="${PROJECT_ROOT}/docker/docker-compose.yml"
GPU_NVIDIA_COMPOSE_FILE="${PROJECT_ROOT}/docker/docker-compose.nvidia.yml"

cd "${PROJECT_ROOT}"

MODE="${1:-local}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${MODE}" in
  local)
    cargo run --bin gpu-preflight -- "$@"
    ;;
  compose)
    docker compose -f "${GPU_COMPOSE_FILE}" run --rm flexpart-gpu cargo run --bin gpu-preflight -- "$@"
    ;;
  nvidia)
    docker compose -f "${GPU_COMPOSE_FILE}" -f "${GPU_NVIDIA_COMPOSE_FILE}" run --rm flexpart-gpu cargo run --bin gpu-preflight -- "$@"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Usage: scripts/gpu-preflight.sh [local|compose|nvidia] [gpu-preflight args...]" >&2
    exit 2
    ;;
esac
