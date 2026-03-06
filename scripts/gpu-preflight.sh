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
#   compose -> runs in default docker-compose setup
#   nvidia  -> runs with docker-compose.nvidia.yml overlay

MODE="${1:-local}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${MODE}" in
  local)
    cargo run --bin gpu-preflight -- "$@"
    ;;
  compose)
    docker compose run --rm flexpart-gpu cargo run --bin gpu-preflight -- "$@"
    ;;
  nvidia)
    docker compose -f docker-compose.yml -f docker-compose.nvidia.yml run --rm flexpart-gpu cargo run --bin gpu-preflight -- "$@"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Usage: scripts/gpu-preflight.sh [local|compose|nvidia] [gpu-preflight args...]" >&2
    exit 2
    ;;
esac
