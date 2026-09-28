#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$ROOT_DIR/.julia_depot}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.mplconfig}"
export JULIA_WORKERS="${JULIA_WORKERS:-2}"

mkdir -p "$JULIA_DEPOT_PATH" "$MPLCONFIGDIR"

julia --project="$ROOT_DIR" -e 'using Pkg; Pkg.instantiate()'
julia --project="$ROOT_DIR" "$ROOT_DIR/run_parallel_rice_mele.jl"
