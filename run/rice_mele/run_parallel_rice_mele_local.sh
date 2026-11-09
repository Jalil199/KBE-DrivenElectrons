#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$ROOT_DIR/.julia_depot}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT_DIR/.mplconfig}"
export JULIA_WORKERS="${JULIA_WORKERS:-2}"

mkdir -p "$JULIA_DEPOT_PATH" "$MPLCONFIGDIR"

julia --project="$ROOT_DIR" -e 'using Pkg; Pkg.instantiate()'
julia --project="$ROOT_DIR" "$SCRIPT_DIR/run_parallel_rice_mele.jl"
