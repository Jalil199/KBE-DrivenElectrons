#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs

export JULIA_WORKERS=3
export JULIA_DEPOT_PATH=/tmp/julia-depot
export MPLCONFIGDIR=/tmp/mpl

exec julia --project=. run/rice_mele/run_parallel_rice_mele_gap_scan_t30.jl
