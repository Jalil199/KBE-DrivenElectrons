#!/usr/bin/env bash
set -euo pipefail

cd /home/jalil/jalil_codes/Projects_2026/KBE-DrivenElectrons
mkdir -p logs

export JULIA_WORKERS=36
export JULIA_DEPOT_PATH=/tmp/julia-depot
export MPLCONFIGDIR=/tmp/mpl

exec julia --project=. run_parallel_rice_mele_best_flat_bath_with_bath1_scan_t60.jl
