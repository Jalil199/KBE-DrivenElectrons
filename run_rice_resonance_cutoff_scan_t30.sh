#!/usr/bin/env bash
set -euo pipefail

cd /home/jalil/jalil_codes/Projects_2026/KBE-DrivenElectrons
mkdir -p logs

export JULIA_WORKERS=15
export JULIA_DEPOT_PATH=/tmp/julia-depot
export MPLCONFIGDIR=/tmp/mpl

exec julia --project=. run_parallel_rice_mele_resonance_cutoff_scan_t30.jl
