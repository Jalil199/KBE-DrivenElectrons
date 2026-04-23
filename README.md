# KBE-DrivenElectrons

KBE-DrivenElectrons provides Julia implementations for studying **nonequilibrium electron dynamics** under external driving fields within the Kadanoff–Baym equation (KBE) framework. The project focuses on real-time Green-function evolution for lattice electrons coupled to a bosonic environment, enabling analysis of transient and driven steady-state behavior.

## Scope
The codebase includes:
- a momentum-space tight-binding electron model with time-dependent driving,
- bath kernels derived from spectral-density and Bose/Fermi occupation functions,
- self-energy construction and momentum-space convolutions for KBE propagation,
- distributed execution utilities for parameter sweeps,
- notebook workflows for post-processing, visualization, and figure generation.

## Repository Structure
- `main.jl` — core model definitions, kernels, self-energy updates, and time-evolution routines.
- `run_parallel.jl` — distributed run script for scanning multiple lattice sizes.
- `main.ipynb`, `Figure.ipynb`, `fig.ipynb`, `main-rice_mele.ipynb` — exploratory analysis and plotting workflows.

## Typical Workflow
1. Configure physical and numerical parameters in `main.jl` (e.g., lattice size, temperature, pulse parameters, bath coupling).
2. Run single configurations directly from Julia, or launch multi-configuration sweeps through `run_parallel.jl`.
3. Use the notebooks to inspect observables and reproduce analysis figures.

## Parallel Notes
The simplest entry points are:

`./run_parallel_local.sh`

and

`./run_parallel_rice_mele_local.sh`

They create local runtime directories for the Julia depot and Matplotlib config if needed, then launch `run_parallel.jl` or `run_parallel_rice_mele.jl`.

If you want to override the worker count, for example:

`JULIA_WORKERS=4 ./run_parallel_local.sh`

or

`JULIA_WORKERS=4 ./run_parallel_rice_mele_local.sh`

This matters in restricted environments where the default Julia depot or Matplotlib config path may be read-only.
Also, parallel jobs must produce distinct output filenames; if two workers run the exact same parameter set, they can collide while writing `Data/*.jld2`.
