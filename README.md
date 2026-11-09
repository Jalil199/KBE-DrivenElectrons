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
- `main.jl` — single-band chain model: kernels, self-energy updates, time evolution.
- `main-rice_mele.jl` — two-band Rice-Mele model (one or two bosonic baths, `init_type`).
- `run/` — batch launchers for parameter sweeps (`run/chain/`, `run/rice_mele/`). See `run/README.md`.
- `scripts/figures/` — analysis and figure scripts, grouped by topic. See `scripts/README.md`.
- `notebooks/` — Jupyter workflows:
  - `active/` — main exploratory notebooks (`main.ipynb`, `main-rice_mele.ipynb`, `Figure.ipynb`, ...).
  - `final_figure/` — final paper figure notebooks and `precompute_wigner.jl`.
  - `figures_electron_bath/` — clean chain/spectral figure notebooks.
  - `kubo_tests_and_figures/`, `archived/` — older tests and notebooks.
- `test/` — unit tests (`julia --project=. test/runtests.jl`).
- `legacy/` — earlier `KadanoffBaym.jl`-based code, kept for reference.

Not tracked by git (see `.gitignore`): simulation output (`Data/`, `Data_rice/`),
logs, and generated figures (`distributions*/`, `spectrums*/`, `sepectrums_sumk*/`, ...).

## Typical Workflow
1. Configure physical and numerical parameters in `main.jl` (e.g., lattice size, temperature, pulse parameters, bath coupling).
2. Run single configurations directly from Julia, or launch multi-configuration sweeps from `run/`.
3. Use the notebooks to inspect observables and reproduce analysis figures.

## Parallel Notes
The simplest entry points are:

`./run/chain/run_parallel_local.sh`

and

`./run/rice_mele/run_parallel_rice_mele_local.sh`

They create local runtime directories for the Julia depot and Matplotlib config if needed, then launch `run_parallel.jl` or `run_parallel_rice_mele.jl`.

If you want to override the worker count, for example:

`JULIA_WORKERS=4 ./run/chain/run_parallel_local.sh`

or

`JULIA_WORKERS=4 ./run/rice_mele/run_parallel_rice_mele_local.sh`

This matters in restricted environments where the default Julia depot or Matplotlib config path may be read-only.
Also, parallel jobs must produce distinct output filenames; if two workers run the exact same parameter set, they can collide while writing `Data/*.jld2`.

## Material recovered from kubo (2026-09-29)

Files that only existed on `kubo:~jalil2` and were added here without changes:

- `notebooks/final_figure/`: `precompute_wigner.jl`, `FinalFigure.ipynb`, `FinalFigure_package_version.ipynb`
  (21 May 2026): Wigner-function precomputation and the two versions of the
  final paper figure notebook (`Desktop/KBE-DrivenElectrons` and
  `Desktop/FinalFigure_package`). The precomputed `wigner_precomputed.jld2`
  (1.7 GB) stays on kubo; the notebooks read it from `../../Data/`.
- `notebooks/kubo_tests_and_figures/`: test notebooks (`KBE-Test*`, `Krylov`,
  `KBE_Orb`, `Figures_comparison`) and paper figures from `Documents/KBA`
  (April 2026), plus `F.png` and `G_lesser.pdf`.
- `legacy/bloch_electrons/`: November 2025 predecessor written with
  `KadanoffBaym.jl` (`Documents/Bloch_Electrons`).
