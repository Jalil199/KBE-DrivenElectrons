# KBE-DrivenElectrons

Real-time **Kadanoff–Baym equations** (KBE) for lattice electrons coupled to a
bosonic bath, written in Julia on top of
[`KadanoffBaym.jl`](https://github.com/NonequilibriumDynamics/KadanoffBaym.jl).

The code propagates the two-time Green functions $G^<(k;t,t')$ and
$G^>(k;t,t')$ of a 1D lattice model under an optional laser pulse, with a
momentum-resolved electron–boson self-energy. Its main use is to study
**relaxation and thermalization** of photoexcited (or initially inverted)
carriers: band populations, effective temperatures, spectral functions and
relaxation times.

---

## Models

| File | Model | Bands |
|---|---|---|
| `main.jl` | Single-band chain, $\epsilon_k = u - 2\gamma\cos k$ | 1 |
| `main-rice_mele.jl` | Rice–Mele chain, $H_k = \mathbf d(k)\cdot\boldsymbol\sigma$ with $d_x = t_1 + t_2\cos k$, $d_y = t_2\sin k$, $d_z = \Delta/2$ | 2 |

Both models share the same structure:

- **Drive.** Peierls substitution $k \to k - A(t)$ with a Gaussian pulse
  $A(t) = A \exp[-(t-t_0)^2/2\sigma^2] \sin(\omega_0 t)$. Set `A = 0.0` to
  study relaxation without light.
- **Bath.** Second-order (SCBA) self-energy with momentum exchange,

  ```math
  \Sigma^{\lessgtr}_{k}(t,t') = \frac{i}{L}\sum_{q} \Xi^{\lessgtr}_{q}(t,t')\, G^{\lessgtr}_{k-q}(t,t')
  ```

  The bosonic kernel $\Xi_q$ can be:
  - `bath_type = :spectral_density`: a local Ohmic-like bath $J(\omega)=\alpha|\omega|^s e^{-|\omega|/\omega_c}$;
  - `bath_type = :dispersion`: dispersive bosons $\omega_q = \omega_{b0} + v_b|q|$ (`:linear`) or $\omega_{b0} + 2v_b|\sin(q/2)|$ (`:sin_lattice`),
    with sharp modes (`boson_kernel = :delta`, damping `η`) or Lorentzian spectral functions (`boson_kernel = :spectral`).
  - Momentum weights $w_q$: `:uniform` or `:power_exp`, $|q|^{s_q} e^{-|q|/\lambda_q}$; coupling $g_q^2 = \alpha w_q$.
- **Second bath** (Rice–Mele only, `use_bath2 = true`): an extra, typically flat
  (Einstein-like) boson branch (`α2`, `ωb0_2`, `η2`, `λ_q2`, …) to provide
  interband decay.
- **Initial state.** Thermal at electron temperature `Te` (bath at `Tb`), or,
  for Rice–Mele, `init_type = :upper_full` (upper band full, lower band empty).
- **Switch-on.** `switch_on = true` turns the coupling on smoothly between `ti` and `to`.

---

## Quick start

```bash
git clone https://github.com/Jalil199/KBE-DrivenElectrons.git
cd KBE-DrivenElectrons
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### A single run

```julia
include("main-rice_mele.jl")

main(; tmax = 30,
       L = 80, t1 = -1.0, t2 = -0.8, Δ = 2.0,      # gapped, non-topological
       Te = 1.0, Tb = 0.1, A = 0.0,                # no pulse: pure relaxation
       bath_type = :dispersion, dispersion_type = :linear,
       boson_kernel = :spectral, η = 0.05,
       α = 2.0, ωb0 = 0.1, v_b = 0.2,
       wq_profile = :power_exp, s_q = 1.0, λ_q = 1.0)
```

Results are written to `Data_rice/` (Rice–Mele) or `Data/` (chain):

| File | Content |
|---|---|
| `GL_<name>.jld2` | $G^<$ on the full two-time grid |
| `GG_<name>.jld2` | $G^>$ on the full two-time grid |
| `ts_<name>.jld2` | solver output (adaptive time grid `sol.t`) |

`<name>` is built by `make_name()` and encodes every parameter, e.g.
`L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.1_dispersion_α2.0_…_tmax60`. A suffix
`_init<type>` appears for non-thermal initial states and `_b2_…` when the second
bath is on. The chain solver also accepts `save_mode = :occupations_only` to
save only the equal-time occupations.

> **Cost.** Memory and time grow as $L \times N_t^2$ (two-time storage). The full
> `GL`/`GG` files for `L = 80`, `tmax = 60` are ~1.6 GB each.

### Parameter sweeps

Sweeps use `Distributed`: each parameter set runs on its own worker. Launchers
live in `run/chain/` and `run/rice_mele/` (see [`run/README.md`](run/README.md)
for what each one scans). They can be started from any directory:

```bash
mkdir -p logs
JULIA_WORKERS=40 nohup julia --project=. \
    run/rice_mele/run_parallel_rice_mele_two_bath_lq_fine_scan_t60.jl \
    > logs/two_bath_lq_fine.log 2>&1 &
```

Set `JULIA_WORKERS` to the number of parameter sets. Existing outputs are
skipped unless `force = true` in the launcher. Two workers must never run the
same parameter set, or they will write to the same file.

### Analysis and figures

Figure and analysis scripts are in `scripts/figures/<topic>/` and write to
`outputs/` (see [`scripts/README.md`](scripts/README.md)):

```bash
julia --project=. scripts/figures/rice_mele/generate_rice_exciton_teff_coherence_lambda_fine_even.jl
```

Typical observables: band occupations $N_\pm(t)$, momentum distributions
$n_k(t)$, effective temperature (Fermi fit or excitonic ansatz), interband
coherences, $A(k,\omega)$ and $F(k,\omega)$ spectra, and relaxation-time
estimates.

---

## Repository layout

```
main.jl                 single-band chain model and solver
main-rice_mele.jl       Rice–Mele model and solver (one or two baths)
run/                    sweep launchers (chain/, rice_mele/, archived/)
scripts/figures/        analysis & figure scripts (chain, rice_mele, spectra, kernels, ...)
notebooks/
  active/               main exploratory notebooks (main.ipynb, main-rice_mele.ipynb, ...)
  final_figure/         final paper-figure notebooks + precompute_wigner.jl
  figures_electron_bath/  clean chain / spectral figure notebooks
  kubo_tests_and_figures/, archived/   older tests and notebooks
test/                   unit tests
legacy/                 earlier KadanoffBaym.jl-based code (Bloch electrons, Nov 2025)
```

Not tracked by git (see `.gitignore`):

```
Data/, Data_rice/       simulation output (.jld2)
outputs/                generated figures, spectra, occupations, analysis caches
logs/                   run logs
HANDOFF_*/              bundles copied to kubo for figure making
```

---

## Tests

```bash
julia --project=. test/runtests.jl
```

They check the KBE two-time storage helpers and the $k-q$ index map on the
$[-\pi,\pi)$ grid.

---

## Notes

- **Normalization.** The momentum weights $w_q$ are *not* normalized by
  $\sum_q w_q$; the overall bath strength is set by `α` (and `η`, `λ_q`). Older
  data (before 2026-04-22) used the summed normalization.
- **k-grid.** $k \in [-\pi, \pi)$ with $L$ points; `kmq_idx` includes the
  half-zone offset needed to map $k-q$ back onto the grid.
- **Restricted environments.** `run/*/run_parallel*_local.sh` set local
  `JULIA_DEPOT_PATH` and `MPLCONFIGDIR` for clusters where the defaults are read-only.
- **Material recovered from kubo (2026-09-29).** `notebooks/final_figure/`
  (Wigner precomputation and the final figure notebooks; the 1.7 GB
  `wigner_precomputed.jld2` stays on kubo and is read from `../../Data/`),
  `notebooks/kubo_tests_and_figures/` and `legacy/bloch_electrons/`.
