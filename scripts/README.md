# Scripts

Analysis and plotting scripts live under `figures/`, grouped by topic:

- `chain/`: momentum distributions, convergence metrics, effective temperatures.
- `rice_mele/`: Rice-Mele band populations and relaxation diagnostics.
- `kernels/`: bath/kernel figures.
- `spectra/`: spectral function and sum-over-momentum plots.
- `distributions/`: occupation/distribution summaries and animations.
- `stationary/`: stationary or static-reference calculations.

Run scripts from the repository root, for example:

```bash
julia --project=. scripts/figures/chain/generate_nk_t60_chain_tb05_scan_panels.jl
```
