# Correct Rice-Mele Analysis Conventions

This note summarizes the corrected conventions used by the accompanying scripts.

## Matrix lesser Green function

For Rice-Mele, do not use `imag.(GL[:, :, ik, it, it])` before projecting to the band basis. That only works for scalar/diagonal components.

The correct density matrix is

```julia
rho_sub = (-1im) .* GL[:, :, ik, it, it]
rho_band = U' * rho_sub * U
nminus = real(rho_band[1, 1])
nplus  = real(rho_band[2, 2])
```

At `t=0`, this reproduces the Fermi initial condition at `Te=1.0` to machine precision.

## Spectrum and lesser component

For the traced orbital Green functions,

```julia
GL_trace = GL[1, 1, :, :, :] .+ GL[2, 2, :, :, :]
GG_trace = GG[1, 1, :, :, :] .+ GG[2, 2, :, :, :]
GR_trace = GG_trace .- GL_trace
```

After Wigner transforming,

```julia
A_kw      = -imag.(GR_Wigner) ./ pi
lesser_kw = real.((-1im) .* GL_Wigner) ./ pi
```

## Momentum average

For plots that compare intensive spectra, use the legal momentum average

```julia
A_avgk      = sum_k(A_kw) / L
lesser_avgk = sum_k(lesser_kw) / L
```

The older labels saying `sum_k` were ambiguous. The corrected script names and labels use `avgk` when dividing by `L`.

## Distribution function

The distribution is computed as

```julia
F_avgk = lesser_avgk ./ A_avgk
```

This ratio is not additionally normalized. For plotting, mask points where `A_avgk` is too small, otherwise the division amplifies numerical noise outside the spectral support.

## Included scripts

- `generate_integrated_dos_best_t30.jl`: computes avg-k spectrum and F near `t=30`.
- `generate_spectrum_sumk_batch.jl`: generic Wigner spectrum batch script, corrected lesser component.
- `generate_spectrum_sumk_single.jl`: generic single-case Wigner spectrum script, corrected lesser component.
- `generate_rice_teff_rmse_lambda_compare.jl`: corrected effective-temperature and Fermi-fit RMSE calculation.
