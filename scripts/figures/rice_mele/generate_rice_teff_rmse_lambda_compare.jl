using JLD2
using LinearAlgebra
using Statistics
using PyPlot

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const DATA_DIR = joinpath(ROOT, "Data_rice")
const OUT_DIR = joinpath(ROOT, "distributions_rice")

mkpath(OUT_DIR)

const SIGMA_X = [0.0 1.0; 1.0 0.0]
const SIGMA_Y = [0.0 -1im; 1im 0.0]
const SIGMA_Z = [1.0 0.0; 0.0 -1.0]

const T1 = -1.0
const T2 = -0.8
const DELTA = 2.0
const TE = 1.0
const TB = 0.1

const HAS_LATEX = Sys.which("latex") !== nothing
rc("text", usetex=HAS_LATEX)
rc("font", family="serif", size=15)
rc("axes", linewidth=1.5, labelsize=18, titlesize=16)
rc("xtick", labelsize=14)
rc("ytick", labelsize=14)
rc("legend", fontsize=12)
rc("xtick", direction="in")
rc("ytick", direction="in")

function H_k(k::Float64; t1::Float64=T1, t2::Float64=T2, Δ::Float64=DELTA)
    dx = t1 + t2 * cos(k + pi)
    dy = t2 * sin(k + pi)
    dz = Δ / 2
    return SIGMA_X * dx + SIGMA_Y * dy + SIGMA_Z * dz
end

array_data(x) = hasproperty(x, :data) ? getproperty(x, :data) : x

function fermi_eps(eps, T, mu)
    x = clamp((eps - mu) / T, -80.0, 80.0)
    return 1 / (exp(x) + 1)
end

function fit_rmse(nocc, eps, T, mu)
    model = fermi_eps.(eps, T, mu)
    return sqrt(mean((nocc .- model) .^ 2))
end

function best_fermi_fit(nocc, eps)
    Tgrid = range(0.02, 2.50; length=100)
    mugrid = range(-3.00, 3.00; length=181)

    best_T = first(Tgrid)
    best_mu = first(mugrid)
    best_err = Inf

    for T in Tgrid, mu in mugrid
        err = fit_rmse(nocc, eps, T, mu)
        if err < best_err
            best_T = T
            best_mu = mu
            best_err = err
        end
    end

    for T in range(max(0.005, best_T - 0.10), best_T + 0.10; length=81),
        mu in range(best_mu - 0.25, best_mu + 0.25; length=81)
        err = fit_rmse(nocc, eps, T, mu)
        if err < best_err
            best_T = T
            best_mu = mu
            best_err = err
        end
    end

    return best_T, best_mu, best_err
end

function band_basis_data(L)
    ks = collect(range(-pi, stop=pi - 2pi / L, length=L))
    Us = Matrix{ComplexF64}[]
    eps_lower = zeros(Float64, L)
    eps_upper = zeros(Float64, L)

    for (ik, k) in enumerate(ks)
        ev = eigen(H_k(k))
        push!(Us, ev.vectors)
        eps_lower[ik] = real(ev.values[1])
        eps_upper[ik] = real(ev.values[2])
    end

    return ks, Us, vcat(eps_lower, eps_upper)
end

function band_occupations_at_time(GL, Us, it)
    L = length(Us)
    nlower = zeros(Float64, L)
    nupper = zeros(Float64, L)

    @inbounds for ik in 1:L
        rho_sub = (-1im) .* GL[:, :, ik, it, it]
        rho_band = Us[ik]' * rho_sub * Us[ik]
        nlower[ik] = real(rho_band[1, 1])
        nupper[ik] = real(rho_band[2, 2])
    end

    return vcat(nlower, nupper)
end

function compute_teff_rmse(dataset)
    println("Loading $(dataset)")
    flush(stdout)

    GL_obj = load(joinpath(DATA_DIR, "GL_$(dataset).jld2"), "GL")
    GL = array_data(GL_obj)
    ts_obj = load(joinpath(DATA_DIR, "ts_$(dataset).jld2"), "sol")
    ts = hasproperty(ts_obj, :t) ? ts_obj.t : ts_obj

    L = size(GL, 3)
    _, Us, eps = band_basis_data(L)
    Nt = length(ts)
    teff = zeros(Float64, Nt)
    mu_eff = zeros(Float64, Nt)
    rmse = zeros(Float64, Nt)

    for it in 1:Nt
        nocc = band_occupations_at_time(GL, Us, it)
        teff[it], mu_eff[it], rmse[it] = best_fermi_fit(nocc, eps)
        if it % 100 == 0 || it == Nt
            println("  t=$(round(ts[it], digits=2))  T_eff=$(round(teff[it], digits=4))  RMSE=$(round(rmse[it], digits=5))")
            flush(stdout)
        end
    end

    GL = nothing
    GC.gc()
    return (; ts, teff, mu_eff, rmse)
end

function dataset_for(lambda_q)
    lambda_text = lambda_q == 1.0 ? "1.0" : "0.5"
    return "L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.1_dispersion_α1.0_s1.0_ωc3.0_linear_spectral_η0.5_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q$(lambda_text)_t020.0_ω02.2_σ2.0_A0.0_switch0_b2_α1.0_ωb0_22.5_η20.1_s_q20.0_λ_q210.0_ti0.5_to5.0_tmax60"
end

cases = [
    (0.5, dataset_for(0.5), "#2b8cbe"),
    (1.0, dataset_for(1.0), "#d95f0e"),
]

results = [(lambda_q, color, compute_teff_rmse(dataset)) for (lambda_q, dataset, color) in cases]

fig, axes = subplots(1, 2; figsize=(11.0, 4.4), sharex=true)

for (lambda_q, color, data) in results
    label = raw"$\lambda_q=" * string(lambda_q) * raw"$"
    axes[1].plot(data.ts, data.teff; color=color, linewidth=2.4, label=label)
    axes[2].plot(data.ts, data.rmse; color=color, linewidth=2.4, label=label)
    println("λ_q=$(lambda_q)  T_eff(tf)=$(round(data.teff[end], digits=5))  RMSE(tf)=$(round(data.rmse[end], digits=6))")
end

axes[1].axhline(TE; color="0.25", linestyle="--", linewidth=1.2, label=raw"$T_e$")
axes[1].axhline(TB; color="0.25", linestyle=":", linewidth=1.5, label=raw"$T_b$")
axes[1].set_ylabel(raw"$T_{\mathrm{eff}}(t)$")
axes[1].set_xlabel(raw"$t$")
axes[1].set_xlim(0, 60)
axes[1].set_ylim(0.0, 1.2)
axes[1].legend(frameon=false, loc="best")
axes[1].set_title(raw"Effective temperature")

axes[2].set_ylabel(raw"$\mathrm{RMSE}_{\mathrm{Fermi}}(t)$")
axes[2].set_xlabel(raw"$t$")
axes[2].set_xlim(0, 60)
axes[2].legend(frameon=false, loc="best")
axes[2].set_title(raw"Fermi-fit error")

fig.suptitle(raw"Rice-Mele, thermal initial state, $\alpha=1,\eta=0.5$; bath 2 fixed at $\omega_{b,2}=2.5,\eta_2=0.1,\lambda_{q,2}=10$", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.92])

outpath = joinpath(OUT_DIR, "rice_teff_rmse_lambda_compare_alpha1_eta05_bath2_t60.png")
fig.savefig(outpath; dpi=300, bbox_inches="tight")
close(fig)
println("Saved $(outpath)")
