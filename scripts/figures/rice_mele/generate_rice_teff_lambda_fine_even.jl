using JLD2
using LinearAlgebra
using Statistics
using Printf
using PyPlot

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const DATA_DIR = joinpath(ROOT, "Data_rice")
const OUT_DIR = joinpath(ROOT, "distributions_rice")
const CACHE_DIR = joinpath(ROOT, "analysis_cache", "rice_teff_lambda_fine_even")

mkpath(OUT_DIR)
mkpath(CACHE_DIR)

const SIGMA_X = [0.0 1.0; 1.0 0.0]
const SIGMA_Y = [0.0 -1im; 1im 0.0]
const SIGMA_Z = [1.0 0.0; 0.0 -1.0]

const T1 = -1.0
const T2 = -0.8
const DELTA = 2.0
const TE = 1.0
const TB = 0.1

const ALPHAS = (1.0, 2.0)
const ETAS = (0.05, 0.5)
const LAMBDAS = collect(0.2:0.2:1.4)

const HAS_LATEX = Sys.which("latex") !== nothing
rc("text", usetex=HAS_LATEX)
rc("font", family="serif", size=13)
rc("axes", linewidth=1.4, labelsize=15, titlesize=14)
rc("xtick", labelsize=12)
rc("ytick", labelsize=12)
rc("legend", fontsize=9)
rc("xtick", direction="in")
rc("ytick", direction="in")

array_data(x) = hasproperty(x, :data) ? getproperty(x, :data) : x

lambda_text(x) = @sprintf("%.1f", x)
alpha_text(x) = @sprintf("%.1f", x)
eta_text(x) = x == 0.05 ? "0.05" : @sprintf("%.1f", x)

function dataset_name(alpha, eta, lambda_q)
    return "L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.1_dispersion_α$(alpha_text(alpha))_s1.0_ωc3.0_linear_spectral_η$(eta_text(eta))_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q$(lambda_text(lambda_q))_t020.0_ω02.2_σ2.0_A0.0_switch0_b2_α1.0_ωb0_22.5_η20.1_s_q20.0_λ_q210.0_ti0.5_to5.0_tmax60"
end

function H_k(k::Float64; t1::Float64=T1, t2::Float64=T2, Δ::Float64=DELTA)
    dx = t1 + t2 * cos(k + pi)
    dy = t2 * sin(k + pi)
    dz = Δ / 2
    return SIGMA_X * dx + SIGMA_Y * dy + SIGMA_Z * dz
end

function fermi_eps(eps, T, mu)
    x = clamp((eps - mu) / T, -80.0, 80.0)
    return 1 / (exp(x) + 1)
end

function fit_rmse(nocc, eps, T, mu)
    model = fermi_eps.(eps, T, mu)
    return sqrt(mean((nocc .- model) .^ 2))
end

function best_fermi_fit(nocc, eps)
    Tgrid = range(0.02, 2.50; length=90)
    mugrid = range(-3.00, 3.00; length=151)

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

    for T in range(max(0.005, best_T - 0.12), best_T + 0.12; length=61),
        mu in range(best_mu - 0.25, best_mu + 0.25; length=61)
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

    return Us, vcat(eps_lower, eps_upper)
end

function band_occupations_at_time(GL, Us, it)
    L = length(Us)
    nocc = zeros(Float64, 2L)

    @inbounds for ik in 1:L
        rho_sub = (-1im) .* GL[:, :, ik, it, it]
        rho_band = Us[ik]' * rho_sub * Us[ik]
        nocc[ik] = real(rho_band[1, 1])
        nocc[L + ik] = real(rho_band[2, 2])
    end

    return nocc
end

function compute_case(alpha, eta, lambda_q)
    cache_path = joinpath(CACHE_DIR, "teff_alpha$(alpha_text(alpha))_eta$(eta_text(eta))_lambda$(lambda_text(lambda_q)).jld2")
    if isfile(cache_path)
        return load(cache_path)
    end

    dataset = dataset_name(alpha, eta, lambda_q)
    gl_path = joinpath(DATA_DIR, "GL_$(dataset).jld2")
    ts_path = joinpath(DATA_DIR, "ts_$(dataset).jld2")
    isfile(gl_path) || error("Missing GL file: $(gl_path)")
    isfile(ts_path) || error("Missing ts file: $(ts_path)")

    println("Computing alpha=$(alpha), eta=$(eta), lambda_q=$(lambda_q)")
    flush(stdout)

    GL = array_data(load(gl_path, "GL"))
    ts_obj = load(ts_path, "sol")
    ts = hasproperty(ts_obj, :t) ? collect(ts_obj.t) : collect(ts_obj)

    L = size(GL, 3)
    Us, eps = band_basis_data(L)
    teff = zeros(Float64, length(ts))
    mu_eff = zeros(Float64, length(ts))
    rmse = zeros(Float64, length(ts))

    for it in eachindex(ts)
        nocc = band_occupations_at_time(GL, Us, it)
        teff[it], mu_eff[it], rmse[it] = best_fermi_fit(nocc, eps)
    end

    result = Dict("ts" => ts, "teff" => teff, "mu_eff" => mu_eff, "rmse" => rmse)
    jldsave(cache_path; ts, teff, mu_eff, rmse)
    GL = nothing
    GC.gc()
    return result
end

colors = get_cmap("viridis")(range(0.08, 0.92; length=length(LAMBDAS)))
fig, axes = subplots(2, 2; figsize=(11.5, 7.4), sharex=true, sharey=true)

summary = String[]

for (row, alpha) in enumerate(ALPHAS), (col, eta) in enumerate(ETAS)
    ax = axes[row, col]
    for (i, lambda_q) in enumerate(LAMBDAS)
        data = compute_case(alpha, eta, lambda_q)
        ts = data["ts"]
        teff = data["teff"]
        rmse = data["rmse"]
        ax.plot(ts, teff; color=colors[i, :], linewidth=2.0, label=raw"$\lambda_q=" * lambda_text(lambda_q) * raw"$")
        push!(summary, @sprintf("alpha=%.1f eta=%s lambda_q=%.1f T_eff(tf)=%.5f RMSE(tf)=%.6f",
                                alpha, eta_text(eta), lambda_q, teff[end], rmse[end]))
    end
    ax.axhline(TE; color="0.35", linestyle="--", linewidth=1.0)
    ax.axhline(TB; color="0.35", linestyle=":", linewidth=1.2)
    ax.set_title(raw"$\alpha=" * alpha_text(alpha) * raw",\ \eta=" * eta_text(eta) * raw"$")
    ax.set_xlim(0, 60)
    ax.set_ylim(0.0, 1.15)
    row == 2 && ax.set_xlabel(raw"$t$")
    col == 1 && ax.set_ylabel(raw"$T_{\mathrm{eff}}(t)$")
end

axes[1, 2].legend(frameon=false, loc="upper right", ncol=1)
fig.suptitle(raw"Rice-Mele, thermal initial state; bath 2 fixed at $\omega_{b,2}=2.5,\eta_2=0.1,\lambda_{q,2}=10$", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])

outpath = joinpath(OUT_DIR, "rice_teff_lambda_fine_even_t60.png")
fig.savefig(outpath; dpi=300, bbox_inches="tight")
close(fig)

summary_path = joinpath(OUT_DIR, "rice_teff_lambda_fine_even_t60_summary.txt")
open(summary_path, "w") do io
    println(io, join(summary, "\n"))
end

println("Saved $(outpath)")
println("Saved $(summary_path)")
println(join(summary, "\n"))
