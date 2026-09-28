using JLD2
using LinearAlgebra
using Statistics
using Printf

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const DATA_DIR = joinpath(ROOT, "Data_rice")
const OUT_DIR = joinpath(ROOT, "distributions_rice")
const CACHE_DIR = joinpath(ROOT, "analysis_cache", "rice_exciton_teff_coherence_lambda_fine_even")

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

# The PRL ansatz uses a smooth selector with alpha=beta. Set
# EXCITON_THETA_FIXED to a positive value to test a fixed selector width.
const THETA_FIXED = tryparse(Float64, get(ENV, "EXCITON_THETA_FIXED", ""))

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

function fermi_centered(omega, T)
    x = clamp(omega / T, -80.0, 80.0)
    return 1 / (exp(x) + 1)
end

function theta_alpha(omega, T)
    alpha = isnothing(THETA_FIXED) ? 1 / T : THETA_FIXED
    return 0.5 * (1 - tanh(alpha * omega / 2))
end

function exciton_distribution(omega, T, mu_ex)
    θ = theta_alpha(omega, T)
    return θ * fermi_centered(omega + mu_ex, T) + (1 - θ) * fermi_centered(omega - mu_ex, T)
end

function exciton_rmse(nocc, eps, T, mu_ex)
    model = exciton_distribution.(eps, T, mu_ex)
    return sqrt(mean((nocc .- model) .^ 2))
end

function best_exciton_fit(nocc, eps)
    Tgrid = range(0.02, 2.50; length=90)
    mugrid = range(0.00, 3.00; length=151)

    best_T = first(Tgrid)
    best_mu = first(mugrid)
    best_err = Inf

    for T in Tgrid, mu_ex in mugrid
        err = exciton_rmse(nocc, eps, T, mu_ex)
        if err < best_err
            best_T = T
            best_mu = mu_ex
            best_err = err
        end
    end

    for T in range(max(0.005, best_T - 0.12), best_T + 0.12; length=61),
        mu_ex in range(max(0.0, best_mu - 0.25), best_mu + 0.25; length=61)
        err = exciton_rmse(nocc, eps, T, mu_ex)
        if err < best_err
            best_T = T
            best_mu = mu_ex
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

function band_observables_at_time(GL, Us, it)
    L = length(Us)
    nocc = zeros(Float64, 2L)
    coherence_abs = zeros(Float64, L)

    @inbounds for ik in 1:L
        rho_sub = (-1im) .* GL[:, :, ik, it, it]
        rho_band = Us[ik]' * rho_sub * Us[ik]
        nocc[ik] = real(rho_band[1, 1])
        nocc[L + ik] = real(rho_band[2, 2])
        coherence_abs[ik] = abs(rho_band[1, 2])
    end

    return nocc, mean(coherence_abs), sqrt(mean(abs2, coherence_abs))
end

function compute_case(alpha, eta, lambda_q)
    cache_path = joinpath(CACHE_DIR, "exciton_alpha$(alpha_text(alpha))_eta$(eta_text(eta))_lambda$(lambda_text(lambda_q)).jld2")
    if isfile(cache_path)
        return load(cache_path)
    end

    dataset = dataset_name(alpha, eta, lambda_q)
    gl_path = joinpath(DATA_DIR, "GL_$(dataset).jld2")
    ts_path = joinpath(DATA_DIR, "ts_$(dataset).jld2")
    isfile(gl_path) || error("Missing GL file: $(gl_path)")
    isfile(ts_path) || error("Missing ts file: $(ts_path)")

    println("Computing exciton fit/coherence alpha=$(alpha), eta=$(eta), lambda_q=$(lambda_q)")
    flush(stdout)

    GL = array_data(load(gl_path, "GL"))
    ts_obj = load(ts_path, "sol")
    ts = hasproperty(ts_obj, :t) ? collect(ts_obj.t) : collect(ts_obj)

    L = size(GL, 3)
    Us, eps = band_basis_data(L)
    teff = zeros(Float64, length(ts))
    mu_ex = zeros(Float64, length(ts))
    rmse = zeros(Float64, length(ts))
    sse = zeros(Float64, length(ts))
    coherence_mean = zeros(Float64, length(ts))
    coherence_rms = zeros(Float64, length(ts))

    for it in eachindex(ts)
        nocc, coherence_mean[it], coherence_rms[it] = band_observables_at_time(GL, Us, it)
        teff[it], mu_ex[it], rmse[it] = best_exciton_fit(nocc, eps)
        sse[it] = rmse[it]^2 * length(nocc)
    end

    jldsave(cache_path; ts, teff, mu_ex, rmse, sse, coherence_mean, coherence_rms)
    result = Dict(
        "ts" => ts,
        "teff" => teff,
        "mu_ex" => mu_ex,
        "rmse" => rmse,
        "sse" => sse,
        "coherence_mean" => coherence_mean,
        "coherence_rms" => coherence_rms,
    )

    GL = nothing
    GC.gc()
    return result
end

function style_axes!(axes)
    for ax in axes
        ax.set_xlim(0, 60)
    end
end

case_list = [(alpha, eta, lambda_q) for alpha in ALPHAS for eta in ETAS for lambda_q in LAMBDAS]
println("Precomputing $(length(case_list)) cases with $(Threads.nthreads()) Julia threads")
Threads.@threads for idx in eachindex(case_list)
    alpha, eta, lambda_q = case_list[idx]
    compute_case(alpha, eta, lambda_q)
end

using PyPlot

const HAS_LATEX = Sys.which("latex") !== nothing
rc("text", usetex=HAS_LATEX)
rc("font", family="serif", size=12)
rc("axes", linewidth=1.3, labelsize=13, titlesize=13)
rc("xtick", labelsize=11)
rc("ytick", labelsize=11)
rc("legend", fontsize=8)
rc("xtick", direction="in")
rc("ytick", direction="in")

colors = get_cmap("viridis")(range(0.08, 0.92; length=length(LAMBDAS)))
summary = String[]

fig, axes = subplots(2, 4; figsize=(15.0, 6.3), sharex=true)

for (icol, (alpha, eta)) in enumerate(Iterators.product(ALPHAS, ETAS))
    axT = axes[1, icol]
    axC = axes[2, icol]

    for (i, lambda_q) in enumerate(LAMBDAS)
        data = compute_case(alpha, eta, lambda_q)
        ts = data["ts"]
        teff = data["teff"]
        mu_ex = data["mu_ex"]
        rmse = data["rmse"]
        coherence_mean = data["coherence_mean"]

        label = raw"$\lambda_q=" * lambda_text(lambda_q) * raw"$"
        axT.plot(ts, teff; color=colors[i, :], linewidth=1.9, label=label)
        axC.plot(ts, coherence_mean; color=colors[i, :], linewidth=1.9)

        push!(summary, @sprintf("alpha=%.1f eta=%s lambda_q=%.1f T_eff(tf)=%.5f mu_ex(tf)=%.5f RMSE(tf)=%.6f Cmean(tf)=%.6e",
                                alpha, eta_text(eta), lambda_q, teff[end], mu_ex[end], rmse[end], coherence_mean[end]))
    end

    axT.axhline(TE; color="0.35", linestyle="--", linewidth=0.9)
    axT.axhline(TB; color="0.35", linestyle=":", linewidth=1.1)
    axT.set_title(raw"$\alpha=" * alpha_text(alpha) * raw",\ \eta=" * eta_text(eta) * raw"$")
    axT.set_ylim(0.0, 1.15)
    axC.set_xlabel(raw"$t$")
    axC.set_ylim(0.0, 0.065)
    icol == 1 && axT.set_ylabel(raw"$T_{\mathrm{eff}}^{\mathrm{ex}}(t)$")
    icol == 1 && axC.set_ylabel(raw"$\langle |\rho_{-+}(k,t)| \rangle_k$")
end

axes[1, 4].legend(frameon=false, loc="upper right", ncol=1)
style_axes!(vec(axes))
theta_desc = isnothing(THETA_FIXED) ? raw"$\alpha_\Theta=\beta$" : raw"$\alpha_\Theta=" * string(THETA_FIXED) * raw"$"
fig.suptitle(raw"Rice-Mele excitonic fit and interband coherence, " * theta_desc, fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.93])

out_main = joinpath(OUT_DIR, "rice_exciton_teff_coherence_lambda_fine_even_t60.png")
fig.savefig(out_main; dpi=300, bbox_inches="tight")
close(fig)

fig2, axes2 = subplots(2, 4; figsize=(15.0, 6.3), sharex=true)

for (icol, (alpha, eta)) in enumerate(Iterators.product(ALPHAS, ETAS))
    axM = axes2[1, icol]
    axE = axes2[2, icol]

    for (i, lambda_q) in enumerate(LAMBDAS)
        data = compute_case(alpha, eta, lambda_q)
        ts = data["ts"]
        axM.plot(ts, data["mu_ex"]; color=colors[i, :], linewidth=1.9, label=raw"$\lambda_q=" * lambda_text(lambda_q) * raw"$")
        axE.plot(ts, data["rmse"]; color=colors[i, :], linewidth=1.9)
    end

    axM.set_title(raw"$\alpha=" * alpha_text(alpha) * raw",\ \eta=" * eta_text(eta) * raw"$")
    axE.set_xlabel(raw"$t$")
    icol == 1 && axM.set_ylabel(raw"$\mu_{\mathrm{ex}}(t)$")
    icol == 1 && axE.set_ylabel(raw"$\mathrm{RMSE}(t)$")
end

axes2[1, 4].legend(frameon=false, loc="upper right", ncol=1)
style_axes!(vec(axes2))
fig2.suptitle(raw"Rice-Mele excitonic fit diagnostics, " * theta_desc, fontsize=14)
fig2.tight_layout(rect=[0, 0, 1, 0.93])

out_diag = joinpath(OUT_DIR, "rice_exciton_muex_rmse_lambda_fine_even_t60.png")
fig2.savefig(out_diag; dpi=300, bbox_inches="tight")
close(fig2)

summary_path = joinpath(OUT_DIR, "rice_exciton_teff_coherence_lambda_fine_even_t60_summary.txt")
open(summary_path, "w") do io
    println(io, join(summary, "\n"))
end

println("Saved $(out_main)")
println("Saved $(out_diag)")
println("Saved $(summary_path)")
println(join(summary, "\n"))
