using JLD2
using LinearAlgebra
using Statistics
using Printf
using PyPlot

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const DATA_DIR = joinpath(ROOT, "Data_rice")
const OUT_DIR = joinpath(ROOT, "distributions_rice")
const CACHE_DIR = joinpath(ROOT, "analysis_cache", "rice_direct_relaxation_metrics_lambdaq")

mkpath(OUT_DIR)
mkpath(CACHE_DIR)

const SIGMA_X = [0.0 1.0; 1.0 0.0]
const SIGMA_Y = [0.0 -1im; 1im 0.0]
const SIGMA_Z = [1.0 0.0; 0.0 -1.0]

const T1 = -1.0
const T2 = -0.8
const DELTA = 2.0

const ALPHAS = (1.0, 2.0)
const ETAS = (0.05, 0.5)
const LAMBDAS = collect(0.1:0.1:1.5)
const THRESHOLD = 0.05

const HAS_LATEX = Sys.which("latex") !== nothing
rc("text", usetex=HAS_LATEX)
rc("font", family="serif", size=12)
rc("axes", linewidth=1.3, labelsize=13, titlesize=13)
rc("xtick", labelsize=11)
rc("ytick", labelsize=11)
rc("legend", fontsize=8)
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

function band_basis_data(L)
    ks = collect(range(-pi, stop=pi - 2pi / L, length=L))
    Us = Matrix{ComplexF64}[]
    for k in ks
        push!(Us, eigen(H_k(k)).vectors)
    end
    return Us
end

function band_occupations(GL, Us)
    L = length(Us)
    Nt = size(GL, 4)
    nminus = zeros(Float64, L, Nt)
    nplus = zeros(Float64, L, Nt)

    @inbounds for it in 1:Nt, ik in 1:L
        rho_sub = (-1im) .* GL[:, :, ik, it, it]
        rho_band = Us[ik]' * rho_sub * Us[ik]
        nminus[ik, it] = real(rho_band[1, 1])
        nplus[ik, it] = real(rho_band[2, 2])
    end

    return nminus, nplus
end

function normalized_distance_to_final(nminus, nplus)
    Nt = size(nminus, 2)
    ref = vcat(nminus[:, 1] .- nminus[:, end], nplus[:, 1] .- nplus[:, end])
    denom = norm(ref)
    R = zeros(Float64, Nt)
    if denom == 0
        return R
    end
    for it in 1:Nt
        diff = vcat(nminus[:, it] .- nminus[:, end], nplus[:, it] .- nplus[:, end])
        R[it] = norm(diff) / denom
    end
    return R
end

function normalized_nplus_distance(nplus_mean)
    denom = abs(nplus_mean[1] - nplus_mean[end])
    denom == 0 && return zeros(Float64, length(nplus_mean))
    return abs.(nplus_mean .- nplus_mean[end]) ./ denom
end

function sustained_threshold_time(ts, R; threshold=THRESHOLD)
    for i in eachindex(ts)
        if maximum(R[i:end]) <= threshold
            return ts[i]
        end
    end
    return NaN
end

function compute_case(alpha, eta, lambda_q)
    cache_path = joinpath(CACHE_DIR, "direct_relax_alpha$(alpha_text(alpha))_eta$(eta_text(eta))_lambda$(lambda_text(lambda_q)).jld2")
    if isfile(cache_path)
        return load(cache_path)
    end

    dataset = dataset_name(alpha, eta, lambda_q)
    gl_path = joinpath(DATA_DIR, "GL_$(dataset).jld2")
    ts_path = joinpath(DATA_DIR, "ts_$(dataset).jld2")
    isfile(gl_path) || error("Missing GL file: $(gl_path)")
    isfile(ts_path) || error("Missing ts file: $(ts_path)")

    println("Computing direct metrics alpha=$(alpha), eta=$(eta), lambda_q=$(lambda_q)")
    flush(stdout)

    GL = array_data(load(gl_path, "GL"))
    ts_obj = load(ts_path, "sol")
    ts = hasproperty(ts_obj, :t) ? collect(ts_obj.t) : collect(ts_obj)

    Us = band_basis_data(size(GL, 3))
    nminus, nplus = band_occupations(GL, Us)
    Rfull = normalized_distance_to_final(nminus, nplus)
    Nminus = vec(mean(nminus; dims=1))
    Nplus = vec(mean(nplus; dims=1))
    Rplus = normalized_nplus_distance(Nplus)

    t_rel_full = sustained_threshold_time(ts, Rfull)
    t_rel_plus = sustained_threshold_time(ts, Rplus)

    jldsave(cache_path; ts, Rfull, Rplus, Nminus, Nplus, t_rel_full, t_rel_plus)
    result = Dict(
        "ts" => ts,
        "Rfull" => Rfull,
        "Rplus" => Rplus,
        "Nminus" => Nminus,
        "Nplus" => Nplus,
        "t_rel_full" => t_rel_full,
        "t_rel_plus" => t_rel_plus,
    )

    GL = nothing
    GC.gc()
    return result
end

cases = [
    (1.0, 0.05, "#1f77b4", "-", raw"$\alpha=1,\eta=0.05$"),
    (1.0, 0.5, "#1f77b4", "--", raw"$\alpha=1,\eta=0.5$"),
    (2.0, 0.05, "#d1495b", "-", raw"$\alpha=2,\eta=0.05$"),
    (2.0, 0.5, "#d1495b", "--", raw"$\alpha=2,\eta=0.5$"),
]

summary = String[]
fig, axes = subplots(1, 3; figsize=(14.0, 4.3), sharex=false)

for (alpha, eta, color, ls, label) in cases
    t_rel_fulls = Float64[]
    t_rel_pluses = Float64[]
    Rfull_tf = Float64[]
    Rplus_tf = Float64[]

    for lambda_q in LAMBDAS
        data = compute_case(alpha, eta, lambda_q)
        push!(t_rel_fulls, data["t_rel_full"])
        push!(t_rel_pluses, data["t_rel_plus"])
        push!(Rfull_tf, data["Rfull"][end])
        push!(Rplus_tf, data["Rplus"][end])
        push!(summary, @sprintf("alpha=%.1f eta=%s lambda_q=%.1f t_rel_full_5pct=%.6g t_rel_Nplus_5pct=%.6g Nplus_i=%.8f Nplus_f=%.8f",
                                alpha, eta_text(eta), lambda_q, data["t_rel_full"], data["t_rel_plus"], data["Nplus"][1], data["Nplus"][end]))
    end

    axes[1].plot(LAMBDAS, t_rel_fulls; color=color, linestyle=ls, linewidth=2.2, alpha=0.9, label=label)
    axes[2].plot(LAMBDAS, t_rel_pluses; color=color, linestyle=ls, linewidth=2.2, alpha=0.9)
    axes[1].scatter(LAMBDAS, t_rel_fulls; color=color, s=34, zorder=3)
    axes[2].scatter(LAMBDAS, t_rel_pluses; color=color, s=34, zorder=3)
end

axes[1].set_ylabel(raw"$t_{\mathrm{rel}}$: full $n_\pm(k,t)$ distance $<5\%$")
axes[2].set_ylabel(raw"$t_{\mathrm{rel}}$: $N_+(t)$ distance $<5\%$")

for ax in axes[1:2]
    ax.set_xlabel(raw"$\lambda_q$")
    ax.set_xlim(0.08, 1.52)
    ax.set_ylim(0, 62)
    ax.grid(true; alpha=0.25)
end

selected_lambdas = (0.2, 0.6, 1.0, 1.4)
selected_colors = get_cmap("viridis")(range(0.12, 0.9; length=length(selected_lambdas)))
alpha_sel = 2.0
eta_sel = 0.05
for (i, lambda_q) in enumerate(selected_lambdas)
    data = compute_case(alpha_sel, eta_sel, lambda_q)
    axes[3].plot(data["ts"], data["Rfull"]; color=selected_colors[i, :], linewidth=2.0,
                 label=raw"$\lambda_q=" * lambda_text(lambda_q) * raw"$")
end
axes[3].axhline(THRESHOLD; color="0.35", linestyle=":", linewidth=1.2)
axes[3].set_xlabel(raw"$t$")
axes[3].set_ylabel(raw"$R_{\mathrm{full}}(t)$")
axes[3].set_xlim(0, 60)
axes[3].set_ylim(0, 1.05)
axes[3].set_title(raw"Example: $\alpha=2,\eta=0.05$")
axes[3].grid(true; alpha=0.25)
axes[3].legend(frameon=false, loc="upper right")

axes[1].legend(frameon=false, loc="upper right")
fig.suptitle(raw"Direct relaxation metrics from Rice-Mele band occupations", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.92])

outpath = joinpath(OUT_DIR, "rice_direct_relaxation_metrics_lambdaq_t60.png")
fig.savefig(outpath; dpi=300, bbox_inches="tight")
close(fig)

summary_path = joinpath(OUT_DIR, "rice_direct_relaxation_metrics_lambdaq_t60_summary.txt")
open(summary_path, "w") do io
    println(io, join(summary, "\n"))
end

println("Saved $(outpath)")
println("Saved $(summary_path)")
println(join(summary, "\n"))
