using JLD2
using LinearAlgebra
using Statistics
using Printf
using PyPlot

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const DATA_DIR = joinpath(ROOT, "Data_rice")
const OUT_DIR = joinpath(ROOT, "distributions_rice")
const CACHE_DIR = joinpath(ROOT, "analysis_cache", "rice_velocity_lambda_fine_even")

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
    occ = zeros(Float64, 2L, Nt)

    @inbounds for it in 1:Nt, ik in 1:L
        rho_sub = (-1im) .* GL[:, :, ik, it, it]
        rho_band = Us[ik]' * rho_sub * Us[ik]
        occ[ik, it] = real(rho_band[1, 1])
        occ[L + ik, it] = real(rho_band[2, 2])
    end

    return occ
end

function velocity_metrics(ts, occ)
    Nt = length(ts)
    dmax = zeros(Float64, Nt)
    d95 = zeros(Float64, Nt)

    for it in 1:Nt
        if it == 1
            dt = ts[2] - ts[1]
            deriv = abs.((occ[:, 2] .- occ[:, 1]) ./ dt)
        elseif it == Nt
            dt = ts[Nt] - ts[Nt - 1]
            deriv = abs.((occ[:, Nt] .- occ[:, Nt - 1]) ./ dt)
        else
            dt = ts[it + 1] - ts[it - 1]
            deriv = abs.((occ[:, it + 1] .- occ[:, it - 1]) ./ dt)
        end
        dmax[it] = maximum(deriv)
        d95[it] = quantile(vec(deriv), 0.95)
    end

    return dmax, d95
end

function compute_case(alpha, eta, lambda_q)
    cache_path = joinpath(CACHE_DIR, "velocity_alpha$(alpha_text(alpha))_eta$(eta_text(eta))_lambda$(lambda_text(lambda_q)).jld2")
    if isfile(cache_path)
        return load(cache_path)
    end

    dataset = dataset_name(alpha, eta, lambda_q)
    gl_path = joinpath(DATA_DIR, "GL_$(dataset).jld2")
    ts_path = joinpath(DATA_DIR, "ts_$(dataset).jld2")
    isfile(gl_path) || error("Missing GL file: $(gl_path)")
    isfile(ts_path) || error("Missing ts file: $(ts_path)")

    println("Computing velocity alpha=$(alpha), eta=$(eta), lambda_q=$(lambda_q)")
    flush(stdout)

    GL = array_data(load(gl_path, "GL"))
    ts_obj = load(ts_path, "sol")
    ts = hasproperty(ts_obj, :t) ? collect(ts_obj.t) : collect(ts_obj)

    Us = band_basis_data(size(GL, 3))
    occ = band_occupations(GL, Us)
    dmax, d95 = velocity_metrics(ts, occ)

    jldsave(cache_path; ts, dmax, d95)
    result = Dict("ts" => ts, "dmax" => dmax, "d95" => d95)
    GL = nothing
    GC.gc()
    return result
end

colors = get_cmap("viridis")(range(0.08, 0.92; length=length(LAMBDAS)))
summary = String[]

fig, axes = subplots(2, 4; figsize=(15.0, 6.3), sharex=true, sharey="row")

for (icol, (alpha, eta)) in enumerate(Iterators.product(ALPHAS, ETAS))
    axD = axes[1, icol]
    axP = axes[2, icol]

    for (i, lambda_q) in enumerate(LAMBDAS)
        data = compute_case(alpha, eta, lambda_q)
        ts = data["ts"]
        dmax = data["dmax"]
        d95 = data["d95"]
        label = raw"$\lambda_q=" * lambda_text(lambda_q) * raw"$"

        axD.plot(ts, dmax; color=colors[i, :], linewidth=1.9, label=label)
        axP.plot(ts, d95; color=colors[i, :], linewidth=1.9)
        push!(summary, @sprintf("alpha=%.1f eta=%s lambda_q=%.1f Dmax(tf)=%.6e D95(tf)=%.6e",
                                alpha, eta_text(eta), lambda_q, dmax[end], d95[end]))
    end

    axD.set_title(raw"$\alpha=" * alpha_text(alpha) * raw",\ \eta=" * eta_text(eta) * raw"$")
    axD.set_xlim(0, 60)
    axP.set_xlim(0, 60)
    axD.set_ylim(0.0, 0.025)
    axP.set_ylim(0.0, 0.025)
    axP.set_xlabel(raw"$t$")
    icol == 1 && axD.set_ylabel(raw"$D_{\max}(t)=\max_{k,b} |\partial_t n_b(k,t)|$")
    icol == 1 && axP.set_ylabel(raw"$D_{95}(t)$")
end

axes[1, 4].legend(frameon=false, loc="upper right", ncol=1)
fig.suptitle(raw"Rice-Mele occupation velocities in band basis", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.93])

outpath = joinpath(OUT_DIR, "rice_velocity_lambda_fine_even_t60.png")
fig.savefig(outpath; dpi=300, bbox_inches="tight")
close(fig)

summary_path = joinpath(OUT_DIR, "rice_velocity_lambda_fine_even_t60_summary.txt")
open(summary_path, "w") do io
    println(io, join(summary, "\n"))
end

println("Saved $(outpath)")
println("Saved $(summary_path)")
println(join(summary, "\n"))
