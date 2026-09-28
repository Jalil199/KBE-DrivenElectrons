using JLD2
using LinearAlgebra
using Statistics
using Interpolations
using PyPlot

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const DATA_DIR = joinpath(ROOT, "Data_rice")
const OUT_DIR = joinpath(ROOT, "distributions_rice")
mkpath(OUT_DIR)

const SIGMA_X = [0.0 1.0; 1.0 0.0]
const SIGMA_Y = [0.0 -1im; 1im 0.0]
const SIGMA_Z = [1.0 0.0; 0.0 -1.0]

rc("font", family="serif", size=14)
rc("axes", labelsize=17, titlesize=16)
rc("xtick", labelsize=13)
rc("ytick", labelsize=13)
rc("legend", fontsize=12)

array_data(x) = hasproperty(x, :data) ? getproperty(x, :data) : x

function H_k(k; t1=-1.0, t2=-0.8, Δ=2.0)
    dx = t1 + t2 * cos(k + pi)
    dy = t2 * sin(k + pi)
    dz = Δ / 2
    return SIGMA_X * dx + SIGMA_Y * dy + SIGMA_Z * dz
end

function band_basis(L)
    ks = collect(range(-pi, stop=pi - 2pi / L, length=L))
    Us = [eigen(H_k(k)).vectors for k in ks]
    return ks, Us
end

function rice_band_timeseries(GL)
    _, _, L, Nt, _ = size(GL)
    _, Us = band_basis(L)
    values = zeros(Float64, 2L, Nt)

    @inbounds for it in 1:Nt
        for ik in 1:L
            rho_sub = (-1im) .* GL[:, :, ik, it, it]
            rho_band = Us[ik]' * rho_sub * Us[ik]
            values[ik, it] = real(rho_band[1, 1])
            values[L + ik, it] = real(rho_band[2, 2])
        end
    end

    return values
end

function interpolated_pointwise_rates(ts, values; n_uniform=601)
    tu = collect(range(first(ts), last(ts); length=n_uniform))
    dim = size(values, 1)
    sampled = zeros(Float64, dim, n_uniform)

    for j in 1:dim
        itp = interpolate((ts,), values[j, :], Gridded(Linear()))
        @inbounds for i in eachindex(tu)
            sampled[j, i] = itp(tu[i])
        end
    end

    pointwise = zeros(Float64, dim, n_uniform - 1)
    @inbounds for i in 1:(n_uniform - 1)
        dt = tu[i + 1] - tu[i]
        pointwise[:, i] .= abs.(sampled[:, i + 1] .- sampled[:, i]) ./ dt
    end
    tmids = (tu[1:end-1] .+ tu[2:end]) ./ 2
    return tmids, pointwise
end

function dataset_for(lambda_q)
    lambda_text = lambda_q == 1.0 ? "1.0" : "0.5"
    return "L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.1_dispersion_α1.0_s1.0_ωc3.0_linear_spectral_η0.5_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q$(lambda_text)_t020.0_ω02.2_σ2.0_A0.0_switch0_b2_α1.0_ωb0_22.5_η20.1_s_q20.0_λ_q210.0_ti0.5_to5.0_tmax60"
end

function compute_case(lambda_q)
    dataset = dataset_for(lambda_q)
    println("Loading $(dataset)")
    GL_obj = load(joinpath(DATA_DIR, "GL_$(dataset).jld2"), "GL")
    GL = array_data(GL_obj)
    ts = load(joinpath(DATA_DIR, "ts_$(dataset).jld2"), "sol").t
    values = rice_band_timeseries(GL)
    tmid, pointwise = interpolated_pointwise_rates(ts, values)
    dmax = vec(maximum(pointwise; dims=1))
    d95 = [quantile(view(pointwise, :, i), 0.95) for i in axes(pointwise, 2)]
    return (; tmid, dmax, d95)
end

cases = [
    (0.5, "#2b8cbe"),
    (1.0, "#d95f0e"),
]

results = [(lambda_q, color, compute_case(lambda_q)) for (lambda_q, color) in cases]

fig, axes = subplots(1, 2; figsize=(11.0, 4.2), sharex=true)

for (lambda_q, color, data) in results
    label = raw"$\lambda_q=" * string(lambda_q) * raw"$"
    axes[1].plot(data.tmid, data.dmax; color=color, linewidth=2.3, label=label)
    axes[2].plot(data.tmid, data.d95; color=color, linewidth=2.3, label=label)
end

axes[1].set_title(raw"Maximum band-occupation rate")
axes[1].set_ylabel(raw"$D_{\max}(t)=\max_{k,\nu}|\partial_t n_\nu(k,t)|$")
axes[1].set_xlabel(raw"$t$")
axes[1].set_xlim(0, 60)
axes[1].grid(alpha=0.18)
axes[1].legend(frameon=false, loc="best")

axes[2].set_title(raw"95th percentile rate")
axes[2].set_ylabel(raw"$D_{95}(t)$")
axes[2].set_xlabel(raw"$t$")
axes[2].set_xlim(0, 60)
axes[2].grid(alpha=0.18)
axes[2].legend(frameon=false, loc="best")

fig.suptitle(raw"Rice-Mele stationarity metric, $\rho=-iG^<$, thermal initial state, $\alpha=1,\eta=0.5$", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.92])

outpath = joinpath(OUT_DIR, "rice_derivative_metric_lambda_compare_alpha1_eta05_bath2_t60.png")
fig.savefig(outpath; dpi=300, bbox_inches="tight")
close(fig)
println("Saved $(outpath)")
