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

rc("font", family="serif", size=13)
rc("axes", labelsize=16, titlesize=15)
rc("xtick", labelsize=12)
rc("ytick", labelsize=12)
rc("legend", fontsize=11)

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

function rice_band_occupations(GL)
    _, _, L, Nt, _ = size(GL)
    ks, Us = band_basis(L)
    nminus = zeros(Float64, L, Nt)
    nplus = zeros(Float64, L, Nt)

    @inbounds for it in 1:Nt
        for ik in 1:L
            rho_sub = (-1im) .* GL[:, :, ik, it, it]
            rho_band = Us[ik]' * rho_sub * Us[ik]
            nminus[ik, it] = real(rho_band[1, 1])
            nplus[ik, it] = real(rho_band[2, 2])
        end
    end

    return ks, nminus, nplus
end

function interpolated_derivative(ts, values; n_uniform=701)
    tu = collect(range(first(ts), last(ts); length=n_uniform))
    L = size(values, 1)
    sampled = zeros(Float64, L, n_uniform)

    for ik in 1:L
        itp = interpolate((ts,), values[ik, :], Gridded(Linear()))
        @inbounds for it in eachindex(tu)
            sampled[ik, it] = itp(tu[it])
        end
    end

    dvals = zeros(Float64, L, n_uniform - 1)
    @inbounds for it in 1:(n_uniform - 1)
        dt = tu[it + 1] - tu[it]
        dvals[:, it] .= abs.(sampled[:, it + 1] .- sampled[:, it]) ./ dt
    end
    tmids = (tu[1:end-1] .+ tu[2:end]) ./ 2
    return tmids, dvals
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
    ks, nminus, nplus = rice_band_occupations(GL)
    tmids, dminus = interpolated_derivative(ts, nminus)
    _, dplus = interpolated_derivative(ts, nplus)
    return (; ks, tmids, dminus, dplus)
end

cases = [
    (0.5, compute_case(0.5)),
    (1.0, compute_case(1.0)),
]

vmax = quantile(vcat([vec(c.dminus) for (_, c) in cases]..., [vec(c.dplus) for (_, c) in cases]...), 0.995)

fig, axs = subplots(2, 2; figsize=(11.5, 7.2), sharex=true, sharey=true)

for (col, (lambda_q, data)) in enumerate(cases)
    for (row, (vals, band_label)) in enumerate(((data.dminus, raw"$|\partial_t n_-(k,t)|$"), (data.dplus, raw"$|\partial_t n_+(k,t)|$")))
        ax = axs[row, col]
        im = ax.imshow(
            vals;
            origin="lower",
            aspect="auto",
            extent=[first(data.tmids), last(data.tmids), first(data.ks), last(data.ks)],
            cmap="magma_r",
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(raw"$\lambda_q=" * string(lambda_q) * raw"$, " * band_label)
        ax.set_yticks([-pi, -pi / 2, 0, pi / 2, pi])
        ax.set_yticklabels([raw"$-\pi$", raw"$-\pi/2$", raw"$0$", raw"$\pi/2$", raw"$\pi$"])
        row == 2 && ax.set_xlabel(raw"$t$")
        col == 1 && ax.set_ylabel(raw"$k$")
    end
end

cbar = fig.colorbar(axs[1, 1].images[1], ax=axs[:]; shrink=0.92, pad=0.02)
cbar.set_label(raw"$|\partial_t n_\nu(k,t)|$")

fig.suptitle(raw"Rice-Mele band-occupation derivative heatmap, $\rho=-iG^<$, $\alpha=1,\eta=0.5$", fontsize=14)
fig.tight_layout(rect=[0, 0, 0.92, 0.94])

outpath = joinpath(OUT_DIR, "rice_derivative_heatmap_lambda_compare_alpha1_eta05_bath2_t60.png")
fig.savefig(outpath; dpi=300, bbox_inches="tight")
close(fig)
println("Saved $(outpath)")
