using JLD2
using PyPlot
using LaTeXStrings
using Interpolations
using FFTW

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const DATA_DIR = joinpath(ROOT, "Data_rice")
const OUT_DIR = joinpath(ROOT, "distributions_rice")
const TARGET_TIME = 30.0
const DOS_MASK_FRACTION = 0.03
const DOS_LOOSE_MASK_FRACTION = 0.003
const NORMALIZE_BY_L = true

mkpath(OUT_DIR)

const HAS_LATEX = Sys.which("latex") !== nothing
rc("text", usetex=HAS_LATEX)
rc("font", family="serif")
if HAS_LATEX
    rc("text.latex", preamble=raw"\usepackage{amsmath}")
end
rc("font", size=15)
rc("axes", labelsize=18, titlesize=17)
rc("xtick", labelsize=14)
rc("ytick", labelsize=14)
rc("legend", fontsize=12)

gauss_window(N::Integer) = begin
    n = collect(0:N-1)
    sigma = max((N - 1) / 6, 1.0)
    center = (N - 1) / 2
    @. exp(-0.5 * ((n - center) / sigma)^2)
end

function ft_points(xs, ys; inverse::Bool=false)
    L = length(xs)
    dx = xs[2] - xs[1]
    xh = fftfreq(L, 2pi / dx)
    phase = (inverse ? 1 / (2pi) : 1.0) * dx * exp.((inverse ? -1.0 : 1.0) * im * xs[1] .* xh)
    data = ComplexF64.(ys)
    yh = phase .* (inverse ? FFTW.fft(data) : FFTW.bfft(data))
    return fftshift(xh), fftshift(yh)
end

function wigner_transform_local(x::AbstractMatrix; ts=collect(1:size(x, 1)), fourier::Bool=true)
    Nt = size(x, 1)
    xW = zero(x)
    for T in 1:Nt
        taumax = minimum([T - 1, Nt - T, Nt ÷ 2 - 1])
        taus = (-taumax):taumax
        indices = 1 .+ rem.(Nt .+ taus, Nt)
        for (i, tau) in zip(indices, taus)
            xW[i, T] = x[T + tau, T - tau]
        end
        tau = Nt ÷ 2
        if T <= Nt - tau && T >= tau + 1
            xW[tau + 1, T] = 0.5 * (x[T + tau, T - tau] + x[T - tau, T + tau])
        end
    end

    xW = fftshift(xW, 1)
    taus = ts .- reverse(ts)
    if iseven(Nt)
        taus = taus .- 0.5 * (taus[2] - taus[1])
    end

    if !fourier
        return xW, (taus, ts)
    end

    omegas = ft_points(taus, taus)[1]
    xWh = mapslices(col -> ft_points(taus, col; inverse=false)[2], xW; dims=1)
    return xWh, (omegas, ts)
end

function wigner_transform_itp(x, ts::Vector; fourier::Bool=true, ts_lin=collect(range(first(ts), last(ts); length=length(ts))), window=gauss_window(length(ts)))
    itp = interpolate((ts, ts), x, Gridded(Linear()))
    sampled = [itp(t1, t2) for t1 in ts_lin, t2 in ts_lin]
    sampled_windowed = sampled .* (window * window')
    return wigner_transform_local(sampled_windowed; ts=ts_lin, fourier=fourier)
end

nearest_time_index(ts, t_target) = argmin(abs.(ts .- t_target))

array_data(x) = hasproperty(x, :data) ? getproperty(x, :data) : x

function rice_trace(GL, GG)
    return GL[1, 1, :, :, :] .+ GL[2, 2, :, :, :],
           GG[1, 1, :, :, :] .+ GG[2, 2, :, :, :]
end

function compute_integrated_dos(GL_data, GG_data, ts)
    L = size(GL_data, 1)
    dim_tavg = size(GL_data, 3)
    GL_Wigner_sumk = zeros(ComplexF64, dim_tavg, dim_tavg)
    A_Wigner_sumk = zeros(Float64, dim_tavg, dim_tavg)
    omegas = nothing
    tavg = nothing

    @inbounds for k in 1:L
        GL_Wigner_FFT, (omegas, tavg) = wigner_transform_itp(GL_data[k, :, :], ts; fourier=true)
        GR_Wigner_FFT, _ = wigner_transform_itp((GG_data - GL_data)[k, :, :], ts; fourier=true)
        GL_Wigner_sumk .+= GL_Wigner_FFT
        A_Wigner_sumk .+= -imag.(GR_Wigner_FFT) ./ pi
        if k % 10 == 0
            println("  processed k = $(k)/$(L)")
            flush(stdout)
        end
    end

    if NORMALIZE_BY_L
        GL_Wigner_sumk ./= L
        A_Wigner_sumk ./= L
    end
    return GL_Wigner_sumk, A_Wigner_sumk, tavg, omegas
end

function load_case(dataset)
    gl_path = joinpath(DATA_DIR, "GL_$(dataset).jld2")
    gg_path = joinpath(DATA_DIR, "GG_$(dataset).jld2")
    ts_path = joinpath(DATA_DIR, "ts_$(dataset).jld2")

    GL_raw = array_data(load(gl_path, "GL"))
    GG_raw = array_data(load(gg_path, "GG"))
    ts = load(ts_path, "sol").t
    GL_data, GG_data = rice_trace(GL_raw, GG_raw)
    return GL_data, GG_data, ts
end

function compute_case(dataset)
    println("Loading $(dataset)")
    flush(stdout)
    GL_data, GG_data, ts = load_case(dataset)
    GL_Wigner_sumk, A_Wigner_sumk, tavg, omegas = compute_integrated_dos(GL_data, GG_data, ts)
    t_idx = nearest_time_index(tavg, TARGET_TIME)
    t_eff = tavg[t_idx]
    result = (
        omegas = omegas,
        t_eff = t_eff,
        A = real.(A_Wigner_sumk[:, t_idx]),
        lesser = real.((-1im) .* GL_Wigner_sumk[:, t_idx]) ./ pi,
    )
    GL_data = nothing
    GG_data = nothing
    GL_Wigner_sumk = nothing
    A_Wigner_sumk = nothing
    GC.gc()
    return result
end

function integrated_distribution(A, lesser; mask_fraction=DOS_MASK_FRACTION)
    threshold = mask_fraction * maximum(abs.(A))
    F = fill(NaN, length(A))
    mask = abs.(A) .> threshold
    F[mask] .= lesser[mask] ./ A[mask]
    return F, threshold
end

const BEST_THERMAL = "L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.1_dispersion_α1.0_s1.0_ωc3.0_linear_spectral_η0.5_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q0.2_t020.0_ω02.2_σ2.0_A0.0_switch0_b2_α1.0_ωb0_22.5_η20.1_s_q20.0_λ_q210.0_ti0.5_to5.0_tmax60"
const BEST_UPPER = "L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.1_dispersion_α1.0_s1.0_ωc3.0_linear_spectral_η0.5_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q0.2_t020.0_ω02.2_σ2.0_A0.0_switch0_initupper_full_b2_α1.0_ωb0_22.5_η20.1_s_q20.0_λ_q210.0_ti0.5_to5.0_tmax60"

cases = [
    ("thermal", BEST_THERMAL),
    ("upper full", BEST_UPPER),
]

results = [(label, compute_case(dataset)) for (label, dataset) in cases]

fig, axes = subplots(1, 2; figsize=(12.0, 4.8), sharex=true, sharey=true)

for (ax, (label, result)) in zip(axes, results)
    ax.plot(result.omegas, result.A; color="black", linewidth=2.2, label=raw"$A_{\mathrm{avg}\,k}(\omega,t)$")
    ax.plot(result.omegas, result.lesser; color="#1f77b4", linewidth=1.8, label=raw"$-iG^<_{\mathrm{avg}\,k}(\omega,t)/\pi$")
    ax.axhline(0.0; color="0.75", linewidth=0.8)
    ax.set_xlim(-5, 5)
    ax.set_title("$(label), t = $(round(result.t_eff, digits=2))")
    ax.set_xlabel(raw"$\omega$")
end

axes[1].set_ylabel(raw"$k$-averaged spectral weight")
axes[1].legend(frameon=false, loc="upper left")
fig.suptitle(raw"Rice-Mele $k$-averaged DOS, best bath set: $\alpha=1,\eta=0.5,\lambda_q=0.2$; bath 2 $\omega_b=2.5,\eta_2=0.1,\lambda_{q,2}=10$", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])

out_path = joinpath(OUT_DIR, "rice_avgk_spectrum_best_bath2_t30.png")
fig.savefig(out_path; dpi=300, bbox_inches="tight")
close(fig)
println("Saved $(out_path)")

figF, axesF = subplots(1, 2; figsize=(12.0, 4.8), sharex=true, sharey=true)

for (ax, (label, result)) in zip(axesF, results)
    F, threshold = integrated_distribution(result.A, result.lesser)
    ax.plot(result.omegas, F; color="#8c2d04", linewidth=2.0)
    ax.axhline(0.0; color="0.8", linewidth=0.8)
    ax.axhline(1.0; color="0.8", linewidth=0.8)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("$(label), t = $(round(result.t_eff, digits=2))")
    ax.set_xlabel(raw"$\omega$")
    ax.text(
        0.03, 0.94,
        raw"$A_{\mathrm{int}} > $" * string(round(threshold, sigdigits=3));
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=Dict("boxstyle" => "round", "facecolor" => "white", "alpha" => 0.85, "edgecolor" => "0.8"),
    )
end

axesF[1].set_ylabel(raw"$F_{\mathrm{avg}\,k}(\omega,t)=(-iG^<_{\mathrm{avg}\,k}/\pi)/A_{\mathrm{avg}\,k}$")
figF.suptitle(raw"Rice-Mele integrated distribution near $t=30$", fontsize=15)
figF.tight_layout(rect=[0, 0, 1, 0.93])

out_dist_path = joinpath(OUT_DIR, "rice_avgk_distribution_best_bath2_t30.png")
figF.savefig(out_dist_path; dpi=300, bbox_inches="tight")
close(figF)
println("Saved $(out_dist_path)")

figReadable, axesReadable = subplots(1, 2; figsize=(12.0, 4.8), sharex=true, sharey=true)

for (ax, (label, result)) in zip(axesReadable, results)
    F, threshold = integrated_distribution(result.A, result.lesser; mask_fraction=DOS_LOOSE_MASK_FRACTION)
    A_norm = result.A ./ maximum(abs.(result.A))
    lesser_norm = result.lesser ./ maximum(abs.(result.A))

    ax.fill_between(result.omegas, 0, A_norm; color="0.88", alpha=0.9, linewidth=0.0, label=raw"$A_{\mathrm{avg}\,k}$ norm.")
    ax.plot(result.omegas, lesser_norm; color="#6baed6", linewidth=1.5, alpha=0.8, label=raw"$-iG^<_{\mathrm{avg}\,k}$ norm.")
    ax.plot(result.omegas, F; color="#8c2d04", linewidth=2.2, label=raw"$F_{\mathrm{avg}\,k}$")
    ax.axhline(0.0; color="0.72", linewidth=0.8)
    ax.axhline(1.0; color="0.72", linewidth=0.8)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.15, 1.15)
    ax.set_title("$(label), t = $(round(result.t_eff, digits=2))")
    ax.set_xlabel(raw"$\omega$")
    ax.text(
        0.03, 0.94,
        raw"$A_{\mathrm{int}} > $" * string(round(threshold, sigdigits=3));
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=Dict("boxstyle" => "round", "facecolor" => "white", "alpha" => 0.85, "edgecolor" => "0.8"),
    )
end

axesReadable[1].set_ylabel(raw"$F_{\mathrm{avg}\,k}$ with normalized spectral background")
axesReadable[1].legend(frameon=false, loc="center right")
figReadable.suptitle(raw"Rice-Mele integrated distribution on the spectral support, $t\simeq 30$", fontsize=15)
figReadable.tight_layout(rect=[0, 0, 1, 0.93])

out_readable_path = joinpath(OUT_DIR, "rice_avgk_distribution_best_bath2_t30_readable.png")
figReadable.savefig(out_readable_path; dpi=300, bbox_inches="tight")
close(figReadable)
println("Saved $(out_readable_path)")

figNatural, axesNatural = subplots(1, 2; figsize=(12.0, 4.8), sharex=true)

for (ax, (label, result)) in zip(axesNatural, results)
    F, threshold = integrated_distribution(result.A, result.lesser; mask_fraction=DOS_LOOSE_MASK_FRACTION)
    axF = ax.twinx()

    ax.plot(result.omegas, result.A; color="black", linewidth=2.0, label=raw"$A_{\mathrm{avg}\,k}$")
    ax.plot(result.omegas, result.lesser; color="#1f77b4", linewidth=1.8, label=raw"$-iG^<_{\mathrm{avg}\,k}/\pi$")
    axF.plot(result.omegas, F; color="#8c2d04", linewidth=2.0, label=raw"$F_{\mathrm{avg}\,k}$")

    ax.axhline(0.0; color="0.8", linewidth=0.8)
    axF.axhline(0.0; color="0.75", linewidth=0.8)
    axF.axhline(1.0; color="0.75", linewidth=0.8)

    ax.set_xlim(-5, 5)
    axF.set_ylim(-0.15, 1.15)
    ax.set_title("$(label), t = $(round(result.t_eff, digits=2))")
    ax.set_xlabel(raw"$\omega$")
    ax.text(
        0.03, 0.94,
        raw"$A_{\mathrm{avg}\,k} > $" * string(round(threshold, sigdigits=3));
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=Dict("boxstyle" => "round", "facecolor" => "white", "alpha" => 0.85, "edgecolor" => "0.8"),
    )

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = axF.get_legend_handles_labels()
    ax.legend(vcat(lines1, lines2), vcat(labels1, labels2); frameon=false, loc="upper right", fontsize=10)
    axF.tick_params(axis="y", colors="#8c2d04")
    axF.spines["right"].set_color("#8c2d04")
end

axesNatural[1].set_ylabel(raw"$A_{\mathrm{avg}\,k},\ -iG^<_{\mathrm{avg}\,k}/\pi$")
axesNatural[2].set_ylabel(raw"$A_{\mathrm{avg}\,k},\ -iG^<_{\mathrm{avg}\,k}/\pi$")
figNatural.text(0.985, 0.5, raw"$F_{\mathrm{avg}\,k}=(-iG^<_{\mathrm{avg}\,k}/\pi)/A_{\mathrm{avg}\,k}$", rotation=-90, va="center", ha="right", color="#8c2d04", fontsize=16)
figNatural.suptitle(raw"Rice-Mele natural $k$-averaged spectrum and distribution, $t\simeq30$", fontsize=15)
figNatural.tight_layout(rect=[0, 0, 0.97, 0.93])

out_natural_path = joinpath(OUT_DIR, "rice_avgk_distribution_best_bath2_t30_natural.png")
figNatural.savefig(out_natural_path; dpi=300, bbox_inches="tight")
close(figNatural)
println("Saved $(out_natural_path)")
