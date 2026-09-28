using JLD2
using PyPlot
using LaTeXStrings
using Interpolations
using FFTW
using Statistics

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const DATA_DIR = joinpath(ROOT, "Data_rice")
const OUT_DIR = joinpath(ROOT, "distributions_rice")
const TARGET_TIME = 30.0

mkpath(OUT_DIR)

const HAS_LATEX = Sys.which("latex") !== nothing
rc("text", usetex=HAS_LATEX)
rc("font", family="serif")
rc("font", size=16)
rc("axes", labelsize=20, titlesize=18)
rc("xtick", labelsize=16)
rc("ytick", labelsize=16)

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

array_data(x) = hasproperty(x, :data) ? getproperty(x, :data) : x
nearest_time_index(ts, target) = argmin(abs.(ts .- target))

function compute_maps(GL_trace, GG_trace, ts)
    L = size(GL_trace, 1)
    Nt = size(GL_trace, 2)
    A_map = zeros(Float64, L, Nt)
    lesser_map = zeros(Float64, L, Nt)
    omegas = nothing
    tavg = nothing
    t_idx = nothing

    @inbounds for ik in 1:L
        GL_Wigner, (omegas, tavg) = wigner_transform_itp(GL_trace[ik, :, :], ts; fourier=true)
        GR_Wigner, _ = wigner_transform_itp((GG_trace - GL_trace)[ik, :, :], ts; fourier=true)
        t_idx = nearest_time_index(tavg, TARGET_TIME)
        A_map[ik, :] .= -imag.(GR_Wigner[:, t_idx]) ./ pi
        lesser_map[ik, :] .= real.((-1im) .* GL_Wigner[:, t_idx]) ./ pi
        if ik % 10 == 0
            println("  processed k = $(ik)/$(L)")
            flush(stdout)
        end
    end

    return A_map, lesser_map, omegas, tavg[t_idx]
end

const DATASET = "L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.1_dispersion_α1.0_s1.0_ωc3.0_linear_spectral_η0.05_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q0.2_t020.0_ω02.2_σ2.0_A0.0_switch0_b2_α1.0_ωb0_22.5_η20.1_s_q20.0_λ_q210.0_ti0.5_to5.0_tmax60"

GL_raw = array_data(load(joinpath(DATA_DIR, "GL_$(DATASET).jld2"), "GL"))
GG_raw = array_data(load(joinpath(DATA_DIR, "GG_$(DATASET).jld2"), "GG"))
ts = load(joinpath(DATA_DIR, "ts_$(DATASET).jld2"), "sol").t

GL_trace = GL_raw[1, 1, :, :, :] .+ GL_raw[2, 2, :, :, :]
GG_trace = GG_raw[1, 1, :, :, :] .+ GG_raw[2, 2, :, :, :]
A_map, lesser_map, omegas, t_eff = compute_maps(GL_trace, GG_trace, ts)

L = size(GL_trace, 1)
ks = collect(range(-pi, stop=pi - 2pi / L, length=L))
A_plot = transpose(A_map[:, end:-1:1])
lesser_plot = transpose(lesser_map[:, end:-1:1])

vmax_A = quantile(abs.(vec(A_plot)), 0.995)
vmax_L = quantile(abs.(vec(lesser_plot)), 0.995)
vmax_A = max(vmax_A, eps())
vmax_L = max(vmax_L, eps())

fig, axs = subplots(1, 2; figsize=(13.5, 5.8), sharex=true, sharey=true)
cmap = plt.get_cmap("RdBu", 3024)

imgA = axs[1].imshow(
    A_plot;
    origin="lower",
    aspect="auto",
    cmap=cmap,
    extent=[first(ks), last(ks), first(omegas), last(omegas)],
    vmin=-vmax_A,
    vmax=vmax_A,
)
imgL = axs[2].imshow(
    lesser_plot;
    origin="lower",
    aspect="auto",
    cmap=cmap,
    extent=[first(ks), last(ks), first(omegas), last(omegas)],
    vmin=-vmax_L,
    vmax=vmax_L,
)

for ax in axs
    ax.set_xlim(-pi, pi)
    ax.set_ylim(-5, 5)
    ax.set_xlabel(raw"$k$")
    ax.set_xticks([-pi, -pi / 2, 0, pi / 2, pi])
    ax.set_xticklabels([L"-\pi", L"-\pi/2", L"0", L"\pi/2", L"\pi"])
end
axs[1].set_ylabel(raw"$\omega$")
axs[1].set_title(raw"$A(k,\omega,t)$")
axs[2].set_title(raw"$-iG^<(k,\omega,t)/\pi$")

cbarA = fig.colorbar(imgA, ax=axs[1], fraction=0.046, pad=0.04)
cbarL = fig.colorbar(imgL, ax=axs[2], fraction=0.046, pad=0.04)
cbarA.set_label(raw"$A$")
cbarL.set_label(raw"$-iG^</\pi$")

fig.suptitle(raw"Rice-Mele spectra, $\eta=0.05$, $\lambda_q=0.2$, $t\simeq$" * string(round(t_eff, digits=2)), fontsize=18)
fig.tight_layout(rect=[0, 0, 1, 0.93])

outpath = joinpath(OUT_DIR, "rice_spectrum_maps_eta005_lambda02_t30_rdbu.png")
fig.savefig(outpath; dpi=300, bbox_inches="tight")
close(fig)
println("Saved $(outpath)")
