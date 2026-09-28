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

function compute_lesser_map(GL_trace, ts)
    L = size(GL_trace, 1)
    Nt = size(GL_trace, 2)
    lesser_map = zeros(Float64, L, Nt)
    omegas = nothing
    tavg = nothing

    @inbounds for ik in 1:L
        GL_Wigner, (omegas, tavg) = wigner_transform_itp(GL_trace[ik, :, :], ts; fourier=true)
        t_idx = nearest_time_index(tavg, TARGET_TIME)
        lesser_map[ik, :] .= real.((-1im) .* GL_Wigner[:, t_idx]) ./ pi
        if ik % 10 == 0
            println("  processed k = $(ik)/$(L)")
            flush(stdout)
        end
    end

    return lesser_map, omegas, tavg
end

const DATASET = "L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.1_dispersion_α1.0_s1.0_ωc3.0_linear_spectral_η0.5_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q0.2_t020.0_ω02.2_σ2.0_A0.0_switch0_b2_α1.0_ωb0_22.5_η20.1_s_q20.0_λ_q210.0_ti0.5_to5.0_tmax60"

GL_raw = array_data(load(joinpath(DATA_DIR, "GL_$(DATASET).jld2"), "GL"))
ts = load(joinpath(DATA_DIR, "ts_$(DATASET).jld2"), "sol").t

GL_trace = GL_raw[1, 1, :, :, :] .+ GL_raw[2, 2, :, :, :]
lesser_map, omegas, tavg = compute_lesser_map(GL_trace, ts)
t_eff = tavg[nearest_time_index(tavg, TARGET_TIME)]

L = size(GL_trace, 1)
ks = collect(range(-pi, stop=pi - 2pi / L, length=L))

plot_data = transpose(lesser_map[:, end:-1:1])
vmax = quantile(vec(plot_data[isfinite.(plot_data)]), 0.995)
vmax = max(vmax, eps())

fig, ax = subplots(figsize=(8.2, 6.2))
img = ax.imshow(
    plot_data;
    origin="lower",
    aspect="auto",
    cmap="magma",
    extent=[first(ks), last(ks), first(omegas), last(omegas)],
    vmin=0,
    vmax=vmax,
)

ax.set_xlim(-pi, pi)
ax.set_ylim(-5, 5)
ax.set_xlabel(raw"$k$")
ax.set_ylabel(raw"$\omega$")
ax.set_xticks([-pi, -pi / 2, 0, pi / 2, pi])
ax.set_xticklabels([L"-\pi", L"-\pi/2", L"0", L"\pi/2", L"\pi"])
ax.set_title(raw"Rice-Mele $-iG^<(k,\omega,t)/\pi$, $\lambda_q=0.2$, $t\simeq$" * string(round(t_eff, digits=2)))

cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(raw"$-iG^</\pi$")

fig.tight_layout()
outpath = joinpath(OUT_DIR, "rice_lesser_spectrum_map_lambda02_t30.png")
fig.savefig(outpath; dpi=300, bbox_inches="tight")
close(fig)
println("Saved $(outpath)")
