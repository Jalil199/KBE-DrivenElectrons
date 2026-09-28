using JLD2
using PyPlot
using LaTeXStrings
using Interpolations
using FFTW

const HAS_LATEX = Sys.which("latex") !== nothing
rc("text", usetex=HAS_LATEX)
rc("font", family="serif")
if HAS_LATEX
    rc("text.latex", preamble=raw"\usepackage{amsmath}")
end
rc("font", size=18)
rc("axes", labelsize=22, titlesize=20)
rc("xtick", labelsize=18)
rc("ytick", labelsize=18)
rc("legend", fontsize=14)

const TARGET_TIMES = [30.0, 40.0, 50.0]
const OUTPUT_DIR = "distributions"
const DEFAULT_DATASET = "L100_Te1.0_Tb0.05_u0.0_γ1.0_dispersion_α0.5_s1.0_ωc10.0_linear_delta_η0.05_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q1.0_t050.0_ω03.141592653589793_σ2.0_A0.0_switch0_ti3.0_to20.0_tmax60"

gauss_window(N::Integer) = begin
    n = collect(0:N-1)
    σ = max((N - 1) / 6, 1.0)
    c = (N - 1) / 2
    @. exp(-0.5 * ((n - c) / σ)^2)
end

function ft_points(xs, ys; inverse::Bool=false)
    L = length(xs)
    dx = xs[2] - xs[1]
    xh = fftfreq(L, 2π / dx)
    phase = (inverse ? 1 / (2π) : 1.0) * dx * exp.((inverse ? -1.0 : 1.0) * im * xs[1] .* xh)
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
        is = 1 .+ rem.(Nt .+ taus, Nt)
        for (i, τ) in zip(is, taus)
            xW[i, T] = x[T + τ, T - τ]
        end
        τ = Nt ÷ 2
        if T <= Nt - τ && T >= τ + 1
            xW[τ + 1, T] = 0.5 * (x[T + τ, T - τ] + x[T - τ, T + τ])
        end
    end

    xW = fftshift(xW, 1)
    τs = ts .- reverse(ts)
    if iseven(Nt)
        τs = τs .- 0.5 * (τs[2] - τs[1])
    end

    if !fourier
        return xW, (τs, ts)
    end

    ωs = ft_points(τs, τs)[1]
    xWh = mapslices(col -> ft_points(τs, col; inverse=false)[2], xW; dims=1)
    return xWh, (ωs, ts)
end

function wigner_transform_itp(x, ts::Vector; fourier::Bool=true, ts_lin=collect(range(first(ts), last(ts); length=length(ts))), window=gauss_window(length(ts)))
    itp = interpolate((ts, ts), x, Gridded(Linear()))
    sampled = [itp(t1, t2) for t1 in ts_lin, t2 in ts_lin]
    sampled_windowed = sampled .* (window * window')
    return wigner_transform_local(sampled_windowed; ts=ts_lin, fourier=fourier)
end

nearest_time_index(ts, t_target) = argmin(abs.(ts .- t_target))
array_data(x) = hasproperty(x, :data) ? getproperty(x, :data) : x

extract_field(name::AbstractString, key::AbstractString) = begin
    m = match(Regex("$(key)([^_]+)"), name)
    m === nothing ? missing : m.captures[1]
end

function pretty_dataset_label(name::AbstractString)
    α = extract_field(name, "α")
    kernel = extract_field(name, "linear_")
    η = extract_field(name, "η")
    λ_q = extract_field(name, "λ_q")
    return "α = $(α), kernel = $(kernel), η = $(η), λ_q = $(λ_q)"
end

fermi(ω, Te) = 1 / (exp(ω / Te) + 1)

function compute_sumk(GL_data, GG_data, ts)
    L = size(GL_data, 1)
    dim_tavg = size(GL_data, 3)
    GL_Wigner_sumk = zeros(ComplexF64, dim_tavg, dim_tavg)
    A_Wigner_sumk = zeros(Float64, dim_tavg, dim_tavg)
    ωs = nothing
    tavg = nothing

    @inbounds for k in 1:L
        GL_Wigner_FFT, (ωs, tavg) = wigner_transform_itp(GL_data[k, :, :], ts; fourier=true)
        GR_Wigner_FFT, _ = wigner_transform_itp((GG_data - GL_data)[k, :, :], ts; fourier=true)
        GL_Wigner_sumk .+= GL_Wigner_FFT
        A_Wigner_sumk .+= -imag.(GR_Wigner_FFT) ./ pi
    end

    GL_Wigner_sumk ./= L
    A_Wigner_sumk ./= L
    return GL_Wigner_sumk, A_Wigner_sumk, tavg, ωs
end

function distribution_curve(GL_sumk, A_sumk, t_idx; δ=1e-6)
    A_col = real.(A_sumk[:, t_idx])
    GL_im = imag.(GL_sumk[:, t_idx]) ./ pi
    ratio = fill(NaN, length(A_col))
    mask = abs.(A_col) .> δ
    ratio[mask] .= GL_im[mask] ./ A_col[mask]
    return ratio
end

dataset = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_DATASET
mkpath(OUTPUT_DIR)

gl_path = joinpath("Data", "GL_$(dataset).jld2")
gg_path = joinpath("Data", "GG_$(dataset).jld2")
ts_path = joinpath("Data", "ts_$(dataset).jld2")

GL_raw = array_data(load(gl_path, "GL"))
GG_raw = array_data(load(gg_path, "GG"))
ts = load(ts_path, "sol").t

GL_sumk, A_sumk, tavg, ωs = compute_sumk(GL_raw, GG_raw, ts)
time_indices = [nearest_time_index(tavg, t) for t in TARGET_TIMES]
time_effective = tavg[time_indices]
curves = [distribution_curve(GL_sumk, A_sumk, idx) for idx in time_indices]

Te = parse(Float64, String(extract_field(dataset, "Te")))
fermi0 = fermi.(ωs, Te)

fig, axs = subplots(1, 2; figsize=(14, 6.5), sharey=true)
colors = ["red", "blue", "gold"]
styles = ["-", "-.", "--"]

for (i, curve) in enumerate(curves)
    axs[1].plot(ωs, curve; color=colors[i], linestyle=styles[i], linewidth=2.2,
        label="\$t_{\\mathrm{avg}} = $(round(time_effective[i], digits=2))\$")
end

axs[2].plot(ωs, curves[end]; color="blue", linewidth=2.2,
    label="\$F(\\omega,t_{\\mathrm{avg}}=$(round(time_effective[end], digits=2)))\$")
axs[2].plot(ωs, fermi0; color="black", linewidth=2.0, linestyle="--",
    label=raw"$f_{\mathrm{FD}}(\omega;T_e^{\mathrm{init}})$")

for ax in axs
    ax.set_xlabel(raw"$\omega\,(\gamma/\hbar)$")
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(frameon=false, loc="best")
end

axs[1].set_ylabel(raw"$F(\omega,t_{\mathrm{avg}})$")
axs[1].set_title("Distribution at multiple times")
axs[2].set_title("Comparison with initial Fermi")

fig.text(0.5, 0.01, pretty_dataset_label(dataset); ha="center", va="bottom", fontsize=13)
fig.tight_layout(rect=[0, 0.04, 1, 1])

out_path = joinpath(OUTPUT_DIR, "distribution_t30_t40_t50_" * dataset * ".png")
fig.savefig(out_path; dpi=300, bbox_inches="tight")
close(fig)

println("Saved $(out_path)")
