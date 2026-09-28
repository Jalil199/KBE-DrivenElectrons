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
rc("legend", fontsize=15)

const OUTPUT_DIR = "sepectrums_sumk"
const TARGET_TIME = 20.0
const DEFAULT_NAME = "L100_Te1.0_Tb0.05_u0.0_γ1.0_dispersion_α0.5_s1.0_ωc10.0_linear_delta_η0.05_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q1.0_t050.0_ω03.141592653589793_σ2.0_A0.0_switch0_ti3.0_to20.0_tmax60"

gauss_window(N::Integer) = begin
    n = collect(0:N-1)
    σ = max((N - 1) / 6, 1.0)
    center = (N - 1) / 2
    @. exp(-0.5 * ((n - center) / σ)^2)
end

function ft_points(xs, ys; inverse::Bool=false)
    @assert issorted(xs)
    L = length(xs)
    dx = xs[2] - xs[1]
    x̂s = fftfreq(L, 2π / dx)
    phase = (inverse ? 1 / (2π) : 1.0) * dx * exp.((inverse ? -1.0 : 1.0) * im * xs[1] .* x̂s)
    data = ComplexF64.(ys)
    ŷs = phase .* (inverse ? FFTW.fft(data) : FFTW.bfft(data))
    return fftshift(x̂s), fftshift(ŷs)
end

function wigner_transform_local(x::AbstractMatrix; ts=collect(1:size(x, 1)), fourier::Bool=true)
    Nt = size(x, 1)
    @assert length(ts) == Nt
    @assert let dts = diff(ts); all(z -> z ≈ dts[1], dts) end

    x_W = zero(x)
    for T in 1:Nt
        τ_max = minimum([T - 1, Nt - T, Nt ÷ 2 - 1])
        τs_idx = (-τ_max):τ_max
        is = 1 .+ rem.(Nt .+ τs_idx, Nt)
        for (i, τᵢ) in zip(is, τs_idx)
            x_W[i, T] = x[T + τᵢ, T - τᵢ]
        end
        τ = Nt ÷ 2
        if T <= Nt - τ && T >= τ + 1
            x_W[τ + 1, T] = 0.5 * (x[T + τ, T - τ] + x[T - τ, T + τ])
        end
    end

    x_W = fftshift(x_W, 1)
    τs = ts .- reverse(ts)
    if iseven(Nt)
        τs = τs .- 0.5 * (τs[2] - τs[1])
    end

    if !fourier
        return x_W, (τs, ts)
    end

    ωs = ft_points(τs, τs)[1]
    x_Ŵ = mapslices(col -> ft_points(τs, col; inverse=false)[2], x_W; dims=1)
    return x_Ŵ, (ωs, ts)
end

function wigner_transform_itp(x, ts::Vector; fourier::Bool=true, ts_lin=collect(range(first(ts), last(ts); length=length(ts))), window=gauss_window(length(ts)))
    itp = interpolate((ts, ts), x, Gridded(Linear()))
    sampled = [itp(t1, t2) for t1 in ts_lin, t2 in ts_lin]
    sampled_input = sampled .* (window * window')
    return wigner_transform_local(sampled_input; ts=ts_lin, fourier=fourier)
end

nearest_time_index(ts, t_target) = argmin(abs.(ts .- t_target))

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

array_data(x) = hasproperty(x, :data) ? getproperty(x, :data) : x

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

    return GL_Wigner_sumk, A_Wigner_sumk, tavg, ωs
end

mkpath(OUTPUT_DIR)

dataset_name = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_NAME
gl_path = joinpath("Data", "GL_$(dataset_name).jld2")
gg_path = joinpath("Data", "GG_$(dataset_name).jld2")
ts_path = joinpath("Data", "ts_$(dataset_name).jld2")

GL = array_data(load(gl_path, "GL"))
GG = array_data(load(gg_path, "GG"))
ts = load(ts_path, "sol").t

println("Computing k-integrated spectrum for $(dataset_name)")
GL_Wigner_sumk, A_Wigner_sumk, tavg, ωs = compute_sumk(GL, GG, ts)
t_idx = nearest_time_index(tavg, TARGET_TIME)
t_eff = tavg[t_idx]

A_col = real.(A_Wigner_sumk[:, t_idx])
Iless_col = real.((-1im) .* GL_Wigner_sumk[:, t_idx]) ./ π

fig, ax = subplots(figsize=(9, 6.5))
ax.plot(ωs, A_col; color="black", linewidth=2.5, label=raw"$A_{\mathrm{sum}\,k}(\omega,t_{\mathrm{avg}})$")
ax.plot(ωs, Iless_col; color="blue", linewidth=1.8, label=raw"$-iG^{<}_{\mathrm{sum}\,k}(\omega,t_{\mathrm{avg}})/\pi$")

ax.set_xlabel(raw"$\omega\,(\gamma/\hbar)$")
ax.set_ylabel(raw"$k$-integrated spectrum")
ax.set_xlim(-4, 4)
ax.legend(frameon=false, loc="best")

textbox = pretty_dataset_label(dataset_name) * "\n" * "t_avg = $(round(t_eff, digits=2))"
ax.text(
    0.03, 0.97, textbox;
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=12,
    bbox=Dict("boxstyle" => "round", "facecolor" => "white", "alpha" => 0.9, "edgecolor" => "0.7"),
)

fig.tight_layout()
out_path = joinpath(OUTPUT_DIR, "spectrum_sumk_t20_" * dataset_name * ".png")
fig.savefig(out_path; dpi=300, bbox_inches="tight")
close(fig)

println("Saved $(out_path)")
