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

const TARGET_TIME = 20.0

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

function spectral_input(GL, GG, mode::Symbol)
    if mode == :chain
        return GL, GG
    end
    return GL[1, 1, :, :, :] .+ GL[2, 2, :, :, :], GG[1, 1, :, :, :] .+ GG[2, 2, :, :, :]
end

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

function datasets_from_occupations(dir::AbstractString, prefix::AbstractString)
    pngs = sort(filter(f -> startswith(f, prefix) && endswith(f, ".png"), readdir(dir)))
    return [replace(replace(f, prefix => ""), ".png" => "") for f in pngs]
end

function output_name(dataset_name::AbstractString, outdir::AbstractString)
    joinpath(outdir, "spectrum_sumk_t20_" * dataset_name * ".png")
end

function make_figure(dataset_name::AbstractString, mode::Symbol, outdir::AbstractString)
    gl_path = joinpath("Data", "GL_$(dataset_name).jld2")
    gg_path = joinpath("Data", "GG_$(dataset_name).jld2")
    ts_path = joinpath("Data", "ts_$(dataset_name).jld2")

    GL_raw = array_data(load(gl_path, "GL"))
    GG_raw = array_data(load(gg_path, "GG"))
    ts = load(ts_path, "sol").t

    GL_data, GG_data = spectral_input(GL_raw, GG_raw, mode)
    GL_Wigner_sumk, A_Wigner_sumk, tavg, ωs = compute_sumk(GL_data, GG_data, ts)
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
    out_path = output_name(dataset_name, outdir)
    fig.savefig(out_path; dpi=300, bbox_inches="tight")
    close(fig)
    println("Saved $(out_path)")
end

mode = length(ARGS) >= 1 && ARGS[1] == "rice" ? :rice : :chain
source_dir = length(ARGS) >= 2 ? ARGS[2] : (mode == :chain ? "occupations" : "occupations_rice")
prefix = mode == :chain ? "occupation_" : "occupation_rice_"
output_dir = length(ARGS) >= 3 ? ARGS[3] : (mode == :chain ? "sepectrums_sumk" : "sepectrums_sumk_rice")
mkpath(output_dir)

dataset_names = datasets_from_occupations(source_dir, prefix)
println("Mode = $(mode), datasets = $(length(dataset_names))")

for dataset_name in dataset_names
    make_figure(dataset_name, mode, output_dir)
end
