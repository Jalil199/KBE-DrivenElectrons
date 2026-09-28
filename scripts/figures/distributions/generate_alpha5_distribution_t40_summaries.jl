using JLD2
using PyPlot
using Interpolations
using FFTW

rc("font", family="serif")
rc("font", size=13)
rc("axes", labelsize=15, titlesize=14)
rc("xtick", labelsize=11)
rc("ytick", labelsize=11)
rc("legend", fontsize=9)

const OUTPUT_DIR = "distributions"
mkpath(OUTPUT_DIR)

gauss_window(N::Integer) = begin
    n = collect(0:N-1)
    sigma = max((N - 1) / 6, 1.0)
    center = (N - 1) / 2
    @. exp(-0.5 * ((n - center) / sigma)^2)
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
        for (i, tau) in zip(is, taus)
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

function extract_field(name::AbstractString, key::AbstractString)
    m = match(Regex("$(key)([^_]+)"), name)
    return m === nothing ? missing : m.captures[1]
end

fermi(omega, Te) = 1 / (exp(omega / Te) + 1)

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
    omegas = nothing
    tavg = nothing

    @inbounds for k in 1:L
        GL_Wigner_FFT, (omegas, tavg) = wigner_transform_itp(GL_data[k, :, :], ts; fourier=true)
        GR_Wigner_FFT, _ = wigner_transform_itp((GG_data - GL_data)[k, :, :], ts; fourier=true)
        GL_Wigner_sumk .+= GL_Wigner_FFT
        A_Wigner_sumk .+= -imag.(GR_Wigner_FFT) ./ pi
    end

    GL_Wigner_sumk ./= L
    A_Wigner_sumk ./= L
    return GL_Wigner_sumk, A_Wigner_sumk, tavg, omegas
end

function distribution_curve(GL_sumk, A_sumk, t_idx; delta=1e-6)
    A_col = real.(A_sumk[:, t_idx])
    GL_im = imag.(GL_sumk[:, t_idx]) ./ pi
    ratio = fill(NaN, length(A_col))
    mask = abs.(A_col) .> delta
    ratio[mask] .= GL_im[mask] ./ A_col[mask]
    return ratio
end

function plot_summary(mode::Symbol, files::Vector{String}, target_time::Float64, outfile::String)
    n = length(files)
    n == 0 && error("No files provided for $(mode)")
    ncols = min(3, n)
    nrows = cld(n, ncols)
    fig, axs = subplots(nrows, ncols; figsize=(4.8ncols, 3.8nrows), squeeze=false, sharex=true, sharey=true)

    for (ax, f) in zip(axs[:], files)
        name = basename(f)
        dataset = replace(replace(name, "GL_" => ""), ".jld2" => "")
        GL_raw = array_data(load(f, "GL"))
        GG_raw = array_data(load(joinpath("Data", "GG_" * dataset * ".jld2"), "GG"))
        ts = load(joinpath("Data", "ts_" * dataset * ".jld2"), "sol").t
        GL_data, GG_data = spectral_input(GL_raw, GG_raw, mode)
        GL_sumk, A_sumk, tavg, omegas = compute_sumk(GL_data, GG_data, ts)
        t_idx = nearest_time_index(tavg, target_time)
        curve = distribution_curve(GL_sumk, A_sumk, t_idx)
        Te = parse(Float64, String(extract_field(dataset, "Te")))
        eta = extract_field(dataset, "η")
        lambda_q = extract_field(dataset, "λ_q")

        ax.plot(omegas, curve; color="#2c7fb8", linewidth=2.0, label="F(ω)")
        ax.plot(omegas, fermi.(omegas, Te); color="black", linestyle="--", linewidth=1.6, label="Fermi init")
        ax.set_title("η = $(eta), p_c = $(lambda_q)/a")
        ax.text(0.04, 0.08, "t ≈ $(round(tavg[t_idx], digits=2))"; transform=ax.transAxes, fontsize=10)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(alpha=0.16)
    end

    for ax in axs[:][n+1:end]
        ax.axis("off")
    end

    axs[1, 1].legend(frameon=false, loc="best")
    for ax in axs[end, :]
        ax.set_xlabel(raw"$\omega$")
    end
    for ax in axs[:, 1]
        ax.set_ylabel(raw"$F(\omega)$")
    end
    fig.tight_layout()
    outpath = joinpath(OUTPUT_DIR, outfile)
    fig.savefig(outpath; dpi=220, bbox_inches="tight")
    close(fig)
    println("Saved " * outpath)
end

chain_files = sort(filter(f -> occursin("GL_L100_Te1.0_Tb0.1", f) && occursin("_α5.0_", f), readdir("Data"; join=true)))
rice_files = sort(filter(f -> occursin("GL_L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.1", f) && occursin("_α5.0_", f), readdir("Data"; join=true)))

plot_summary(:chain, chain_files, 40.0, "chain_alpha50_distributions_t40.png")
plot_summary(:rice, rice_files, 40.0, "rice_alpha50_distributions_t40.png")
