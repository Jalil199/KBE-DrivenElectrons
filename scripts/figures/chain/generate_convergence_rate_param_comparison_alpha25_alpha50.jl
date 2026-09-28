using JLD2
using PyPlot
using Interpolations
using LinearAlgebra

rc("font", family="serif")
rc("font", size=13)
rc("axes", labelsize=15, titlesize=14)
rc("xtick", labelsize=11)
rc("ytick", labelsize=11)
rc("legend", fontsize=8)

const OUTPUT_DIR = "distributions"
mkpath(OUTPUT_DIR)

array_data(x) = hasproperty(x, :data) ? getproperty(x, :data) : x

function extract_field(name::AbstractString, key::AbstractString)
    m = match(Regex("$(key)([^_]+)"), name)
    return m === nothing ? missing : m.captures[1]
end

function interpolated_pointwise_rates(times::AbstractVector, values::AbstractMatrix; n_uniform=401)
    tu = collect(range(first(times), last(times); length=n_uniform))
    dim = size(values, 1)
    sampled = zeros(Float64, dim, n_uniform)
    for j in 1:dim
        itp = interpolate((times,), values[j, :], Gridded(Linear()))
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

function chain_timeseries(GL_data)
    L, Nt, _ = size(GL_data)
    nk_t = zeros(Float64, L, Nt)
    @inbounds for it in 1:Nt
        nk_t[:, it] .= imag.(GL_data[:, it, it])
    end
    nk_t
end

ks_from_L(L) = collect(range(-pi, stop=pi - 2pi / L; length=L))

function H_rice(k, t1, t2, Δ)
    dx = t1 + t2 * cos(k + pi)
    dy = t2 * sin(k + pi)
    ComplexF64[Δ / 2 dx - 1im * dy; dx + 1im * dy -Δ / 2]
end

function rice_timeseries(GL_data; t1=-1.0, t2=-0.8, Δ=2.0)
    _, _, L, Nt, _ = size(GL_data)
    ks = ks_from_L(L)
    vals = zeros(Float64, 2L, Nt)
    for it in 1:Nt
        for (ik, k) in enumerate(ks)
            _, vecs = eigen(Hermitian(H_rice(k, t1, t2, Δ)))
            rho = imag.(GL_data[:, :, ik, it, it])
            rho_band = adjoint(vecs) * rho * vecs
            vals[ik, it] = real(rho_band[1, 1])
            vals[L + ik, it] = real(rho_band[2, 2])
        end
    end
    vals
end

function plot_param_subset(mode::Symbol, outfile::String)
    fig, axs = subplots(1, 2; figsize=(12.4, 4.6), squeeze=false, sharex=true, sharey=true)
    axs = axs[1, :]
    alphas = [2.5, 5.0]
    prefix = mode == :chain ? "GL_L100_Te1.0_Tb0.1" : "GL_L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.1"
    colors = Dict("0.2" => "#1f77b4", "0.5" => "#2ca02c", "1.0" => "#d62728")
    styles = Dict("0.05" => "-", "0.5" => "--")

    for (ax, α) in zip(axs, alphas)
        files = sort(filter(f -> occursin(prefix, f) && occursin("_α$(α)_", f), readdir("Data"; join=true)))
        for f in files
            name = basename(f)
            η = String(extract_field(name, "η"))
            λ = String(extract_field(name, "λ_q"))
            GL_data = array_data(load(f, "GL"))
            ts = load(replace(f, "GL_" => "ts_"), "sol").t
            values = mode == :chain ? chain_timeseries(GL_data) : rice_timeseries(GL_data)
            tmid, pointwise = interpolated_pointwise_rates(ts, values)
            dmax = vec(maximum(pointwise; dims=1))
            label = "η = $(η), p_c = $(λ)/a"
            ax.plot(tmid, dmax; color=get(colors, λ, "black"), linestyle=get(styles, η, "-"),
                linewidth=2.0, label=label)
        end
        ax.set_title(raw"$\alpha = " * string(α) * raw"$")
        ax.set_xlabel(raw"$t$")
        ax.grid(alpha=0.18)
    end

    ylabel = mode == :chain ? raw"$D_{\max}(t)$" : raw"$D_{\max}^{\mathrm{bands}}(t)$"
    axs[1].set_ylabel(ylabel)
    axs[2].legend(frameon=false, loc="best")
    fig.tight_layout()
    outpath = joinpath(OUTPUT_DIR, outfile)
    fig.savefig(outpath; dpi=220, bbox_inches="tight")
    close(fig)
    println("Saved " * outpath)
end

plot_param_subset(:chain, "chain_convergence_rate_param_scan_alpha25_alpha50.png")
plot_param_subset(:rice, "rice_convergence_rate_param_scan_alpha25_alpha50.png")
