using JLD2
using PyPlot
using Interpolations
using LinearAlgebra

rc("font", family="serif")
rc("font", size=13)
rc("axes", labelsize=15, titlesize=14)
rc("xtick", labelsize=11)
rc("ytick", labelsize=11)
rc("legend", fontsize=9)

const OUTPUT_DIR = "distributions"
mkpath(OUTPUT_DIR)

array_data(x) = hasproperty(x, :data) ? getproperty(x, :data) : x

function extract_field(name::AbstractString, key::AbstractString)
    m = match(Regex("$(key)([^_]+)"), name)
    return m === nothing ? missing : m.captures[1]
end

ks_from_L(L) = collect(range(-pi, stop=pi - 2pi / L; length=L))

function H_rice(k, t1, t2, Δ)
    dx = t1 + t2 * cos(k + pi)
    dy = t2 * sin(k + pi)
    ComplexF64[Δ / 2 dx - 1im * dy; dx + 1im * dy -Δ / 2]
end

function chain_observable(GL_data, it)
    imag.(GL_data[:, it, it])
end

function rice_observable(GL_data, it; t1=-1.0, t2=-0.8, Δ=2.0)
    _, _, L, _, _ = size(GL_data)
    ks = ks_from_L(L)
    nlower = zeros(Float64, L)
    nupper = zeros(Float64, L)
    for (ik, k) in enumerate(ks)
        _, vecs = eigen(Hermitian(H_rice(k, t1, t2, Δ)))
        rho = imag.(GL_data[:, :, ik, it, it])
        rho_band = adjoint(vecs) * rho * vecs
        nlower[ik] = real(rho_band[1, 1])
        nupper[ik] = real(rho_band[2, 2])
    end
    vcat(nlower, nupper)
end

function interpolated_rate(times::AbstractVector, vectors::AbstractMatrix; n_uniform=401)
    tu = collect(range(first(times), last(times); length=n_uniform))
    dim = size(vectors, 1)
    sampled = zeros(Float64, dim, n_uniform)
    for j in 1:dim
        itp = interpolate((times,), vectors[j, :], Gridded(Linear()))
        @inbounds for i in eachindex(tu)
            sampled[j, i] = itp(tu[i])
        end
    end

    rates = zeros(Float64, n_uniform - 1)
    tmids = (tu[1:end-1] .+ tu[2:end]) ./ 2
    @inbounds for i in 1:(n_uniform - 1)
        dt = tu[i + 1] - tu[i]
        rates[i] = norm(sampled[:, i + 1] - sampled[:, i]) / dt
    end
    return collect(tmids), rates
end

function build_time_series(mode::Symbol, GL_data, ts)
    Nt = length(ts)
    if mode == :chain
        L = size(GL_data, 1)
        vals = zeros(Float64, L, Nt)
        for it in 1:Nt
            vals[:, it] .= chain_observable(GL_data, it)
        end
        return vals
    end

    _, _, L, _, _ = size(GL_data)
    vals = zeros(Float64, 2L, Nt)
    for it in 1:Nt
        vals[:, it] .= rice_observable(GL_data, it)
    end
    return vals
end

function plot_panel_set(mode::Symbol, alphas::Vector{Float64}, outfile::String)
    fig, axs = subplots(1, length(alphas); figsize=(6.2 * length(alphas), 4.4), squeeze=false, sharey=true)
    axs = axs[1, :]

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
            values = build_time_series(mode, GL_data, ts)
            tmid, rates = interpolated_rate(ts, values)
            label = "η = $(η), p_c = $(λ)/a"
            ax.plot(tmid, rates; color=get(colors, λ, "black"), linestyle=get(styles, η, "-"),
                linewidth=2.0, label=label)
        end
        ax.set_title(raw"$\alpha = " * string(α) * raw"$")
        ax.set_xlabel(raw"$t$")
        ax.set_yscale("log")
        ax.grid(alpha=0.18)
    end

    ylabel = mode == :chain ? raw"$\|\partial_t n_k\|$" : raw"$\|\partial_t (n_{-}\oplus n_{+})\|$"
    axs[1].set_ylabel(ylabel)
    axs[end].legend(frameon=false, loc="best")
    fig.tight_layout()
    outpath = joinpath(OUTPUT_DIR, outfile)
    fig.savefig(outpath; dpi=220, bbox_inches="tight")
    close(fig)
    println("Saved " * outpath)
end

plot_panel_set(:chain, [2.5, 5.0], "chain_convergence_rate_alpha25_alpha50.png")
plot_panel_set(:rice, [2.5, 5.0], "rice_convergence_rate_alpha25_alpha50.png")
