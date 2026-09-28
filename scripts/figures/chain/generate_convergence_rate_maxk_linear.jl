using JLD2
using PyPlot
using Interpolations
using Statistics

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

function choose_representative(files::Vector{String})
    for tag in [("_η0.5_", "_λ_q1.0_"), ("_η0.5_", "_λ_q0.5_"), ("_η0.05_", "_λ_q1.0_")]
        idx = findfirst(f -> occursin(tag[1], f) && occursin(tag[2], f), files)
        idx !== nothing && return files[idx]
    end
    return files[end]
end

function plot_maxk_linear(alphas::Vector{Float64}, outfile::String)
    fig, axs = subplots(1, length(alphas); figsize=(6.2 * length(alphas), 4.4), squeeze=false, sharey=true)
    axs = axs[1, :]

    prefix = "GL_L100_Te1.0_Tb0.1"
    for (ax, α) in zip(axs, alphas)
        files = sort(filter(f -> occursin(prefix, f) && occursin("_α$(α)_", f), readdir("Data"; join=true)))
        isempty(files) && continue
        f = choose_representative(files)
        name = basename(f)
        η = String(extract_field(name, "η"))
        λ = String(extract_field(name, "λ_q"))
        GL_data = array_data(load(f, "GL"))
        ts = load(replace(f, "GL_" => "ts_"), "sol").t
        nk_t = chain_timeseries(GL_data)
        tmid, pointwise = interpolated_pointwise_rates(ts, nk_t)
        dmax = vec(maximum(pointwise; dims=1))
        d95 = [quantile(view(pointwise, :, i), 0.95) for i in axes(pointwise, 2)]

        ax.plot(tmid, dmax; color="#d62728", linewidth=2.1, label=raw"$D_{\max}(t)$")
        ax.plot(tmid, d95; color="#1f77b4", linewidth=2.0, linestyle="--", label=raw"$D_{95}(t)$")
        ax.set_title(raw"$\alpha = " * string(α) * raw"$" * "\nη = $(η), p_c = $(λ)/a")
        ax.set_xlabel(raw"$t$")
        ax.grid(alpha=0.18)
    end

    axs[1].set_ylabel(raw"$|\partial_t n_k|$")
    axs[end].legend(frameon=false, loc="best")
    fig.tight_layout()
    outpath = joinpath(OUTPUT_DIR, outfile)
    fig.savefig(outpath; dpi=220, bbox_inches="tight")
    close(fig)
    println("Saved " * outpath)
end

plot_maxk_linear([2.5, 5.0], "chain_convergence_rate_maxk_alpha25_alpha50_linear.png")
