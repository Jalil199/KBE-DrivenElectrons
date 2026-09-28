using JLD2
using PyPlot
using Interpolations

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

function interpolated_local_rate(times::AbstractVector, values::AbstractVector; n_uniform=401)
    tu = collect(range(first(times), last(times); length=n_uniform))
    itp = interpolate((times,), values, Gridded(Linear()))
    yu = [itp(t) for t in tu]
    rates = abs.(diff(yu)) ./ diff(tu)
    tmids = (tu[1:end-1] .+ tu[2:end]) ./ 2
    return tmids, rates
end

function ks_from_L(L)
    collect(range(-pi, stop=pi - 2pi / L; length=L))
end

function nearest_k_indices(ks)
    targets = [0.0, pi / 2, pi - 2pi / length(ks)]
    idxs = [argmin(abs.(ks .- t)) for t in targets]
    labels = ["k ≈ 0", raw"$k \approx \pi/2$", raw"$k \approx \pi$"]
    return idxs, labels
end

function plot_selected_k_rates(alphas::Vector{Float64}, outfile::String)
    fig, axs = subplots(1, length(alphas); figsize=(6.2 * length(alphas), 4.4), squeeze=false, sharey=true)
    axs = axs[1, :]

    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    prefix = "GL_L100_Te1.0_Tb0.1"

    for (ax, α) in zip(axs, alphas)
        files = sort(filter(f -> occursin(prefix, f) && occursin("_α$(α)_", f), readdir("Data"; join=true)))
        isempty(files) && continue

        # Use the most strongly changing available case in this alpha family:
        # prefer η=0.5, λ_q=1.0, then η=0.5, λ_q=0.5, then any available.
        preferred = nothing
        for tag in [("_η0.5_", "_λ_q1.0_"), ("_η0.5_", "_λ_q0.5_"), ("_η0.05_", "_λ_q1.0_")]
            match_file = findfirst(f -> occursin(tag[1], f) && occursin(tag[2], f), files)
            if match_file !== nothing
                preferred = files[match_file]
                break
            end
        end
        preferred === nothing && (preferred = files[end])

        name = basename(preferred)
        η = String(extract_field(name, "η"))
        λ = String(extract_field(name, "λ_q"))
        GL_data = array_data(load(preferred, "GL"))
        ts = load(replace(preferred, "GL_" => "ts_"), "sol").t
        L = size(GL_data, 1)
        Nt = length(ts)
        nk_t = zeros(Float64, L, Nt)
        @inbounds for it in 1:Nt
            nk_t[:, it] .= imag.(GL_data[:, it, it])
        end

        ks = ks_from_L(L)
        idxs, labels = nearest_k_indices(ks)
        for (i, (ik, label)) in enumerate(zip(idxs, labels))
            tmid, rate = interpolated_local_rate(ts, nk_t[ik, :])
            ax.plot(tmid, rate; color=colors[i], linewidth=2.0, label=label)
        end

        ax.set_title(raw"$\alpha = " * string(α) * raw"$" * "\nη = $(η), p_c = $(λ)/a")
        ax.set_xlabel(raw"$t$")
        ax.set_yscale("log")
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

plot_selected_k_rates([2.5, 5.0], "chain_convergence_rate_selected_k_alpha25_alpha50.png")
