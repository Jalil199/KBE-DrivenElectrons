using JLD2
using PyPlot
using Interpolations

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

files = sort(filter(f ->
    startswith(basename(f), "GL_L100_Te1.0_Tb0.5_u0.0_γ1.0_dispersion_α") &&
    occursin("_linear_spectral_", basename(f)) &&
    endswith(basename(f), "_tmax60.jld2"),
    readdir("Data"; join=true)))

isempty(files) && error("No completed chain GL files found for Tb=0.5")

fig, axs = subplots(3, 2; figsize=(12.8, 10.2), sharex=true, sharey=true)
alphas = ["0.2", "0.4", "0.6", "0.8", "1.0"]
colors = Dict("0.2" => "#1f77b4", "0.5" => "#2ca02c", "1.0" => "#d62728")
styles = Dict("0.05" => "-", "0.5" => "--")

for (idx, α) in enumerate(alphas)
    ax = axs[idx]
    α_files = filter(f -> occursin("_α$(α)_", basename(f)), files)
    for f in α_files
        name = basename(f)
        η = String(extract_field(name, "η"))
        λ = String(extract_field(name, "λ_q"))
        GL_data = array_data(load(f, "GL"))
        tsobj = load(replace(f, "GL_" => "ts_"), "sol")
        ts = hasproperty(tsobj, :t) ? tsobj.t : tsobj
        values = chain_timeseries(GL_data)
        tmid, pointwise = interpolated_pointwise_rates(ts, values)
        dmax = vec(maximum(pointwise; dims=1))
        label = raw"$\eta = " * η * raw",\ p_c = " * λ * raw"/a$"
        ax.plot(tmid, dmax; color=get(colors, λ, "black"), linestyle=get(styles, η, "-"),
            linewidth=2.0, label=label)
    end
    ax.set_title(raw"$\alpha = " * α * raw"$")
    ax.set_ylim(-0.001, 0.02)
    ax.grid(alpha=0.18)
end

axs[6].axis("off")

for ax in axs[3, :]
    ax.set_xlabel(raw"$t$")
end
for ax in axs[:, 1]
    ax.set_ylabel(raw"$\max_k |\partial_t n_k(t)|$")
end

axs[1, 2].legend(frameon=false, loc="best")

fig.tight_layout()
outpath = joinpath(OUTPUT_DIR, "chain_dmax_completed_tb05.png")
fig.savefig(outpath; dpi=220, bbox_inches="tight")
close(fig)
println("Saved " * outpath)
