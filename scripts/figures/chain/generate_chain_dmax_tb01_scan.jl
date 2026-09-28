using JLD2
using PyPlot
using Interpolations

rc("font", family="serif")
rc("font", size=12)
rc("axes", labelsize=14, titlesize=14)
rc("xtick", labelsize=10)
rc("ytick", labelsize=10)
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
    startswith(basename(f), "GL_L100_Te1.0_Tb0.1_u0.0_γ1.0_dispersion_α") &&
    occursin("_linear_spectral_", basename(f)) &&
    endswith(basename(f), "_tmax60.jld2"),
    readdir("Data"; join=true)))

isempty(files) && error("No completed chain GL files found for Tb=0.1")

alpha_order = ["0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.4", "1.6", "1.8", "2.0"]
colors = Dict("0.2" => "#1f77b4", "0.5" => "#d1495b", "1.0" => "#2a9d8f")
styles = Dict("0.05" => "-", "0.5" => "--")

fig, axs = subplots(2, 5; figsize=(21.5, 8.0), sharex=true, sharey=true)
axs = vec(axs)

for (idx, α) in enumerate(alpha_order)
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
        ax.plot(tmid, dmax;
            color=get(colors, λ, "black"),
            linestyle=get(styles, η, "-"),
            linewidth=2.0,
            label=label)
    end
    ax.text(0.05, 0.92, raw"$\alpha = " * α * raw"$"; transform=ax.transAxes, fontsize=15)
    ax.grid(alpha=0.18)
    ax.legend(frameon=false, loc="upper right")
end

for i in 6:10
    axs[i].set_xlabel(raw"$t$")
end
for i in (1, 6)
    axs[i].set_ylabel(raw"$\max_k |\partial_t n_k(t)|$")
end

fig.text(0.5, 0.015, raw"Chain, $T_b = 0.1$"; ha="center", va="bottom", fontsize=15)
fig.tight_layout(rect=[0, 0.03, 1, 1])

outpath = joinpath(OUTPUT_DIR, "chain_dmax_tb01_scan.png")
fig.savefig(outpath; dpi=220, bbox_inches="tight")
close(fig)
println("Saved " * outpath)
