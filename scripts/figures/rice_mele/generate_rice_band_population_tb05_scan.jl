using JLD2
using LinearAlgebra
using PyPlot

include(joinpath(@__DIR__, "main-rice_mele.jl"))

const OUT_DIR = joinpath(@__DIR__, "distributions")
mkpath(OUT_DIR)

array_data(x) = hasproperty(x, :data) ? getproperty(x, :data) : x

function extract_field(name::AbstractString, key::AbstractString)
    m = match(Regex(key * "([^_]+)"), name)
    return m === nothing ? missing : m.captures[1]
end

function band_populations(GLdata, ks, it; t1, t2, Δ)
    lower_sum = 0.0
    upper_sum = 0.0
    for (ik, k) in enumerate(ks)
        _, U = eigen(H_k(k; t1=t1, t2=t2, Δ=Δ))
        ρsub = imag.(GLdata[:, :, ik, it, it])
        ρband = U' * ρsub * U
        lower_sum += real(ρband[1, 1])
        upper_sum += real(ρband[2, 2])
    end
    L = length(ks)
    return lower_sum / L, upper_sum / L
end

files = sort(filter(f ->
    startswith(basename(f), "GL_L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.5_dispersion_α") &&
    occursin("_linear_spectral_", basename(f)) &&
    endswith(basename(f), "_tmax60.jld2"),
    readdir("Data"; join=true)))

isempty(files) && error("No completed Rice-Mele GL files found for Tb=0.5")

plt.rc("font", family="serif", size=12)
plt.rc("axes", linewidth=1.8)
plt.rc("xtick.major", width=1.8, size=7)
plt.rc("ytick.major", width=1.8, size=7)
plt.rc("xtick", direction="in")
plt.rc("ytick", direction="in")

alpha_order = ["0.2", "0.4", "0.6", "0.8", "1.0"]
colors = Dict("0.2" => "#1f77b4", "0.5" => "#d1495b", "1.0" => "#2a9d8f")
styles = Dict("0.05" => "-", "0.5" => "--")

fig, axs = subplots(2, 5; figsize=(21.5, 8.0), sharex=true)
axs = reshape(collect(axs), 2, 5)

for (idx, α) in enumerate(alpha_order)
    α_files = filter(f -> occursin("_α$(α)_", basename(f)), files)
    ax_upper = axs[1, idx]
    ax_lower = axs[2, idx]

    for f in α_files
        base = basename(f)
        η = String(extract_field(base, "η"))
        λ = String(extract_field(base, "λ_q"))
        t1 = parse(Float64, String(extract_field(base, "t1")))
        t2 = parse(Float64, String(extract_field(base, "t2")))
        Δ = parse(Float64, String(extract_field(base, "Δ")))

        GL = array_data(load(f, "GL"))
        tsobj = load(replace(f, "GL_" => "ts_"), "sol")
        ts = hasproperty(tsobj, :t) ? tsobj.t : tsobj
        L = size(GL, 3)
        ks = collect(range(-π, stop=π - 2π / L, length=L))

        nlower = zeros(Float64, length(ts))
        nupper = zeros(Float64, length(ts))
        @inbounds for it in eachindex(ts)
            nlower[it], nupper[it] = band_populations(GL, ks, it; t1=t1, t2=t2, Δ=Δ)
        end

        label = raw"$\eta = " * η * raw",\ p_c = " * λ * raw"/a$"
        kwargs = (; color=get(colors, λ, "black"), linestyle=get(styles, η, "-"), linewidth=2.0, label=label)
        ax_upper.plot(ts, nupper; kwargs...)
        ax_lower.plot(ts, nlower; kwargs...)
    end

    ax_upper.text(0.05, 0.90, raw"$\alpha = " * α * raw"$"; transform=ax_upper.transAxes, fontsize=15)
    ax_upper.set_ylim(-0.02, 0.35)
    ax_lower.set_ylim(0.55, 1.02)
    ax_upper.grid(alpha=0.18)
    ax_lower.grid(alpha=0.18)
    ax_upper.legend(frameon=false, loc="best", fontsize=8)
end

for idx in 1:5
    axs[2, idx].set_xlabel(raw"$t$")
end
axs[1, 1].set_ylabel(raw"$N_+(t)$")
axs[2, 1].set_ylabel(raw"$N_-(t)$")

fig.text(0.5, 0.015, raw"Rice-Mele, $T_b = 0.5$, band populations"; ha="center", va="bottom", fontsize=15)
fig.tight_layout(rect=[0, 0.03, 1, 1])

outfile = joinpath(OUT_DIR, "rice_band_populations_tb05_scan.png")
fig.savefig(outfile; dpi=220, bbox_inches="tight")
close(fig)
println("Saved $(relpath(outfile, @__DIR__))")
