using JLD2
using PyPlot
using Statistics

rc("font", family="serif")
rc("font", size=12)
rc("axes", labelsize=14, titlesize=14)
rc("xtick", labelsize=10)
rc("ytick", labelsize=10)
rc("legend", fontsize=8)

const OUTPUT_DIR = "distributions"
mkpath(OUTPUT_DIR)

array_data(x) = hasproperty(x, :data) ? getproperty(x, :data) : x

fermi(ϵ, T, μ) = 1 / (exp((ϵ - μ) / T) + 1)

function extract_field(name::AbstractString, key::AbstractString)
    m = match(Regex("$(key)([^_]+)"), name)
    return m === nothing ? missing : m.captures[1]
end

function local_fit_rmse(nk, ϵs, T, μ)
    sqrt(mean((nk .- fermi.(ϵs, T, μ)).^2))
end

function best_fermi_rmse(nk, ϵs)
    best_rmse = Inf
    best_T = NaN
    best_μ = NaN

    for T in 0.02:0.02:1.50, μ in -3.0:0.05:3.0
        fit_rmse = local_fit_rmse(nk, ϵs, T, μ)
        if fit_rmse < best_rmse
            best_rmse = fit_rmse
            best_T = T
            best_μ = μ
        end
    end

    for T in max(0.01, best_T - 0.04):0.002:min(1.50, best_T + 0.04),
        μ in max(-3.0, best_μ - 0.10):0.01:min(3.0, best_μ + 0.10)
        fit_rmse = local_fit_rmse(nk, ϵs, T, μ)
        if fit_rmse < best_rmse
            best_rmse = fit_rmse
        end
    end

    return best_rmse
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
        L = size(values, 1)
        ks = collect(range(-π, stop=π - 2π / L, length=L))
        ϵs = -2 .* cos.(ks)

        fit_rmse = zeros(Float64, length(ts))
        @inbounds for it in eachindex(ts)
            fit_rmse[it] = best_fermi_rmse(values[:, it], ϵs)
        end

        label = raw"$\eta = " * η * raw",\ p_c = " * λ * raw"/a$"
        ax.plot(ts, fit_rmse; color=get(colors, λ, "black"), linestyle=get(styles, η, "-"),
            linewidth=2.0, label=label)
    end
    ax.text(0.05, 0.92, raw"$\alpha = " * α * raw"$"; transform=ax.transAxes, fontsize=15)
    ax.set_ylim(-0.002, 0.10)
    ax.grid(alpha=0.18)
    ax.legend(frameon=false, loc="upper right")
end

for i in 6:10
    axs[i].set_xlabel(raw"$t$")
end
for i in (1, 6)
    axs[i].set_ylabel(raw"$\mathrm{RMSE}(t)$")
end

fig.text(0.5, 0.015, raw"Chain, $T_b = 0.1$"; ha="center", va="bottom", fontsize=15)
fig.tight_layout(rect=[0, 0.03, 1, 1])

outpath = joinpath(OUTPUT_DIR, "chain_fit_rmse_tb01_scan.png")
fig.savefig(outpath; dpi=220, bbox_inches="tight")
close(fig)
println("Saved " * outpath)
