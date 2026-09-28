using JLD2
using PyPlot
using Statistics

rc("font", family="serif")
rc("font", size=13)
rc("axes", labelsize=15, titlesize=14)
rc("xtick", labelsize=11)
rc("ytick", labelsize=11)
rc("legend", fontsize=10)

const OUTPUT_DIR = "distributions_chain"
mkpath(OUTPUT_DIR)

array_data(x) = hasproperty(x, :data) ? getproperty(x, :data) : x

fermi(ϵ, T, μ) = 1 / (exp((ϵ - μ) / T) + 1)

function extract_field(name, key)
    m = match(Regex("$(key)([^_]+)"), name)
    m === nothing ? missing : m.captures[1]
end

function local_fit_rmse(nk, ϵs, T, μ)
    sqrt(mean((nk .- fermi.(ϵs, T, μ)).^2))
end

function best_fermi_rmse(nk, ϵs)
    best_rmse = Inf
    best_T, best_μ = NaN, NaN
    for T in 0.02:0.02:1.50, μ in -3.0:0.05:3.0
        rmse = local_fit_rmse(nk, ϵs, T, μ)
        if rmse < best_rmse
            best_rmse = rmse; best_T = T; best_μ = μ
        end
    end
    for T in max(0.01, best_T-0.04):0.002:min(1.50, best_T+0.04),
        μ in max(-3.0, best_μ-0.10):0.01:min(3.0, best_μ+0.10)
        rmse = local_fit_rmse(nk, ϵs, T, μ)
        if rmse < best_rmse
            best_rmse = rmse
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

α_vals  = ["1.0", "2.0"]
λq_vals = ["0.2", "0.5", "1.0"]
colors  = Dict("1.0" => "#2a9d8f", "2.0" => "#e76f51")
styles  = Dict("0.2" => "-", "0.5" => "--", "1.0" => ":")

files = sort(filter(f ->
    startswith(basename(f), "GL_L100_Te1.0_Tb0.1_u0.0_γ1.0_dispersion_") &&
    occursin("_linear_spectral_η0.5_", basename(f)) &&
    endswith(basename(f), "_tmax60.jld2"),
    readdir("Data"; join=true)))

L  = 100
ks = collect(range(-π, stop=π - 2π/L, length=L))
ϵs = -2.0 .* cos.(ks)

fig, axs = subplots(3, 2; figsize=(11.0, 10.0), sharey=false)

for (row, λ) in enumerate(λq_vals), (col, α) in enumerate(α_vals)
    ax = axs[row, col]
    f_match = filter(f -> occursin("_α$(α)_", basename(f)) &&
                          occursin("λ_q$(λ)_", basename(f)), files)
    isempty(f_match) && continue
    f = first(f_match)

    GL_data = array_data(load(f, "GL"))
    tsobj   = load(replace(f, "GL_" => "ts_"), "sol")
    ts      = hasproperty(tsobj, :t) ? tsobj.t : tsobj
    nk_t    = chain_timeseries(GL_data)

    rmse = zeros(Float64, length(ts))
    @inbounds for it in eachindex(ts)
        rmse[it] = best_fermi_rmse(nk_t[:, it], ϵs)
    end

    ax.plot(ts, rmse; color=colors[α], linewidth=2.2)
    ax.set_title(raw"$\alpha=" * α * raw",\ \lambda_q=" * λ * raw"$", fontsize=12)
    ax.set_xlabel(row == 3 ? raw"$t$" : "")
    ax.set_ylabel(col == 1 ? raw"$\mathrm{RMSE}(t)$" : "")
    ax.grid(alpha=0.18)
    println("α=$(α)  λ_q=$(λ)  RMSE(tf)=$(round(rmse[end], digits=6))")
end

fig.text(0.5, 0.01,
    raw"Chain 1D: $\eta=0.5$, $T_b=0.1$, $T_e=1.0$, $v_b=0.2$, $\omega_{b0}=0.1$, $t_f=60$";
    ha="center", va="bottom", fontsize=11)
fig.tight_layout(rect=[0, 0.04, 1, 1])

outfile = joinpath(OUTPUT_DIR, "chain_rmse_alpha1_2_eta05_tf60_panels.png")
fig.savefig(outfile; dpi=240, bbox_inches="tight")
close(fig)
println("\nSaved $(outfile)")
