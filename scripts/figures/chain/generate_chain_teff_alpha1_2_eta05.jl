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

function local_fit_rmse(nk, ϵs, T, μ)
    sqrt(mean((nk .- fermi.(ϵs, T, μ)).^2))
end

function best_fermi_fit(nk, ϵs)
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
            best_rmse = rmse; best_T = T; best_μ = μ
        end
    end
    return best_T, best_μ, best_rmse
end

function chain_timeseries(GL_data)
    L, Nt, _ = size(GL_data)
    nk_t = zeros(Float64, L, Nt)
    @inbounds for it in 1:Nt
        nk_t[:, it] .= imag.(GL_data[:, it, it])
    end
    nk_t
end

function extract_field(name, key)
    m = match(Regex("$(key)([^_]+)"), name)
    m === nothing ? missing : m.captures[1]
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

L   = 100
Δk  = 2π / L
ks  = collect(range(-π, stop=π - Δk, length=L))
ϵs  = -2.0 .* cos.(ks)

fig, axs = subplots(1, 2; figsize=(12.0, 4.8), sharey=true)

for (col, α) in enumerate(α_vals)
    ax = axs[col]
    ax.axhline(1.0; color="gray",  linewidth=1.0, linestyle="--", alpha=0.5)
    ax.axhline(0.1; color="black", linewidth=1.2, linestyle=":")

    α_files = filter(f -> occursin("_α$(α)_", basename(f)), files)
    for f in α_files
        name = basename(f)
        λ = String(extract_field(name, "λ_q"))
        λ in λq_vals || continue

        GL_data = array_data(load(f, "GL"))
        tsobj   = load(replace(f, "GL_" => "ts_"), "sol")
        ts      = hasproperty(tsobj, :t) ? tsobj.t : tsobj
        nk_t    = chain_timeseries(GL_data)

        teff = zeros(Float64, length(ts))
        @inbounds for it in eachindex(ts)
            teff[it], _, _ = best_fermi_fit(nk_t[:, it], ϵs)
        end

        ax.plot(ts, teff;
            color=colors[α], linestyle=styles[λ], linewidth=2.2,
            label=raw"$\lambda_q = " * λ * raw"$")
        println("α=$(α)  λ_q=$(λ)  T_eff(tf)=$(round(teff[end], digits=4))")
    end

    ax.set_xlabel(raw"$t$")
    ax.set_ylabel(col == 1 ? raw"$T_{\mathrm{eff}}(t)$" : "")
    ax.set_title(raw"$\alpha = " * α * raw"$", fontsize=14)
    ax.set_ylim(0.0, 1.1)
    ax.legend(frameon=false, loc="upper right")
    ax.grid(alpha=0.18)
end

fig.text(0.5, 0.01,
    raw"Chain 1D: $\eta=0.5$, $T_b=0.1$, $T_e=1.0$, $v_b=0.2$, $\omega_{b0}=0.1$, $t_f=60$";
    ha="center", va="bottom", fontsize=11)
fig.tight_layout(rect=[0, 0.06, 1, 1])

outfile = joinpath(OUTPUT_DIR, "chain_teff_alpha1_2_eta05_tf60.png")
fig.savefig(outfile; dpi=240, bbox_inches="tight")
close(fig)
println("\nSaved $(outfile)")
