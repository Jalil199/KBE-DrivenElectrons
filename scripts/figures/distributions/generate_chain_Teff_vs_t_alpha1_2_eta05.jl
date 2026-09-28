using JLD2
using LinearAlgebra
using Statistics
using PyPlot

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

const DATA_DIR = joinpath(ROOT, "Data")
const OUT_DIR  = joinpath(ROOT, "distributions_chain")
mkpath(OUT_DIR)

fermi_dirac(ε, T, μ=0.0) = 1 / (exp((ε - μ) / T) + 1)

function find_mu(ε_k, T, N_target; tol=1e-10)
    f(μ) = mean(fermi_dirac.(ε_k, T, μ)) - N_target
    μlo, μhi = -10.0, 10.0
    for _ in 1:100
        μmid = (μlo + μhi) / 2
        f(μmid) > 0 ? (μhi = μmid) : (μlo = μmid)
        μhi - μlo < tol && break
    end
    return (μlo + μhi) / 2
end

function fit_fermi(nk, ε_k; T_range=(0.01, 5.0), n_T=500)
    N = mean(nk)
    T_vals = exp.(LinRange(log(T_range[1]), log(T_range[2]), n_T))
    best_rms, best_T = Inf, NaN
    for T in T_vals
        μ   = find_mu(ε_k, T, N)
        rms = sqrt(mean((nk .- fermi_dirac.(ε_k, T, μ)).^2))
        if rms < best_rms
            best_rms, best_T = rms, T
        end
    end
    return best_T, best_rms
end

function load_Teff_vs_t(dataset, ε_k)
    GL_raw = load(joinpath(DATA_DIR, "GL_" * dataset * ".jld2"), "GL")
    GL = hasproperty(GL_raw, :data) ? GL_raw.data : GL_raw
    ts_obj = load(joinpath(DATA_DIR, "ts_" * dataset * ".jld2"), "sol")
    ts = hasproperty(ts_obj, :t) ? collect(ts_obj.t) : collect(ts_obj)

    nd = ndims(GL)
    L  = nd == 3 ? size(GL, 1) : size(GL, 3)
    nt = length(ts)

    T_eff = zeros(Float64, nt)
    rms   = zeros(Float64, nt)
    for it in 1:nt
        nk = nd == 3 ? imag.(GL[:, it, it]) : [imag(GL[1, 1, ik, it, it]) for ik in 1:L]
        T_eff[it], rms[it] = fit_fermi(nk, ε_k)
    end
    return ts, T_eff, rms
end

function name_str(α, λ_q; Tb=0.1, η=0.5)
    "L100_Te1.0_Tb$(Tb)_u0.0_γ1.0_dispersion_α$(α)_s1.0_ωc10.0_linear_spectral_" *
    "η$(η)_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q$(λ_q)_t050.0_ω0$(Float64(pi))_σ2.0_A0.0_switch0_ti3.0_to20.0_tmax60"
end

L   = 100
Δk  = 2π / L
ks  = collect(range(-π, stop=π - Δk, length=L))
ε_k = -2 .* cos.(ks)

α_vals  = [1.0, 2.0]
λq_vals = [0.2, 0.5, 1.0]
ls_λ    = Dict(0.2 => "-", 0.5 => "--", 1.0 => ":")
colors  = Dict(1.0 => "#2a9d8f", 2.0 => "#e76f51")

println("Computing T_eff(t) — this may take a minute...")
results = Dict()
for α in α_vals, λ_q in λq_vals
    ds = name_str(α, λ_q)
    isfile(joinpath(DATA_DIR, "GL_" * ds * ".jld2")) || (println("  MISSING α=$(α) λ_q=$(λ_q)"); continue)
    print("  α=$(α) λ_q=$(λ_q) ...")
    ts, T_eff, rms = load_Teff_vs_t(ds, ε_k)
    println("  done  T_eff(tf)=$(round(T_eff[end],digits=4))")
    results[(α, λ_q)] = (; ts, T_eff, rms)
end

plt.rc("font", family="serif", size=12)
plt.rc("axes", linewidth=1.6)
plt.rc("xtick.major", width=1.4, size=5)
plt.rc("ytick.major", width=1.4, size=5)
plt.rc("xtick", direction="in")
plt.rc("ytick", direction="in")

fig, axs = subplots(1, 2; figsize=(12.0, 4.6))

for (col, α) in enumerate(α_vals)
    ax = axs[col]
    ax.axhline(0.1; color="black", linewidth=1.2, linestyle="-", label=raw"$T_b = 0.1$", zorder=5)
    ax.axhline(1.0; color="gray",  linewidth=1.0, linestyle="--", alpha=0.5, label=raw"$T_e = 1.0$")
    for λ_q in λq_vals
        haskey(results, (α, λ_q)) || continue
        d = results[(α, λ_q)]
        ax.plot(d.ts, d.T_eff; color=colors[α], linewidth=2.2, linestyle=ls_λ[λ_q],
                label=raw"$\lambda_q=" * "$(λ_q)" * raw"$")
    end
    ax.set_xlabel(raw"$t$")
    ax.set_ylabel(col == 1 ? raw"$T_\mathrm{eff}(t)$" : "")
    ax.set_title(raw"$\alpha = " * "$(α)" * raw"$", fontsize=13)
    ax.legend(frameon=false, fontsize=10)
    ax.grid(alpha=0.18)
end

fig.text(0.5, 0.01,
    raw"Chain 1D: $\eta=0.5$, $T_b=0.1$, $T_e=1.0$, $v_b=0.2$, $\omega_{b0}=0.1$, $t_f=60$";
    ha="center", va="bottom", fontsize=11)
fig.tight_layout(rect=[0, 0.07, 1, 1])

outfile = joinpath(OUT_DIR, "chain_Teff_vs_t_alpha1_2_eta05_tf60.png")
fig.savefig(outfile; dpi=240, bbox_inches="tight")
close(fig)
println("\nSaved $(relpath(outfile, ROOT))")
