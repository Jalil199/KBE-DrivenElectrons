using JLD2
using LinearAlgebra
using PyPlot

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "main-rice_mele.jl"))

const DATA_DIR = joinpath(ROOT, "Data_rice")
const OUT_DIR  = joinpath(ROOT, "distributions_rice")
mkpath(OUT_DIR)

array_data(x) = hasproperty(x, :data) ? getproperty(x, :data) : x

function band_occupations(GLdata, Us, it)
    L = size(GLdata, 3)
    nminus = zeros(Float64, L)
    nplus  = zeros(Float64, L)
    @inbounds for ik in 1:L
        ρsub  = imag.(GLdata[:, :, ik, it, it])
        ρband = Us[ik]' * ρsub * Us[ik]
        nminus[ik] = real(ρband[1, 1])
        nplus[ik]  = real(ρband[2, 2])
    end
    return nminus, nplus
end

function load_case(dataset; t1=-1.0, t2=-0.8, Δ=2.0)
    GL     = array_data(load(joinpath(DATA_DIR, "GL_" * dataset * ".jld2"), "GL"))
    ts_obj = load(joinpath(DATA_DIR, "ts_" * dataset * ".jld2"), "sol")
    ts     = hasproperty(ts_obj, :t) ? collect(ts_obj.t) : collect(ts_obj)

    L  = size(GL, 3)
    ks = collect(range(-π, stop=π - 2π/L, length=L))
    Us = [eigen(H_k(k; t1, t2, Δ)).vectors for k in ks]

    Nminus = zeros(Float64, length(ts))
    Nplus  = zeros(Float64, length(ts))
    for it in eachindex(ts)
        nminus, nplus = band_occupations(GL, Us, it)
        Nminus[it] = sum(nminus) / L
        Nplus[it]  = sum(nplus)  / L
    end
    return (; ts, Nminus, Nplus)
end

name_str(η, ωb0, λ_q) =
    "L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.5_dispersion_α1.0_s1.0_ωc3.0_linear_spectral_" *
    "η$(η)_v_b0.0_ωb0$(ωb0)_power_exp_s_q1.0_λ_q$(λ_q)_t020.0_ω02.2_σ2.0_A0.0_switch0_ti0.5_to5.0_tmax30"

ωb0_vals   = [0.5, 1.0, 1.5, 2.0, 2.04, 2.2, 2.5, 2.8, 3.0]
gap_ωb0    = 2.04  # ≈ minimum band gap energy

bath_configs = [
    (; λ_q=3.0,  η=0.05, title=raw"$\lambda_q=3,\;\eta=0.05$"),
    (; λ_q=10.0, η=0.05, title=raw"$\lambda_q=10,\;\eta=0.05$"),
    (; λ_q=10.0, η=0.2,  title=raw"$\lambda_q=10,\;\eta=0.2$"),
]

cmap   = matplotlib.cm.get_cmap("plasma_r")
colors = [cmap(v) for v in LinRange(0.05, 0.95, length(ωb0_vals))]

println("Loading data...")
data_all = Dict()
for bc in bath_configs, ωb0 in ωb0_vals
    key = (bc.λ_q, bc.η, ωb0)
    data_all[key] = load_case(name_str(bc.η, ωb0, bc.λ_q))
    d = data_all[key]
    println("  ωb0=$(rpad(ωb0,4))  λ_q=$(bc.λ_q)  η=$(bc.η)" *
            "  N+(0)=$(round(d.Nplus[1],digits=5))  N+(tf)=$(round(d.Nplus[end],digits=5))" *
            "  ΔN+=$(round(d.Nplus[end]-d.Nplus[1],digits=5))")
end

plt.rc("font", family="serif", size=12)
plt.rc("axes", linewidth=1.6)
plt.rc("xtick.major", width=1.4, size=5)
plt.rc("ytick.major", width=1.4, size=5)
plt.rc("xtick", direction="in")
plt.rc("ytick", direction="in")

fig, axs = subplots(1, 3; figsize=(15.0, 4.4))

for (col, bc) in enumerate(bath_configs)
    ax = axs[col]
    for (ci, ωb0) in enumerate(ωb0_vals)
        d  = data_all[(bc.λ_q, bc.η, ωb0)]
        lw = ωb0 == gap_ωb0 ? 2.8 : 2.0
        ls = ωb0 == gap_ωb0 ? "--" : "-"
        ax.plot(d.ts, d.Nplus; color=colors[ci], linewidth=lw, linestyle=ls,
                label=raw"$\omega_{b0}=" * "$(ωb0)" * raw"$")
    end
    ax.set_xlabel(raw"$t$")
    ax.set_ylabel(col == 1 ? raw"$N_+(t)$" : "")
    ax.set_title(bc.title, fontsize=12)
    ax.grid(alpha=0.18)
    col == 1 && ax.legend(frameon=false, fontsize=8)
end

fig.text(0.5, 0.01,
    raw"Rice-Mele gap-threshold scan: $\alpha=1.0$, $v_b=0$, $T_b=0.5$, $T_e=1.0$, $t_f=30$" *
    raw"  (dashed: $\omega_{b0}\approx E_\mathrm{gap}$)";
    ha="center", va="bottom", fontsize=11)
fig.tight_layout(rect=[0, 0.07, 1, 1])

outfile = joinpath(OUT_DIR, "rice_gap_threshold_scan_t30.png")
fig.savefig(outfile; dpi=240, bbox_inches="tight")
close(fig)
println("\nSaved $(relpath(outfile, ROOT))")
