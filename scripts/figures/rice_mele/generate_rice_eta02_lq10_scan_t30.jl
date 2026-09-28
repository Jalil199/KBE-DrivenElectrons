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

name_str(ωb0) =
    "L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.5_dispersion_α1.0_s1.0_ωc3.0_linear_spectral_" *
    "η0.2_v_b0.0_ωb0$(ωb0)_power_exp_s_q1.0_λ_q10.0_t020.0_ω02.2_σ2.0_A0.0_switch0_ti0.5_to5.0_tmax30"

ωb0_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

cmap   = matplotlib.cm.get_cmap("plasma_r")
colors = [cmap(v) for v in LinRange(0.05, 0.95, length(ωb0_vals))]

println("Loading data...")
cases = map(ωb0_vals) do ωb0
    d = load_case(name_str(ωb0))
    println("  ωb0=$(ωb0)  N+(0)=$(round(d.Nplus[1],digits=5))  N+(tf)=$(round(d.Nplus[end],digits=5))  ΔN+=$(round(d.Nplus[end]-d.Nplus[1],digits=5))")
    d
end

plt.rc("font", family="serif", size=13)
plt.rc("axes", linewidth=1.7)
plt.rc("xtick.major", width=1.5, size=5)
plt.rc("ytick.major", width=1.5, size=5)
plt.rc("xtick", direction="in")
plt.rc("ytick", direction="in")

fig, ax = subplots(1, 1; figsize=(6.5, 4.8))

for (ci, (ωb0, d)) in enumerate(zip(ωb0_vals, cases))
    ax.plot(d.ts, d.Nplus; color=colors[ci], linewidth=2.2,
            label=raw"$\omega_{b0}=" * "$(ωb0)" * raw"$")
end

ax.set_xlabel(raw"$t$")
ax.set_ylabel(raw"$N_+(t)$")
ax.set_title(raw"$\lambda_q=10,\;\eta=0.2$", fontsize=13)
ax.legend(frameon=false, fontsize=10)
ax.grid(alpha=0.18)

fig.text(0.5, 0.01,
    raw"Rice-Mele: $\alpha=1.0$, $v_b=0$, $T_b=0.5$, $T_e=1.0$, $t_f=30$";
    ha="center", va="bottom", fontsize=11)
fig.tight_layout(rect=[0, 0.06, 1, 1])

outfile = joinpath(OUT_DIR, "rice_eta02_lq10_scan_t30.png")
fig.savefig(outfile; dpi=240, bbox_inches="tight")
close(fig)
println("\nSaved $(relpath(outfile, ROOT))")
