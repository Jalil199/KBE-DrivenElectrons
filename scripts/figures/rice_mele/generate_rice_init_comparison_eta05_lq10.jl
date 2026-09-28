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

base_str(ωb0; init_suffix="") =
    "L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.5_dispersion_α1.0_s1.0_ωc3.0_linear_spectral_" *
    "η0.5_v_b0.0_ωb0$(ωb0)_power_exp_s_q1.0_λ_q10.0_t020.0_ω02.2_σ2.0_A0.0_switch0" *
    init_suffix * "_ti0.5_to5.0_tmax60"

ωb0_vals = round.(0.2:0.2:4.0; digits=1)
gap_ωb0  = 2.04  # ≈ minimum band gap energy

println("Loading data...")
data_th = Dict{Float64, NamedTuple}()
data_uf = Dict{Float64, NamedTuple}()
for ωb0 in ωb0_vals
    data_th[ωb0] = load_case(base_str(ωb0))
    data_uf[ωb0] = load_case(base_str(ωb0; init_suffix="_initupper_full"))
    dth = data_th[ωb0]
    duf = data_uf[ωb0]
    println("  ωb0=$(rpad(ωb0, 4))" *
            "  thermal:    N+(0)=$(round(dth.Nplus[1],digits=4))  N+(tf)=$(round(dth.Nplus[end],digits=4))" *
            "  upper_full: N+(0)=$(round(duf.Nplus[1],digits=4))  N+(tf)=$(round(duf.Nplus[end],digits=4))")
end

# ── matplotlib style ──────────────────────────────────────────────────────────
plt.rc("font", family="serif", size=12)
plt.rc("axes", linewidth=1.6)
plt.rc("xtick.major", width=1.4, size=5)
plt.rc("ytick.major", width=1.4, size=5)
plt.rc("xtick", direction="in")
plt.rc("ytick", direction="in")

cmap   = matplotlib.cm.get_cmap("plasma_r")
colors = [cmap(v) for v in LinRange(0.05, 0.95, length(ωb0_vals))]

# ── figure 1: N+(t) time traces, two panels ───────────────────────────────────
fig, axs = subplots(1, 2; figsize=(12.0, 4.6), sharey=false)

for (ax, data, title) in [
        (axs[1], data_th, raw"thermal init ($N_+^0 \approx 0.15$)"),
        (axs[2], data_uf, raw"upper-band init ($N_+^0 = 1$)"),
    ]

    for (ci, ωb0) in enumerate(ωb0_vals)
        d  = data[ωb0]
        lw = abs(ωb0 - gap_ωb0) < 0.05 ? 2.8 : 1.8
        ls = abs(ωb0 - gap_ωb0) < 0.05 ? "--" : "-"
        ax.plot(d.ts, d.Nplus; color=colors[ci], linewidth=lw, linestyle=ls,
                label=raw"$\omega_{b0}=" * "$(ωb0)" * raw"$")
    end
    ax.set_xlabel(raw"$t$")
    ax.set_ylabel(raw"$N_+(t)$")
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.18)
end
axs[1].legend(frameon=false, fontsize=7.5, ncol=2, loc="best")

fig.text(0.5, 0.01,
    raw"Rice-Mele, $\eta=0.5$, $\lambda_q=10$, $v_b=0$, $T_b=0.5$, $T_e=1.0$, $t_f=60$" *
    raw"  (dashed: $\omega_{b0}\approx E_\mathrm{gap}$)";
    ha="center", va="bottom", fontsize=11)
fig.tight_layout(rect=[0, 0.07, 1, 1])

outfile1 = joinpath(OUT_DIR, "rice_init_comparison_eta05_lq10_traces.png")
fig.savefig(outfile1; dpi=240, bbox_inches="tight")
close(fig)
println("Saved $(relpath(outfile1, ROOT))")

# ── figure 2: ΔN+ vs ωb0, thermal vs upper_full ──────────────────────────────
fig2, ax2 = subplots(1, 1; figsize=(6.5, 4.5))

ΔN_th = [data_th[ωb0].Nplus[end] - data_th[ωb0].Nplus[1]   for ωb0 in ωb0_vals]
ΔN_uf = [data_uf[ωb0].Nplus[end] - data_uf[ωb0].Nplus[1]   for ωb0 in ωb0_vals]
Nf_th = [data_th[ωb0].Nplus[end]                             for ωb0 in ωb0_vals]
Nf_uf = [data_uf[ωb0].Nplus[end]                             for ωb0 in ωb0_vals]

ax2.plot(ωb0_vals, ΔN_th, "o-"; color="#1f77b4", linewidth=2.0, markersize=5,
         label=raw"thermal $\Delta N_+$")
ax2.plot(ωb0_vals, ΔN_uf, "s-"; color="#d1495b", linewidth=2.0, markersize=5,
         label=raw"upper-full $\Delta N_+$")
ax2.axvline(gap_ωb0; color="gray", linestyle="--", linewidth=1.2, label=raw"$E_\mathrm{gap}$")
ax2.axhline(0.0;     color="black", linestyle=":", linewidth=1.0)
ax2.set_xlabel(raw"$\omega_{b0}$")
ax2.set_ylabel(raw"$\Delta N_+ = N_+(t_f) - N_+(0)$")
ax2.set_title(raw"Relaxation vs bath frequency: $\eta=0.5$, $\lambda_q=10$", fontsize=12)
ax2.legend(frameon=false, fontsize=11)
ax2.grid(alpha=0.18)

fig2.tight_layout()
outfile2 = joinpath(OUT_DIR, "rice_init_comparison_eta05_lq10_delta.png")
fig2.savefig(outfile2; dpi=240, bbox_inches="tight")
close(fig2)
println("Saved $(relpath(outfile2, ROOT))")

# ── figure 3: N+(tf) vs ωb0 for both inits ───────────────────────────────────
fig3, ax3 = subplots(1, 1; figsize=(6.5, 4.5))

ax3.plot(ωb0_vals, Nf_th, "o-"; color="#1f77b4", linewidth=2.0, markersize=5,
         label=raw"thermal $N_+(t_f)$")
ax3.plot(ωb0_vals, Nf_uf, "s-"; color="#d1495b", linewidth=2.0, markersize=5,
         label=raw"upper-full $N_+(t_f)$")
ax3.axvline(gap_ωb0; color="gray", linestyle="--", linewidth=1.2, label=raw"$E_\mathrm{gap}$")
ax3.set_xlabel(raw"$\omega_{b0}$")
ax3.set_ylabel(raw"$N_+(t_f)$")
ax3.set_title(raw"Final upper-band occupation: $\eta=0.5$, $\lambda_q=10$", fontsize=12)
ax3.legend(frameon=false, fontsize=11)
ax3.grid(alpha=0.18)

fig3.tight_layout()
outfile3 = joinpath(OUT_DIR, "rice_init_comparison_eta05_lq10_final.png")
fig3.savefig(outfile3; dpi=240, bbox_inches="tight")
close(fig3)
println("Saved $(relpath(outfile3, ROOT))")
