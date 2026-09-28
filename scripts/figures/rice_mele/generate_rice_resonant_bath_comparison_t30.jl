using JLD2
using LinearAlgebra
using PyPlot

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "main-rice_mele.jl"))

const DATA_DIR = joinpath(ROOT, "Data_rice")
const OUT_DIR = joinpath(ROOT, "distributions_rice")
mkpath(OUT_DIR)

array_data(x) = hasproperty(x, :data) ? getproperty(x, :data) : x

function band_occupations(GLdata, Us, it)
    L = size(GLdata, 3)
    nminus = zeros(Float64, L)
    nplus = zeros(Float64, L)
    @inbounds for ik in 1:L
        ρsub = imag.(GLdata[:, :, ik, it, it])
        ρband = Us[ik]' * ρsub * Us[ik]
        nminus[ik] = real(ρband[1, 1])
        nplus[ik] = real(ρband[2, 2])
    end
    return nminus, nplus
end

function load_case(dataset; t1=-1.0, t2=-0.8, Δ=2.0)
    GL = array_data(load(joinpath(DATA_DIR, "GL_" * dataset * ".jld2"), "GL"))
    ts_obj = load(joinpath(DATA_DIR, "ts_" * dataset * ".jld2"), "sol")
    ts = hasproperty(ts_obj, :t) ? collect(ts_obj.t) : collect(ts_obj)

    L = size(GL, 3)
    ks = collect(range(-π, stop=π - 2π / L, length=L))
    Us = [eigen(H_k(k; t1=t1, t2=t2, Δ=Δ)).vectors for k in ks]

    Nminus = zeros(Float64, length(ts))
    Nplus = zeros(Float64, length(ts))
    nminus_final = zeros(Float64, L)
    nplus_final = zeros(Float64, L)

    for it in eachindex(ts)
        nminus, nplus = band_occupations(GL, Us, it)
        Nminus[it] = sum(nminus) / L
        Nplus[it] = sum(nplus) / L
        if it == lastindex(ts)
            nminus_final .= nminus
            nplus_final .= nplus
        end
    end

    return (; ts, ks, Nminus, Nplus, nminus_final, nplus_final)
end

base = "L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.5_dispersion_α1.0_s1.0_ωc3.0_linear_spectral_η0.05_"
suffix = "_power_exp_s_q1.0_λ_q1.0_t020.0_ω02.2_σ2.0_A0.0_switch0_ti0.5_to5.0_tmax30"

cases = [
    (; label="current bath", short=raw"$\omega_b=0.1+0.2|q|$", color="#2a9d8f",
       dataset=base * "v_b0.2_ωb00.1" * suffix),
    (; label="flat gap", short=raw"$\omega_b=4.0$", color="#d1495b",
       dataset=base * "v_b0.0_ωb04.0" * suffix),
    (; label="flat high", short=raw"$\omega_b=5.0$", color="#3a5a98",
       dataset=base * "v_b0.0_ωb05.0" * suffix),
]

results = [(case=case, data=load_case(case.dataset)) for case in cases]

plt.rc("font", family="serif", size=13)
plt.rc("axes", linewidth=1.7)
plt.rc("xtick.major", width=1.6, size=6)
plt.rc("ytick.major", width=1.6, size=6)
plt.rc("xtick", direction="in")
plt.rc("ytick", direction="in")

fig, axs = subplots(1, 3; figsize=(15.8, 4.6))

for r in results
    c = r.case
    d = r.data
    axs[1].plot(d.ts, d.Nplus; color=c.color, linewidth=2.4, label=c.short)
    axs[2].plot(d.ts, d.Nminus; color=c.color, linewidth=2.4, label=c.short)
    axs[3].plot(d.ks, d.nplus_final; color=c.color, linewidth=2.2, label=c.short)

    println(rpad(c.label, 14),
            "  N+(0)=", round(d.Nplus[1], digits=6),
            "  N+(tf)=", round(d.Nplus[end], digits=6),
            "  ΔN+=", round(d.Nplus[end] - d.Nplus[1], digits=6),
            "  tf=", round(d.ts[end], digits=4))
end

axs[1].set_xlabel(raw"$t$")
axs[1].set_ylabel(raw"$N_+(t)$")
axs[1].grid(alpha=0.18)
axs[1].legend(frameon=false, fontsize=10)

axs[2].set_xlabel(raw"$t$")
axs[2].set_ylabel(raw"$N_-(t)$")
axs[2].grid(alpha=0.18)

axs[3].set_xlabel(raw"$k$")
axs[3].set_ylabel(raw"$n_+(k,t_f)$")
axs[3].set_xlim(-π, π)
axs[3].set_xticks([-π, -π/2, 0, π/2, π])
axs[3].set_xticklabels([raw"$-\pi$", raw"$-\pi/2$", raw"$0$", raw"$\pi/2$", raw"$\pi$"])
axs[3].grid(alpha=0.18)

fig.text(0.5, 0.02,
    raw"Rice-Mele bath test: $\alpha=1.0$, $\eta=0.05$, $p_c=1.0/a$, $T_b=0.5$, $t_f=30$";
    ha="center", va="bottom", fontsize=13)
fig.tight_layout(rect=[0, 0.07, 1, 1])

outfile = joinpath(OUT_DIR, "rice_resonant_bath_comparison_t30.png")
fig.savefig(outfile; dpi=240, bbox_inches="tight")
close(fig)
println("Saved $(relpath(outfile, ROOT))")
