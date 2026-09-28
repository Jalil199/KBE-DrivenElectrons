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
        ρsub   = imag.(GLdata[:, :, ik, it, it])
        ρband  = Us[ik]' * ρsub * Us[ik]
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
    nplus_final = zeros(Float64, L)

    for it in eachindex(ts)
        nminus, nplus = band_occupations(GL, Us, it)
        Nminus[it] = sum(nminus) / L
        Nplus[it]  = sum(nplus)  / L
        if it == lastindex(ts)
            nplus_final .= nplus
        end
    end
    return (; ts, ks, Nminus, Nplus, nplus_final)
end

# ── Dataset names ──────────────────────────────────────────────────────────────

base(η, ωb0, λ_q, v_b=0.0) =
    "L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.5_dispersion_α1.0_s1.0_ωc3.0_linear_spectral_" *
    "η$(η)_v_b$(v_b)_ωb0$(ωb0)_power_exp_s_q1.0_λ_q$(λ_q)_t020.0_ω02.2_σ2.0_A0.0_switch0_ti0.5_to5.0_tmax30"

ωb0_vals = [2.2, 2.5, 2.8, 3.0]
λ_q_vals = [1.0, 3.0, 10.0]
η_vals   = [0.02, 0.05, 0.1, 0.2]

ωb0_colors = ["#2a9d8f", "#d1495b", "#7b3294", "#e76f51"]
η_colors   = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]

# ── Load data ─────────────────────────────────────────────────────────────────

println("Loading ωb0 × λ_q scan...")
data_wb = Dict()
for ωb0 in ωb0_vals, λ_q in λ_q_vals
    key = (ωb0, λ_q)
    data_wb[key] = load_case(base(0.05, ωb0, λ_q))
    println("  ωb0=$(ωb0), λ_q=$(λ_q)  N+(tf)=$(round(data_wb[key].Nplus[end], digits=5))")
end

println("\nLoading η scan (ωb0=2.5, λ_q=10.0)...")
data_eta = Dict()
for η in η_vals
    data_eta[η] = load_case(base(η, 2.5, 10.0))
    println("  η=$(η)  N+(tf)=$(round(data_eta[η].Nplus[end], digits=5))")
end

# ── Plot ──────────────────────────────────────────────────────────────────────

plt.rc("font", family="serif", size=12)
plt.rc("axes", linewidth=1.6)
plt.rc("xtick.major", width=1.4, size=5)
plt.rc("ytick.major", width=1.4, size=5)
plt.rc("xtick", direction="in")
plt.rc("ytick", direction="in")

fig, axs = subplots(1, 4; figsize=(18.0, 4.4))

# Panels 0-2: ωb0 scan for each λ_q
for (col, λ_q) in enumerate(λ_q_vals)
    ax = axs[col]
    for (ci, ωb0) in enumerate(ωb0_vals)
        d = data_wb[(ωb0, λ_q)]
        ax.plot(d.ts, d.Nplus; color=ωb0_colors[ci], linewidth=2.2,
                label=raw"$\omega_{b0}=" * "$(ωb0)" * raw"$")
    end
    ax.set_xlabel(raw"$t$")
    ax.set_ylabel(col == 1 ? raw"$N_+(t)$" : "")
    ax.set_title(raw"$\lambda_q = " * "$(λ_q)" * raw"$", fontsize=12)
    ax.grid(alpha=0.18)
    col == 1 && ax.legend(frameon=false, fontsize=9)
end

# Panel 3: η scan at ωb0=2.5, λ_q=10
ax = axs[4]
for (ci, η) in enumerate(η_vals)
    d = data_eta[η]
    ax.plot(d.ts, d.Nplus; color=η_colors[ci], linewidth=2.2,
            label=raw"$\eta = " * "$(η)" * raw"$")
end
ax.set_xlabel(raw"$t$")
ax.set_title(raw"$\eta$ scan  ($\omega_{b0}=2.5,\,\lambda_q=10$)", fontsize=12)
ax.grid(alpha=0.18)
ax.legend(frameon=false, fontsize=9)

fig.text(0.5, 0.01,
    raw"Rice-Mele flat-bath resonance scan: $\alpha=1.0$, $v_b=0$, $T_b=0.5$, $T_e=1.0$, $t_f=30$";
    ha="center", va="bottom", fontsize=11)
fig.tight_layout(rect=[0, 0.07, 1, 1])

outfile = joinpath(OUT_DIR, "rice_resonance_cutoff_scan_t30.png")
fig.savefig(outfile; dpi=240, bbox_inches="tight")
close(fig)
println("\nSaved $(relpath(outfile, ROOT))")
