using JLD2
using LinearAlgebra
using PyPlot
using Statistics

include(joinpath(@__DIR__, "main-rice_mele.jl"))

const OUT_DIR = joinpath(@__DIR__, "distributions")
mkpath(OUT_DIR)

array_data(x) = hasproperty(x, :data) ? getproperty(x, :data) : x
nearest_index(xs, x) = argmin(abs.(xs .- x))

function band_occupations(GLdata, ks, it; t1, t2, Δ)
    L = length(ks)
    nlower = zeros(Float64, L)
    nupper = zeros(Float64, L)
    for (ik, k) in enumerate(ks)
        _, U = eigen(H_k(k; t1=t1, t2=t2, Δ=Δ))
        ρsub = imag.(GLdata[:, :, ik, it, it])
        ρband = U' * ρsub * U
        nlower[ik] = real(ρband[1, 1])
        nupper[ik] = real(ρband[2, 2])
    end
    return nlower, nupper
end

dataset = "L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.5_dispersion_α1.0_s1.0_ωc3.0_linear_spectral_η0.05_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q1.0_t020.0_ω02.2_σ2.0_A0.0_switch0_ti0.5_to5.0_tmax60"
gl_path = joinpath(@__DIR__, "Data", "GL_" * dataset * ".jld2")
ts_path = joinpath(@__DIR__, "Data", "ts_" * dataset * ".jld2")

GL = array_data(load(gl_path, "GL"))
ts_obj = load(ts_path, "sol")
ts = hasproperty(ts_obj, :t) ? ts_obj.t : ts_obj

t1 = -1.0
t2 = -0.8
Δ = 2.0
L = size(GL, 3)
ks = collect(range(-π, stop=π - 2π / L, length=L))

nlower_t = zeros(Float64, length(ts), L)
nupper_t = zeros(Float64, length(ts), L)
for it in eachindex(ts)
    nlower_t[it, :], nupper_t[it, :] = band_occupations(GL, ks, it; t1=t1, t2=t2, Δ=Δ)
end

Nlower = vec(mean(nlower_t; dims=2))
Nupper = vec(mean(nupper_t; dims=2))
time_targets = [0.0, 20.0, 40.0, 60.0]
time_idxs = [nearest_index(ts, t) for t in time_targets]

plt.rc("font", family="serif", size=15)
plt.rc("axes", linewidth=1.8)
plt.rc("xtick.major", width=1.8, size=7)
plt.rc("ytick.major", width=1.8, size=7)
plt.rc("xtick", direction="in")
plt.rc("ytick", direction="in")

fig, axs = subplots(1, 3; figsize=(16.5, 4.8))

axs[1].plot(ts, Nupper; color="#d1495b", linewidth=2.4, label=raw"$N_+(t)$")
axs[1].plot(ts, Nlower; color="#2a9d8f", linewidth=2.4, label=raw"$N_-(t)$")
axs[1].set_xlabel(raw"$t$")
axs[1].set_ylabel(raw"$N_\pm(t)$")
axs[1].legend(frameon=false, loc="best")
axs[1].grid(alpha=0.18)

colors = ["black", "#2c7fb8", "#d1495b", "#7b3294"]
for (i, it) in enumerate(time_idxs)
    label = raw"$t = " * string(round(ts[it], digits=1)) * raw"$"
    axs[2].plot(ks, nupper_t[it, :]; color=colors[i], linewidth=2.0, label=label)
    axs[3].plot(ks, nlower_t[it, :]; color=colors[i], linewidth=2.0, label=label)
end

axs[2].set_xlabel(raw"$k$")
axs[2].set_ylabel(raw"$n_+(k,t)$")
axs[2].set_xlim(-π, π)
axs[2].set_ylim(-0.02, 0.55)
axs[2].legend(frameon=false, loc="best")
axs[2].grid(alpha=0.18)

axs[3].set_xlabel(raw"$k$")
axs[3].set_ylabel(raw"$n_-(k,t)$")
axs[3].set_xlim(-π, π)
axs[3].set_ylim(0.45, 1.02)
axs[3].legend(frameon=false, loc="best")
axs[3].grid(alpha=0.18)

for ax in axs[2:3]
    ax.set_xticks([-π, -π/2, 0, π/2, π])
    ax.set_xticklabels([raw"$-\pi$", raw"$-\pi/2$", raw"$0$", raw"$\pi/2$", raw"$\pi$"])
end

fig.text(0.5, 0.02,
    raw"Rice-Mele: $\alpha=1.0$, $\eta=0.05$, $p_c=1.0/a$, $T_b=0.5$";
    ha="center", va="bottom", fontsize=14)
fig.tight_layout(rect=[0, 0.07, 1, 1])

outfile = joinpath(OUT_DIR, "rice_clear_relaxation_alpha1_eta005_pc1.png")
fig.savefig(outfile; dpi=240, bbox_inches="tight")
close(fig)
println("Saved $(relpath(outfile, @__DIR__))")
