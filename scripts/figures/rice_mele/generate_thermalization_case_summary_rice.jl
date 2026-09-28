using JLD2
using LinearAlgebra
using Statistics
using PyPlot

include(joinpath(@__DIR__, "main-rice_mele.jl"))

const TARGET_TIMES = [0.0, 20.0, 60.0]
const HAS_LATEX = Sys.which("latex") !== nothing

rc("text", usetex=HAS_LATEX)
rc("font", family="serif", size=18)
rc("axes", linewidth=2.0, labelsize=30)
rc("xtick", labelsize=24)
rc("ytick", labelsize=24)
rc("legend", fontsize=18)
rc("xtick.major", width=2.0, size=8)
rc("ytick.major", width=2.0, size=8)
rc("xtick", direction="in")
rc("ytick", direction="in")

nearest_index(ts, target) = argmin(abs.(ts .- target))

function extract_field(name::AbstractString, key::AbstractString)
    m = match(Regex(key * "([^_]+)"), name)
    return m === nothing ? missing : m.captures[1]
end

fermi_eps(ϵ, T, μ) = 1 / (exp((ϵ - μ) / T) + 1)

function best_fermi_fit(nocc, ϵs)
    Tgrid = collect(range(0.02, 1.50; length=80))
    μgrid = collect(range(-2.5, 2.5; length=161))

    best_rmse = Inf
    best_T = Tgrid[1]
    best_μ = μgrid[1]

    for T in Tgrid, μ in μgrid
        model = fermi_eps.(ϵs, T, μ)
        rmse = sqrt(mean((nocc .- model) .^ 2))
        if rmse < best_rmse
            best_rmse = rmse
            best_T = T
            best_μ = μ
        end
    end

    Tlo = max(0.01, best_T - 0.08)
    Thi = best_T + 0.08
    μlo = best_μ - 0.20
    μhi = best_μ + 0.20

    for T in range(Tlo, Thi; length=81), μ in range(μlo, μhi; length=81)
        model = fermi_eps.(ϵs, T, μ)
        rmse = sqrt(mean((nocc .- model) .^ 2))
        if rmse < best_rmse
            best_rmse = rmse
            best_T = T
            best_μ = μ
        end
    end

    return best_T, best_μ, best_rmse
end

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

dataset = length(ARGS) >= 1 ? ARGS[1] : error("Pass the Rice-Mele dataset stem without GL_/ts_ prefix")
gl_path = joinpath("Data", "GL_" * dataset * ".jld2")
ts_path = joinpath("Data", "ts_" * dataset * ".jld2")

GL_obj = load(gl_path, "GL")
GL = hasproperty(GL_obj, :data) ? GL_obj.data : GL_obj
ts_obj = load(ts_path, "sol")
ts = hasproperty(ts_obj, :t) ? ts_obj.t : ts_obj

t1 = parse(Float64, string(extract_field(dataset, "t1")))
t2 = parse(Float64, string(extract_field(dataset, "t2")))
Δ = parse(Float64, string(extract_field(dataset, "Δ")))
Te = parse(Float64, string(extract_field(dataset, "Te")))
Tb = parse(Float64, string(extract_field(dataset, "Tb")))
η = string(extract_field(dataset, "η"))
α = string(extract_field(dataset, "α"))
λ_q = string(extract_field(dataset, "λ_q"))

L = size(GL, 3)
ks = collect(range(-π, stop=π - 2π / L, length=L))
Nt = length(ts)

nlower_t = zeros(Float64, Nt, L)
nupper_t = zeros(Float64, Nt, L)
ϵs = zeros(Float64, 2L)

for it in 1:Nt
    nlower_t[it, :], nupper_t[it, :] = band_occupations(GL, ks, it; t1=t1, t2=t2, Δ=Δ)
end

for (ik, k) in enumerate(ks)
    vals = eigen(H_k(k; t1=t1, t2=t2, Δ=Δ)).values
    ϵs[2ik - 1] = vals[1]
    ϵs[2ik] = vals[2]
end

fit_T = zeros(Float64, Nt)
fit_μ = zeros(Float64, Nt)
fit_rmse = zeros(Float64, Nt)

for it in 1:Nt
    nocc = vcat(vec(nlower_t[it, :]), vec(nupper_t[it, :]))
    fit_T[it], fit_μ[it], fit_rmse[it] = best_fermi_fit(nocc, ϵs)
end

time_idxs = [nearest_index(ts, t) for t in TARGET_TIMES]
time_eff = ts[time_idxs]

fig, axs = subplots(1, 3; figsize=(15.2, 4.7))
colors = ["black", "#2c7fb8", "#d1495b"]
styles = ["--", "-", "-."]

for (i, it) in enumerate(time_idxs)
    axs[1].plot(ks, vec(nlower_t[it, :]); color=colors[i], linestyle=styles[i], linewidth=2.2,
        label="\$t = $(round(time_eff[i], digits=2))\$")
end
axs[1].set_xlabel(raw"$k$")
axs[1].set_ylabel(raw"$n_{-}(k,t)$")
axs[1].set_xlim(-π, π)
axs[1].set_ylim(-0.02, 1.02)
axs[1].set_xticks([-π, -π / 2, 0, π / 2, π])
axs[1].set_xticklabels([raw"$-\pi$", raw"$-\pi/2$", raw"$0$", raw"$\pi/2$", raw"$\pi$"])
axs[1].legend(frameon=false, loc="lower center")
axs[1].text(0.03, 0.93, raw"$n_{-}$"; transform=axs[1].transAxes, fontsize=26)

axs[2].plot(ts, fit_T; color="#d1495b", linewidth=2.4)
axs[2].axhline(Te; color="black", linestyle="--", linewidth=1.6, label="\$T_e = $(Te)\$")
axs[2].axhline(Tb; color="#377eb8", linestyle=":", linewidth=2.0, label="\$T_B = $(Tb)\$")
axs[2].set_xlabel(raw"$t$")
axs[2].set_ylabel(raw"$T_{\mathrm{eff}}(t)$")
axs[2].set_xlim(ts[1], ts[end])
axs[2].legend(frameon=false, loc="lower right")
axs[2].text(0.06, 0.93, raw"$T_{\mathrm{eff}}$"; transform=axs[2].transAxes, fontsize=26)

axs[3].plot(ts, fit_rmse; color="#5e3c99", linewidth=2.4)
axs[3].set_xlabel(raw"$t$")
axs[3].set_ylabel(raw"$\mathrm{RMSE}(t)$")
axs[3].set_xlim(ts[1], ts[end])
axs[3].text(0.05, 0.93, "Fermi fit error"; transform=axs[3].transAxes, fontsize=24)

fig.text(0.5, -0.02,
    "Rice-Mele, spectral, η = $(η), α = $(α), T_B = $(Tb), p_c = $(λ_q)/a";
    ha="center", va="top", fontsize=20)

fig.tight_layout()
fig.subplots_adjust(bottom=0.20, wspace=0.32)

outfile = joinpath("distributions", "thermalization_case_summary_rice_" * dataset * ".png")
fig.savefig(outfile; dpi=240, bbox_inches="tight")
println("Saved " * outfile)
