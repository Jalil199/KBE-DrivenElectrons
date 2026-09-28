using JLD2
using PyPlot
using LaTeXStrings
using Statistics

const HAS_LATEX = Sys.which("latex") !== nothing
rc("text", usetex=HAS_LATEX)
rc("font", family="serif")
if HAS_LATEX
    rc("text.latex", preamble=raw"\usepackage{amsmath}")
end
rc("font", size=16)
rc("axes", labelsize=18, titlesize=18)
rc("xtick", labelsize=14)
rc("ytick", labelsize=14)
rc("legend", fontsize=12)

const OUTPUT_DIR = "distributions"
mkpath(OUTPUT_DIR)

const DEFAULT_DATASET = "L100_Te1.0_Tb0.2_u0.0_γ1.0_dispersion_α1.0_s1.0_ωc10.0_linear_spectral_η0.5_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q0.5_t050.0_ω03.141592653589793_σ2.0_A0.0_switch0_ti3.0_to20.0_tmax60"
const TARGET_TIMES = [0.0, 20.0, 60.0]

fermi(ϵ, T, μ) = 1 / (exp((ϵ - μ) / T) + 1)
nearest_index(xs, x) = argmin(abs.(xs .- x))

function local_fit_rmse(nk, ϵs, T, μ)
    ff = fermi.(ϵs, T, μ)
    sqrt(mean((nk .- ff).^2))
end

function best_fermi_fit(nk, ϵs)
    coarse_T = 0.02:0.02:1.50
    coarse_μ = -3.0:0.05:3.0

    best_rmse = Inf
    best_T = NaN
    best_μ = NaN
    for T in coarse_T, μ in coarse_μ
        rmse = local_fit_rmse(nk, ϵs, T, μ)
        if rmse < best_rmse
            best_rmse = rmse
            best_T = T
            best_μ = μ
        end
    end

    fine_T_min = max(0.01, best_T - 0.04)
    fine_T_max = min(1.50, best_T + 0.04)
    fine_μ_min = max(-3.0, best_μ - 0.10)
    fine_μ_max = min(3.0, best_μ + 0.10)

    for T in fine_T_min:0.002:fine_T_max, μ in fine_μ_min:0.01:fine_μ_max
        rmse = local_fit_rmse(nk, ϵs, T, μ)
        if rmse < best_rmse
            best_rmse = rmse
            best_T = T
            best_μ = μ
        end
    end

    return best_T, best_μ, best_rmse
end

function panel_label!(ax, txt)
    ax.text(0.03, 0.94, txt; transform=ax.transAxes, ha="left", va="top", fontsize=13)
end

dataset = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_DATASET

occ_path = joinpath("Data", "occ_$(dataset).jld2")
gl_path = joinpath("Data", "GL_$(dataset).jld2")
ts_path = joinpath("Data", "ts_$(dataset).jld2")

if isfile(occ_path)
    d = load(occ_path)
    nk_t = d["nk_t"]
    ks = d["ks"]
    p = d["params"]
elseif isfile(gl_path)
    d = load(gl_path)
    GL = hasproperty(d["GL"], :data) ? d["GL"].data : d["GL"]
    L = size(GL, 1)
    Nt = size(GL, 2)
    nk_t = zeros(Float64, Nt, L)
    @inbounds for it in 1:Nt
        nk_t[it, :] .= imag.(GL[:, it, it])
    end
    p = (; Te=1.0, Tb=0.2, u=0.0, γ=1.0, η=missing, α=missing, λ_q=missing)
    ks = collect(range(-pi, stop=pi - 2pi / size(GL, 1), length=size(GL, 1)))
else
    error("No occ_*.jld2 or GL_*.jld2 file found for dataset $(dataset)")
end

function extract_field(name::AbstractString, key::AbstractString)
    m = match(Regex("$(key)([^_]+)"), name)
    return m === nothing ? missing : tryparse(Float64, m.captures[1])
end

if ismissing(p.η)
    p = merge(p, (; Te = something(p.Te, extract_field(dataset, "Te")),
                    Tb = something(p.Tb, extract_field(dataset, "Tb")),
                    u = p.u,
                    γ = p.γ,
                    η = extract_field(dataset, "η"),
                    α = extract_field(dataset, "α"),
                    λ_q = extract_field(dataset, "λ_q")))
end

ts_obj = load(ts_path, "sol")
ts = hasproperty(ts_obj, :t) ? ts_obj.t : ts_obj

ϵs = p.u .- 2p.γ .* cos.(ks)
Nt = length(ts)

fit_T = zeros(Float64, Nt)
fit_μ = zeros(Float64, Nt)
fit_rmse = zeros(Float64, Nt)

for it in 1:Nt
    fit_T[it], fit_μ[it], fit_rmse[it] = best_fermi_fit(vec(nk_t[it, :]), ϵs)
end

time_idxs = [nearest_index(ts, t) for t in TARGET_TIMES]
time_eff = ts[time_idxs]

fig, axs = subplots(1, 3; figsize=(15.2, 4.7))
colors = ["black", "#2c7fb8", "#d1495b"]
styles = ["--", "-", "-."]

for (i, it) in enumerate(time_idxs)
    axs[1].plot(ks, vec(nk_t[it, :]); color=colors[i], linestyle=styles[i], linewidth=2.2,
        label="\$t = $(round(time_eff[i], digits=2))\$")
end
axs[1].set_xlabel(raw"$k$")
axs[1].set_ylabel(raw"$n_k(t)$")
axs[1].set_xlim(-π, π)
axs[1].set_ylim(-0.02, 1.02)
axs[1].set_xticks([-π, -π / 2, 0, π / 2, π])
axs[1].set_xticklabels([L"-\pi", L"-\pi/2", L"0", L"\pi/2", L"\pi"])
axs[1].legend(frameon=false, loc="best")
panel_label!(axs[1], raw"$n_k$")

axs[2].plot(ts, fit_T; color="#d1495b", linewidth=2.2)
axs[2].axhline(p.Te; color="black", linewidth=1.6, linestyle="--", label="\$T_e^{\\rm init} = $(p.Te)\$")
axs[2].axhline(p.Tb; color="#2c7fb8", linewidth=1.6, linestyle=":", label="\$T_B = $(p.Tb)\$")
axs[2].set_xlabel(raw"$t$")
axs[2].set_ylabel(raw"$T_{\mathrm{eff}}(t)$")
axs[2].set_xlim(first(ts), last(ts))
axs[2].legend(frameon=false, loc="best")
panel_label!(axs[2], raw"$T_{\rm eff}$")

axs[3].plot(ts, fit_rmse; color="#5e3c99", linewidth=2.2)
axs[3].set_xlabel(raw"$t$")
axs[3].set_ylabel(raw"$\mathrm{RMSE}(t)$")
axs[3].set_xlim(first(ts), last(ts))
panel_label!(axs[3], raw"$\mathrm{Fermi\ fit\ error}$")

for ax in axs
    ax.tick_params(axis="both", which="both", direction="out", length=4, width=1)
    for spine in ax.spines.values()
        spine.set_linewidth(1.0)
    end
end

fig.text(0.5, 0.01,
    "spectral, η = $(p.η), α = $(p.α), T_B = $(p.Tb), λ_q = $(p.λ_q)";
    ha="center", va="bottom", fontsize=12)
fig.tight_layout(rect=[0, 0.05, 1, 1])

out_path = joinpath(OUTPUT_DIR, "thermalization_case_summary_" * dataset * ".png")
fig.savefig(out_path; dpi=300, bbox_inches="tight")
close(fig)

println("Saved $(out_path)")
