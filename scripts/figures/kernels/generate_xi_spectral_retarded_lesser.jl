using PyPlot
using LaTeXStrings

include("./main.jl")

const HAS_LATEX = Sys.which("latex") !== nothing
rc("text", usetex=HAS_LATEX)
rc("font", family="serif")
if HAS_LATEX
    rc("text.latex", preamble=raw"\usepackage{amsmath}")
end
rc("font", size=18)
rc("axes", labelsize=22, titlesize=20)
rc("xtick", labelsize=18)
rc("ytick", labelsize=18)
rc("legend", fontsize=14)

const TIMES = [0.0, 5.0, 10.0]
const OUTPUT_DIR = "kernels"

mkpath(OUTPUT_DIR)

eta = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.5

base_kwargs = (
    L = 100,
    Te = 1.0,
    Tb = 0.05,
    u = 0.0,
    γ = 1.0,
    α = 0.5,
    s = 1.0,
    ωc = 10.0,
    t0 = 50.0,
    ω0 = Float64(pi),
    σ = 2.0,
    A = 0.0,
    switch_on = false,
    ti = 3.0,
    to = 20.0,
    bath_type = :dispersion,
    dispersion_type = :linear,
    boson_kernel = :spectral,
    η = eta,
    ωA_max = 20.0,
    dωA = 0.01,
    ωb0 = 0.1,
    v_b = 0.2,
    wq_profile = :power_exp,
    s_q = 1.0,
    λ_q = 1.0,
)

model = ModelElectronBath(; base_kwargs...)

colors = ["red", "blue", "gold"]
styles = ["-", "-.", "--"]

fig, axs = subplots(2, 2; figsize=(14, 10), sharex=true)

for (i, t) in enumerate(TIMES)
    ξ_less = Xi_k_at_time(t; model=model, t′=0.0, greater=false, apply_switch=false)
    ξ_great = Xi_k_at_time(t; model=model, t′=0.0, greater=true, apply_switch=false)
    ξ_ret = (t >= 0.0 ? 1.0 : 0.0) .* (ξ_great .- ξ_less)

    axs[1, 1].plot(model.ks, real.(ξ_ret); color=colors[i], linestyle=styles[i], linewidth=2.2,
        label="\$t = $(round(t, digits=2))\$")
    axs[2, 1].plot(model.ks, imag.(ξ_ret); color=colors[i], linestyle=styles[i], linewidth=2.2,
        label="\$t = $(round(t, digits=2))\$")
    axs[1, 2].plot(model.ks, real.(ξ_less); color=colors[i], linestyle=styles[i], linewidth=2.2,
        label="\$t = $(round(t, digits=2))\$")
    axs[2, 2].plot(model.ks, imag.(ξ_less); color=colors[i], linestyle=styles[i], linewidth=2.2,
        label="\$t = $(round(t, digits=2))\$")
end

axs[1, 1].set_title(raw"$\Xi^{R}(q,t,0)$")
axs[2, 1].set_title(raw"$\Xi^{R}(q,t,0)$")
axs[1, 2].set_title(raw"$\Xi^{<}(q,t,0)$")
axs[2, 2].set_title(raw"$\Xi^{<}(q,t,0)$")

axs[1, 1].set_ylabel(raw"$\mathrm{Re}\,\Xi^{R}(q,t,0)$")
axs[2, 1].set_ylabel(raw"$\mathrm{Im}\,\Xi^{R}(q,t,0)$")
axs[1, 2].set_ylabel(raw"$\mathrm{Re}\,\Xi^{<}(q,t,0)$")
axs[2, 2].set_ylabel(raw"$\mathrm{Im}\,\Xi^{<}(q,t,0)$")

axs[2, 1].set_xlabel(raw"$q$")
axs[2, 2].set_xlabel(raw"$q$")

for ax in axs
    ax.set_xlim(-π, π)
    ax.set_xticks([-π, -π / 2, 0, π / 2, π])
    ax.set_xticklabels([L"-\pi", L"-\pi/2", L"0", L"\pi/2", L"\pi"])
    ax.legend(frameon=false, loc="best")
end

fig.text(
    0.5, 0.01,
    "spectral kernel, linear dispersion, η = $(eta), Tb = $(base_kwargs.Tb), λ_q = $(base_kwargs.λ_q), switch off";
    ha="center", va="bottom", fontsize=13
)
fig.tight_layout(rect=[0, 0.04, 1, 1])

out_path = joinpath(OUTPUT_DIR, "xi_spectral_retarded_lesser_linear_eta$(eta).png")
fig.savefig(out_path; dpi=300, bbox_inches="tight")
close(fig)

println("Saved $(out_path)")
