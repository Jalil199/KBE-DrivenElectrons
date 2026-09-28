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
    η = eta,
    ωA_max = 20.0,
    dωA = 0.01,
    ωb0 = 0.1,
    v_b = 0.2,
    wq_profile = :power_exp,
    s_q = 1.0,
    λ_q = 1.0,
)

model_delta = ModelElectronBath(; base_kwargs..., boson_kernel=:delta)
model_spectral = ModelElectronBath(; base_kwargs..., boson_kernel=:spectral)

colors = ["red", "blue", "gold"]
styles = ["-", "-.", "--"]

fig, axs = subplots(2, 2; figsize=(14, 10), sharex=true)

for (col, (title, model)) in enumerate([("delta", model_delta), ("spectral", model_spectral)])
    for (i, t) in enumerate(TIMES)
        ξk = Xi_k_at_time(t; model=model, t′=0.0, greater=false, apply_switch=false)
        axs[1, col].plot(model.ks, real.(ξk); color=colors[i], linestyle=styles[i], linewidth=2.2,
            label="\$t = $(round(t, digits=2))\$")
        axs[2, col].plot(model.ks, imag.(ξk); color=colors[i], linestyle=styles[i], linewidth=2.2,
            label="\$t = $(round(t, digits=2))\$")
    end

    axs[1, col].set_title("kernel = $(title)")
    axs[1, col].set_ylabel(raw"$\mathrm{Re}\,\Xi^{<}(k,t,0)$")
    axs[2, col].set_ylabel(raw"$\mathrm{Im}\,\Xi^{<}(k,t,0)$")
    axs[2, col].set_xlabel(raw"$k$")

    for row in 1:2
        axs[row, col].set_xlim(-π, π)
        axs[row, col].set_xticks([-π, -π / 2, 0, π / 2, π])
        axs[row, col].set_xticklabels([L"-\pi", L"-\pi/2", L"0", L"\pi/2", L"\pi"])
        axs[row, col].legend(frameon=false, loc="best")
    end
end

fig.text(
    0.5, 0.01,
    "linear dispersion, η = $(eta), Tb = $(base_kwargs.Tb), λ_q = $(base_kwargs.λ_q), lesser kernel, switch off";
    ha="center", va="bottom", fontsize=13
)
fig.tight_layout(rect=[0, 0.04, 1, 1])

out_path = joinpath(OUTPUT_DIR, "xi_kernel_comparison_linear_eta$(eta).png")
fig.savefig(out_path; dpi=300, bbox_inches="tight")
close(fig)

println("Saved $(out_path)")
