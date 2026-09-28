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
rc("legend", fontsize=13)

const OUTPUT_DIR = "kernels"
mkpath(OUTPUT_DIR)

eta = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.5

model = ModelElectronBath(;
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
    ωA_max = 2.0,
    dωA = 0.01,
    ωb0 = 0.1,
    v_b = 0.2,
    wq_profile = :power_exp,
    s_q = 1.0,
    λ_q = 1.0,
)

ωs = collect(-model.ωA_max:model.dωA:model.ωA_max)
q_indices = [
    argmin(abs.(model.ks .- 0.0)),
    argmin(abs.(model.ks .- (pi / 2))),
    argmin(abs.(model.ks .- pi)),
]
q_labels = [
    raw"$q \approx 0$",
    raw"$q \approx \pi/2$",
    raw"$q \approx \pi$",
]
colors = ["red", "blue", "gold"]
styles = ["-", "-.", "--"]

function xi_retarded_freq(ω, ωq, g2q, η)
    return g2q * (1 / (ω - ωq + 1im * η) - 1 / (ω + ωq + 1im * η))
end

fig, axs = subplots(1, 2; figsize=(14, 5), sharex=true)

for (i, qidx) in enumerate(q_indices)
    ωq = model.ωq[qidx]
    g2q = model.g2q[qidx]
    ξω = xi_retarded_freq.(ωs, ωq, g2q, eta)

    axs[1].plot(ωs, real.(ξω); color=colors[i], linestyle=styles[i], linewidth=2.2, label=q_labels[i])
    axs[2].plot(ωs, imag.(ξω); color=colors[i], linestyle=styles[i], linewidth=2.2, label=q_labels[i])
end

axs[1].set_title(raw"$\mathrm{Re}\,\Xi^{R}(q,\omega)$")
axs[2].set_title(raw"$\mathrm{Im}\,\Xi^{R}(q,\omega)$")
axs[1].set_ylabel(raw"$\Xi^{R}(q,\omega)$")
axs[2].set_ylabel(raw"$\Xi^{R}(q,\omega)$")
axs[1].set_xlabel(raw"$\omega$")
axs[2].set_xlabel(raw"$\omega$")

for ax in axs
    ax.axhline(0.0; color="black", linewidth=0.8, alpha=0.5)
    ax.legend(frameon=false, loc="best")
end

fig.text(
    0.5, 0.01,
    "spectral kernel, linear dispersion, η = $(eta), Tb = $(model.Tb), λ_q = $(model.λ_q), representative q values";
    ha="center", va="bottom", fontsize=13
)
fig.tight_layout(rect=[0, 0.05, 1, 1])

out_path = joinpath(OUTPUT_DIR, "xi_spectral_retarded_frequency_linear_eta$(eta).png")
fig.savefig(out_path; dpi=300, bbox_inches="tight")
close(fig)

println("Saved $(out_path)")
