const LOCAL_LATEX_BIN = joinpath(@__DIR__, ".latex-env", "bin")
const LOCAL_LATEX_BIN_DEFAULTS = joinpath(@__DIR__, ".latex-env-defaults", "bin")
if isdir(LOCAL_LATEX_BIN_DEFAULTS)
    ENV["PATH"] = LOCAL_LATEX_BIN_DEFAULTS * ":" * ENV["PATH"]
elseif isdir(LOCAL_LATEX_BIN)
    ENV["PATH"] = LOCAL_LATEX_BIN * ":" * ENV["PATH"]
end

using PyPlot
using LaTeXStrings

include("./main.jl")

function latex_usable()
    latex = Sys.which("latex")
    latex === nothing && return false
    dir = mktempdir()
    texfile = joinpath(dir, "test.tex")
    write(texfile, raw"\documentclass{article}\begin{document}$x_1$\end{document}")
    ok = try
        success(`$(latex) -interaction=nonstopmode -halt-on-error -output-directory=$(dir) $(texfile)`)
    catch
        false
    end
    rm(dir; recursive=true, force=true)
    return ok
end

const HAS_LATEX = latex_usable()
rc("text", usetex=HAS_LATEX)
rc("font", family="serif")
if HAS_LATEX
    rc("text.latex", preamble=raw"\usepackage{amsmath}")
end
rc("font", size=16)
rc("axes", labelsize=16, titlesize=16)
rc("xtick", labelsize=13)
rc("ytick", labelsize=13)

const OUTPUT_DIR = "kernels"
mkpath(OUTPUT_DIR)

eta = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.5
λ_fixed = 1.0

function make_model(; η, λ_q)
    ModelElectronBath(;
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
        η = η,
        ωA_max = 2.0,
        dωA = 0.01,
        ωb0 = 0.1,
        v_b = 0.2,
        wq_profile = :power_exp,
        s_q = 1.0,
        λ_q = λ_q,
    )
end

model = make_model(; η=eta, λ_q=λ_fixed)
ωs = collect(-model.ωA_max:model.dωA:model.ωA_max)
qs = model.ks
τmax = 15.0
dτ = 0.05
τs = collect(0.0:dτ:τmax)

function xi_retarded_freq(ω, ωq, g2q, η)
    g2q * (1 / (ω - ωq + 1im * η) - 1 / (ω + ωq + 1im * η))
end

heat_im = zeros(Float64, length(ωs), length(qs))
for iq in eachindex(qs)
    ξω = xi_retarded_freq.(ωs, model.ωq[iq], model.g2q[iq], eta)
    heat_im[:, iq] .= -imag.(ξω)
end

heat_tau_ret = zeros(Float64, length(τs), length(qs))
for (it, τ) in enumerate(τs)
    ξ_less = Xi_k_at_time(τ; model=model, t′=0.0, greater=false, apply_switch=false)
    ξ_great = Xi_k_at_time(τ; model=model, t′=0.0, greater=true, apply_switch=false)
    ξ_ret = ξ_great .- ξ_less
    heat_tau_ret[it, :] .= abs.(ξ_ret)
end

cmap = plt.get_cmap("RdBu_r", 3024)
cmap_tau = plt.get_cmap("Reds", 3024)
vabs_im = maximum(abs.(heat_im))
vmax_tau_ret = maximum(heat_tau_ret)

fig, axs = subplots(1, 2; figsize=(11.4, 4.8))

im_tau = axs[1].imshow(
    heat_tau_ret[end:-1:1, :];
    extent=[minimum(qs), maximum(qs), minimum(τs), maximum(τs)],
    aspect="auto",
    cmap=cmap_tau,
    interpolation="nearest",
    vmin=0.0,
    vmax=vmax_tau_ret,
)

im_im = axs[2].imshow(
    heat_im[end:-1:1, :];
    extent=[minimum(qs), maximum(qs), minimum(ωs), maximum(ωs)],
    aspect="auto",
    cmap=cmap,
    interpolation="nearest",
    vmin=-vabs_im,
    vmax=vabs_im,
)

for ax in axs
    ax.set_xlim(-π, π)
    ax.set_xticks([-π, -π / 2, 0, π / 2, π])
    ax.set_xticklabels([L"-\pi", L"-\pi/2", L"0", L"\pi/2", L"\pi"])
    ax.set_xlabel(raw"$q$")
    ax.tick_params(axis="both", which="both", direction="out", length=4, width=1)
    for spine in ax.spines.values()
        spine.set_linewidth(1.0)
    end
end

axs[1].set_ylabel(raw"$\tau$")
axs[1].set_ylim(0.0, τmax)
axs[1].text(0.05, 0.92, "\$\\eta = $(eta)\$"; transform=axs[1].transAxes, fontsize=11)
axs[1].text(0.05, 0.84, "\$\\lambda_q = $(λ_fixed)\$"; transform=axs[1].transAxes, fontsize=11)

axs[2].set_ylabel(raw"$\omega$")
axs[2].set_ylim(-2.0, 2.0)
axs[2].plot(qs, model.ωq; color="black", linewidth=1.3, linestyle="-", alpha=0.9)
axs[2].text(0.05, 0.92, "\$\\eta = $(eta)\$"; transform=axs[2].transAxes, fontsize=11)
axs[2].text(0.05, 0.84, "\$\\lambda_q = $(λ_fixed)\$"; transform=axs[2].transAxes, fontsize=11)

for (ax, img, labeltxt) in [
    (axs[1], im_tau, raw"$|\Xi_q^{R}(\tau)|$"),
    (axs[2], im_im, raw"$-\Im\,\Xi_q^{R}(\omega)$"),
]
    box = ax.get_position()
    cax = fig.add_axes([box.x0 + 0.14 * box.width, box.y1 + 0.01, 0.72 * box.width, 0.018])
    cbar = fig.colorbar(img, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cax.text(0.5, -1.9, labeltxt; transform=cax.transAxes, ha="center", va="top", fontsize=14)
end

fig.tight_layout(rect=[0, 0.02, 1, 0.93])

out_path = joinpath(OUTPUT_DIR, "xi_kernel_2panel_summary_linear_eta$(eta)_lambda$(λ_fixed).png")
fig.savefig(out_path; dpi=300, bbox_inches="tight")
close(fig)

println("Saved $(out_path)")
