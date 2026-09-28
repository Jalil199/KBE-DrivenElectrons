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
rc("font", size=24)
rc("axes", labelsize=26, titlesize=22)
rc("xtick", labelsize=20)
rc("ytick", labelsize=20)
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
qs = model.ks
τmax = 12.0
dτ = 0.05
τs = collect(0.0:dτ:τmax)

function xi_retarded_freq(ω, ωq, g2q, η)
    return g2q * (1 / (ω - ωq + 1im * η) - 1 / (ω + ωq + 1im * η))
end

heat_re = zeros(Float64, length(ωs), length(qs))
heat_im = zeros(Float64, length(ωs), length(qs))
for (iq, q) in enumerate(qs)
    ωq = model.ωq[iq]
    g2q = model.g2q[iq]
    ξω = xi_retarded_freq.(ωs, ωq, g2q, eta)
    heat_re[:, iq] .= real.(ξω)
    heat_im[:, iq] .= -imag.(ξω)
end

heat_tau_ret = zeros(Float64, length(τs), length(qs))
heat_tau_less = zeros(Float64, length(τs), length(qs))
for (it, τ) in enumerate(τs)
    ξ_less = Xi_k_at_time(τ; model=model, t′=0.0, greater=false, apply_switch=false)
    ξ_great = Xi_k_at_time(τ; model=model, t′=0.0, greater=true, apply_switch=false)
    ξ_ret = ξ_great .- ξ_less
    heat_tau_ret[it, :] .= abs.(ξ_ret)
    heat_tau_less[it, :] .= imag.(ξ_less)
end

cmap = plt.get_cmap("RdBu", 3024)
fs = 22
vabs_re = maximum(abs.(heat_re))
vabs_im = maximum(abs.(heat_im))
vmax_tau_ret = maximum(heat_tau_ret)
vabs_tau_less = maximum(abs.(heat_tau_less))

fig, axs = subplots(2, 2; figsize=(12.8, 10.2), sharex=true)

im_re = axs[1, 1].imshow(
    heat_re[end:-1:1, :];
    extent=[minimum(qs), maximum(qs), minimum(ωs), maximum(ωs)],
    aspect="auto",
    cmap=cmap,
    interpolation="nearest",
    vmin=-vabs_re,
    vmax=vabs_re,
)

im_im = axs[1, 2].imshow(
    heat_im[end:-1:1, :];
    extent=[minimum(qs), maximum(qs), minimum(ωs), maximum(ωs)],
    aspect="auto",
    cmap=cmap,
    interpolation="nearest",
    vmin=-vabs_im,
    vmax=vabs_im,
)

im_tau_ret = axs[2, 1].imshow(
    heat_tau_ret[end:-1:1, :];
    extent=[minimum(qs), maximum(qs), minimum(τs), maximum(τs)],
    aspect="auto",
    cmap=cmap,
    interpolation="nearest",
    vmin=0.0,
    vmax=vmax_tau_ret,
)

im_tau_less = axs[2, 2].imshow(
    heat_tau_less[end:-1:1, :];
    extent=[minimum(qs), maximum(qs), minimum(τs), maximum(τs)],
    aspect="auto",
    cmap=cmap,
    interpolation="nearest",
    vmin=-vabs_tau_less,
    vmax=vabs_tau_less,
)

for ax in axs[1, :]
    ax.plot(qs, model.ωq; color="black", linewidth=1.6, linestyle="-", alpha=0.9, label=raw"$\omega_q$")
    ax.set_xlim(-π, π)
    ax.set_ylim(-2.0, 2.0)
    ax.set_xticks([-π, -π / 2, 0, π / 2, π])
    ax.set_xticklabels([L"-\pi", L"-\pi/2", L"0", L"\pi/2", L"\pi"])
    ax.set_xlabel(raw"$q$")
    ax.tick_params(axis="both", which="both", labelsize=fs, direction="out", length=6, width=1)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 2), useMathText=true)
    ax.yaxis.offsetText.set_fontsize(fs)
    for spine in ax.spines.values()
        spine.set_linewidth(1.0)
    end
end

for ax in axs[2, :]
    ax.set_xlim(-π, π)
    ax.set_ylim(0.0, τmax)
    ax.set_xticks([-π, -π / 2, 0, π / 2, π])
    ax.set_xticklabels([L"-\pi", L"-\pi/2", L"0", L"\pi/2", L"\pi"])
    ax.set_xlabel(raw"$q$")
    ax.tick_params(axis="both", which="both", labelsize=fs, direction="out", length=6, width=1)
    for spine in ax.spines.values()
        spine.set_linewidth(1.0)
    end
end

axs[1, 1].set_ylabel(raw"$\omega\;(\gamma/\hbar)$")
axs[2, 1].set_ylabel(raw"$\tau$")
axs[1, 2].legend(frameon=false, loc="upper right", handlelength=2.0)

box1 = axs[1, 1].get_position()
box2 = axs[1, 2].get_position()
box3 = axs[2, 1].get_position()
box4 = axs[2, 2].get_position()
cbar_ax1 = fig.add_axes([box1.x0 + 0.16 * box1.width, box1.y1 + 0.012, 0.68 * box1.width, 0.015])
cbar_ax2 = fig.add_axes([box2.x0 + 0.16 * box2.width, box2.y1 + 0.012, 0.68 * box2.width, 0.015])
cbar_ax3 = fig.add_axes([box3.x0 + 0.16 * box3.width, box3.y1 + 0.012, 0.68 * box3.width, 0.015])
cbar_ax4 = fig.add_axes([box4.x0 + 0.16 * box4.width, box4.y1 + 0.012, 0.68 * box4.width, 0.015])
cbar_re = fig.colorbar(im_re, cax=cbar_ax1, orientation="horizontal")
cbar_im = fig.colorbar(im_im, cax=cbar_ax2, orientation="horizontal")
cbar_tau_ret = fig.colorbar(im_tau_ret, cax=cbar_ax3, orientation="horizontal")
cbar_tau_less = fig.colorbar(im_tau_less, cax=cbar_ax4, orientation="horizontal")
for cbar in (cbar_re, cbar_im, cbar_tau_ret, cbar_tau_less)
    cbar.ax.tick_params(labelsize=fs - 4)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
end
cbar_ax1.text(0.5, -1.9, raw"$\Re\,\Xi_q^{R}(\omega)$"; transform=cbar_ax1.transAxes,
    ha="center", va="top", fontsize=fs)
cbar_ax2.text(0.5, -1.9, raw"$-\Im\,\Xi_q^{R}(\omega)$"; transform=cbar_ax2.transAxes,
    ha="center", va="top", fontsize=fs)
cbar_ax3.text(0.5, -1.9, raw"$|\Xi_q^{R}(\tau)|$"; transform=cbar_ax3.transAxes,
    ha="center", va="top", fontsize=fs)
cbar_ax4.text(0.5, -1.9, raw"$\Im\,\Xi_q^{<}(\tau)$"; transform=cbar_ax4.transAxes,
    ha="center", va="top", fontsize=fs)
fig.tight_layout(rect=[0, 0.05, 1, 0.96])

out_path = joinpath(OUTPUT_DIR, "xi_spectral_retarded_heatmap_linear_eta$(eta).png")
fig.savefig(out_path; dpi=300, bbox_inches="tight")
close(fig)

println("Saved $(out_path)")
