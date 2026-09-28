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
rc("legend", fontsize=10)

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
τ_fixed = 5.0

function xi_retarded_freq(ω, ωq, g2q, η)
    g2q * (1 / (ω - ωq + 1im * η) - 1 / (ω + ωq + 1im * η))
end

heat_re = zeros(Float64, length(ωs), length(qs))
heat_im = zeros(Float64, length(ωs), length(qs))
for iq in eachindex(qs)
    ξω = xi_retarded_freq.(ωs, model.ωq[iq], model.g2q[iq], eta)
    heat_re[:, iq] .= real.(ξω)
    heat_im[:, iq] .= -imag.(ξω)
end

heat_tau_ret = zeros(Float64, length(τs), length(qs))
for (it, τ) in enumerate(τs)
    ξ_less = Xi_k_at_time(τ; model=model, t′=0.0, greater=false, apply_switch=false)
    ξ_great = Xi_k_at_time(τ; model=model, t′=0.0, greater=true, apply_switch=false)
    ξ_ret = ξ_great .- ξ_less
    heat_tau_ret[it, :] .= abs.(ξ_ret)
end

function memory_curve(model, τs)
    memory = zeros(Float64, length(τs))
    for (it, τ) in enumerate(τs)
        ξ_less = Xi_k_at_time(τ; model=model, t′=0.0, greater=false, apply_switch=false)
        ξ_great = Xi_k_at_time(τ; model=model, t′=0.0, greater=true, apply_switch=false)
        ξ_ret = ξ_great .- ξ_less
        memory[it] = sum(abs.(ξ_ret)) / length(ξ_ret)
    end
    memory
end

λqs = [0.25, 0.5, 1.0]
ηs = [0.05, 0.25, 0.5, 1.0]
colors_λ = ["#3b4cc0", "#1f9e89", "#d1495b"]
styles_λ = ["-", "--", "-."]
colors_η = ["#3b4cc0", "#2c7fb8", "#1f9e89", "#d1495b"]
styles_η = ["-", "--", "-.", ":"]

cmap = plt.get_cmap("RdBu_r", 3024)
cmap_tau = plt.get_cmap("Reds", 3024)
vabs_im = maximum(abs.(heat_im))
vmax_tau_ret = maximum(heat_tau_ret)

fig, axs = subplots(2, 2; figsize=(11.5, 9.2))

im_tau = axs[1, 1].imshow(
    heat_tau_ret[end:-1:1, :];
    extent=[minimum(qs), maximum(qs), minimum(τs), maximum(τs)],
    aspect="auto",
    cmap=cmap_tau,
    interpolation="nearest",
    vmin=0.0,
    vmax=vmax_tau_ret,
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

axs[1, 1].set_xlim(-π, π)
axs[1, 1].set_ylim(0.0, τmax)
axs[1, 1].set_xticks([-π, -π / 2, 0, π / 2, π])
axs[1, 1].set_xticklabels([L"-\pi", L"-\pi/2", L"0", L"\pi/2", L"\pi"])
axs[1, 1].set_xlabel(raw"$q$")
axs[1, 1].set_ylabel(raw"$\tau$")
axs[1, 1].tick_params(axis="both", which="both", direction="out", length=4, width=1)

axs[1, 2].plot(qs, model.ωq; color="black", linewidth=1.3, linestyle="-", alpha=0.9, label=raw"$\omega_q$")
axs[1, 2].set_xlim(-π, π)
axs[1, 2].set_ylim(-2.0, 2.0)
axs[1, 2].set_xticks([-π, -π / 2, 0, π / 2, π])
axs[1, 2].set_xticklabels([L"-\pi", L"-\pi/2", L"0", L"\pi/2", L"\pi"])
axs[1, 2].set_xlabel(raw"$q$")
axs[1, 2].set_ylabel(raw"$\omega$")
axs[1, 2].tick_params(axis="both", which="both", direction="out", length=4, width=1)
axs[1, 2].ticklabel_format(axis="y", style="sci", scilimits=(-1, 2), useMathText=true)
for ax in axs[1, :]
    for spine in ax.spines.values()
        spine.set_linewidth(1.0)
    end
end
axs[1, 2].legend(frameon=false, loc="upper right", handlelength=1.8)
axs[1, 1].text(0.05, 0.92, "\$\\eta = $(eta)\$"; transform=axs[1, 1].transAxes, fontsize=11)
axs[1, 1].text(0.05, 0.84, "\$\\lambda_q = $(λ_fixed)\$"; transform=axs[1, 1].transAxes, fontsize=11)
axs[1, 2].text(0.05, 0.92, "\$\\eta = $(eta)\$"; transform=axs[1, 2].transAxes, fontsize=11)
axs[1, 2].text(0.05, 0.84, "\$\\lambda_q = $(λ_fixed)\$"; transform=axs[1, 2].transAxes, fontsize=11)

for (i, λ_q) in enumerate(λqs)
    model_i = make_model(; η=eta, λ_q=λ_q)
    ξ_less = Xi_k_at_time(τ_fixed; model=model_i, t′=0.0, greater=false, apply_switch=false)
    ξ_great = Xi_k_at_time(τ_fixed; model=model_i, t′=0.0, greater=true, apply_switch=false)
    ξ_ret = ξ_great .- ξ_less
    axs[2, 1].plot(model_i.ks, abs.(ξ_ret); color=colors_λ[i], linestyle=styles_λ[i], linewidth=2.0,
        label="\$\\lambda_q = $(λ_q)\$")
end

for (i, ηi) in enumerate(ηs)
    mem = memory_curve(make_model(; η=ηi, λ_q=λ_fixed), τs)
    axs[2, 2].plot(τs, mem; color=colors_η[i], linestyle=styles_η[i], linewidth=2.0,
        label="\$\\eta = $(ηi)\$")
end

axs[2, 1].set_xlim(-π, π)
axs[2, 1].set_xlabel(raw"$q$")
axs[2, 1].set_ylabel(raw"$|\Xi_q^R(\tau)|$")
axs[2, 1].set_xticks([-π, -π / 2, 0, π / 2, π])
axs[2, 1].set_xticklabels([L"-\pi", L"-\pi/2", L"0", L"\pi/2", L"\pi"])
axs[2, 1].tick_params(axis="both", which="both", direction="out", length=4, width=1)
axs[2, 1].grid(alpha=0.18, linewidth=0.6)

axs[2, 2].set_xlim(0.0, τmax)
axs[2, 2].set_xlabel(raw"$\tau$")
axs[2, 2].set_ylabel(raw"$M(\tau)$")
axs[2, 2].tick_params(axis="both", which="both", direction="out", length=4, width=1)
axs[2, 2].grid(alpha=0.18, linewidth=0.6)

for ax in axs[2, :]
    for spine in ax.spines.values()
        spine.set_linewidth(1.0)
    end
end

axs[2, 1].legend(frameon=false, loc="upper right", handlelength=1.8, borderaxespad=0.2, labelspacing=0.3)
axs[2, 2].legend(frameon=false, loc="upper right", handlelength=1.8, borderaxespad=0.2, labelspacing=0.3)
axs[2, 1].text(0.05, 0.92, "\$\\eta = $(eta)\$"; transform=axs[2, 1].transAxes, fontsize=11)
axs[2, 1].text(0.05, 0.84, "\$\\tau = $(τ_fixed)\$"; transform=axs[2, 1].transAxes, fontsize=11)
axs[2, 2].text(0.05, 0.92, "\$\\lambda_q = $(λ_fixed)\$"; transform=axs[2, 2].transAxes, fontsize=11)

for (ax, img, labeltxt) in [
    (axs[1, 1], im_tau, raw"$|\Xi_q^{R}(\tau)|$"),
    (axs[1, 2], im_im, raw"$-\Im\,\Xi_q^{R}(\omega)$"),
]
    box = ax.get_position()
    cax = fig.add_axes([box.x0 + 0.16 * box.width, box.y1 + 0.01, 0.68 * box.width, 0.012])
    cbar = fig.colorbar(img, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cax.text(0.5, -1.7, labeltxt; transform=cax.transAxes, ha="center", va="top", fontsize=14)
end

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

out_path = joinpath(OUTPUT_DIR, "xi_kernel_4panel_summary_linear_eta$(eta)_lambda$(λ_fixed).png")
fig.savefig(out_path; dpi=300, bbox_inches="tight")
close(fig)

println("Saved $(out_path)")
