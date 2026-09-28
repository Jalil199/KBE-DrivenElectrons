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
λ_q = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0

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

function xi_retarded_freq(ω, ωq, g2q, η)
    g2q * (1 / (ω - ωq + 1im * η) - 1 / (ω + ωq + 1im * η))
end

model = make_model(; η=eta, λ_q=λ_q)
ωs = collect(-model.ωA_max:model.dωA:model.ωA_max)
qs = model.ks

heat_im = zeros(Float64, length(ωs), length(qs))
for iq in eachindex(qs)
    ξω = xi_retarded_freq.(ωs, model.ωq[iq], model.g2q[iq], eta)
    heat_im[:, iq] .= -imag.(ξω)
end

vabs_im = maximum(abs.(heat_im))
cmap = plt.get_cmap("RdBu_r", 3024)

fig, ax = subplots(figsize=(6.6, 5.0))
im = ax.imshow(
    heat_im[end:-1:1, :];
    extent=[minimum(qs), maximum(qs), minimum(ωs), maximum(ωs)],
    aspect="auto",
    cmap=cmap,
    interpolation="nearest",
    vmin=-vabs_im,
    vmax=vabs_im,
)

ax.plot(qs, model.ωq; color="black", linewidth=1.3, linestyle="-", alpha=0.9)
ax.set_xlim(-π, π)
ax.set_ylim(-2.0, 2.0)
ax.set_xticks([-π, -π / 2, 0, π / 2, π])
ax.set_xticklabels([L"-\pi", L"-\pi/2", L"0", L"\pi/2", L"\pi"])
ax.set_xlabel(raw"$q$")
ax.set_ylabel(raw"$\omega$")
ax.tick_params(axis="both", which="both", direction="out", length=4, width=1)
for spine in ax.spines.values()
    spine.set_linewidth(1.0)
end
ax.text(0.05, 0.92, "\$\\eta = $(eta)\$"; transform=ax.transAxes, fontsize=11)
ax.text(0.05, 0.84, "\$\\lambda_q = $(λ_q)\$"; transform=ax.transAxes, fontsize=11)
ax.text(0.05, 0.76, raw"$w_q \propto (\eta/\lambda_q)\,|q|^{s_q} e^{-|q|/\lambda_q}$";
    transform=ax.transAxes, fontsize=10)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(raw"$-\Im\,\Xi_q^{R}(\omega)$")
cbar.ax.tick_params(labelsize=11)

fig.tight_layout()
out_path = joinpath(OUTPUT_DIR, "xi_kernel_im_panel_linear_eta$(eta)_lambda$(λ_q).png")
fig.savefig(out_path; dpi=300, bbox_inches="tight")
close(fig)

println("Saved $(out_path)")
