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
rc("legend", fontsize=11)

const OUTPUT_DIR = "kernels"
mkpath(OUTPUT_DIR)

λ_q = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.5
q_fixed = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0
ηs = (0.05, 0.5)
τmax = 100.0
dτ = 0.05
τs = collect(0.0:dτ:τmax)

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

fig, ax = subplots(figsize=(7.2, 4.8))
colors = ["#1f77b4", "#d62728"]

q_actual = NaN
for (η, color) in zip(ηs, colors)
    global q_actual
    model = make_model(; η, λ_q)
    iq = argmin(abs.(model.ks .- q_fixed))
    q_actual = model.ks[iq]
    ξτ = zeros(ComplexF64, length(τs))
    for (it, τ) in enumerate(τs)
        ξ_less = Xi_k_at_time(τ; model=model, t′=0.0, greater=false, apply_switch=false)
        ξ_great = Xi_k_at_time(τ; model=model, t′=0.0, greater=true, apply_switch=false)
        ξτ[it] = ξ_great[iq] - ξ_less[iq]
    end
    ax.plot(τs, .-real.(ξτ); color=color, linewidth=2.2, label="\$\\eta = $(η)\$")
end

ax.axhline(0.0; color="black", linewidth=0.9, alpha=0.6)
ax.set_xlabel(raw"$\tau$")
ax.set_ylabel(raw"$-\Xi_q^R(\tau)$")
ax.grid(alpha=0.18)
ax.legend(frameon=false, loc="best")
ax.text(0.05, 0.92, "\$\\lambda_q = $(λ_q)\$"; transform=ax.transAxes, fontsize=11)
ax.text(0.05, 0.84, "\$q \\approx $(round(q_actual, digits=3))\$"; transform=ax.transAxes, fontsize=11)
ax.text(0.05, 0.76, raw"$w_q \propto (\eta/\lambda_q)\,|q|^{s_q} e^{-|q|/\lambda_q}$";
    transform=ax.transAxes, fontsize=10)

fig.tight_layout()
out_path = joinpath(OUTPUT_DIR, "xi_kernel_time_eta_compare_minus_real_lambda$(λ_q)_q$(round(q_actual, digits=3)).png")
fig.savefig(out_path; dpi=300, bbox_inches="tight")
close(fig)

println("Saved $(out_path)")
