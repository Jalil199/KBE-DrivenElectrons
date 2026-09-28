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
rc("font", size=18)
rc("axes", labelsize=18, titlesize=18)
rc("xtick", labelsize=15)
rc("ytick", labelsize=15)
rc("legend", fontsize=11)

const OUTPUT_DIR = "kernels"
mkpath(OUTPUT_DIR)

eta = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.5
λ_q = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.5
times = [0.0, 5.0, 10.0]
colors = ["#3b4cc0", "#1f9e89", "#d1495b"]
styles = ["-", "--", "-."]

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
    λ_q = λ_q,
)

fig, ax = subplots(1, 1; figsize=(5.6, 4.2))

for (i, τ) in enumerate(times)
    ξ_less = Xi_k_at_time(τ; model=model, t′=0.0, greater=false, apply_switch=false)
    ξ_great = Xi_k_at_time(τ; model=model, t′=0.0, greater=true, apply_switch=false)
    ξ_ret = ξ_great .- ξ_less
    ax.plot(model.ks, abs.(ξ_ret); color=colors[i], linestyle=styles[i], linewidth=2.2,
        label="\$\\tau = $(τ)\$")
end

ax.set_xlabel(raw"$q$")
ax.set_ylabel(raw"$|\Xi_q^R(\tau)|$")
ax.set_xlim(-π, π)
ax.set_xticks([-π, -π / 2, 0, π / 2, π])
ax.set_xticklabels([L"-\pi", L"-\pi/2", L"0", L"\pi/2", L"\pi"])
ax.legend(frameon=false, loc="upper right", handlelength=1.8, borderaxespad=0.2, labelspacing=0.3)
ax.tick_params(axis="both", which="both", direction="out", length=4, width=1)
ax.grid(alpha=0.18, linewidth=0.6)
ax.text(0.05, 0.92, "\$\\eta = $(eta)\$"; transform=ax.transAxes, fontsize=12)
ax.text(0.05, 0.84, "\$\\lambda_q = $(λ_q)\$"; transform=ax.transAxes, fontsize=12)
for spine in ax.spines.values()
    spine.set_linewidth(1.0)
end

fig.tight_layout(pad=0.4)

out_path = joinpath(OUTPUT_DIR, "xi_retarded_qcuts_times_linear_eta$(eta)_lambda$(λ_q).png")
fig.savefig(out_path; dpi=300, bbox_inches="tight")
close(fig)

println("Saved $(out_path)")
