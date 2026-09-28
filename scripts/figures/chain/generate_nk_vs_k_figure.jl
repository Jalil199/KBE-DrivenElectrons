using JLD2
using PyPlot
using LaTeXStrings

# Match the visual style used in main.ipynb.
# Fall back to Matplotlib mathtext when a LaTeX binary is unavailable.
const HAS_LATEX = Sys.which("latex") !== nothing
rc("text", usetex=HAS_LATEX)
rc("font", family="serif")
if HAS_LATEX
    rc("text.latex", preamble=raw"\usepackage{amsmath}")
end
rc("font", size=18)
rc("axes", labelsize=22, titlesize=22)
rc("xtick", labelsize=18)
rc("ytick", labelsize=18)
rc("legend", fontsize=16)

const DATA_NAME = "L100_Te1.0_Tb0.05_u0.0_γ1.0_dispersion_α0.5_s1.0_ωc10.0_linear_delta_η0.05_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q1.0_t050.0_ω03.141592653589793_σ2.0_A0.0_switch0_ti3.0_to20.0_tmax60"
const TARGET_TIMES = [0.0, 20.0, 60.0]
const OUTPUT_FILE = "nk_vs_k_three_times.png"

function nearest_time_index(ts, t_target)
    return argmin(abs.(ts .- t_target))
end

GL = load("Data/GL_$(DATA_NAME).jld2", "GL")
ts = load("Data/ts_$(DATA_NAME).jld2", "sol").t

L = size(GL.data, 1)
Δk = 2π / L
ks = collect(range(-π, stop = π - Δk, length = L))

indices = [nearest_time_index(ts, t) for t in TARGET_TIMES]
actual_times = ts[indices]
nk_curves = [imag.(GL.data[:, it, it]) for it in indices]

fig, ax = subplots(figsize = (8, 6))

colors = ["red", "blue", "gold"]
styles = ["-", "-.", "--"]

for (i, nk) in enumerate(nk_curves)
    ax.plot(
        ks,
        nk;
        color = colors[i],
        linestyle = styles[i],
        linewidth = 2,
        label = "\$t = $(round(actual_times[i], digits=2))\$",
    )
end

ax.set_xlabel(raw"$k$")
ax.set_ylabel(raw"$n_k(t)$")
ax.set_xlim(-π, π)
ax.set_xticks([-π, -π / 2, 0, π / 2, π])
ax.set_xticklabels([L"-\pi", L"-\pi/2", L"0", L"\pi/2", L"\pi"])
ax.legend(frameon = false)
fig.tight_layout()
fig.savefig(OUTPUT_FILE; dpi = 300, bbox_inches = "tight")

println("Saved $(OUTPUT_FILE)")
println("Requested times: $(TARGET_TIMES)")
println("Actual times used: $(round.(actual_times; digits=6))")
