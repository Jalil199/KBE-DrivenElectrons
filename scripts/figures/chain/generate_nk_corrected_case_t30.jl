using JLD2
using PyPlot
using LaTeXStrings

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

const DATA_NAME = "L100_Te1.0_Tb0.1_u0.0_γ1.0_dispersion_α1.0_s1.0_ωc10.0_linear_spectral_η0.05_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q1.0_t050.0_ω03.141592653589793_σ2.0_A0.0_switch0_ti3.0_to20.0_tmax30"
const TARGET_TIMES = [0.0, 20.0, 30.0]
const OUTPUT_FILE = "distributions/nk_corrected_case_t30.png"

nearest_time_index(ts, t_target) = argmin(abs.(ts .- t_target))

occ = load("Data/occ_$(DATA_NAME).jld2")
ts_obj = load("Data/ts_$(DATA_NAME).jld2", "sol")
ts = hasproperty(ts_obj, :t) ? ts_obj.t : ts_obj

nk_t = occ["nk_t"]
ks = occ["ks"]

indices = [nearest_time_index(ts, t) for t in TARGET_TIMES]
actual_times = ts[indices]
nk_curves = [nk_t[:, it] for it in indices]

fig, ax = subplots(figsize=(8, 6))

colors = ["red", "blue", "gold"]
styles = ["-", "-.", "--"]

for (i, nk) in enumerate(nk_curves)
    ax.plot(
        ks,
        nk;
        color=colors[i],
        linestyle=styles[i],
        linewidth=2.2,
        label="\$t = $(round(actual_times[i], digits=2))\$",
    )
end

ax.set_xlabel(raw"$k$")
ax.set_ylabel(raw"$n_k(t)$")
ax.set_xlim(-π, π)
ax.set_xticks([-π, -π / 2, 0, π / 2, π])
ax.set_xticklabels([L"-\pi", L"-\pi/2", L"0", L"\pi/2", L"\pi"])
ax.legend(frameon=false, loc="upper right")

textbox = "corrected k-q wrapping\n" *
          "spectral, η = 0.05, p_c = 1.0/a, α = 1.0"
ax.text(
    0.03, 0.97, textbox;
    transform=ax.transAxes,
    va="top", ha="left", fontsize=14,
    bbox=Dict("boxstyle" => "round", "facecolor" => "white", "alpha" => 0.9, "edgecolor" => "0.7"),
)

fig.tight_layout()
fig.savefig(OUTPUT_FILE; dpi=300, bbox_inches="tight")

println("Saved $(OUTPUT_FILE)")
println("Requested times: $(TARGET_TIMES)")
println("Actual times used: $(round.(actual_times; digits=6))")
