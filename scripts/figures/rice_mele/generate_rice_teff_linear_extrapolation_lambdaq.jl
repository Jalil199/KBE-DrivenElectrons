using JLD2
using Statistics
using Printf
using PyPlot

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const OUT_DIR = joinpath(ROOT, "distributions_rice")
const EXCITON_CACHE = joinpath(ROOT, "analysis_cache", "rice_exciton_teff_coherence_lambda_fine_even")

mkpath(OUT_DIR)

const ALPHAS = (1.0, 2.0)
const ETAS = (0.05, 0.5)
const LAMBDAS = collect(0.2:0.2:1.4)
const FIT_WINDOW = (50.0, 60.0)
const TB = 0.1
const R2_GOOD = 0.8
const TTARGET_YMAX = 300.0

const HAS_LATEX = Sys.which("latex") !== nothing
rc("text", usetex=HAS_LATEX)
rc("font", family="serif", size=13)
rc("axes", linewidth=1.4, labelsize=15, titlesize=15)
rc("xtick", labelsize=12)
rc("ytick", labelsize=12)
rc("legend", fontsize=10)
rc("xtick", direction="in")
rc("ytick", direction="in")

lambda_text(x) = @sprintf("%.1f", x)
alpha_text(x) = @sprintf("%.1f", x)
eta_text(x) = x == 0.05 ? "0.05" : @sprintf("%.1f", x)

function cache_path(alpha, eta, lambda_q)
    suffix = "alpha$(alpha_text(alpha))_eta$(eta_text(eta))_lambda$(lambda_text(lambda_q)).jld2"
    return joinpath(EXCITON_CACHE, "exciton_$(suffix)")
end

function linear_fit(x, y)
    xm = mean(x)
    ym = mean(y)
    denom = sum((x .- xm) .^ 2)
    slope = denom == 0 ? NaN : sum((x .- xm) .* (y .- ym)) / denom
    intercept = ym - slope * xm
    pred = intercept .+ slope .* x
    ss_res = sum((y .- pred) .^ 2)
    ss_tot = sum((y .- ym) .^ 2)
    r2 = ss_tot == 0 ? NaN : 1 - ss_res / ss_tot
    return intercept, slope, r2
end

function fit_target_time(ts, teff)
    tmin, tmax = FIT_WINDOW
    mask = (ts .>= tmin) .& (ts .<= tmax) .& isfinite.(teff)
    if count(mask) < 5
        return NaN, NaN, NaN, NaN
    end

    a, b, r2 = linear_fit(ts[mask], teff[mask])
    ttarget = b == 0 ? NaN : (TB - a) / b

    # Keep only forward crossings with a tail that moves toward the bath temperature.
    current = teff[findlast(isfinite, teff)]
    moves_toward_target = (current > TB && b < 0) || (current < TB && b > 0)
    if !moves_toward_target || ttarget < maximum(ts[mask])
        ttarget = NaN
    end
    return ttarget, r2, b, current
end

function load_case(alpha, eta, lambda_q)
    path = cache_path(alpha, eta, lambda_q)
    isfile(path) || error("Missing excitonic fit cache: $(path)")
    data = load(path)
    return data["ts"], data["teff"], data["rmse"]
end

cases = [
    (1.0, 0.05, "#1f77b4", "-", raw"$\alpha=1,\eta=0.05$"),
    (1.0, 0.5, "#1f77b4", "--", raw"$\alpha=1,\eta=0.5$"),
    (2.0, 0.05, "#d1495b", "-", raw"$\alpha=2,\eta=0.05$"),
    (2.0, 0.5, "#d1495b", "--", raw"$\alpha=2,\eta=0.5$"),
]

fig, axes = subplots(1, 3; figsize=(14.0, 4.4), sharex=true)
summary = String[]

for (alpha, eta, color, ls, label) in cases
    ttargets = Float64[]
    r2s = Float64[]
    slopes = Float64[]
    tefftf = Float64[]
    rmsetf = Float64[]

    for lambda_q in LAMBDAS
        ts, teff, rmse = load_case(alpha, eta, lambda_q)
        ttarget, r2, slope, current = fit_target_time(ts, teff)
        push!(ttargets, ttarget)
        push!(r2s, r2)
        push!(slopes, slope)
        push!(tefftf, current)
        push!(rmsetf, rmse[end])
        push!(summary, @sprintf("alpha=%.1f eta=%s lambda_q=%.1f t_Tb_linear=%.6g R2_linear=%.5f slope=%.6e Teff_tf=%.6f RMSE_tf=%.6f",
                                alpha, eta_text(eta), lambda_q, ttarget, r2, slope, current, rmse[end]))
    end

    target_plot = [isfinite(t) ? min(t, TTARGET_YMAX) : NaN for t in ttargets]
    axes[1].plot(LAMBDAS, target_plot; color=color, linestyle=ls, linewidth=2.2, alpha=0.9, label=label)
    axes[2].plot(LAMBDAS, tefftf; color=color, linestyle=ls, linewidth=2.2, alpha=0.9)
    axes[3].plot(LAMBDAS, slopes; color=color, linestyle=ls, linewidth=2.2, alpha=0.9)

    finite = isfinite.(ttargets)
    clipped = finite .& (ttargets .> TTARGET_YMAX)
    visible = finite .& .!clipped
    no_cross = .!finite
    good_visible = visible .& (r2s .>= R2_GOOD)
    bad_visible = visible .& (r2s .< R2_GOOD)

    axes[1].scatter(LAMBDAS[good_visible], target_plot[good_visible]; color=color, s=42, marker="o", zorder=4)
    axes[1].scatter(LAMBDAS[bad_visible], target_plot[bad_visible]; edgecolors=color, facecolors="none", s=54, marker="o", linewidths=1.6, zorder=5)
    axes[1].scatter(LAMBDAS[clipped], fill(TTARGET_YMAX, count(clipped)); edgecolors=color, facecolors="none", s=72, marker="^", linewidths=1.7, zorder=6)
    axes[1].scatter(LAMBDAS[no_cross], fill(0.98 * TTARGET_YMAX, count(no_cross)); color=color, s=62, marker="x", linewidths=1.8, zorder=7)

    axes[2].scatter(LAMBDAS, tefftf; color=color, s=36, marker="o", zorder=4)
    axes[3].scatter(LAMBDAS, slopes; color=color, s=36, marker="o", zorder=4)
end

axes[1].set_ylabel(raw"$t(T_{\mathrm{eff}}=T_b)$")
axes[2].set_ylabel(raw"$T_{\mathrm{eff}}(t=60)$")
axes[3].set_ylabel(raw"$dT_{\mathrm{eff}}/dt$ tail")

axes[1].set_ylim(0, TTARGET_YMAX)
axes[2].axhline(TB; color="0.35", linestyle=":", linewidth=1.2)
axes[3].axhline(0.0; color="0.35", linestyle=":", linewidth=1.2)

for ax in axes
    ax.set_xlabel(raw"$\lambda_q$")
    ax.set_xlim(0.18, 1.42)
    ax.grid(true; alpha=0.25)
end

axes[1].legend(frameon=false, loc="upper left")
axes[1].text(0.98, 0.97, raw"$\triangle:\ t>300,\quad \times:\ no\ crossing$";
             transform=axes[1].transAxes, ha="right", va="top", fontsize=9)
fig.suptitle(raw"Linear extrapolation of excitonic $T_{\mathrm{eff}}(t)$ to $T_b=0.1$, fit window $[50,60]$", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.92])

outpath = joinpath(OUT_DIR, "rice_teff_linear_extrapolation_lambdaq_t50_60.png")
fig.savefig(outpath; dpi=300, bbox_inches="tight")
close(fig)

summary_path = joinpath(OUT_DIR, "rice_teff_linear_extrapolation_lambdaq_t50_60_summary.txt")
open(summary_path, "w") do io
    println(io, join(summary, "\n"))
end

println("Saved $(outpath)")
println("Saved $(summary_path)")
println(join(summary, "\n"))
