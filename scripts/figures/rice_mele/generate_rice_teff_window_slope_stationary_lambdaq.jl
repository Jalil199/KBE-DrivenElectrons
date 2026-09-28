using JLD2
using Statistics
using Printf
using PyPlot

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const OUT_DIR = joinpath(ROOT, "distributions_rice")
const EXCITON_CACHE = joinpath(ROOT, "analysis_cache", "rice_exciton_teff_coherence_lambda_fine_even")

mkpath(OUT_DIR)

const LAMBDAS = collect(0.2:0.2:1.4)
const SLOPE_WINDOW = 10.0
const FIT_WINDOW = (50.0, 60.0)
const SLOPE_THRESHOLD = 1e-4
const TRELAX_YMAX = 300.0
const R2_GOOD = 0.8

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

function window_slopes(ts, ys)
    tmid = Float64[]
    slopes = Float64[]
    for i in eachindex(ts)
        mask = (ts .>= ts[i] - SLOPE_WINDOW) .& (ts .<= ts[i])
        count(mask) < 5 && continue
        _, b, _ = linear_fit(ts[mask], ys[mask])
        push!(tmid, ts[i])
        push!(slopes, abs(b))
    end
    return tmid, slopes
end

function stationary_time_from_window_slope(ts, teff)
    tmid, abs_slopes = window_slopes(ts, teff)
    tail = (tmid .>= FIT_WINDOW[1]) .& (tmid .<= FIT_WINDOW[2]) .&
           isfinite.(abs_slopes) .& (abs_slopes .> 0)
    current = abs_slopes[end]

    if current <= SLOPE_THRESHOLD
        return ts[end], NaN, NaN, current
    end
    if count(tail) < 5
        return NaN, NaN, NaN, current
    end

    a, b, r2 = linear_fit(tmid[tail], log.(abs_slopes[tail]))
    if !(b < 0)
        return NaN, r2, b, current
    end

    trelax = (log(SLOPE_THRESHOLD) - a) / b
    trelax < maximum(tmid[tail]) && return NaN, r2, b, current
    return trelax, r2, b, current
end

function load_case(alpha, eta, lambda_q)
    path = cache_path(alpha, eta, lambda_q)
    isfile(path) || error("Missing excitonic fit cache: $(path)")
    data = load(path)
    return data["ts"], data["teff"]
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
    trelaxs = Float64[]
    r2s = Float64[]
    log_decay = Float64[]
    slopes_tf = Float64[]

    for lambda_q in LAMBDAS
        ts, teff = load_case(alpha, eta, lambda_q)
        trelax, r2, b, slope_tf = stationary_time_from_window_slope(ts, teff)
        push!(trelaxs, trelax)
        push!(r2s, r2)
        push!(log_decay, b)
        push!(slopes_tf, slope_tf)
        push!(summary, @sprintf("alpha=%.1f eta=%s lambda_q=%.1f t_stationary=%.6g R2_log_window_slope=%.5f log_decay_slope=%.6e abs_window_dTeff_dt_tf=%.6e",
                                alpha, eta_text(eta), lambda_q, trelax, r2, b, slope_tf))
    end

    trelax_plot = [isfinite(t) ? min(t, TRELAX_YMAX) : NaN for t in trelaxs]
    axes[1].plot(LAMBDAS, trelax_plot; color=color, linestyle=ls, linewidth=2.2, alpha=0.9, label=label)
    axes[2].plot(LAMBDAS, slopes_tf; color=color, linestyle=ls, linewidth=2.2, alpha=0.9)
    axes[3].plot(LAMBDAS, r2s; color=color, linestyle=ls, linewidth=2.2, alpha=0.9)

    finite = isfinite.(trelaxs)
    clipped = finite .& (trelaxs .> TRELAX_YMAX)
    visible = finite .& .!clipped
    no_cross = .!finite
    good_visible = visible .& (r2s .>= R2_GOOD)
    bad_visible = visible .& ((r2s .< R2_GOOD) .| .!isfinite.(r2s))

    axes[1].scatter(LAMBDAS[good_visible], trelax_plot[good_visible]; color=color, s=42, marker="o", zorder=4)
    axes[1].scatter(LAMBDAS[bad_visible], trelax_plot[bad_visible]; edgecolors=color, facecolors="none", s=54, marker="o", linewidths=1.6, zorder=5)
    axes[1].scatter(LAMBDAS[clipped], fill(TRELAX_YMAX, count(clipped)); edgecolors=color, facecolors="none", s=72, marker="^", linewidths=1.7, zorder=6)
    axes[1].scatter(LAMBDAS[no_cross], fill(0.98 * TRELAX_YMAX, count(no_cross)); color=color, s=62, marker="x", linewidths=1.8, zorder=7)
    axes[2].scatter(LAMBDAS, slopes_tf; color=color, s=36, marker="o", zorder=4)
    axes[3].scatter(LAMBDAS, r2s; color=color, s=36, marker="o", zorder=4)
end

axes[1].set_ylabel(raw"$t_{\mathrm{stat}}$ from window slope")
axes[2].set_ylabel(raw"$|\mathrm{slope}_{10}(T_{\mathrm{eff}})|(t=60)$")
axes[3].set_ylabel(raw"$R^2$ of tail extrapolation")

axes[1].set_ylim(0, TRELAX_YMAX)
axes[2].set_yscale("log")
axes[2].axhline(SLOPE_THRESHOLD; color="0.35", linestyle=":", linewidth=1.2)
axes[3].set_ylim(0, 1.05)
axes[3].axhline(R2_GOOD; color="0.35", linestyle=":", linewidth=1.2)

for ax in axes
    ax.set_xlabel(raw"$\lambda_q$")
    ax.set_xlim(0.18, 1.42)
    ax.grid(true; alpha=0.25)
end

axes[1].legend(frameon=false, loc="upper left")
axes[1].text(0.98, 0.97, raw"$\triangle:\ t>300,\quad \times:\ no\ decay$";
             transform=axes[1].transAxes, ha="right", va="top", fontsize=9)
fig.suptitle(raw"Stationary-time estimate from 10-time-unit slope of $T_{\mathrm{eff}}$, threshold $10^{-4}$", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.92])

outpath = joinpath(OUT_DIR, "rice_teff_window_slope_stationary_lambdaq_t50_60.png")
fig.savefig(outpath; dpi=300, bbox_inches="tight")
close(fig)

summary_path = joinpath(OUT_DIR, "rice_teff_window_slope_stationary_lambdaq_t50_60_summary.txt")
open(summary_path, "w") do io
    println(io, join(summary, "\n"))
end

println("Saved $(outpath)")
println("Saved $(summary_path)")
println(join(summary, "\n"))
