using JLD2
using Statistics
using Printf
using PyPlot

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const OUT_DIR = joinpath(ROOT, "distributions_rice")
const VELOCITY_CACHE = joinpath(ROOT, "analysis_cache", "rice_velocity_lambda_fine_even")

mkpath(OUT_DIR)

const LAMBDAS = collect(0.1:0.1:1.5)
const FIT_WINDOW = (50.0, 60.0)
const D_THRESHOLD = 1e-4
const R2_GOOD = 0.8
const TMAX_PLOT = 300.0

const HAS_LATEX = Sys.which("latex") !== nothing
rc("text", usetex=HAS_LATEX)
rc("font", family="serif", size=12)
rc("axes", linewidth=1.3, labelsize=13, titlesize=13)
rc("xtick", labelsize=11)
rc("ytick", labelsize=11)
rc("legend", fontsize=8)
rc("xtick", direction="in")
rc("ytick", direction="in")

lambda_text(x) = @sprintf("%.1f", x)
alpha_text(x) = @sprintf("%.1f", x)
eta_text(x) = x == 0.05 ? "0.05" : @sprintf("%.1f", x)

function cache_path(alpha, eta, lambda_q)
    suffix = "alpha$(alpha_text(alpha))_eta$(eta_text(eta))_lambda$(lambda_text(lambda_q)).jld2"
    return joinpath(VELOCITY_CACHE, "velocity_$(suffix)")
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

function tail_stationary_time(ts, d95)
    tail = (ts .>= FIT_WINDOW[1]) .& (ts .<= FIT_WINDOW[2]) .& isfinite.(d95) .& (d95 .> 0)
    count(tail) < 5 && return (NaN, NaN, NaN, d95[end], "too_few_points")

    # Exponential tail model: D95(t) = exp(a + b t).
    a, b, r2 = linear_fit(ts[tail], log.(d95[tail]))
    current = d95[end]

    if current <= D_THRESHOLD
        return (ts[end], r2, b, current, "already_below_threshold")
    end
    if !(b < 0)
        return (NaN, r2, b, current, "not_decaying")
    end

    t_cross = (log(D_THRESHOLD) - a) / b
    if t_cross < maximum(ts[tail])
        return (NaN, r2, b, current, "crosses_in_past")
    end
    if r2 < R2_GOOD
        return (t_cross, r2, b, current, "weak_fit")
    end
    return (t_cross, r2, b, current, "reliable")
end

function load_case(alpha, eta, lambda_q)
    path = cache_path(alpha, eta, lambda_q)
    isfile(path) || error("Missing velocity cache: $(path)")
    data = load(path)
    return data["ts"], data["d95"]
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
    tstats = Float64[]
    r2s = Float64[]
    slopes = Float64[]
    d95tf = Float64[]
    statuses = String[]

    for lambda_q in LAMBDAS
        ts, d95 = load_case(alpha, eta, lambda_q)
        tstat, r2, slope, current, status = tail_stationary_time(ts, d95)
        push!(tstats, tstat)
        push!(r2s, r2)
        push!(slopes, slope)
        push!(d95tf, current)
        push!(statuses, status)
        push!(summary, @sprintf("alpha=%.1f eta=%s lambda_q=%.1f t_stat_D95=%.6g R2_log_tail=%.5f log_decay_slope=%.6e D95_tf=%.6e status=%s",
                                alpha, eta_text(eta), lambda_q, tstat, r2, slope, current, status))
    end

    tplot = [isfinite(t) ? min(t, TMAX_PLOT) : NaN for t in tstats]
    axes[1].plot(LAMBDAS, tplot; color=color, linestyle=ls, linewidth=2.2, alpha=0.9, label=label)
    axes[2].plot(LAMBDAS, d95tf; color=color, linestyle=ls, linewidth=2.2, alpha=0.9)
    axes[3].plot(LAMBDAS, r2s; color=color, linestyle=ls, linewidth=2.2, alpha=0.9)

    reliable = statuses .== "reliable"
    weak = statuses .== "weak_fit"
    clipped = isfinite.(tstats) .& (tstats .> TMAX_PLOT)
    lower_bound = .!isfinite.(tstats)
    visible_reliable = reliable .& .!clipped
    visible_weak = weak .& .!clipped

    axes[1].scatter(LAMBDAS[visible_reliable], tplot[visible_reliable]; color=color, s=42, marker="o", zorder=4)
    axes[1].scatter(LAMBDAS[visible_weak], tplot[visible_weak]; edgecolors=color, facecolors="none", s=54, marker="o", linewidths=1.6, zorder=5)
    axes[1].scatter(LAMBDAS[clipped], fill(TMAX_PLOT, count(clipped)); edgecolors=color, facecolors="none", s=72, marker="^", linewidths=1.7, zorder=6)
    axes[1].scatter(LAMBDAS[lower_bound], fill(0.98 * TMAX_PLOT, count(lower_bound)); color=color, s=62, marker="x", linewidths=1.8, zorder=7)
    axes[2].scatter(LAMBDAS, d95tf; color=color, s=34, marker="o", zorder=4)
    axes[3].scatter(LAMBDAS, r2s; color=color, s=34, marker="o", zorder=4)
end

axes[1].set_ylabel(raw"$t_{\mathrm{stat}}$ from $D_{95}<10^{-4}$")
axes[2].set_ylabel(raw"$D_{95}(t=60)$")
axes[3].set_ylabel(raw"$R^2$ of log-tail fit")

axes[1].set_ylim(0, TMAX_PLOT)
axes[2].set_yscale("log")
axes[2].axhline(D_THRESHOLD; color="0.35", linestyle=":", linewidth=1.2)
axes[3].set_ylim(0, 1.05)
axes[3].axhline(R2_GOOD; color="0.35", linestyle=":", linewidth=1.2)

for ax in axes
    ax.set_xlabel(raw"$\lambda_q$")
    ax.set_xlim(0.08, 1.52)
    ax.grid(true; alpha=0.25)
end

axes[1].legend(frameon=false, loc="upper left")
axes[1].text(0.98, 0.97, raw"$\circ$: reliable, open: weak, $\triangle$: $>300$, $\times$: no decay";
             transform=axes[1].transAxes, ha="right", va="top", fontsize=8)

fig.suptitle(raw"Stationary-time extrapolation from $D_{95}(t)$ tail, fit window $[50,60]$", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.92])

outpath = joinpath(OUT_DIR, "rice_d95_tail_stationary_lambdaq_t50_60.png")
fig.savefig(outpath; dpi=300, bbox_inches="tight")
close(fig)

summary_path = joinpath(OUT_DIR, "rice_d95_tail_stationary_lambdaq_t50_60_summary.txt")
open(summary_path, "w") do io
    println(io, join(summary, "\n"))
end

println("Saved $(outpath)")
println("Saved $(summary_path)")
println(join(summary, "\n"))
