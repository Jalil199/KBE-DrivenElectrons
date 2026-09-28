using JLD2
using Statistics
using Printf
using PyPlot

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const OUT_DIR = joinpath(ROOT, "distributions_rice")
const EXCITON_CACHE = joinpath(ROOT, "analysis_cache", "rice_exciton_teff_coherence_lambda_fine_even")

mkpath(OUT_DIR)

const LAMBDAS = collect(0.2:0.2:1.4)
const FIT_WINDOW = (50.0, 60.0)
const DT_THRESHOLD = 1e-4
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

function central_derivative(ts, ys)
    dys = similar(ys)
    n = length(ys)
    dys[1] = (ys[2] - ys[1]) / (ts[2] - ts[1])
    dys[n] = (ys[n] - ys[n - 1]) / (ts[n] - ts[n - 1])
    for i in 2:n-1
        dys[i] = (ys[i + 1] - ys[i - 1]) / (ts[i + 1] - ts[i - 1])
    end
    return dys
end

function stationary_time(ts, teff)
    dteff = abs.(central_derivative(ts, teff))
    tail = (ts .>= FIT_WINDOW[1]) .& (ts .<= FIT_WINDOW[2]) .& isfinite.(dteff) .& (dteff .> 0)
    if count(tail) < 5
        return NaN, NaN, NaN, dteff[end]
    end

    y = log.(dteff[tail])
    a, b, r2 = linear_fit(ts[tail], y)
    current = dteff[end]

    if current <= DT_THRESHOLD
        return ts[end], r2, b, current
    end
    if !(b < 0)
        return NaN, r2, b, current
    end

    trelax = (log(DT_THRESHOLD) - a) / b
    trelax < maximum(ts[tail]) && return NaN, r2, b, current
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
    decay_slopes = Float64[]
    dTtf = Float64[]

    for lambda_q in LAMBDAS
        ts, teff = load_case(alpha, eta, lambda_q)
        trelax, r2, slope, current_dT = stationary_time(ts, teff)
        push!(trelaxs, trelax)
        push!(r2s, r2)
        push!(decay_slopes, slope)
        push!(dTtf, current_dT)
        push!(summary, @sprintf("alpha=%.1f eta=%s lambda_q=%.1f t_stationary=%.6g R2_log_dT=%.5f log_decay_slope=%.6e abs_dTeff_dt_tf=%.6e",
                                alpha, eta_text(eta), lambda_q, trelax, r2, slope, current_dT))
    end

    trelax_plot = [isfinite(t) ? min(t, TRELAX_YMAX) : NaN for t in trelaxs]
    axes[1].plot(LAMBDAS, trelax_plot; color=color, linestyle=ls, linewidth=2.2, alpha=0.9, label=label)
    axes[2].plot(LAMBDAS, dTtf; color=color, linestyle=ls, linewidth=2.2, alpha=0.9)
    axes[3].plot(LAMBDAS, r2s; color=color, linestyle=ls, linewidth=2.2, alpha=0.9)

    finite = isfinite.(trelaxs)
    clipped = finite .& (trelaxs .> TRELAX_YMAX)
    visible = finite .& .!clipped
    no_cross = .!finite
    good_visible = visible .& (r2s .>= R2_GOOD)
    bad_visible = visible .& (r2s .< R2_GOOD)

    axes[1].scatter(LAMBDAS[good_visible], trelax_plot[good_visible]; color=color, s=42, marker="o", zorder=4)
    axes[1].scatter(LAMBDAS[bad_visible], trelax_plot[bad_visible]; edgecolors=color, facecolors="none", s=54, marker="o", linewidths=1.6, zorder=5)
    axes[1].scatter(LAMBDAS[clipped], fill(TRELAX_YMAX, count(clipped)); edgecolors=color, facecolors="none", s=72, marker="^", linewidths=1.7, zorder=6)
    axes[1].scatter(LAMBDAS[no_cross], fill(0.98 * TRELAX_YMAX, count(no_cross)); color=color, s=62, marker="x", linewidths=1.8, zorder=7)
    axes[2].scatter(LAMBDAS, dTtf; color=color, s=36, marker="o", zorder=4)
    axes[3].scatter(LAMBDAS, r2s; color=color, s=36, marker="o", zorder=4)
end

axes[1].set_ylabel(raw"$t_{\mathrm{stat}}$ from $|\partial_tT_{\mathrm{eff}}|<10^{-4}$")
axes[2].set_ylabel(raw"$|\partial_tT_{\mathrm{eff}}|(t=60)$")
axes[3].set_ylabel(raw"$R^2$ of $\log|\partial_tT_{\mathrm{eff}}|$")

axes[1].set_ylim(0, TRELAX_YMAX)
axes[2].set_yscale("log")
axes[2].axhline(DT_THRESHOLD; color="0.35", linestyle=":", linewidth=1.2)
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
fig.suptitle(raw"Stationary-time estimate from excitonic $T_{\mathrm{eff}}(t)$, fit window $[50,60]$", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.92])

outpath = joinpath(OUT_DIR, "rice_teff_stationary_time_lambdaq_t50_60.png")
fig.savefig(outpath; dpi=300, bbox_inches="tight")
close(fig)

summary_path = joinpath(OUT_DIR, "rice_teff_stationary_time_lambdaq_t50_60_summary.txt")
open(summary_path, "w") do io
    println(io, join(summary, "\n"))
end

println("Saved $(outpath)")
println("Saved $(summary_path)")
println(join(summary, "\n"))
