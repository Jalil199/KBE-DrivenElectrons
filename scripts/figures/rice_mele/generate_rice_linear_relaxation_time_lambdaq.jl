using JLD2
using Statistics
using Printf
using PyPlot

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const OUT_DIR = joinpath(ROOT, "distributions_rice")
const VELOCITY_CACHE = joinpath(ROOT, "analysis_cache", "rice_velocity_lambda_fine_even")

mkpath(OUT_DIR)

const ALPHAS = (1.0, 2.0)
const ETAS = (0.05, 0.5)
const LAMBDAS = collect(0.1:0.1:1.5)
const FIT_WINDOW = (50.0, 60.0)
const R2_GOOD = 0.8
const T0_YMAX = 300.0

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

function fit_linear_zero(ts, d95)
    tmin, tmax = FIT_WINDOW
    mask = (ts .>= tmin) .& (ts .<= tmax) .& isfinite.(d95)
    if count(mask) < 5
        return NaN, NaN, NaN
    end
    a, b, r2 = linear_fit(ts[mask], d95[mask])
    tzero = b < 0 ? -a / b : NaN
    return tzero, r2, b
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
    tzeros = Float64[]
    r2s = Float64[]
    d95tf = Float64[]
    slopes = Float64[]

    for lambda_q in LAMBDAS
        ts, d95 = load_case(alpha, eta, lambda_q)
        tzero, r2, slope = fit_linear_zero(ts, d95)
        push!(tzeros, tzero)
        push!(r2s, r2)
        push!(d95tf, d95[end])
        push!(slopes, slope)
        push!(summary, @sprintf("alpha=%.1f eta=%s lambda_q=%.1f t_zero_linear=%.6g R2_linear=%.5f slope=%.6e D95_tf=%.6e",
                                alpha, eta_text(eta), lambda_q, tzero, r2, slope, d95[end]))
    end

    tzero_plot = [isfinite(t) ? min(t, T0_YMAX) : NaN for t in tzeros]
    axes[1].plot(LAMBDAS, tzero_plot; color=color, linestyle=ls, linewidth=2.2, alpha=0.9, label=label)
    axes[2].plot(LAMBDAS, r2s; color=color, linestyle=ls, linewidth=2.2, alpha=0.9)
    axes[3].plot(LAMBDAS, d95tf; color=color, linestyle=ls, linewidth=2.2, alpha=0.9)

    finite = isfinite.(tzeros)
    clipped = finite .& (tzeros .> T0_YMAX)
    visible = finite .& .!clipped
    no_cross = .!finite
    good_visible = visible .& (r2s .>= R2_GOOD)
    bad_visible = visible .& (r2s .< R2_GOOD)
    axes[1].scatter(LAMBDAS[good_visible], tzero_plot[good_visible]; color=color, s=42, marker="o", zorder=4)
    axes[1].scatter(LAMBDAS[bad_visible], tzero_plot[bad_visible]; edgecolors=color, facecolors="none", s=54, marker="o", linewidths=1.6, zorder=5)
    axes[1].scatter(LAMBDAS[clipped], fill(T0_YMAX, count(clipped)); edgecolors=color, facecolors="none", s=72, marker="^", linewidths=1.7, zorder=6)
    axes[1].scatter(LAMBDAS[no_cross], fill(0.98 * T0_YMAX, count(no_cross)); color=color, s=62, marker="x", linewidths=1.8, zorder=7)
    axes[2].scatter(LAMBDAS, r2s; color=color, s=36, marker="o", zorder=4)
    axes[3].scatter(LAMBDAS, d95tf; color=color, s=36, marker="o", zorder=4)
end

axes[1].set_ylabel(raw"$t_0^{\mathrm{linear}}$ from $D_{95}=0$")
axes[2].set_ylabel(raw"$R^2_{\mathrm{linear}}$")
axes[3].set_ylabel(raw"$D_{95}(t=60)$")

axes[1].set_ylim(0, T0_YMAX)
axes[2].set_ylim(0, 1.05)
axes[3].set_ylim(0, 0.0028)
axes[2].axhline(R2_GOOD; color="0.35", linestyle=":", linewidth=1.2)

for ax in axes
    ax.set_xlabel(raw"$\lambda_q$")
    ax.set_xlim(0.08, 1.52)
    ax.grid(true; alpha=0.25)
end

axes[1].legend(frameon=false, loc="upper left")
axes[1].text(0.98, 0.97, raw"$\triangle:\ t_0>300,\quad \times:\ no\ crossing$";
             transform=axes[1].transAxes, ha="right", va="top", fontsize=9)
fig.suptitle(raw"Linear relaxation-time extrapolation from $D_{95}(t)$, fit window $[50,60]$", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.92])

outpath = joinpath(OUT_DIR, "rice_linear_relaxation_time_lambdaq_t50_60.png")
fig.savefig(outpath; dpi=300, bbox_inches="tight")
close(fig)

summary_path = joinpath(OUT_DIR, "rice_linear_relaxation_time_lambdaq_t50_60_summary.txt")
open(summary_path, "w") do io
    println(io, join(summary, "\n"))
end

println("Saved $(outpath)")
println("Saved $(summary_path)")
println(join(summary, "\n"))
