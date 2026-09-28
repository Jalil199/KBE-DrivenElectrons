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
const WINDOWS = ((20.0, 60.0), (30.0, 60.0), (40.0, 60.0))
const BASE_WINDOW = (30.0, 60.0)
const EPSILON = parse(Float64, get(ENV, "RELAX_EPSILON", "1e-4"))

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

function fit_window(ts, d95, window)
    tmin, tmax = window
    mask = (ts .>= tmin) .& (ts .<= tmax) .& (d95 .> 0)
    if count(mask) < 5
        return (; t_exp=NaN, tau=NaN, r2_exp=NaN, t_zero=NaN, r2_lin=NaN, slope_lin=NaN, n=count(mask))
    end

    t = ts[mask]
    d = d95[mask]

    a_log, b_log, r2_exp = linear_fit(t, log.(d))
    tau = b_log < 0 ? -1 / b_log : NaN
    t_exp = b_log < 0 ? (log(EPSILON) - a_log) / b_log : NaN

    a_lin, b_lin, r2_lin = linear_fit(t, d)
    t_zero = b_lin < 0 ? -a_lin / b_lin : NaN

    return (; t_exp, tau, r2_exp, t_zero, r2_lin, slope_lin=b_lin, n=count(mask))
end

function load_case(alpha, eta, lambda_q)
    path = cache_path(alpha, eta, lambda_q)
    isfile(path) || error("Missing velocity cache: $(path)")
    data = load(path)
    return data["ts"], data["d95"]
end

colors = Dict(1.0 => "#1f77b4", 2.0 => "#d1495b")
styles = Dict(0.05 => "-", 0.5 => "--")
labels = Dict(
    (1.0, 0.05) => raw"$\alpha=1,\eta=0.05$",
    (1.0, 0.5) => raw"$\alpha=1,\eta=0.5$",
    (2.0, 0.05) => raw"$\alpha=2,\eta=0.05$",
    (2.0, 0.5) => raw"$\alpha=2,\eta=0.5$",
)

function window_alpha(window)
    window == BASE_WINDOW && return 1.0
    window == (20.0, 60.0) && return 0.35
    return 0.60
end

fig, axes = subplots(2, 3; figsize=(12.8, 7.2), sharex=true)
summary = String[]

for alpha in ALPHAS, eta in ETAS
    color = colors[alpha]
    ls = styles[eta]
    label = labels[(alpha, eta)]

    base_exp = Float64[]
    base_zero = Float64[]
    base_tau = Float64[]
    base_r2_exp = Float64[]
    base_r2_lin = Float64[]
    base_d95tf = Float64[]

    for window in WINDOWS
        t_exp_vals = Float64[]
        t_zero_vals = Float64[]

        for lambda_q in LAMBDAS
            ts, d95 = load_case(alpha, eta, lambda_q)
            fit = fit_window(ts, d95, window)
            push!(t_exp_vals, fit.t_exp)
            push!(t_zero_vals, fit.t_zero)

            if window == BASE_WINDOW
                push!(base_exp, fit.t_exp)
                push!(base_zero, fit.t_zero)
                push!(base_tau, fit.tau)
                push!(base_r2_exp, fit.r2_exp)
                push!(base_r2_lin, fit.r2_lin)
                push!(base_d95tf, d95[end])
                push!(summary, @sprintf("alpha=%.1f eta=%s lambda_q=%.1f window=[%.0f,%.0f] tau_exp=%.6g t_exp_eps%.0e=%.6g R2_exp=%.5f t_zero_linear=%.6g R2_lin=%.5f D95_tf=%.6e",
                                        alpha, eta_text(eta), lambda_q, window[1], window[2],
                                        fit.tau, EPSILON, fit.t_exp, fit.r2_exp, fit.t_zero, fit.r2_lin, d95[end]))
            end
        end

        axes[1, 1].plot(LAMBDAS, t_exp_vals; color=color, linestyle=ls, marker="o",
                        linewidth=2.0, alpha=window_alpha(window), label=window == BASE_WINDOW ? label : nothing)
        axes[1, 2].plot(LAMBDAS, t_zero_vals; color=color, linestyle=ls, marker="o",
                        linewidth=2.0, alpha=window_alpha(window))
    end

    axes[1, 3].plot(LAMBDAS, base_tau; color=color, linestyle=ls, marker="o", linewidth=2.0)
    axes[2, 1].plot(LAMBDAS, base_r2_exp; color=color, linestyle=ls, marker="o", linewidth=2.0)
    axes[2, 2].plot(LAMBDAS, base_r2_lin; color=color, linestyle=ls, marker="o", linewidth=2.0)
    axes[2, 3].plot(LAMBDAS, base_d95tf; color=color, linestyle=ls, marker="o", linewidth=2.0)
end

axes[1, 1].set_ylabel(raw"$t_{\epsilon}^{\exp}$ from $D_{95}=\epsilon$")
axes[1, 2].set_ylabel(raw"$t_0^{\mathrm{linear}}$ from $D_{95}=0$")
axes[1, 3].set_ylabel(raw"$\tau_{\exp}$")
axes[2, 1].set_ylabel(raw"$R^2_{\exp}$")
axes[2, 2].set_ylabel(raw"$R^2_{\mathrm{linear}}$")
axes[2, 3].set_ylabel(raw"$D_{95}(t=60)$")

for ax in axes
    ax.set_xlabel(raw"$\lambda_q$")
    ax.set_xlim(0.15, 1.45)
    ax.grid(true; alpha=0.25)
end

axes[1, 1].legend(frameon=false, loc="best")
axes[1, 1].set_title(raw"Exponential threshold, $\epsilon=" * @sprintf("%.0e", EPSILON) * raw"$")
axes[1, 2].set_title(raw"Direct linear extrapolation")
axes[1, 3].set_title(raw"Base window $[30,60]$")
axes[2, 1].set_ylim(0, 1.05)
axes[2, 2].set_ylim(0, 1.05)

fig.suptitle(raw"Relaxation-time extrapolations from late-time $D_{95}(t)$; faded curves use windows $[20,60]$ and $[40,60]$", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])

outpath = joinpath(OUT_DIR, "rice_relaxation_time_lambdaq_t60.png")
fig.savefig(outpath; dpi=300, bbox_inches="tight")
close(fig)

summary_path = joinpath(OUT_DIR, "rice_relaxation_time_lambdaq_t60_summary.txt")
open(summary_path, "w") do io
    println(io, join(summary, "\n"))
end

println("Saved $(outpath)")
println("Saved $(summary_path)")
println(join(summary, "\n"))
