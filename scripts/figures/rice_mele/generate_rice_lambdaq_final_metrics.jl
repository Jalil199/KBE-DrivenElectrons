using JLD2
using Printf
using PyPlot

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
const OUT_DIR = joinpath(ROOT, "distributions_rice")
const EXCITON_CACHE = joinpath(ROOT, "analysis_cache", "rice_exciton_teff_coherence_lambda_fine_even")
const VELOCITY_CACHE = joinpath(ROOT, "analysis_cache", "rice_velocity_lambda_fine_even")

mkpath(OUT_DIR)

const ALPHAS = (1.0, 2.0)
const ETAS = (0.05, 0.5)
const LAMBDAS = collect(0.2:0.2:1.4)

const HAS_LATEX = Sys.which("latex") !== nothing
rc("text", usetex=HAS_LATEX)
rc("font", family="serif", size=12)
rc("axes", linewidth=1.3, labelsize=13, titlesize=13)
rc("xtick", labelsize=11)
rc("ytick", labelsize=11)
rc("legend", fontsize=9)
rc("xtick", direction="in")
rc("ytick", direction="in")

lambda_text(x) = @sprintf("%.1f", x)
alpha_text(x) = @sprintf("%.1f", x)
eta_text(x) = x == 0.05 ? "0.05" : @sprintf("%.1f", x)

function cache_paths(alpha, eta, lambda_q)
    suffix = "alpha$(alpha_text(alpha))_eta$(eta_text(eta))_lambda$(lambda_text(lambda_q)).jld2"
    return (
        exciton = joinpath(EXCITON_CACHE, "exciton_$(suffix)"),
        velocity = joinpath(VELOCITY_CACHE, "velocity_$(suffix)"),
    )
end

function final_metrics(alpha, eta)
    teff = Float64[]
    mu_ex = Float64[]
    rmse = Float64[]
    coherence = Float64[]
    dmax = Float64[]
    d95 = Float64[]

    for lambda_q in LAMBDAS
        paths = cache_paths(alpha, eta, lambda_q)
        isfile(paths.exciton) || error("Missing cache: $(paths.exciton)")
        isfile(paths.velocity) || error("Missing cache: $(paths.velocity)")
        ex = load(paths.exciton)
        vel = load(paths.velocity)
        push!(teff, ex["teff"][end])
        push!(mu_ex, ex["mu_ex"][end])
        push!(rmse, ex["rmse"][end])
        push!(coherence, ex["coherence_mean"][end])
        push!(dmax, vel["dmax"][end])
        push!(d95, vel["d95"][end])
    end

    return (; teff, mu_ex, rmse, coherence, dmax, d95)
end

colors = Dict(1.0 => "#1f77b4", 2.0 => "#d1495b")
styles = Dict(0.05 => "-", 0.5 => "--")
labels = Dict(
    (1.0, 0.05) => raw"$\alpha=1,\eta=0.05$",
    (1.0, 0.5) => raw"$\alpha=1,\eta=0.5$",
    (2.0, 0.05) => raw"$\alpha=2,\eta=0.05$",
    (2.0, 0.5) => raw"$\alpha=2,\eta=0.5$",
)

fig, axes = subplots(2, 3; figsize=(12.5, 7.0), sharex=true)
summary = String[]

for alpha in ALPHAS, eta in ETAS
    m = final_metrics(alpha, eta)
    color = colors[alpha]
    ls = styles[eta]
    label = labels[(alpha, eta)]

    axes[1, 1].plot(LAMBDAS, m.teff; color=color, linestyle=ls, marker="o", linewidth=2.0, label=label)
    axes[1, 2].plot(LAMBDAS, m.mu_ex; color=color, linestyle=ls, marker="o", linewidth=2.0)
    axes[1, 3].plot(LAMBDAS, m.rmse; color=color, linestyle=ls, marker="o", linewidth=2.0)
    axes[2, 1].plot(LAMBDAS, m.coherence; color=color, linestyle=ls, marker="o", linewidth=2.0)
    axes[2, 2].plot(LAMBDAS, m.dmax; color=color, linestyle=ls, marker="o", linewidth=2.0)
    axes[2, 3].plot(LAMBDAS, m.d95; color=color, linestyle=ls, marker="o", linewidth=2.0)

    for (i, lambda_q) in enumerate(LAMBDAS)
        push!(summary, @sprintf("alpha=%.1f eta=%s lambda_q=%.1f Teff=%.5f mu_ex=%.5f RMSE=%.6f Cmean=%.6e Dmax=%.6e D95=%.6e",
                                alpha, eta_text(eta), lambda_q, m.teff[i], m.mu_ex[i], m.rmse[i],
                                m.coherence[i], m.dmax[i], m.d95[i]))
    end
end

axes[1, 1].set_ylabel(raw"$T_{\mathrm{eff}}^{\mathrm{ex}}(t=60)$")
axes[1, 2].set_ylabel(raw"$\mu_{\mathrm{ex}}(t=60)$")
axes[1, 3].set_ylabel(raw"$\mathrm{RMSE}(t=60)$")
axes[2, 1].set_ylabel(raw"$\langle |\rho_{-+}| \rangle_k(t=60)$")
axes[2, 2].set_ylabel(raw"$D_{\max}(t=60)$")
axes[2, 3].set_ylabel(raw"$D_{95}(t=60)$")

for ax in axes
    ax.set_xlabel(raw"$\lambda_q$")
    ax.set_xlim(0.15, 1.45)
    ax.grid(true; alpha=0.25)
end

axes[1, 1].legend(frameon=false, loc="best")
fig.suptitle(raw"Final Rice-Mele metrics vs $\lambda_q$ at $t=60$", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.94])

outpath = joinpath(OUT_DIR, "rice_lambdaq_final_metrics_t60.png")
fig.savefig(outpath; dpi=300, bbox_inches="tight")
close(fig)

summary_path = joinpath(OUT_DIR, "rice_lambdaq_final_metrics_t60_summary.txt")
open(summary_path, "w") do io
    println(io, join(summary, "\n"))
end

println("Saved $(outpath)")
println("Saved $(summary_path)")
println(join(summary, "\n"))
