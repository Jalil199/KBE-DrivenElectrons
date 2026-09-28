using PyPlot

include(joinpath(@__DIR__, "main-rice_mele.jl"))

rc("font", family="serif")
rc("font", size=14)
rc("axes", labelsize=16, titlesize=15)
rc("xtick", labelsize=12)
rc("ytick", labelsize=12)
rc("legend", fontsize=11)

const OUTPUT_DIR = "distributions"
mkpath(OUTPUT_DIR)

fermi_fd(omega, T, mu) = 1 / (exp((omega - mu) / T) + 1)

function parse_arg(args, i, default)
    return length(args) >= i ? parse(typeof(default), args[i]) : default
end

function rice_static_occupations(; L=80, t1=-1.0, t2=-0.8, Δ=2.0, T=0.1, μ=0.0)
    ks = collect(range(-π, stop=π - 2π / L, length=L))
    nminus = zeros(Float64, L)
    nplus = zeros(Float64, L)

    for (ik, k) in enumerate(ks)
        evals = sort(real.(eigvals(H_k(k; t1=t1, t2=t2, Δ=Δ))))
        nminus[ik] = fermi_fd(evals[1], T, μ)
        nplus[ik] = fermi_fd(evals[2], T, μ)
    end

    return ks, nminus, nplus
end

L = parse_arg(ARGS, 1, 80)
t1 = parse_arg(ARGS, 2, -1.0)
t2 = parse_arg(ARGS, 3, -0.8)
Δ = parse_arg(ARGS, 4, 2.0)
T = parse_arg(ARGS, 5, 0.1)
μ = parse_arg(ARGS, 6, 0.0)

ks, nminus, nplus = rice_static_occupations(; L=L, t1=t1, t2=t2, Δ=Δ, T=T, μ=μ)

fig, ax = subplots(figsize=(7.2, 4.8))
ax.plot(ks, nminus; color="#2c7fb8", linewidth=2.2, label=raw"$n_{-}^{\mathrm{th}}(k)$")
ax.plot(ks, nplus; color="#d95f0e", linewidth=2.0, linestyle="--", label=raw"$n_{+}^{\mathrm{th}}(k)$")
ax.set_xlabel(raw"$k$")
ax.set_ylabel(raw"$n_\pm(k)$")
ax.set_xlim(first(ks), last(ks))
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.18)
ax.legend(frameon=false, loc="best")
fig.suptitle("Rice-Mele thermal occupations", fontsize=16, y=1.02)
fig.text(0.5, -0.02, "t1=$(t1), t2=$(t2), Δ=$(Δ), T=$(T), μ=$(μ), L=$(L)"; ha="center", va="top", fontsize=12)
fig.tight_layout()

outfile = joinpath(OUTPUT_DIR, "rice_static_occupations_t1$(t1)_t2$(t2)_Δ$(Δ)_T$(T)_μ$(μ).png")
fig.savefig(outfile; dpi=220, bbox_inches="tight")
println("Saved " * outfile)
