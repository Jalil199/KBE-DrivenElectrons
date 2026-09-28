using JLD2
using LinearAlgebra
using PyPlot

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "main-rice_mele.jl"))

const DATA_DIR = joinpath(ROOT, "Data_rice")
const OUT_DIR = joinpath(ROOT, "distributions_rice")
mkpath(OUT_DIR)

array_data(x) = hasproperty(x, :data) ? getproperty(x, :data) : x

function capture_float(name::AbstractString, pattern::AbstractString)
    m = match(Regex(pattern), name)
    m === nothing && error("Could not parse $(pattern) from $(name)")
    return parse(Float64, m.captures[1])
end

function parse_params(path::AbstractString)
    b = basename(path)
    return (;
        α = capture_float(b, "_dispersion_α([^_]+)_"),
        η = capture_float(b, "_η([^_]+)_v_b"),
        λq = capture_float(b, "_λ_q([^_]+)_t0"),
        α2 = capture_float(b, "_b2_α([^_]+)_"),
        ωb2 = capture_float(b, "_ωb0_2([^_]+)_"),
        η2 = capture_float(b, "_η2([^_]+)_"),
        λq2 = capture_float(b, "_λ_q2([^_]+)_"),
    )
end

function final_upper_population(path::AbstractString)
    p = parse_params(path)
    GL = array_data(load(path, "GL"))
    L = size(GL, 3)
    it0 = 1
    itf = size(GL, 4)
    ks = collect(range(-π, stop=π - 2π / L, length=L))
    n0 = 0.0
    nf = 0.0
    for (ik, k) in enumerate(ks)
        U = eigen(H_k(k; t1=-1.0, t2=-0.8, Δ=2.0)).vectors
        ρ0 = U' * imag.(GL[:, :, ik, it0, it0]) * U
        ρf = U' * imag.(GL[:, :, ik, itf, itf]) * U
        n0 += real(ρ0[2, 2])
        nf += real(ρf[2, 2])
    end
    return merge(p, (; Nplus0=n0 / L, Nplusf=nf / L, ΔNplus=(nf - n0) / L, path))
end

function approxin(x, vals; atol=1e-10)
    any(v -> isapprox(x, v; atol), vals)
end

files = sort(filter(f ->
    startswith(basename(f), "GL_L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.1_dispersion_α") &&
    occursin("_b2_", basename(f)) &&
    endswith(basename(f), "_tmax60.jld2"),
    readdir(DATA_DIR; join=true)))

records = final_upper_population.(files)

freq_records = filter(r ->
    approxin(r.α, (0.0, 1.0, 4.0)) &&
    isapprox(r.η, 0.05; atol=1e-10) &&
    isapprox(r.λq, 1.0; atol=1e-10) &&
    isapprox(r.α2, 0.5; atol=1e-10) &&
    isapprox(r.η2, 0.1; atol=1e-10) &&
    isapprox(r.λq2, 10.0; atol=1e-10) &&
    approxin(r.ωb2, (1.0, 1.5, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0)),
    records)

eta_records = filter(r ->
    isapprox(r.α, 0.0; atol=1e-10) &&
    isapprox(r.α2, 0.5; atol=1e-10) &&
    isapprox(r.λq2, 10.0; atol=1e-10) &&
    approxin(r.ωb2, (2.0, 2.5, 3.0)),
    records)

lq_records = filter(r ->
    isapprox(r.α, 0.0; atol=1e-10) &&
    isapprox(r.α2, 0.5; atol=1e-10) &&
    isapprox(r.η2, 0.1; atol=1e-10) &&
    approxin(r.ωb2, (2.0, 2.5, 3.0)),
    records)

a2_records = filter(r ->
    isapprox(r.α, 0.0; atol=1e-10) &&
    isapprox(r.η2, 0.1; atol=1e-10) &&
    isapprox(r.λq2, 10.0; atol=1e-10) &&
    approxin(r.ωb2, (2.0, 2.5, 3.0)),
    records)

println("Frequency scan")
for r in sort(freq_records, by=r -> (r.α, r.ωb2))
    println("alpha=$(r.α) wb2=$(r.ωb2) Nplusf=$(round(r.Nplusf, digits=6)) Δ=$(round(r.ΔNplus, digits=6))")
end

println("\nEta2 scan")
for r in sort(eta_records, by=r -> (r.ωb2, r.η2))
    println("wb2=$(r.ωb2) eta2=$(r.η2) Nplusf=$(round(r.Nplusf, digits=6))")
end

println("\nLambdaq2 scan")
for r in sort(lq_records, by=r -> (r.ωb2, r.λq2))
    println("wb2=$(r.ωb2) lambdaq2=$(r.λq2) Nplusf=$(round(r.Nplusf, digits=6))")
end

println("\nAlpha2 scan")
for r in sort(a2_records, by=r -> (r.ωb2, r.α2))
    println("wb2=$(r.ωb2) alpha2=$(r.α2) Nplusf=$(round(r.Nplusf, digits=6))")
end

plt.rc("font", family="serif", size=12)
plt.rc("axes", linewidth=1.5)
plt.rc("xtick", direction="in")
plt.rc("ytick", direction="in")

colors = Dict(2.0 => "#2a9d8f", 2.5 => "#3a5a98", 3.0 => "#7b3294")
αcolors = Dict(0.0 => "#2a9d8f", 1.0 => "#e07a2f", 4.0 => "#3a5a98")

fig, axs = subplots(2, 2; figsize=(12.8, 8.6))
axs = reshape(collect(axs), 2, 2)

for α in (0.0, 1.0, 4.0)
    rs = sort(filter(r -> isapprox(r.α, α; atol=1e-10), freq_records), by=r -> r.ωb2)
    axs[1, 1].plot([r.ωb2 for r in rs], [r.Nplusf for r in rs]; marker="o", linewidth=2.1,
                   color=αcolors[α], label=raw"$\alpha=" * string(α) * raw"$")
end

for ωb2 in (2.0, 2.5, 3.0)
    rs = sort(filter(r -> isapprox(r.ωb2, ωb2; atol=1e-10), eta_records), by=r -> r.η2)
    axs[1, 2].plot([r.η2 for r in rs], [r.Nplusf for r in rs]; marker="o", linewidth=2.1,
                   color=colors[ωb2], label=raw"$\omega_{b,2}=" * string(ωb2) * raw"$")

    rs = sort(filter(r -> isapprox(r.ωb2, ωb2; atol=1e-10), lq_records), by=r -> r.λq2)
    axs[2, 1].plot([r.λq2 for r in rs], [r.Nplusf for r in rs]; marker="o", linewidth=2.1,
                   color=colors[ωb2], label=raw"$\omega_{b,2}=" * string(ωb2) * raw"$")

    rs = sort(filter(r -> isapprox(r.ωb2, ωb2; atol=1e-10), a2_records), by=r -> r.α2)
    axs[2, 2].plot([r.α2 for r in rs], [r.Nplusf for r in rs]; marker="o", linewidth=2.1,
                   color=colors[ωb2], label=raw"$\omega_{b,2}=" * string(ωb2) * raw"$")
end

axs[1, 1].set_xlabel(raw"$\omega_{b,2}$")
axs[1, 1].set_ylabel(raw"$N_+(60)$")
axs[1, 1].set_title("Flat bath frequency scan")
axs[1, 1].legend(frameon=false, fontsize=9)

axs[1, 2].set_xlabel(raw"$\eta_2$")
axs[1, 2].set_ylabel(raw"$N_+(60)$")
axs[1, 2].set_title(raw"$\eta_2$ scan, $\alpha=0$")
axs[1, 2].set_xscale("log")
axs[1, 2].legend(frameon=false, fontsize=9)

axs[2, 1].set_xlabel(raw"$\lambda_{q,2}$")
axs[2, 1].set_ylabel(raw"$N_+(60)$")
axs[2, 1].set_title(raw"$\lambda_{q,2}$ scan, $\alpha=0$")
axs[2, 1].set_xscale("log")
axs[2, 1].legend(frameon=false, fontsize=9)

axs[2, 2].set_xlabel(raw"$\alpha_2$")
axs[2, 2].set_ylabel(raw"$N_+(60)$")
axs[2, 2].set_title(raw"$\alpha_2$ scan, $\alpha=0$")
axs[2, 2].set_xscale("log")
axs[2, 2].legend(frameon=false, fontsize=9)

for ax in axs
    ax.grid(alpha=0.18)
end

fig.suptitle(raw"Rice-Mele flat bath parameter scans, $T_b=0.1$, $t_{\max}=60$", y=0.995, fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.96])
outfile = joinpath(OUT_DIR, "rice_flat_bath_scan_summary_t60.png")
fig.savefig(outfile; dpi=230, bbox_inches="tight")
close(fig)
println("\nSaved $(relpath(outfile, ROOT))")

