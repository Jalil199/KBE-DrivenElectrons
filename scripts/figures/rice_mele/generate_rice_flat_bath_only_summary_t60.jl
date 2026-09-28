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

function upper_populations(path::AbstractString)
    p = parse_params(path)
    GL = array_data(load(path, "GL"))
    ts_path = joinpath(DATA_DIR, replace(basename(path), "GL_" => "ts_"))
    ts_obj = load(ts_path, "sol")
    ts = hasproperty(ts_obj, :t) ? collect(ts_obj.t) : collect(ts_obj)

    L = size(GL, 3)
    ks = collect(range(-π, stop=π - 2π / L, length=L))
    Us = [eigen(H_k(k; t1=-1.0, t2=-0.8, Δ=2.0)).vectors for k in ks]

    Nplus = zeros(Float64, length(ts))
    nplus_final = zeros(Float64, L)
    for it in eachindex(ts)
        acc = 0.0
        for ik in eachindex(ks)
            ρband = Us[ik]' * imag.(GL[:, :, ik, it, it]) * Us[ik]
            val = real(ρband[2, 2])
            acc += val
            it == lastindex(ts) && (nplus_final[ik] = val)
        end
        Nplus[it] = acc / L
    end
    return merge(p, (; ts, ks, Nplus, nplus_final, Nplusf=Nplus[end], path))
end

function approxin(x, vals; atol=1e-10)
    any(v -> isapprox(x, v; atol), vals)
end

files = sort(filter(f ->
    startswith(basename(f), "GL_L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.1_dispersion_α0.0_") &&
    occursin("_b2_", basename(f)) &&
    endswith(basename(f), "_tmax60.jld2"),
    readdir(DATA_DIR; join=true)))

records = upper_populations.(files)

freq_records = sort(filter(r ->
    isapprox(r.α2, 0.5; atol=1e-10) &&
    isapprox(r.η2, 0.1; atol=1e-10) &&
    isapprox(r.λq2, 10.0; atol=1e-10) &&
    approxin(r.ωb2, (1.0, 1.5, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0)),
    records), by=r -> r.ωb2)

eta_records = sort(filter(r ->
    isapprox(r.α2, 0.5; atol=1e-10) &&
    isapprox(r.λq2, 10.0; atol=1e-10) &&
    approxin(r.ωb2, (2.0, 2.5, 3.0)),
    records), by=r -> (r.ωb2, r.η2))

lq_records = sort(filter(r ->
    isapprox(r.α2, 0.5; atol=1e-10) &&
    isapprox(r.η2, 0.1; atol=1e-10) &&
    approxin(r.ωb2, (2.0, 2.5, 3.0)),
    records), by=r -> (r.ωb2, r.λq2))

a2_records = sort(filter(r ->
    isapprox(r.η2, 0.1; atol=1e-10) &&
    isapprox(r.λq2, 10.0; atol=1e-10) &&
    approxin(r.ωb2, (2.0, 2.5, 3.0)),
    records), by=r -> (r.ωb2, r.α2))

best = argmin([r.Nplusf for r in records])
println("Best flat-only case:")
println("ωb2=$(records[best].ωb2), α2=$(records[best].α2), η2=$(records[best].η2), λq2=$(records[best].λq2), Nplusf=$(round(records[best].Nplusf, digits=6))")

plt.rc("font", family="serif", size=12)
plt.rc("axes", linewidth=1.5)
plt.rc("xtick", direction="in")
plt.rc("ytick", direction="in")

colors = Dict(2.0 => "#2a9d8f", 2.5 => "#3a5a98", 3.0 => "#7b3294")

fig, axs = subplots(1, 3; figsize=(16.0, 4.7))

axs[1].plot([r.ωb2 for r in freq_records], [r.Nplusf for r in freq_records];
            marker="o", linewidth=2.4, color="#2a9d8f")
axs[1].set_xlabel(raw"$\omega_{b,2}$")
axs[1].set_ylabel(raw"$N_+(60)$")
axs[1].set_title(raw"Frequency scan, $\alpha_2=0.5,\eta_2=0.1,\lambda_{q,2}=10$")

for ωb2 in (2.0, 2.5, 3.0)
    rs = filter(r -> isapprox(r.ωb2, ωb2; atol=1e-10), eta_records)
    axs[2].plot([r.η2 for r in rs], [r.Nplusf for r in rs];
                marker="o", linewidth=2.2, color=colors[ωb2],
                label=raw"$\omega_{b,2}=" * string(ωb2) * raw"$")
end
axs[2].set_xscale("log")
axs[2].set_xlabel(raw"$\eta_2$")
axs[2].set_ylabel(raw"$N_+(60)$")
axs[2].set_title(raw"Bath width scan")
axs[2].legend(frameon=false, fontsize=9)

for ωb2 in (2.0, 2.5, 3.0)
    rs = filter(r -> isapprox(r.ωb2, ωb2; atol=1e-10), a2_records)
    axs[3].plot([r.α2 for r in rs], [r.Nplusf for r in rs];
                marker="o", linewidth=2.2, color=colors[ωb2],
                label=raw"$\omega_{b,2}=" * string(ωb2) * raw"$")
end
axs[3].set_xscale("log")
axs[3].set_xlabel(raw"$\alpha_2$")
axs[3].set_ylabel(raw"$N_+(60)$")
axs[3].set_title(raw"Flat-bath coupling scan")
axs[3].legend(frameon=false, fontsize=9)

for ax in axs
    ax.grid(alpha=0.18)
end

fig.suptitle(raw"Rice-Mele with flat bath only, $T_b=0.1$, $t_{\max}=60$", y=1.02, fontsize=15)
fig.tight_layout()
outfile = joinpath(OUT_DIR, "rice_flat_bath_only_summary_t60.png")
fig.savefig(outfile; dpi=230, bbox_inches="tight")
close(fig)
println("Saved $(relpath(outfile, ROOT))")

best_record = records[best]
fig2, axs2 = subplots(1, 2; figsize=(10.5, 4.4))
axs2[1].plot(best_record.ts, best_record.Nplus; color="#3a5a98", linewidth=2.4)
axs2[1].set_xlabel(raw"$t$")
axs2[1].set_ylabel(raw"$N_+(t)$")
axs2[1].set_title(raw"Best flat-only case")
axs2[1].grid(alpha=0.18)

axs2[2].plot(best_record.ks, best_record.nplus_final; color="#3a5a98", linewidth=2.2)
axs2[2].set_xlabel(raw"$k$")
axs2[2].set_ylabel(raw"$n_+(k,60)$")
axs2[2].set_xlim(-π, π)
axs2[2].set_xticks([-π, -π/2, 0, π/2, π])
axs2[2].set_xticklabels([raw"$-\pi$", raw"$-\pi/2$", raw"$0$", raw"$\pi/2$", raw"$\pi$"])
axs2[2].grid(alpha=0.18)

fig2.suptitle(raw"$\omega_{b,2}=" * string(best_record.ωb2) *
              raw",\ \alpha_2=" * string(best_record.α2) *
              raw",\ \eta_2=" * string(best_record.η2) *
              raw",\ \lambda_{q,2}=" * string(best_record.λq2) * raw"$",
              y=1.03, fontsize=14)
fig2.tight_layout()
outfile2 = joinpath(OUT_DIR, "rice_flat_bath_only_best_case_t60.png")
fig2.savefig(outfile2; dpi=230, bbox_inches="tight")
close(fig2)
println("Saved $(relpath(outfile2, ROOT))")

