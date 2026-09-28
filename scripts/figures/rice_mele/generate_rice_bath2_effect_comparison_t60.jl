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
        init = occursin("_initupper_full_", b) ? :upper_full : :thermal,
        has_bath2 = occursin("_b2_", b),
        α = capture_float(b, "_dispersion_α([^_]+)_"),
        η = capture_float(b, "_η([^_]+)_v_b"),
        λq = capture_float(b, "_λ_q([^_]+)_t0"),
    )
end

function load_Nplus(path::AbstractString)
    p = parse_params(path)
    GL = array_data(load(path, "GL"))
    ts_obj = load(joinpath(DATA_DIR, replace(basename(path), "GL_" => "ts_")), "sol")
    ts = hasproperty(ts_obj, :t) ? collect(ts_obj.t) : collect(ts_obj)
    L = size(GL, 3)
    ks = collect(range(-π, stop=π - 2π / L, length=L))
    Us = [eigen(H_k(k; t1=-1.0, t2=-0.8, Δ=2.0)).vectors for k in ks]
    Nplus = zeros(Float64, length(ts))
    for it in eachindex(ts)
        acc = 0.0
        for ik in eachindex(ks)
            ρband = Us[ik]' * imag.(GL[:, :, ik, it, it]) * Us[ik]
            acc += real(ρband[2, 2])
        end
        Nplus[it] = acc / L
    end
    return merge(p, (; ts, Nplus, Nplus0=Nplus[1], Nplusf=Nplus[end], path))
end

files_b2 = sort(filter(f ->
    startswith(basename(f), "GL_L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.1_dispersion_α") &&
    occursin("_b2_α1.0_ωb0_22.5_η20.1_s_q20.0_λ_q210.0_", basename(f)) &&
    endswith(basename(f), "_tmax60.jld2") &&
    !occursin("_dispersion_α0.0_", basename(f)),
    readdir(DATA_DIR; join=true)))

files_no = sort(filter(f ->
    startswith(basename(f), "GL_L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.1_dispersion_α") &&
    !occursin("_b2_", basename(f)) &&
    endswith(basename(f), "_tmax60.jld2"),
    readdir(DATA_DIR; join=true)))

records = vcat(load_Nplus.(files_b2), load_Nplus.(files_no))

key(r) = (r.init, r.α, r.η, r.λq)
b2 = Dict(key(r) => r for r in records if r.has_bath2)
no = Dict(key(r) => r for r in records if !r.has_bath2)
common_keys = sort(collect(intersect(keys(b2), keys(no))))

println("Comparable cases: $(length(common_keys))")
for init in (:thermal, :upper_full)
    println("\n$(init):")
    rows = []
    for k in common_keys
        k[1] == init || continue
        rb = b2[k]
        rn = no[k]
        push!(rows, (; init=k[1], α=k[2], η=k[3], λq=k[4], with=rb.Nplusf, without=rn.Nplusf, improvement=rn.Nplusf-rb.Nplusf))
    end
    rows = sort(rows, by=r -> -r.improvement)
    for r in rows
        println("α=$(r.α) η=$(r.η) λq=$(r.λq) without=$(round(r.without,digits=6)) with=$(round(r.with,digits=6)) improvement=$(round(r.improvement,digits=6))")
    end
end

plt.rc("font", family="serif", size=12)
plt.rc("axes", linewidth=1.5)
plt.rc("xtick", direction="in")
plt.rc("ytick", direction="in")

fig, axs = subplots(1, 2; figsize=(12.8, 4.8), sharey=true)
for (iax, init) in enumerate((:thermal, :upper_full))
    ax = axs[iax]
    rows = [(; α=k[2], η=k[3], λq=k[4], with=b2[k].Nplusf, without=no[k].Nplusf)
            for k in common_keys if k[1] == init]
    ax.scatter([r.without for r in rows], [r.with for r in rows]; s=70, color=(init == :thermal ? "#2a9d8f" : "#3a5a98"))
    lim0 = min(minimum([r.without for r in rows]), minimum([r.with for r in rows])) - 0.01
    lim1 = max(maximum([r.without for r in rows]), maximum([r.with for r in rows])) + 0.01
    ax.plot([lim0, lim1], [lim0, lim1]; color="black", linestyle="--", linewidth=1.5)
    ax.set_xlim(lim0, lim1)
    ax.set_ylim(lim0, lim1)
    ax.set_xlabel(raw"$N_+(60)$ without bath 2")
    ax.set_ylabel(raw"$N_+(60)$ with bath 2")
    ax.set_title(init == :thermal ? "thermal" : "upper-full")
    ax.grid(alpha=0.18)
end
fig.suptitle(raw"Effect of optimized flat bath: points below diagonal improve relaxation", y=1.03, fontsize=14)
fig.tight_layout()
outfile = joinpath(OUT_DIR, "rice_bath2_effect_final_scatter_t60.png")
fig.savefig(outfile; dpi=230, bbox_inches="tight")
close(fig)
println("Saved $(relpath(outfile, ROOT))")

selected = [
    (:thermal, 1.0, 0.5, 0.2),
    (:thermal, 2.0, 0.5, 0.2),
    (:upper_full, 1.0, 0.5, 0.5),
    (:upper_full, 2.0, 0.05, 1.0),
]

fig2, axs2 = subplots(2, 2; figsize=(12.8, 7.2), sharex=true)
axs2 = reshape(collect(axs2), 2, 2)
for (i, k) in enumerate(selected)
    ax = axs2[i]
    rb = b2[k]
    rn = no[k]
    ax.plot(rn.ts, rn.Nplus; color="#d95f02", linewidth=2.3, label="without bath 2")
    ax.plot(rb.ts, rb.Nplus; color="#1b9e77", linewidth=2.3, label="with bath 2")
    ax.set_title("$(k[1]), α=$(k[2]), η=$(k[3]), λq=$(k[4])")
    ax.set_xlabel(raw"$t$")
    ax.set_ylabel(raw"$N_+(t)$")
    ax.grid(alpha=0.18)
    ax.legend(frameon=false, fontsize=9)
end
fig2.tight_layout()
outfile2 = joinpath(OUT_DIR, "rice_bath2_effect_representative_traces_t60.png")
fig2.savefig(outfile2; dpi=230, bbox_inches="tight")
close(fig2)
println("Saved $(relpath(outfile2, ROOT))")

