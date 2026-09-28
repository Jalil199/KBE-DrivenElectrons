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
    init = occursin("_initupper_full_", b) ? :upper_full : :thermal
    return (;
        init,
        α = capture_float(b, "_dispersion_α([^_]+)_"),
        η = capture_float(b, "_η([^_]+)_v_b"),
        λq = capture_float(b, "_λ_q([^_]+)_t0"),
        α2 = capture_float(b, "_b2_α([^_]+)_"),
        ωb2 = capture_float(b, "_ωb0_2([^_]+)_"),
        η2 = capture_float(b, "_η2([^_]+)_"),
        λq2 = capture_float(b, "_λ_q2([^_]+)_"),
    )
end

function band_trace(path::AbstractString)
    p = parse_params(path)
    GL = array_data(load(path, "GL"))
    ts_path = joinpath(DATA_DIR, replace(basename(path), "GL_" => "ts_"))
    ts_obj = load(ts_path, "sol")
    ts = hasproperty(ts_obj, :t) ? collect(ts_obj.t) : collect(ts_obj)

    L = size(GL, 3)
    ks = collect(range(-π, stop=π - 2π / L, length=L))
    Us = [eigen(H_k(k; t1=-1.0, t2=-0.8, Δ=2.0)).vectors for k in ks]

    Nminus = zeros(Float64, length(ts))
    Nplus = zeros(Float64, length(ts))
    nplus_final = zeros(Float64, L)
    for it in eachindex(ts)
        accm = 0.0
        accp = 0.0
        for ik in eachindex(ks)
            ρband = Us[ik]' * imag.(GL[:, :, ik, it, it]) * Us[ik]
            nm = real(ρband[1, 1])
            np = real(ρband[2, 2])
            accm += nm
            accp += np
            it == lastindex(ts) && (nplus_final[ik] = np)
        end
        Nminus[it] = accm / L
        Nplus[it] = accp / L
    end

    return merge(p, (; ts, ks, Nminus, Nplus, nplus_final, Nplus0=Nplus[1], Nplusf=Nplus[end], ΔNplus=Nplus[end] - Nplus[1], path))
end

files = sort(filter(f ->
    startswith(basename(f), "GL_L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.1_dispersion_α") &&
    occursin("_b2_α1.0_ωb0_22.5_η20.1_s_q20.0_λ_q210.0_", basename(f)) &&
    endswith(basename(f), "_tmax60.jld2") &&
    !occursin("_dispersion_α0.0_", basename(f)),
    readdir(DATA_DIR; join=true)))

records = band_trace.(files)
println("Loaded $(length(records)) best-bath2 with bath1 cases")

for init in (:thermal, :upper_full)
    rs = sort(filter(r -> r.init == init, records), by=r -> r.Nplusf)
    println("\n$(init) best to worst by Nplus(60):")
    for r in rs
        println("α=$(r.α) η=$(r.η) λq=$(r.λq) Nplus0=$(round(r.Nplus0,digits=6)) Nplusf=$(round(r.Nplusf,digits=6)) Δ=$(round(r.ΔNplus,digits=6))")
    end
end

plt.rc("font", family="serif", size=12)
plt.rc("axes", linewidth=1.5)
plt.rc("xtick", direction="in")
plt.rc("ytick", direction="in")

colors = Dict(0.2 => "#2a9d8f", 0.5 => "#e07a2f", 1.0 => "#3a5a98")
styles = Dict(0.05 => "-", 0.5 => "--")
alphas = [1.0, 2.0, 4.0]

fig, axs = subplots(2, 3; figsize=(16.5, 7.4), sharex=true)
axs = reshape(collect(axs), 2, 3)

for (icol, α) in enumerate(alphas)
    for (irow, init) in enumerate((:thermal, :upper_full))
        ax = axs[irow, icol]
        rs = sort(filter(r -> r.init == init && isapprox(r.α, α; atol=1e-10), records), by=r -> (r.η, r.λq))
        for r in rs
            label = raw"$\eta=" * string(r.η) * raw",\lambda_q=" * string(r.λq) * raw"$"
            ax.plot(r.ts, r.Nplus; color=colors[r.λq], linestyle=styles[r.η], linewidth=2.0, label=label)
        end
        ax.set_title((init == :thermal ? "thermal" : "upper-full") * raw", $\alpha=" * string(α) * raw"$")
        ax.set_xlabel(raw"$t$")
        ax.set_ylabel(raw"$N_+(t)$")
        ax.grid(alpha=0.18)
        ax.legend(frameon=false, fontsize=8, loc="best")
    end
end

fig.suptitle(raw"Rice-Mele best flat bath plus bath-1 scan, $\alpha_2=1,\omega_{b,2}=2.5,\eta_2=0.1,\lambda_{q,2}=10$", y=0.995, fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
outfile = joinpath(OUT_DIR, "rice_best_bath2_with_bath1_populations_t60.png")
fig.savefig(outfile; dpi=230, bbox_inches="tight")
close(fig)
println("Saved $(relpath(outfile, ROOT))")

fig2, axs2 = subplots(1, 2; figsize=(12.0, 4.8), sharey=false)

for (iax, init) in enumerate((:thermal, :upper_full))
    ax = axs2[iax]
    for η in (0.05, 0.5)
        xs = Float64[]
        ys = Float64[]
        cs = String[]
        for r in filter(r -> r.init == init && isapprox(r.η, η; atol=1e-10), records)
            push!(xs, r.α + 0.05 * (r.λq - 0.5))
            push!(ys, r.Nplusf)
            push!(cs, colors[r.λq])
        end
        ax.scatter(xs, ys; s=70, c=cs, marker=(η == 0.05 ? "o" : "s"), label=raw"$\eta=" * string(η) * raw"$")
    end
    ax.set_xlabel(raw"$\alpha$ with small $\lambda_q$ offset")
    ax.set_ylabel(raw"$N_+(60)$")
    ax.set_title(init == :thermal ? "thermal" : "upper-full")
    ax.grid(alpha=0.18)
    ax.legend(frameon=false, fontsize=9)
end

fig2.suptitle(raw"Final upper-band population; colors: $\lambda_q=0.2,0.5,1.0$", y=1.03, fontsize=14)
fig2.tight_layout()
outfile2 = joinpath(OUT_DIR, "rice_best_bath2_with_bath1_final_scatter_t60.png")
fig2.savefig(outfile2; dpi=230, bbox_inches="tight")
close(fig2)
println("Saved $(relpath(outfile2, ROOT))")

