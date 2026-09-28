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
    return merge(p, (; ts, Nplus, Nplusf=Nplus[end]))
end

files = sort(filter(f ->
    startswith(basename(f), "GL_L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.1_dispersion_α") &&
    occursin("_b2_α1.0_ωb0_22.5_η20.1_s_q20.0_λ_q210.0_", basename(f)) &&
    endswith(basename(f), "_tmax60.jld2") &&
    !occursin("_dispersion_α0.0_", basename(f)),
    readdir(DATA_DIR; join=true)))

records = load_Nplus.(files)

plt.rc("font", family="serif", size=12)
plt.rc("axes", linewidth=1.6)
plt.rc("xtick", direction="in")
plt.rc("ytick", direction="in")

colors = Dict(0.2 => "#2a9d8f", 0.5 => "#e07a2f", 1.0 => "#3a5a98")
styles = Dict(0.05 => "-", 0.5 => "--")
alphas = [1.0, 2.0, 4.0]
inits = [:thermal, :upper_full]

fig, axs = subplots(2, 3; figsize=(16.8, 7.8), sharex=true)
axs = reshape(collect(axs), 2, 3)

for (irow, init) in enumerate(inits), (icol, α) in enumerate(alphas)
    ax = axs[irow, icol]
    rs = sort(filter(r -> r.init == init && isapprox(r.α, α; atol=1e-10), records), by=r -> (r.η, r.λq))
    for r in rs
        label = raw"$\eta=" * string(r.η) * raw",\ \lambda_q=" * string(r.λq) * raw"$"
        ax.plot(r.ts, r.Nplus; color=colors[r.λq], linestyle=styles[r.η], linewidth=2.2, label=label)
    end
    ax.set_title((init == :thermal ? "thermal" : "upper full") * raw", $\alpha=" * string(α) * raw"$")
    ax.set_xlabel(raw"$t$")
    ax.set_ylabel(raw"$N_+(t)$")
    ax.grid(alpha=0.18)
    ax.legend(frameon=false, fontsize=8, loc="best")
end

fig.suptitle(raw"Rice-Mele relaxation with optimized flat bath: $\alpha_2=1,\omega_{b,2}=2.5,\eta_2=0.1,\lambda_{q,2}=10$", y=0.995, fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.96])
outfile = joinpath(OUT_DIR, "rice_best_bath2_relaxation_time_panels_t60.png")
fig.savefig(outfile; dpi=230, bbox_inches="tight")
close(fig)
println("Saved $(relpath(outfile, ROOT))")

# A compact figure with only the best final curve per initial condition and alpha.
fig2, axs2 = subplots(1, 2; figsize=(12.2, 4.6), sharex=true)
for (iax, init) in enumerate(inits)
    ax = axs2[iax]
    for α in alphas
        rs = filter(r -> r.init == init && isapprox(r.α, α; atol=1e-10), records)
        best = rs[argmin([r.Nplusf for r in rs])]
        label = raw"$\alpha=" * string(α) * raw",\eta=" * string(best.η) * raw",\lambda_q=" * string(best.λq) * raw"$"
        ax.plot(best.ts, best.Nplus; linewidth=2.5, label=label)
    end
    ax.set_title(init == :thermal ? "thermal best curves" : "upper-full best curves")
    ax.set_xlabel(raw"$t$")
    ax.set_ylabel(raw"$N_+(t)$")
    ax.grid(alpha=0.18)
    ax.legend(frameon=false, fontsize=8, loc="best")
end
fig2.tight_layout()
outfile2 = joinpath(OUT_DIR, "rice_best_bath2_relaxation_best_curves_t60.png")
fig2.savefig(outfile2; dpi=230, bbox_inches="tight")
close(fig2)
println("Saved $(relpath(outfile2, ROOT))")

