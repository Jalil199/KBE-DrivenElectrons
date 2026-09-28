using JLD2
using LinearAlgebra
using Statistics
using PyPlot

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "main-rice_mele.jl"))

const DATA_DIR = joinpath(ROOT, "Data_rice")
const OUT_DIR = joinpath(ROOT, "distributions_rice")
mkpath(OUT_DIR)

array_data(x) = hasproperty(x, :data) ? getproperty(x, :data) : x

function field_value(name::AbstractString, key::AbstractString)
    m = match(Regex(key * "([^_]+)"), name)
    m === nothing && error("Could not read field $(key) from $(name)")
    return parse(Float64, m.captures[1])
end

function band_occupations(GLdata, Us, it)
    L = size(GLdata, 3)
    nminus = zeros(Float64, L)
    nplus = zeros(Float64, L)
    @inbounds for ik in 1:L
        ρsub = imag.(GLdata[:, :, ik, it, it])
        ρband = Us[ik]' * ρsub * Us[ik]
        nminus[ik] = real(ρband[1, 1])
        nplus[ik] = real(ρband[2, 2])
    end
    return nminus, nplus
end

function load_case(path::AbstractString)
    base = basename(path)
    dataset = replace(replace(base, "GL_" => ""), ".jld2" => "")
    GL = array_data(load(path, "GL"))
    ts_obj = load(joinpath(DATA_DIR, "ts_" * dataset * ".jld2"), "sol")
    ts = hasproperty(ts_obj, :t) ? collect(ts_obj.t) : collect(ts_obj)

    L = size(GL, 3)
    ks = collect(range(-π, stop=π - 2π / L, length=L))
    t1 = field_value(base, "t1")
    t2 = field_value(base, "t2")
    Δ = field_value(base, "Δ")
    Us = [eigen(H_k(k; t1=t1, t2=t2, Δ=Δ)).vectors for k in ks]

    _, nplus0 = band_occupations(GL, Us, 1)
    _, nplusf = band_occupations(GL, Us, lastindex(ts))
    gap = zeros(Float64, L)
    for (ik, k) in enumerate(ks)
        evals = eigen(H_k(k; t1=t1, t2=t2, Δ=Δ)).values
        gap[ik] = evals[2] - evals[1]
    end

    return (;
        base, ts, ks, nplus0, nplusf, dnplus=nplusf .- nplus0, gap,
        α=field_value(base, "α"),
        η=field_value(base, "η"),
        λq=field_value(base, "λ_q"),
        ωb2=field_value(base, "ωb0_2"),
    )
end

files = sort(filter(f ->
    startswith(basename(f), "GL_L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.1_dispersion_α") &&
    occursin("_η0.05_", basename(f)) &&
    occursin("_λ_q1.0_", basename(f)) &&
    occursin("_b2_α0.5_", basename(f)) &&
    endswith(basename(f), "_tmax60.jld2"),
    readdir(DATA_DIR; join=true)))

cases = [load_case(f) for f in files]
cases = filter(c -> c.α in (1.0, 4.0) && c.ωb2 in (1.0, 2.0, 2.5, 3.0), cases)
isempty(cases) && error("No frequency-scan two-bath cases found")

println("Loaded $(length(cases)) frequency-scan cases")
for c in sort(cases, by=c -> (c.α, c.ωb2))
    println("alpha=$(c.α) wb2=$(c.ωb2) min Δn+=$(round(minimum(c.dnplus), digits=5)) ",
            "at k=$(round(c.ks[argmin(c.dnplus)], digits=4)) ",
            "gap(k)=$(round(c.gap[argmin(c.dnplus)], digits=4)) ",
            "mean Δn+=$(round(mean(c.dnplus), digits=6))")
end

plt.rc("font", family="serif", size=12)
plt.rc("axes", linewidth=1.6)
plt.rc("xtick.major", width=1.4, size=6)
plt.rc("ytick.major", width=1.4, size=6)
plt.rc("xtick", direction="in")
plt.rc("ytick", direction="in")

colors = Dict(1.0 => "#2a9d8f", 2.0 => "#e07a2f", 2.5 => "#3a5a98", 3.0 => "#7b3294")
fig, axs = subplots(2, 2; figsize=(13.6, 8.4), sharex=true)
axs = reshape(collect(axs), 2, 2)

for (icol, α) in enumerate([1.0, 4.0])
    α_cases = sort(filter(c -> isapprox(c.α, α; atol=1e-10), cases), by=c -> c.ωb2)
    ref = first(α_cases)
    axd = axs[1, icol]
    axg = axs[2, icol]

    for c in α_cases
        color = colors[c.ωb2]
        label = raw"$\omega_{b,2}=" * string(c.ωb2) * raw"$"
        axd.plot(c.ks, c.dnplus; color=color, linewidth=2.2, label=label)
        axg.axhline(c.ωb2; color=color, linestyle="--", linewidth=1.6, alpha=0.9)
    end

    axg.plot(ref.ks, ref.gap; color="black", linewidth=2.2, label=raw"$E_+(k)-E_-(k)$")
    axd.axhline(0.0; color="black", linewidth=1.0, alpha=0.5)
    axd.set_title(raw"$\alpha=" * string(α) * raw",\ \eta=0.05,\ \lambda_q=1.0$")
    axd.set_ylabel(raw"$\Delta n_+(k)$")
    axg.set_ylabel(raw"$\omega$")
    axg.set_xlabel(raw"$k$")
    axd.grid(alpha=0.18)
    axg.grid(alpha=0.18)
    axd.legend(frameon=false, fontsize=9, loc="best")
    axg.legend(frameon=false, fontsize=9, loc="best")
    axg.set_xlim(-π, π)
    axg.set_xticks([-π, -π/2, 0, π/2, π])
    axg.set_xticklabels([raw"$-\pi$", raw"$-\pi/2$", raw"$0$", raw"$\pi/2$", raw"$\pi$"])
end

fig.suptitle(raw"Momentum-resolved relaxation versus Rice-Mele interband gap, $T_b=0.1$", y=0.995, fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.96])
outfile = joinpath(OUT_DIR, "rice_two_bath_resonance_kresolved_t60.png")
fig.savefig(outfile; dpi=230, bbox_inches="tight")
close(fig)
println("Saved $(relpath(outfile, ROOT))")
