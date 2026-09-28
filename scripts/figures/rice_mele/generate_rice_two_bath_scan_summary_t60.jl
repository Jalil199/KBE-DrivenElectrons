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
    ts_path = joinpath(DATA_DIR, "ts_" * dataset * ".jld2")

    GL = array_data(load(path, "GL"))
    ts_obj = load(ts_path, "sol")
    ts = hasproperty(ts_obj, :t) ? collect(ts_obj.t) : collect(ts_obj)

    L = size(GL, 3)
    ks = collect(range(-π, stop=π - 2π / L, length=L))
    t1 = field_value(base, "t1")
    t2 = field_value(base, "t2")
    Δ = field_value(base, "Δ")
    Us = [eigen(H_k(k; t1=t1, t2=t2, Δ=Δ)).vectors for k in ks]

    Nminus = zeros(Float64, length(ts))
    Nplus = zeros(Float64, length(ts))
    nplus_final = zeros(Float64, L)
    nminus_final = zeros(Float64, L)

    for it in eachindex(ts)
        nminus, nplus = band_occupations(GL, Us, it)
        Nminus[it] = sum(nminus) / L
        Nplus[it] = sum(nplus) / L
        if it == lastindex(ts)
            nminus_final .= nminus
            nplus_final .= nplus
        end
    end

    return (;
        base, dataset, ts, ks, Nminus, Nplus, nminus_final, nplus_final,
        α=field_value(base, "α"),
        η=field_value(base, "η"),
        λq=field_value(base, "λ_q"),
        ωb2=field_value(base, "ωb0_2"),
    )
end

fermi_fd(ω, T, μ=0.0) = 1 / (exp((ω - μ) / T) + 1)

function thermal_upper_reference(ks; t1=-1.0, t2=-0.8, Δ=2.0, T=0.1, μ=0.0)
    nplus = zeros(Float64, length(ks))
    for (ik, k) in enumerate(ks)
        evals = eigen(H_k(k; t1=t1, t2=t2, Δ=Δ)).values
        nplus[ik] = fermi_fd(evals[2], T, μ)
    end
    return nplus, mean(nplus)
end

function get_cases()
    files = sort(filter(f ->
        startswith(basename(f), "GL_L80_t1-1.0_t2-0.8_Δ2.0_Te1.0_Tb0.1_dispersion_α") &&
        occursin("_b2_α0.5_", basename(f)) &&
        endswith(basename(f), "_tmax60.jld2"),
        readdir(DATA_DIR; join=true)))
    isempty(files) && error("No two-bath Rice-Mele tmax60 files found in $(DATA_DIR)")
    return [load_case(f) for f in files]
end

function style_axes!(axs)
    for ax in axs
        ax.grid(alpha=0.18)
        ax.tick_params(direction="in", width=1.4, length=5)
    end
end

plt.rc("font", family="serif", size=12)
plt.rc("axes", linewidth=1.6)
plt.rc("xtick.major", width=1.4, size=6)
plt.rc("ytick.major", width=1.4, size=6)
plt.rc("xtick", direction="in")
plt.rc("ytick", direction="in")

cases = get_cases()
println("Loaded $(length(cases)) two-bath cases")
ks_ref = first(cases).ks
nplus_th, Nplus_th = thermal_upper_reference(ks_ref; T=0.1)
println("Nplus thermal Tb=0.1 = $(Nplus_th)")

colors_λ = Dict(0.2 => "#2a9d8f", 0.5 => "#e07a2f", 1.0 => "#3a5a98")
styles_η = Dict(0.05 => "-", 0.5 => "--")
alphas = [1.0, 2.0, 4.0]

# Main sweep: second bath fixed at ωb2 = 2.5.
main_cases = filter(c -> isapprox(c.ωb2, 2.5; atol=1e-10), cases)
fig, axs = subplots(2, 3; figsize=(16.8, 7.6), sharex=true)
axs = reshape(collect(axs), 2, 3)

for (icol, α) in enumerate(alphas)
    α_cases = sort(filter(c -> isapprox(c.α, α; atol=1e-10), main_cases), by=c -> (c.η, c.λq))
    for c in α_cases
        color = get(colors_λ, c.λq, "black")
        ls = get(styles_η, c.η, "-")
        label = raw"$\eta=" * string(c.η) * raw",\ \lambda_q=" * string(c.λq) * raw"$"
        axs[1, icol].plot(c.ts, c.Nplus; color=color, linestyle=ls, linewidth=2.1, label=label)
        axs[2, icol].plot(c.ts, c.Nminus; color=color, linestyle=ls, linewidth=2.1, label=label)
    end
    axs[1, icol].axhline(Nplus_th; color="black", linestyle=":", linewidth=1.7,
                         label=icol == 1 ? raw"$N_+^{\mathrm{th}}$" : nothing)
    axs[1, icol].set_title(raw"$\alpha=" * string(α) * raw"$")
    axs[1, icol].set_ylabel(raw"$N_+(t)$")
    axs[2, icol].set_ylabel(raw"$N_-(t)$")
    axs[2, icol].set_xlabel(raw"$t$")
    axs[1, icol].legend(frameon=false, fontsize=8, loc="best")
end
style_axes!(axs[:])
fig.suptitle(raw"Rice-Mele two-bath scan, $\omega_{b,2}=2.5$, $T_b=0.1$", y=0.995, fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.96])
outfile = joinpath(OUT_DIR, "rice_two_bath_main_populations_t60.png")
fig.savefig(outfile; dpi=230, bbox_inches="tight")
close(fig)
println("Saved $(relpath(outfile, ROOT))")

# Frequency scan: compare resonant bath frequencies at fixed η=0.05, λq=1.0.
freqs = [1.0, 2.0, 2.5, 3.0]
colors_ω = Dict(1.0 => "#2a9d8f", 2.0 => "#e07a2f", 2.5 => "#3a5a98", 3.0 => "#7b3294")
freq_cases = filter(c -> c.α in (1.0, 4.0) && isapprox(c.η, 0.05; atol=1e-10) && isapprox(c.λq, 1.0; atol=1e-10), cases)

fig2, axs2 = subplots(2, 2; figsize=(12.8, 8.2))
axs2 = reshape(collect(axs2), 2, 2)

for (icol, α) in enumerate([1.0, 4.0])
    α_cases = sort(filter(c -> isapprox(c.α, α; atol=1e-10), freq_cases), by=c -> c.ωb2)
    for c in α_cases
        color = get(colors_ω, c.ωb2, "black")
        label = raw"$\omega_{b,2}=" * string(c.ωb2) * raw"$"
        axs2[1, icol].plot(c.ts, c.Nplus; color=color, linewidth=2.3, label=label)
        axs2[2, icol].plot(c.ks, c.nplus_final; color=color, linewidth=2.1, label=label)
        println("alpha=$(c.α) eta=$(c.η) lambda=$(c.λq) wb2=$(c.ωb2) ",
                "Nplus0=$(round(c.Nplus[1], digits=6)) ",
                "Nplusf=$(round(c.Nplus[end], digits=6)) ",
                "Delta=$(round(c.Nplus[end] - c.Nplus[1], digits=6))")
    end
    axs2[1, icol].axhline(Nplus_th; color="black", linestyle=":", linewidth=1.7,
                          label=icol == 1 ? raw"$N_+^{\mathrm{th}}$" : nothing)
    axs2[2, icol].plot(ks_ref, nplus_th; color="black", linestyle=":", linewidth=1.7,
                       label=icol == 1 ? raw"$n_+^{\mathrm{th}}(k)$" : nothing)
    axs2[1, icol].set_title(raw"$\alpha=" * string(α) * raw",\ \eta=0.05,\ \lambda_q=1.0$")
    axs2[1, icol].set_ylabel(raw"$N_+(t)$")
    axs2[2, icol].set_ylabel(raw"$n_+(k,t=60)$")
    axs2[2, icol].set_xlabel(raw"$k$")
    axs2[2, icol].set_xlim(-π, π)
    axs2[2, icol].set_xticks([-π, -π/2, 0, π/2, π])
    axs2[2, icol].set_xticklabels([raw"$-\pi$", raw"$-\pi/2$", raw"$0$", raw"$\pi/2$", raw"$\pi$"])
    axs2[1, icol].legend(frameon=false, fontsize=9, loc="best")
end
style_axes!(axs2[:])
fig2.suptitle(raw"Second-bath frequency scan, $T_b=0.1$", y=0.995, fontsize=15)
fig2.tight_layout(rect=[0, 0, 1, 0.96])
outfile2 = joinpath(OUT_DIR, "rice_two_bath_frequency_scan_t60.png")
fig2.savefig(outfile2; dpi=230, bbox_inches="tight")
close(fig2)
println("Saved $(relpath(outfile2, ROOT))")
