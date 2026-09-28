using JLD2
using PyPlot
using Interpolations

rc("font", family="serif")
rc("font", size=13)
rc("axes", labelsize=15, titlesize=14)
rc("xtick", labelsize=11)
rc("ytick", labelsize=11)
rc("legend", fontsize=9)

const OUTPUT_DIR = "distributions"
mkpath(OUTPUT_DIR)

array_data(x) = hasproperty(x, :data) ? getproperty(x, :data) : x

function chain_timeseries(GL_data)
    L, Nt, _ = size(GL_data)
    nk_t = zeros(Float64, L, Nt)
    @inbounds for it in 1:Nt
        nk_t[:, it] .= imag.(GL_data[:, it, it])
    end
    nk_t
end

ks_from_L(L) = collect(range(-pi, stop=pi - 2pi / L; length=L))

function kmin_hard_soft_full(ks::AbstractVector, ts::AbstractVector, nk_t::AbstractMatrix;
    nt_uniform::Int=801, nk_uniform::Int=1601, power::Int=4)
    tu = collect(range(first(ts), last(ts); length=nt_uniform))
    ku = collect(range(first(ks), last(ks); length=nk_uniform))
    itp = interpolate((ks, ts), nk_t, Gridded(Linear()))

    sampled = zeros(Float64, nk_uniform, nt_uniform)
    @inbounds for ik in eachindex(ku), it in eachindex(tu)
        sampled[ik, it] = itp(ku[ik], tu[it])
    end

    dnk = abs.(diff(sampled; dims=2)) ./ reshape(diff(tu), 1, :)
    tmids = (tu[1:end-1] .+ tu[2:end]) ./ 2

    khard = zeros(Float64, length(tmids))
    ksoft = zeros(Float64, length(tmids))
    @inbounds for it in eachindex(tmids)
        loc = view(dnk, :, it)
        imin = argmin(loc)
        khard[it] = ku[imin]
        invloc = maximum(loc) .- loc
        weights = invloc .^ power
        sw = sum(weights)
        ksoft[it] = sw <= 0 ? ku[1] : sum(ku .* weights) / sw
    end
    return tmids, khard, ksoft
end

cases = [
    (
        raw"$\eta=0.05,\ p_c=1.0/a$",
        "Data/GL_L100_Te1.0_Tb0.1_u0.0_γ1.0_dispersion_α5.0_s1.0_ωc10.0_linear_spectral_η0.05_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q1.0_t050.0_ω03.141592653589793_σ2.0_A0.0_switch0_ti3.0_to20.0_tmax60.jld2",
        "#1f77b4",
    ),
    (
        raw"$\eta=0.5,\ p_c=0.5/a$",
        "Data/GL_L100_Te1.0_Tb0.1_u0.0_γ1.0_dispersion_α5.0_s1.0_ωc10.0_linear_spectral_η0.5_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q0.5_t050.0_ω03.141592653589793_σ2.0_A0.0_switch0_ti3.0_to20.0_tmax60.jld2",
        "#d62728",
    ),
]

fig, ax = subplots(figsize=(7.1, 4.7))

for (label, file, color) in cases
    GL_data = array_data(load(file, "GL"))
    ts = load(replace(file, "GL_" => "ts_"), "sol").t
    nk_t = chain_timeseries(GL_data)
    ks = ks_from_L(size(nk_t, 1))
    tmid, khard, ksoft = kmin_hard_soft_full(ks, ts, nk_t; power=4)
    ax.plot(tmid, khard; color=color, linewidth=1.0, alpha=0.35, label=label * raw" : $k_{\min}$")
    ax.plot(tmid, ksoft; color=color, linewidth=2.2, label=label * raw" : $k_{*}^{\min}$")
end

ax.set_xlabel(raw"$t$")
ax.set_ylabel(raw"$k(t)$")
ax.set_ylim(-pi, pi)
ax.set_yticks([-pi, -pi/2, 0.0, pi/2, pi])
ax.set_yticklabels([raw"$-\pi$", raw"$-\pi/2$", raw"$0$", raw"$\pi/2$", raw"$\pi$"])
ax.grid(alpha=0.18)
ax.legend(frameon=false, loc="best")

fig.tight_layout()
outpath = joinpath(OUTPUT_DIR, "chain_alpha5_kmin_fullrange.png")
fig.savefig(outpath; dpi=220, bbox_inches="tight")
close(fig)
println("Saved " * outpath)
