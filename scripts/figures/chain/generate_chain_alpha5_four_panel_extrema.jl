using JLD2
using PyPlot
using Interpolations

rc("font", family="serif")
rc("font", size=13)
rc("axes", labelsize=15, titlesize=14)
rc("xtick", labelsize=11)
rc("ytick", labelsize=11)
rc("legend", fontsize=8)

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

function interpolated_pointwise_rates(times::AbstractVector, values::AbstractMatrix; n_uniform=801)
    tu = collect(range(first(times), last(times); length=n_uniform))
    dim = size(values, 1)
    sampled = zeros(Float64, dim, n_uniform)
    for j in 1:dim
        itp = interpolate((times,), values[j, :], Gridded(Linear()))
        @inbounds for i in eachindex(tu)
            sampled[j, i] = itp(tu[i])
        end
    end
    pointwise = zeros(Float64, dim, n_uniform - 1)
    @inbounds for i in 1:(n_uniform - 1)
        dt = tu[i + 1] - tu[i]
        pointwise[:, i] .= abs.(sampled[:, i + 1] .- sampled[:, i]) ./ dt
    end
    tmids = (tu[1:end-1] .+ tu[2:end]) ./ 2
    return tmids, pointwise
end

function kextrema_hard_soft(ks::AbstractVector, ts::AbstractVector, nk_t::AbstractMatrix;
    kmin::Real, kmax::Real, nt_uniform::Int=801, nk_uniform::Int=1601, power::Int=4, mode::Symbol=:max)
    keep = findall(k -> 0.0 <= k <= (pi - 1e-12), ks)
    kpos = ks[keep]
    nk_pos = nk_t[keep, :]

    tu = collect(range(first(ts), last(ts); length=nt_uniform))
    ku = collect(range(first(kpos), last(kpos); length=nk_uniform))
    itp = interpolate((kpos, ts), nk_pos, Gridded(Linear()))

    sampled = zeros(Float64, nk_uniform, nt_uniform)
    @inbounds for ik in eachindex(ku), it in eachindex(tu)
        sampled[ik, it] = itp(ku[ik], tu[it])
    end

    dnk = abs.(diff(sampled; dims=2)) ./ reshape(diff(tu), 1, :)
    tmids = (tu[1:end-1] .+ tu[2:end]) ./ 2
    krange = findall(k -> kmin <= k <= kmax, ku)

    khard = zeros(Float64, length(tmids))
    ksoft = zeros(Float64, length(tmids))
    @inbounds for it in eachindex(tmids)
        loc = view(dnk, krange, it)
        if mode == :max
            khard[it] = ku[krange[argmax(loc)]]
            weights = loc .^ power
        else
            khard[it] = ku[krange[argmin(loc)]]
            invloc = maximum(loc) .- loc
            weights = invloc .^ power
        end
        sw = sum(weights)
        ksoft[it] = sw <= 0 ? ku[krange[1]] : sum(ku[krange] .* weights) / sw
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

ranges = [
    (0.0, pi / 2, "-"),
    (pi / 2, pi, "--"),
]

fig, axs = subplots(2, 2; figsize=(12.6, 8.8), squeeze=false)
axa, axb = axs[1, 1], axs[1, 2]
axc, axd = axs[2, 1], axs[2, 2]

for (case_label, file, color) in cases
    GL_data = array_data(load(file, "GL"))
    ts = load(replace(file, "GL_" => "ts_"), "sol").t
    nk_t = chain_timeseries(GL_data)
    ks = ks_from_L(size(nk_t, 1))

    tmid_d, pointwise = interpolated_pointwise_rates(ts, nk_t)
    dmax = vec(maximum(pointwise; dims=1))
    dmin = vec(minimum(pointwise; dims=1))
    axa.plot(tmid_d, dmax; color=color, linewidth=2.2, label=case_label)
    axc.plot(tmid_d, dmin; color=color, linewidth=2.2, label=case_label)

    for (kmin, kmax, style) in ranges
        tmid_k, khard_max, ksoft_max = kextrema_hard_soft(ks, ts, nk_t; kmin=kmin, kmax=kmax, power=4, mode=:max)
        axb.plot(tmid_k, khard_max; color=color, linestyle=style, linewidth=1.0, alpha=0.35)
        axb.plot(tmid_k, ksoft_max; color=color, linestyle=style, linewidth=2.2)

        tmid_k, khard_min, ksoft_min = kextrema_hard_soft(ks, ts, nk_t; kmin=kmin, kmax=kmax, power=4, mode=:min)
        axd.plot(tmid_k, khard_min; color=color, linestyle=style, linewidth=1.0, alpha=0.35)
        axd.plot(tmid_k, ksoft_min; color=color, linestyle=style, linewidth=2.2)
    end
end

for (ax, title) in zip([axa, axb, axc, axd], ["a", "b", "c", "d"])
    ax.set_title(title)
    ax.grid(alpha=0.18)
end

axa.set_xlabel(raw"$t$")
axa.set_ylabel(raw"$\max_k\, |\partial_t n_k(t)|$")
axa.legend(frameon=false, loc="best")

axb.set_xlabel(raw"$t$")
axb.set_ylabel(raw"$k(t)$")
axb.set_ylim(0, pi)
axb.set_yticks([0.0, pi/4, pi/2, 3pi/4, pi])
axb.set_yticklabels([raw"$0$", raw"$\pi/4$", raw"$\pi/2$", raw"$3\pi/4$", raw"$\pi$"])
axb.axhline(1.0; color="black", linestyle=":", linewidth=1.2, alpha=0.7)
axb.axhline(0.5; color="black", linestyle="--", linewidth=1.0, alpha=0.55)

axc.set_xlabel(raw"$t$")
axc.set_ylabel(raw"$\min_k\, |\partial_t n_k(t)|$")
axc.legend(frameon=false, loc="best")

axd.set_xlabel(raw"$t$")
axd.set_ylabel(raw"$k(t)$")
axd.set_ylim(0, pi)
axd.set_yticks([0.0, pi/4, pi/2, 3pi/4, pi])
axd.set_yticklabels([raw"$0$", raw"$\pi/4$", raw"$\pi/2$", raw"$3\pi/4$", raw"$\pi$"])
axd.axhline(1.0; color="black", linestyle=":", linewidth=1.2, alpha=0.7)
axd.axhline(0.5; color="black", linestyle="--", linewidth=1.0, alpha=0.55)

fig.tight_layout()
outpath = joinpath(OUTPUT_DIR, "chain_alpha5_four_panel_extrema.png")
fig.savefig(outpath; dpi=220, bbox_inches="tight")
close(fig)
println("Saved " * outpath)
