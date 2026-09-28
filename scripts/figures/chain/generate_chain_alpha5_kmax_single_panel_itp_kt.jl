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

function moving_average(y::AbstractVector, halfwidth::Int)
    n = length(y)
    ys = similar(y, Float64)
    @inbounds for i in 1:n
        lo = max(1, i - halfwidth)
        hi = min(n, i + halfwidth)
        ys[i] = sum(@view y[lo:hi]) / (hi - lo + 1)
    end
    ys
end

function interpolated_kmax_kt(ks::AbstractVector, ts::AbstractVector, nk_t::AbstractMatrix;
    kmin::Real, kmax::Real, nt_uniform::Int=401, nk_uniform::Int=801)
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
    argidx = zeros(Int, length(tmids))
    @inbounds for it in eachindex(tmids)
        local_idx = argmax(view(dnk, krange, it))
        argidx[it] = krange[local_idx]
    end

    return tmids, ku[argidx]
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
    (0.0, pi / 2, raw"$0 \leq k \leq \pi/2$", "-"),
    (pi / 2, pi, raw"$\pi/2 \leq k \leq \pi$", "--"),
]

fig, ax = subplots(figsize=(7.3, 4.8))

for (case_label, file, color) in cases
    GL_data = array_data(load(file, "GL"))
    ts = load(replace(file, "GL_" => "ts_"), "sol").t
    nk_t = chain_timeseries(GL_data)
    ks = ks_from_L(size(nk_t, 1))
    for (kmin, kmax, range_label, style) in ranges
        tmid, kvals = interpolated_kmax_kt(ks, ts, nk_t; kmin=kmin, kmax=kmax)
        kvals_smooth = moving_average(kvals, 6)
        ax.plot(tmid, kvals_smooth; color=color, linestyle=style, linewidth=2.0,
            label=case_label * " , " * range_label)
    end
end

ax.set_xlabel(raw"$t$")
ax.set_ylabel(raw"$k_{\max}(t)$")
ax.set_ylim(0, pi)
ax.set_yticks([0.0, pi/4, pi/2, 3pi/4, pi])
ax.set_yticklabels([raw"$0$", raw"$\pi/4$", raw"$\pi/2$", raw"$3\pi/4$", raw"$\pi$"])
ax.grid(alpha=0.18)
ax.legend(frameon=false, loc="best")

fig.tight_layout()
outpath = joinpath(OUTPUT_DIR, "chain_alpha5_kmax_single_panel_itp_kt.png")
fig.savefig(outpath; dpi=220, bbox_inches="tight")
close(fig)
println("Saved " * outpath)
