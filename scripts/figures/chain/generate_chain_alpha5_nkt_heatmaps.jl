using JLD2
using PyPlot
using Interpolations

rc("font", family="serif")
rc("font", size=13)
rc("axes", labelsize=15, titlesize=15)
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

function interpolate_nk(ks::AbstractVector, ts::AbstractVector, nk_t::AbstractMatrix;
    nk_uniform::Int=401, nt_uniform::Int=401)
    keep = findall(k -> 0.0 <= k <= (pi - 1e-12), ks)
    kpos = ks[keep]
    nk_pos = nk_t[keep, :]

    ku = collect(range(first(kpos), last(kpos); length=nk_uniform))
    tu = collect(range(first(ts), last(ts); length=nt_uniform))
    itp = interpolate((kpos, ts), nk_pos, Gridded(Linear()))

    sampled = zeros(Float64, nk_uniform, nt_uniform)
    @inbounds for ik in eachindex(ku), it in eachindex(tu)
        sampled[ik, it] = itp(ku[ik], tu[it])
    end
    return ku, tu, sampled
end

cases = [
    (
        raw"$\eta=0.05,\ p_c=1.0/a$",
        "Data/GL_L100_Te1.0_Tb0.1_u0.0_γ1.0_dispersion_α5.0_s1.0_ωc10.0_linear_spectral_η0.05_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q1.0_t050.0_ω03.141592653589793_σ2.0_A0.0_switch0_ti3.0_to20.0_tmax60.jld2",
    ),
    (
        raw"$\eta=0.5,\ p_c=0.5/a$",
        "Data/GL_L100_Te1.0_Tb0.1_u0.0_γ1.0_dispersion_α5.0_s1.0_ωc10.0_linear_spectral_η0.5_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q0.5_t050.0_ω03.141592653589793_σ2.0_A0.0_switch0_ti3.0_to20.0_tmax60.jld2",
    ),
]

maps = let maps_acc = Any[]
    for (_, file) in cases
        GL_data = array_data(load(file, "GL"))
        ts = load(replace(file, "GL_" => "ts_"), "sol").t
        nk_t = chain_timeseries(GL_data)
        ks = ks_from_L(size(nk_t, 1))
        ku, tu, nkmap = interpolate_nk(ks, ts, nk_t)
        push!(maps_acc, (ku, tu, nkmap))
    end
    maps_acc
end

fig, axs = subplots(1, 2; figsize=(12.6, 4.9), squeeze=false, sharex=true, sharey=true)
axs = axs[1, :]
images = Any[]

for (ax, (label, _), (ku, tu, nkmap)) in zip(axs, cases, maps)
    im = ax.imshow(
        nkmap;
        origin="lower",
        aspect="auto",
        extent=(first(tu), last(tu), first(ku), last(ku)),
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    push!(images, im)
    ax.set_title(label)
    ax.set_xlabel(raw"$t$")
    ax.axhline(pi / 2; color="black", linestyle="-", linewidth=1.0, alpha=0.9)
    ax.axhline(1.0; color="#1f3b73", linestyle=":", linewidth=1.2, alpha=0.95)
    ax.axhline(0.5; color="#1f3b73", linestyle="--", linewidth=1.1, alpha=0.85)
    ax.contour(tu, ku, nkmap, levels=[0.5], colors="white", linewidths=1.0, alpha=0.9)
    ax.text(0.03, 0.95, label; transform=ax.transAxes, ha="left", va="top",
        fontsize=12, bbox=Dict("facecolor" => "white", "alpha" => 0.75, "edgecolor" => "none", "pad" => 2.5))
end

axs[1].set_ylabel(raw"$k$")
axs[1].set_ylim(0, pi)
axs[1].set_yticks([0.0, pi/4, pi/2, 3pi/4, pi])
axs[1].set_yticklabels([raw"$0$", raw"$\pi/4$", raw"$\pi/2$", raw"$3\pi/4$", raw"$\pi$"])

cbar = fig.colorbar(images[end], ax=axs, fraction=0.045, pad=0.03)
cbar.set_label(raw"$n_k(t)$")

fig.subplots_adjust(wspace=0.08, right=0.88)
outpath = joinpath(OUTPUT_DIR, "chain_alpha5_nkt_heatmaps.png")
fig.savefig(outpath; dpi=220, bbox_inches="tight")
close(fig)
println("Saved " * outpath)
