using JLD2
using PyPlot
using Interpolations
using FFTW
using Statistics

rc("font", family="serif")
rc("font", size=13)
rc("axes", labelsize=15, titlesize=15)
rc("xtick", labelsize=11)
rc("ytick", labelsize=11)
rc("legend", fontsize=9)

const OUTPUT_DIR = "distributions"
const OMEGA_B0 = 0.1
const V_B = 0.2
const GAMMA = 1.0
mkpath(OUTPUT_DIR)

array_data(x) = hasproperty(x, :data) ? getproperty(x, :data) : x
ks_from_L(L) = collect(range(-pi, stop=pi - 2pi / L; length=L))

function chain_timeseries(GL_data)
    L, Nt, _ = size(GL_data)
    nk_t = zeros(Float64, L, Nt)
    @inbounds for it in 1:Nt
        nk_t[:, it] .= imag.(GL_data[:, it, it])
    end
    nk_t
end

function interpolate_nk_time(ks::AbstractVector, ts::AbstractVector, nk_t::AbstractMatrix;
    nk_uniform::Int=801, nt_uniform::Int=1024)
    ku = collect(range(first(ks), last(ks); length=nk_uniform))
    tu = collect(range(first(ts), last(ts); length=nt_uniform))
    itp = interpolate((ks, ts), nk_t, Gridded(Linear()))

    sampled = zeros(Float64, nk_uniform, nt_uniform)
    @inbounds for ik in eachindex(ku), it in eachindex(tu)
        sampled[ik, it] = itp(ku[ik], tu[it])
    end
    return ku, tu, sampled
end

function temporal_fft_map(ku::AbstractVector, tu::AbstractVector, nkmap::AbstractMatrix)
    nk_centered = nkmap .- mean(nkmap; dims=2)
    window = reshape(0.5 .- 0.5 .* cos.(2pi .* (0:length(tu)-1) ./ (length(tu)-1)), 1, :)
    weighted = nk_centered .* window

    dt = tu[2] - tu[1]
    fft_raw = fft(weighted, 2)
    fft_shifted = fftshift(fft_raw, 2)
    Ω = 2pi .* fftshift(fftfreq(length(tu), 1 / dt))
    amp = abs.(fft_shifted)
    return Ω, amp
end

cases = [
    (
        raw"$\eta=0.05,\ p_c=0.5/a$",
        "Data/GL_L100_Te1.0_Tb0.1_u0.0_γ1.0_dispersion_α5.0_s1.0_ωc10.0_linear_spectral_η0.05_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q0.5_t050.0_ω03.141592653589793_σ2.0_A0.0_switch0_ti3.0_to20.0_tmax60.jld2",
    ),
    (
        raw"$\eta=0.5,\ p_c=0.5/a$",
        "Data/GL_L100_Te1.0_Tb0.1_u0.0_γ1.0_dispersion_α5.0_s1.0_ωc10.0_linear_spectral_η0.5_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q0.5_t050.0_ω03.141592653589793_σ2.0_A0.0_switch0_ti3.0_to20.0_tmax60.jld2",
    ),
]

maps = let acc = Any[]
    for (_, file) in cases
        GL_data = array_data(load(file, "GL"))
        ts = load(replace(file, "GL_" => "ts_"), "sol").t
        nk_t = chain_timeseries(GL_data)
        ks = ks_from_L(size(nk_t, 1))
        ku, tu, nkmap = interpolate_nk_time(ks, ts, nk_t)
        Ω, amp = temporal_fft_map(ku, tu, nkmap)
        push!(acc, (ku, Ω, amp))
    end
    acc
end

Ωmax = 2.5
trimmed = let acc = Any[]
    for (ku, Ω, amp) in maps
        keep = findall(ω -> 1e-8 < ω <= Ωmax, Ω)
        push!(acc, (ku, Ω[keep], amp[:, keep]))
    end
    acc
end

vals = vcat([vec(log10.(amp .+ 1e-8)) for (_, _, amp) in trimmed]...)
vmin = quantile(vals, 0.10)
vmax = quantile(vals, 0.995)

fig, axs = subplots(1, 2; figsize=(12.8, 5.0), squeeze=false, sharex=true, sharey=true)
axs = axs[1, :]
images = Any[]

for (ax, (label, _), (ku, Ω, amp)) in zip(axs, cases, trimmed)
    z = log10.(amp .+ 1e-8)
    im = ax.imshow(
        z;
        origin="lower",
        aspect="auto",
        extent=(first(Ω), last(Ω), first(ku), last(ku)),
        cmap="magma_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    push!(images, im)
    ax.set_title(label)
    ax.set_xlabel(raw"$\Omega$")
    kline = collect(range(-pi, pi; length=1200))
    ωline = OMEGA_B0 .+ V_B .* abs.(kline)
    ax.plot(ωline, kline; color="cyan", linewidth=1.35, alpha=0.95)
    εline = abs.(-2 .* GAMMA .* cos.(kline))
    ax.plot(εline, kline; color="#7CFC00", linewidth=1.25, linestyle="--", alpha=0.95)
    ax.text(0.03, 0.95, label; transform=ax.transAxes, ha="left", va="top",
        fontsize=12, bbox=Dict("facecolor" => "white", "alpha" => 0.78, "edgecolor" => "none", "pad" => 2.5))
end

axs[1].set_ylabel(raw"$k$")
axs[1].set_xlim(0, Ωmax)
axs[1].set_ylim(-pi, pi)
axs[1].set_yticks([-pi, -pi/2, 0.0, pi/2, pi])
axs[1].set_yticklabels([raw"$-\pi$", raw"$-\pi/2$", raw"$0$", raw"$\pi/2$", raw"$\pi$"])

cbar = fig.colorbar(images[end], ax=axs, fraction=0.045, pad=0.03)
cbar.set_label(raw"$\log_{10} |\tilde n(k,\Omega)|$")

fig.subplots_adjust(wspace=0.08, right=0.88)
outpath = joinpath(OUTPUT_DIR, "chain_alpha5_nkw_temporal_fft.png")
fig.savefig(outpath; dpi=220, bbox_inches="tight")
close(fig)
println("Saved " * outpath)
