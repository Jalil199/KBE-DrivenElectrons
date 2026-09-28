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

function dnkdt_map(ks::AbstractVector, ts::AbstractVector, nk_t::AbstractMatrix;
    nt_uniform::Int=801, nk_uniform::Int=801)
    tu = collect(range(first(ts), last(ts); length=nt_uniform))
    ku = collect(range(first(ks), last(ks); length=nk_uniform))
    itp = interpolate((ks, ts), nk_t, Gridded(Linear()))

    sampled = zeros(Float64, nk_uniform, nt_uniform)
    @inbounds for ik in eachindex(ku), it in eachindex(tu)
        sampled[ik, it] = itp(ku[ik], tu[it])
    end

    dnk = abs.(diff(sampled; dims=2)) ./ reshape(diff(tu), 1, :)
    tmids = (tu[1:end-1] .+ tu[2:end]) ./ 2
    return ku, tmids, dnk
end

cases = [
    (
        raw"$\eta=0.05,\ p_c=0.2/a$",
        "Data/GL_L100_Te1.0_Tb0.1_u0.0_γ1.0_dispersion_α2.5_s1.0_ωc10.0_linear_spectral_η0.05_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q0.2_t050.0_ω03.141592653589793_σ2.0_A0.0_switch0_ti3.0_to20.0_tmax60.jld2",
        "#1f77b4",
    ),
    (
        raw"$\eta=0.5,\ p_c=0.2/a$",
        "Data/GL_L100_Te1.0_Tb0.1_u0.0_γ1.0_dispersion_α2.5_s1.0_ωc10.0_linear_spectral_η0.5_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q0.2_t050.0_ω03.141592653589793_σ2.0_A0.0_switch0_ti3.0_to20.0_tmax60.jld2",
        "#d62728",
    ),
]

fig, axs = subplots(1, 2; figsize=(12.6, 4.8), squeeze=false)
ax1, ax2 = axs[1, 1], axs[1, 2]
images = Any[]

for (case_label, file, color) in cases
    GL_data = array_data(load(file, "GL"))
    ts = load(replace(file, "GL_" => "ts_"), "sol").t
    nk_t = chain_timeseries(GL_data)
    ks = ks_from_L(size(nk_t, 1))

    tmid_d, pointwise = interpolated_pointwise_rates(ts, nk_t)
    dmax = vec(maximum(pointwise; dims=1))
    ax1.plot(tmid_d, dmax; color=color, linewidth=2.2, label=case_label)
end

heat_label, heat_file, _ = cases[1]
GL_data = array_data(load(heat_file, "GL"))
ts = load(replace(heat_file, "GL_" => "ts_"), "sol").t
nk_t = chain_timeseries(GL_data)
ks = ks_from_L(size(nk_t, 1))
ku, tmids, dnk = dnkdt_map(ks, ts, nk_t)
im = ax2.imshow(
    dnk;
    origin="lower",
    aspect="auto",
    extent=(first(tmids), last(tmids), first(ku), last(ku)),
    cmap="magma_r",
    vmin=0.0,
    vmax=maximum(dnk),
    interpolation="nearest",
)
push!(images, im)

ax1.set_title("a")
ax1.set_xlabel(raw"$t$")
ax1.set_ylabel(raw"$\max_k\, |\partial_t n_k(t)|$")
ax1.grid(alpha=0.18)
ax1.legend(frameon=false, loc="best")

ax2.set_title("b")
ax2.set_xlabel(raw"$t$")
ax2.set_ylabel(raw"$k$")
ax2.set_ylim(0, pi)
ax2.set_yticks([0.0, pi/2, pi])
ax2.set_yticklabels([raw"$0$", raw"$\pi/2$", raw"$\pi$"])
ax2.text(0.03, 0.95, heat_label; transform=ax2.transAxes, ha="left", va="top",
    fontsize=11, bbox=Dict("facecolor" => "white", "alpha" => 0.78, "edgecolor" => "none", "pad" => 2.5))

cbar = fig.colorbar(images[end], ax=ax2, fraction=0.046, pad=0.03)
cbar.set_label(raw"$|\partial_t n_k(t)|$")

fig.tight_layout()
outpath = joinpath(OUTPUT_DIR, "chain_alpha25_pc02_two_panel_dmax_kpanel.png")
fig.savefig(outpath; dpi=220, bbox_inches="tight")
close(fig)
println("Saved " * outpath)
