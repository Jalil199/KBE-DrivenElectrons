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

function interpolated_pointwise_rates(times::AbstractVector, values::AbstractMatrix; n_uniform=401)
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

function chain_timeseries(GL_data)
    L, Nt, _ = size(GL_data)
    nk_t = zeros(Float64, L, Nt)
    @inbounds for it in 1:Nt
        nk_t[:, it] .= imag.(GL_data[:, it, it])
    end
    nk_t
end

cases = [
    (
        raw"$\alpha=2.5,\ \eta=0.05,\ p_c=1.0/a$",
        "Data/GL_L100_Te1.0_Tb0.1_u0.0_γ1.0_dispersion_α2.5_s1.0_ωc10.0_linear_spectral_η0.05_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q1.0_t050.0_ω03.141592653589793_σ2.0_A0.0_switch0_ti3.0_to20.0_tmax60.jld2",
        "#1f77b4",
    ),
    (
        raw"$\alpha=2.5,\ \eta=0.5,\ p_c=0.2/a$",
        "Data/GL_L100_Te1.0_Tb0.1_u0.0_γ1.0_dispersion_α2.5_s1.0_ωc10.0_linear_spectral_η0.5_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q0.2_t050.0_ω03.141592653589793_σ2.0_A0.0_switch0_ti3.0_to20.0_tmax60.jld2",
        "#2ca02c",
    ),
    (
        raw"$\alpha=5.0,\ \eta=0.05,\ p_c=1.0/a$",
        "Data/GL_L100_Te1.0_Tb0.1_u0.0_γ1.0_dispersion_α5.0_s1.0_ωc10.0_linear_spectral_η0.05_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q1.0_t050.0_ω03.141592653589793_σ2.0_A0.0_switch0_ti3.0_to20.0_tmax60.jld2",
        "#d62728",
    ),
    (
        raw"$\alpha=5.0,\ \eta=0.5,\ p_c=0.5/a$",
        "Data/GL_L100_Te1.0_Tb0.1_u0.0_γ1.0_dispersion_α5.0_s1.0_ωc10.0_linear_spectral_η0.5_v_b0.2_ωb00.1_power_exp_s_q1.0_λ_q0.5_t050.0_ω03.141592653589793_σ2.0_A0.0_switch0_ti3.0_to20.0_tmax60.jld2",
        "#9467bd",
    ),
]

fig, ax = subplots(figsize=(7.2, 4.8))

for (label, file, color) in cases
    GL_data = array_data(load(file, "GL"))
    ts = load(replace(file, "GL_" => "ts_"), "sol").t
    nk_t = chain_timeseries(GL_data)
    tmid, pointwise = interpolated_pointwise_rates(ts, nk_t)
    dmax = vec(maximum(pointwise; dims=1))
    ax.plot(tmid, dmax; color=color, linewidth=2.2, label=label)
end

ax.set_xlabel(raw"$t$")
ax.set_ylabel(raw"$D_{\max}(t)$")
ax.grid(alpha=0.18)
ax.legend(frameon=false, loc="best")
fig.tight_layout()
outpath = joinpath(OUTPUT_DIR, "chain_convergence_dmax_two_alphas.png")
fig.savefig(outpath; dpi=220, bbox_inches="tight")
close(fig)
println("Saved " * outpath)
