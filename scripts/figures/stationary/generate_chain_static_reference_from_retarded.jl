using JLD2
using PyPlot

include(joinpath(@__DIR__, "main.jl"))

rc("font", family="serif")
rc("font", size=14)
rc("axes", labelsize=16, titlesize=15)
rc("xtick", labelsize=12)
rc("ytick", labelsize=12)
rc("legend", fontsize=11)

const OUTPUT_DIR = "distributions"
mkpath(OUTPUT_DIR)

function fermi_fd(omega, T, mu)
    return 1 / (exp((omega - mu) / T) + 1)
end

function parse_arg(args, i, default)
    return length(args) >= i ? parse(typeof(default), args[i]) : default
end

function chain_static_reference(; L=100, u=0.0, γ=1.0, T=1.0, μ=0.0, η=0.08, ωmin=-3.0, ωmax=3.0, Nω=1200)
    ks = collect(range(-π, stop=π - 2π / L, length=L))
    ϵs = ϵ_k(ks; u, γ)
    nk_eq = fermi_fd.(ϵs, T, μ)
    ωs = collect(range(ωmin, ωmax, length=Nω))

    A_sum = zeros(Float64, Nω)
    Gless_im_sum = zeros(Float64, Nω)

    for (iw, ω) in enumerate(ωs)
        fω = fermi_fd(ω, T, μ)
        accA = 0.0
        for ϵ in ϵs
            Gr = inv(complex(ω, η) - ϵ)
            accA += -2 * imag(Gr)
        end
        A_sum[iw] = accA / L
        Gless_im_sum[iw] = fω * A_sum[iw]
    end

    F_num = fill(NaN, Nω)
    mask = abs.(A_sum) .> 1e-10
    F_num[mask] .= Gless_im_sum[mask] ./ A_sum[mask]

    return (; ks, ϵs, nk_eq, ωs, A_sum, Gless_im_sum, F_num)
end

L = parse_arg(ARGS, 1, 100)
u = parse_arg(ARGS, 2, 0.0)
γ = parse_arg(ARGS, 3, 1.0)
T = parse_arg(ARGS, 4, 1.0)
μ = parse_arg(ARGS, 5, 0.0)
η = parse_arg(ARGS, 6, 0.08)

out = chain_static_reference(; L=L, u=u, γ=γ, T=T, μ=μ, η=η)
F_fd = fermi_fd.(out.ωs, T, μ)

fig, axs = subplots(1, 3; figsize=(17.2, 4.8))

axs[1].plot(out.ks, out.ϵs; color="black", linewidth=1.7, label=raw"$\epsilon_k$")
axs1_twin = axs[1].twinx()
axs1_twin.plot(out.ks, out.nk_eq; color="#2c7fb8", linewidth=2.0, label=raw"$n_k^{\mathrm{th}}$")
axs[1].set_xlabel(raw"$k$")
axs[1].set_ylabel(raw"$\epsilon_k$")
axs1_twin.set_ylabel(raw"$n_k$")
axs[1].set_xlim(first(out.ks), last(out.ks))
axs[1].grid(alpha=0.18)
lines1, labels1 = axs[1].get_legend_handles_labels()
lines2, labels2 = axs1_twin.get_legend_handles_labels()
axs[1].legend(vcat(lines1, lines2), vcat(labels1, labels2); frameon=false, loc="best")
axs[1].text(0.04, 0.93, "Thermal chain state"; transform=axs[1].transAxes, ha="left", va="top", fontsize=12)

axs[2].plot(out.ωs, out.A_sum; color="black", linewidth=2.0, label=raw"$A_{\mathrm{sum}\,k}(\omega)$")
axs[2].plot(out.ωs, out.Gless_im_sum; color="#2c7fb8", linewidth=1.8, label=raw"$iG^{<}_{\mathrm{sum}\,k}(\omega)$")
axs[2].set_xlabel(raw"$\omega$")
axs[2].set_ylabel(raw"$A,\; iG^<$")
axs[2].set_xlim(first(out.ωs), last(out.ωs))
axs[2].grid(alpha=0.18)
axs[2].legend(frameon=false, loc="best")

axs[3].plot(out.ωs, out.F_num; color="#2c7fb8", linewidth=2.1, label=raw"$F(\omega)$ from $G^R$")
axs[3].plot(out.ωs, F_fd; color="black", linestyle="--", linewidth=1.6, label="Fermi-Dirac")
axs[3].set_xlabel(raw"$\omega$")
axs[3].set_ylabel(raw"$F(\omega)$")
axs[3].set_xlim(first(out.ωs), last(out.ωs))
axs[3].set_ylim(-0.05, 1.05)
axs[3].grid(alpha=0.18)
axs[3].legend(frameon=false, loc="best")

fig.suptitle("Chain static equilibrium reference", fontsize=16, y=1.02)
fig.text(0.5, -0.02, "u=$(u), γ=$(γ), T=$(T), μ=$(μ), η=$(η), L=$(L)"; ha="center", va="top", fontsize=12)
fig.tight_layout()

png_out = joinpath(OUTPUT_DIR, "chain_static_reference_from_retarded_u$(u)_γ$(γ)_T$(T)_μ$(μ)_η$(η).png")
fig.savefig(png_out; dpi=220, bbox_inches="tight")

jld2_out = joinpath(OUTPUT_DIR, "chain_static_reference_from_retarded_u$(u)_γ$(γ)_T$(T)_μ$(μ)_η$(η).jld2")
@save jld2_out out

println("Saved " * png_out)
println("Saved " * jld2_out)
