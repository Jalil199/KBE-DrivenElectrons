using JLD2
using LinearAlgebra
using PyPlot

include(joinpath(@__DIR__, "main-rice_mele.jl"))

rc("font", family="serif")
rc("font", size=14)
rc("axes", labelsize=16, titlesize=15)
rc("xtick", labelsize=12)
rc("ytick", labelsize=12)
rc("legend", fontsize=11)

const OUTPUT_DIR = "distributions"
mkpath(OUTPUT_DIR)

fermi_fd(omega, T, mu) = 1 / (exp((omega - mu) / T) + 1)

function parse_arg(args, i, default)
    return length(args) >= i ? parse(typeof(default), args[i]) : default
end

function static_rice_reference(; L=80, t1=-1.0, t2=-0.8, Δ=2.0, T=1.0, μ=0.0, η=0.08, ωmin=-4.0, ωmax=4.0, Nω=1200)
    ks = collect(range(-π, stop=π - 2π / L, length=L))
    ωs = collect(range(ωmin, ωmax, length=Nω))

    Eminus = zeros(Float64, L)
    Eplus = zeros(Float64, L)
    nminus = zeros(Float64, L)
    nplus = zeros(Float64, L)

    A_sum = zeros(Float64, Nω)
    Gless_im_sum = zeros(Float64, Nω)

    for (ik, k) in enumerate(ks)
        evals = sort(real.(eigvals(H_k(k; t1=t1, t2=t2, Δ=Δ))))
        Eminus[ik], Eplus[ik] = evals
        nminus[ik] = fermi_fd(Eminus[ik], T, μ)
        nplus[ik] = fermi_fd(Eplus[ik], T, μ)
    end

    for (iw, ω) in enumerate(ωs)
        fω = fermi_fd(ω, T, μ)
        accA = 0.0
        for k in ks
            Gr = inv((ω + 1im * η) * I(2) - H_k(k; t1=t1, t2=t2, Δ=Δ))
            A_mat = -2 .* imag.(Gr)
            accA += real(tr(A_mat))
        end
        A_sum[iw] = accA / L
        Gless_im_sum[iw] = fω * A_sum[iw]
    end

    F_num = fill(NaN, Nω)
    mask = abs.(A_sum) .> 1e-10
    F_num[mask] .= Gless_im_sum[mask] ./ A_sum[mask]

    return (; ks, Eminus, Eplus, nminus, nplus, ωs, A_sum, Gless_im_sum, F_num)
end

L = parse_arg(ARGS, 1, 80)
t1 = parse_arg(ARGS, 2, -1.0)
t2 = parse_arg(ARGS, 3, -0.8)
Δ = parse_arg(ARGS, 4, 2.0)
T = parse_arg(ARGS, 5, 1.0)
μ = parse_arg(ARGS, 6, 0.0)
η = parse_arg(ARGS, 7, 0.08)

out = static_rice_reference(; L=L, t1=t1, t2=t2, Δ=Δ, T=T, μ=μ, η=η)
F_fd = fermi_fd.(out.ωs, T, μ)

fig, axs = subplots(1, 3; figsize=(17.2, 4.8))

axs[1].plot(out.ks, out.nminus; color="#2c7fb8", linewidth=2.2, label=raw"$n_-^{\mathrm{th}}(k)$")
axs[1].plot(out.ks, out.nplus; color="#d95f0e", linewidth=2.0, linestyle="--", label=raw"$n_+^{\mathrm{th}}(k)$")
axs[1].set_xlabel(raw"$k$")
axs[1].set_ylabel(raw"$n_\pm(k)$")
axs[1].set_ylim(-0.05, 1.05)
axs1_twin = axs[1].twinx()
axs1_twin.plot(out.ks, out.Eminus; color="black", linewidth=1.1, alpha=0.35, label=raw"$E_-(k)$")
axs1_twin.plot(out.ks, out.Eplus; color="#4d4d4d", linewidth=1.1, linestyle="--", alpha=0.35, label=raw"$E_+(k)$")
axs1_twin.set_ylabel(raw"$E_\pm(k)$", color="#666666")
axs1_twin.tick_params(axis="y", colors="#666666")
axs[1].set_xlim(first(out.ks), last(out.ks))
axs[1].grid(alpha=0.18)
lines1, labels1 = axs[1].get_legend_handles_labels()
lines2, labels2 = axs1_twin.get_legend_handles_labels()
axs[1].legend(vcat(lines1, lines2), vcat(labels1, labels2); frameon=false, loc="best")
axs[1].text(0.04, 0.93, "Thermal Rice-Mele state"; transform=axs[1].transAxes, ha="left", va="top", fontsize=12)

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

fig.suptitle("Rice-Mele static equilibrium reference", fontsize=16, y=1.02)
fig.text(0.5, -0.02, "t1=$(t1), t2=$(t2), Δ=$(Δ), T=$(T), μ=$(μ), η=$(η), L=$(L)"; ha="center", va="top", fontsize=12)
fig.tight_layout()

png_out = joinpath(OUTPUT_DIR, "rice_static_reference_from_retarded_t1$(t1)_t2$(t2)_Δ$(Δ)_T$(T)_μ$(μ)_η$(η).png")
fig.savefig(png_out; dpi=220, bbox_inches="tight")

jld2_out = joinpath(OUTPUT_DIR, "rice_static_reference_from_retarded_t1$(t1)_t2$(t2)_Δ$(Δ)_T$(T)_μ$(μ)_η$(η).jld2")
@save jld2_out out

println("Saved " * png_out)
println("Saved " * jld2_out)
