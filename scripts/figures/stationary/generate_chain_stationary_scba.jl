using JLD2
using PyPlot

include(joinpath(@__DIR__, "main.jl"))

rc("font", family="serif")
rc("font", size=13)
rc("axes", labelsize=15, titlesize=14)
rc("xtick", labelsize=11)
rc("ytick", labelsize=11)
rc("legend", fontsize=10)

const OUTPUT_DIR = "distributions"
mkpath(OUTPUT_DIR)

fermi_fd(ω, T, μ) = 1 / (exp((ω - μ) / T) + 1)

function parse_arg(args, i, default)
    return length(args) >= i ? parse(typeof(default), args[i]) : default
end

function build_bath_kernels(ωs, model)
    L = model.L
    Nω = length(ωs)
    ΞLω = zeros(ComplexF64, L, Nω)
    ΞGω = zeros(ComplexF64, L, Nω)
    for q in 1:L
        for (iω, ω) in enumerate(ωs)
            Aω = boson_spectral_A(ω, model.ωq[q], model.η)
            ΞLω[q, iω] = (-1im) * bose(ω; model) * Aω * model.g2q[q]
            ΞGω[q, iω] = (-1im) * (bose(ω; model) + 1) * Aω * model.g2q[q]
        end
    end
    return ΞLω, ΞGω
end

function update_sigma_lesser_greater!(ΣL, ΣG, GL, GG, ΞLω, ΞGω, kmq_idx, dω)
    L, Nω = size(GL)
    mid = Nω ÷ 2 + 1
    fill!(ΣL, 0)
    fill!(ΣG, 0)

    @inbounds for k in 1:L
        for q in 1:L
            kq = kmq_idx[k, q]
            for iω in 1:Nω
                accL = zero(ComplexF64)
                accG = zero(ComplexF64)
                for jω in 1:Nω
                    lω = iω - jω + mid
                    if 1 <= lω <= Nω
                        accL += ΞLω[q, jω] * GL[kq, lω]
                        accG += ΞGω[q, jω] * GG[kq, lω]
                    end
                end
                ΣL[k, iω] += 1im * accL * dω / (2π * L)
                ΣG[k, iω] += 1im * accG * dω / (2π * L)
            end
        end
    end
    return ΣL, ΣG
end

function kk_real_part(Aspec, ωs, dω)
    Nω = length(ωs)
    ReΣ = zeros(Float64, Nω)
    @inbounds for i in 1:Nω
        acc = 0.0
        ωi = ωs[i]
        for j in 1:Nω
            j == i && continue
            acc += Aspec[j] / (ωi - ωs[j])
        end
        ReΣ[i] = acc * dω / (2π)
    end
    return ReΣ
end

function stationary_scba_chain(; L=60, u=0.0, γ=1.0, α=2.5, Tb=0.1, η=0.05, λ_q=0.5, s_q=1.0,
                               ωb0=0.1, v_b=0.2, ωmax=4.0, Nω=301, Tinit=1.0, μ=0.0,
                               maxiter=20, mix=0.5, tol=1e-5, δ_reg=0.01)
    isodd(Nω) || throw(ArgumentError("Nω must be odd so the ω-grid is centered at zero"))
    model = ModelElectronBath(; L=L, u=u, γ=γ, α=α, Tb=Tb, η=η, λ_q=λ_q, s_q=s_q,
                              ωb0=ωb0, v_b=v_b, bath_type=:dispersion,
                              boson_kernel=:spectral, wq_profile=:power_exp)
    ks = model.ks
    ϵs = ϵ_k(ks; u, γ)
    ωs = collect(range(-ωmax, ωmax, length=Nω))
    dω = ωs[2] - ωs[1]

    ΞLω, ΞGω = build_bath_kernels(ωs, model)

    GR = zeros(ComplexF64, L, Nω)
    GL = zeros(ComplexF64, L, Nω)
    GG = zeros(ComplexF64, L, Nω)
    ΣL = zeros(ComplexF64, L, Nω)
    ΣG = zeros(ComplexF64, L, Nω)
    ΣR = zeros(ComplexF64, L, Nω)

    for k in 1:L, iω in 1:Nω
        GR[k, iω] = inv(complex(ωs[iω], δ_reg) - ϵs[k])
        Akw = -2 * imag(GR[k, iω])
        fω = fermi_fd(ωs[iω], Tinit, μ)
        GL[k, iω] = 1im * fω * Akw
        GG[k, iω] = -1im * (1 - fω) * Akw
    end

    errors = Float64[]
    for iter in 1:maxiter
        oldGL = copy(GL)
        oldGG = copy(GG)

        update_sigma_lesser_greater!(ΣL, ΣG, GL, GG, ΞLω, ΞGω, model.kmq_idx, dω)

        for k in 1:L
            AΣ = 1im .* (ΣG[k, :] .- ΣL[k, :])
            ReΣ = kk_real_part(real.(AΣ), ωs, dω)
            ImΣ = -0.5 .* real.(AΣ)
            ΣR[k, :] .= ReΣ .+ 1im .* ImΣ
        end

        for k in 1:L, iω in 1:Nω
            GR_new = inv(complex(ωs[iω], δ_reg) - ϵs[k] - ΣR[k, iω])
            GL_new = abs2(GR_new) * ΣL[k, iω]
            GG_new = abs2(GR_new) * ΣG[k, iω]
            GR[k, iω] = (1 - mix) * GR[k, iω] + mix * GR_new
            GL[k, iω] = (1 - mix) * GL[k, iω] + mix * GL_new
            GG[k, iω] = (1 - mix) * GG[k, iω] + mix * GG_new
        end

        err = max(maximum(abs.(GL .- oldGL)), maximum(abs.(GG .- oldGG)))
        push!(errors, err)
        println("iter=$(iter) err=$(err)")
        err < tol && break
    end

    Akw = real.(1im .* (GG .- GL))
    A_sum = vec(sum(Akw; dims=1)) ./ L
    minus_iGless_sum = vec(real.((-1im) .* sum(GL; dims=1))) ./ L
    F_num = fill(NaN, Nω)
    mask = abs.(A_sum) .> 1e-10
    F_num[mask] .= minus_iGless_sum[mask] ./ A_sum[mask]
    nk = vec(sum(real.((-1im) .* GL); dims=2)) .* dω ./ (2π)

    return (; model, ks, ϵs, ωs, GR, GL, GG, ΣR, ΣL, ΣG, Akw, A_sum, minus_iGless_sum, F_num, nk, errors)
end

L = parse_arg(ARGS, 1, 60)
α = parse_arg(ARGS, 2, 2.5)
Tb = parse_arg(ARGS, 3, 0.1)
η = parse_arg(ARGS, 4, 0.05)
λ_q = parse_arg(ARGS, 5, 0.5)
ωmax = parse_arg(ARGS, 6, 4.0)
Nω = parse_arg(ARGS, 7, 301)
maxiter = parse_arg(ARGS, 8, 20)
δ_reg = parse_arg(ARGS, 9, 0.01)

out = stationary_scba_chain(; L=L, α=α, Tb=Tb, η=η, λ_q=λ_q, ωmax=ωmax, Nω=Nω, maxiter=maxiter, δ_reg=δ_reg)
F_fd_bath = fermi_fd.(out.ωs, Tb, 0.0)

fig, axs = subplots(1, 3; figsize=(17.2, 4.8))

axs[1].plot(out.ks, out.ϵs; color="black", linewidth=1.4, alpha=0.45, label=raw"$\epsilon_k$")
axs1_twin = axs[1].twinx()
axs1_twin.plot(out.ks, out.nk; color="#2c7fb8", linewidth=2.1, label=raw"$n_k^{\mathrm{SCBA}}$")
axs[1].set_xlabel(raw"$k$")
axs[1].set_ylabel(raw"$\epsilon_k$", color="#666666")
axs1_twin.set_ylabel(raw"$n_k$")
axs[1].tick_params(axis="y", colors="#666666")
axs[1].set_xlim(first(out.ks), last(out.ks))
axs[1].grid(alpha=0.18)
lines1, labels1 = axs[1].get_legend_handles_labels()
lines2, labels2 = axs1_twin.get_legend_handles_labels()
axs[1].legend(vcat(lines1, lines2), vcat(labels1, labels2); frameon=false, loc="best")

axs[2].plot(out.ωs, out.A_sum; color="black", linewidth=2.0, label=raw"$A_{\mathrm{sum}\,k}(\omega)$")
axs[2].plot(out.ωs, out.minus_iGless_sum; color="#2c7fb8", linewidth=1.8, label=raw"$-iG^{<}_{\mathrm{sum}\,k}(\omega)$")
axs[2].set_xlabel(raw"$\omega$")
axs[2].set_ylabel(raw"$A,\; -iG^<$")
axs[2].set_xlim(first(out.ωs), last(out.ωs))
axs[2].grid(alpha=0.18)
axs[2].legend(frameon=false, loc="best")

axs[3].plot(out.ωs, out.F_num; color="#2c7fb8", linewidth=2.1, label=raw"$F(\omega)$")
axs[3].plot(out.ωs, F_fd_bath; color="black", linestyle="--", linewidth=1.6, label=raw"$f_{\mathrm{FD}}(\omega;T_b)$")
axs[3].set_xlabel(raw"$\omega$")
axs[3].set_ylabel(raw"$F(\omega)$")
axs[3].set_xlim(first(out.ωs), last(out.ωs))
axs[3].set_ylim(-0.05, 1.05)
axs[3].grid(alpha=0.18)
axs[3].legend(frameon=false, loc="best")

fig.suptitle("Chain stationary SCBA candidate", fontsize=16, y=1.02)
fig.text(0.5, -0.02, "L=$(L), α=$(α), T_b=$(Tb), η=$(η), p_c=$(λ_q), Nω=$(Nω), δ=$(δ_reg), iters=$(length(out.errors))", ha="center", va="top", fontsize=12)
fig.tight_layout()

tag = "L$(L)_α$(α)_Tb$(Tb)_η$(η)_pc$(λ_q)_Nω$(Nω)_δ$(δ_reg)"
png_out = joinpath(OUTPUT_DIR, "chain_stationary_scba_" * tag * ".png")
jld2_out = joinpath(OUTPUT_DIR, "chain_stationary_scba_" * tag * ".jld2")
fig.savefig(png_out; dpi=220, bbox_inches="tight")
out_save = (; ks=out.ks, ϵs=out.ϵs, ωs=out.ωs, GR=out.GR, GL=out.GL, GG=out.GG,
             ΣR=out.ΣR, ΣL=out.ΣL, ΣG=out.ΣG, Akw=out.Akw, A_sum=out.A_sum,
             minus_iGless_sum=out.minus_iGless_sum, F_num=out.F_num, nk=out.nk, errors=out.errors)
@save jld2_out out_save
println("Saved " * png_out)
println("Saved " * jld2_out)
