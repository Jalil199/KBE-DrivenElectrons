using KadanoffBaym, LinearAlgebra, BlockArrays
using StaticArrays
using LaTeXStrings
using FFTW, Interpolations
using Tullio
using JLD2

import LinearAlgebra: adjoint

# Define A† for a 3D tensor A[i,j,k] as conj(A[j,i,k])
adjoint(A::AbstractArray{<:Number,3}) = permutedims(conj.(A), (2,1,3))

# Pauli matrices
const σ_0 = [1.0 0.0; 0.0 1.0]
const σ_x = [0.0 1.0; 1.0 0.0]
const σ_y = [0.0 -1im; 1im 0.0]
const σ_z = [1.0 0.0; 0.0 -1.0]

# ── Bath helper functions ──────────────────────────────────────────────────────

function make_momentum_weights(profile::Symbol; ks, s_q::Float64, λ_q::Float64)
    if profile == :uniform
        wq_raw = ones(Float64, length(ks))
    elseif profile == :power_exp
        wq_raw = abs.(ks) .^ s_q .* exp.(-abs.(ks) ./ λ_q)
    else
        throw(ArgumentError("Unknown momentum weight profile: $profile"))
    end
    wq_sum = sum(wq_raw)
    wq_sum > 0 || throw(ArgumentError("Momentum weights must sum to a positive value before normalization"))
    return wq_raw ./ wq_sum
end

function ωbath_value(q::Real; dispersion_type::Symbol, ωb0::Float64, v_b::Float64)
    if dispersion_type == :linear
        return ωb0 + v_b * abs(q)
    elseif dispersion_type == :sin_lattice
        return ωb0 + 2v_b * abs(sin(q / 2))
    else
        throw(ArgumentError("Unknown dispersion_type: $dispersion_type"))
    end
end

function make_bath_dispersion(dispersion_type::Symbol; qs, ωb0::Float64, v_b::Float64)
    return [ωbath_value(q; dispersion_type, ωb0, v_b) for q in qs]
end

function make_bath_coupling2(; qs, g_b::Float64)
    return fill(g_b^2, length(qs))
end

# ── Model struct ───────────────────────────────────────────────────────────────

Base.@kwdef struct ModelElectronBath
    L::Int = 80

    # Rice-Mele tight-binding parameters
    t1::Float64 = -1.0
    t2::Float64 = -0.6
    Δ::Float64  = 2.0

    # Temperatures
    Te::Float64 = 0.2   # electron temperature
    Tb::Float64 = 0.8   # bath temperature

    # Spectral-density bath parameters J(ω) ~ α|ω|^s exp(-|ω|/ωc)
    α::Float64  = 0.3
    s::Float64  = 1.0
    ωc::Float64 = 3.0

    # Drive envelope: A(t) = A exp[-(t-t0)^2/(2σ^2)] sin(ω0 t)
    t0::Float64 = 20.0
    ω0::Float64 = 2.2
    σ::Float64  = 2.0
    A::Float64  = 0.5

    # Smooth switch-on window
    ti::Float64 = 0.5
    to::Float64 = 5.0

    # Bath implementation choice
    bath_type::Symbol      = :dispersion
    dispersion_type::Symbol = :sin_lattice
    ωb0::Float64 = 0.1
    v_b::Float64 = 0.2
    g_b::Float64 = 0.85

    # Brillouin-zone grid
    Δk::Float64         = 2*pi/L
    ks::Vector{Float64} = collect(range(-pi, stop=pi-Δk, length=L))

    # Bath momentum weights and arrays
    wq_profile::Symbol   = :uniform
    s_q::Float64         = 0.2
    λ_q::Float64         = 0.5
    wq::Vector{Float64}  = make_momentum_weights(wq_profile; ks, s_q, λ_q)
    bath_qs::Vector{Float64} = copy(ks)
    ωq::Vector{Float64}  = make_bath_dispersion(dispersion_type; qs=bath_qs, ωb0, v_b)
    g2q::Vector{Float64} = make_bath_coupling2(; qs=bath_qs, g_b)
    nBq::Vector{Float64} = bose.(ωq; model=(; Tb))
    kmq_idx::Matrix{Int} = [mod1(k - q, L) for k in 1:L, q in 1:L]
end

Base.@kwdef struct DataElectronBath{T1,T2}
    GL::T1
    GG::T1
    ΣL_F::T1
    ΣG_F::T1
    workspace::T2
end

# ── Physics functions ──────────────────────────────────────────────────────────

function fermi(ϵ::Float64, model::ModelElectronBath)
    (; Te) = model
    β = 1/Te
    1/(exp(β*ϵ)+1)
end

function fermi(ϵ::Float64, T::Float64)
    β = 1/T
    1/(exp(β*ϵ)+1)
end

function bose(ϵ::Float64; model)
    (; Tb) = model
    β = 1/Tb
    return 1/(exp(β*ϵ)-1)
end

function H_k(k::Float64; t1::Float64, t2::Float64, Δ::Float64)
    dx = t1 + t2*cos(k + pi)
    dy = t2*sin(k + pi)
    dz = Δ/2
    return σ_x*dx + σ_y*dy + σ_z*dz
end

function E_k(k::Float64, t1::Float64, t2::Float64, Δ::Float64)
    dx = t1 + t2*cos(k)
    dy = t2*sin(k)
    dz = Δ/2
    return sqrt(dx^2 + dy^2 + dz^2)
end

function pulse_Gaussian_sin(t::Float64; t0::Float64, ω0::Float64, σ::Float64, A::Float64)
    A * exp(-0.5 * (t - t0)^2 / σ^2) * sin(t * ω0)
end

function g0l_kt(k::Float64, t::Float64; t1::Float64, t2::Float64, Δ::Float64, Te::Float64)
    vals, U = eigen(H_k(k; t1=t1, t2=t2, Δ=Δ))
    return 1im * (U * Diagonal(fermi.(vals, Te) .* exp.(-1im*vals*t)) * U')
end

function g0g_kt(k::Float64, t::Float64; t1::Float64, t2::Float64, Δ::Float64, Te::Float64)
    vals, U = eigen(H_k(k; t1=t1, t2=t2, Δ=Δ))
    return 1im * U * (Diagonal((fermi.(vals, Te) .- ones(2)) .* exp.(-1im*vals*t)) * U')
end

# ── Bath kernels ───────────────────────────────────────────────────────────────

function J(ω::Float64; model)
    (; α, s, ωc) = model
    sign = ω ≥ 0 ? 1 : -1
    return sign * α * ωc^(1-s) * (abs(ω))^s * exp(-sign*ω/ωc)
end

function Ξl(t::Float64; model::ModelElectronBath)
    dω = 0.01
    ωmax = 100.0
    ωs = vcat(-ωmax:dω:-dω, dω:dω:ωmax)
    return -1im/(2pi) * sum(J.(ωs; model) .* bose.(ωs; model) .* exp.(-1im .* ωs .* t)) * dω
end

function Ξg(t::Float64; model::ModelElectronBath)
    dω = 0.01
    ωmax = 100.0
    ωs = vcat(-ωmax:dω:-dω, dω:dω:ωmax)
    return -1im/(2pi) * sum(J.(ωs; model) .* (bose.(ωs; model) .+ 1) .* exp.(-1im .* ωs .* t)) * dω
end

function stepp(t; model)
    (; ti, to) = model
    1.0 #/(1+exp(-(t-to)/ti))
end

# ── Validation ─────────────────────────────────────────────────────────────────

function validate_bath_config(model::ModelElectronBath)
    L = model.L
    @assert model.bath_type in (:spectral_density, :dispersion) "bath_type must be :spectral_density or :dispersion"
    @assert model.dispersion_type in (:linear, :sin_lattice) "dispersion_type must be :linear or :sin_lattice"
    @assert length(model.ks) == L "ks must have length L"
    @assert length(model.wq) == L "wq must have length L"
    @assert isapprox(sum(model.wq), 1.0; atol=1e-12) "wq must satisfy sum(wq) = 1"
    @assert length(model.ωq) == L "ωq must have length L"
    @assert length(model.g2q) == L "g2q must have length L"
    @assert length(model.nBq) == L "nBq must have length L"
    @assert size(model.kmq_idx) == (L, L) "kmq_idx must be an L×L lookup table"
end

function validate_workspace!(workspace, Gtt_template, model::ModelElectronBath)
    L = model.L
    @assert length(workspace.tmpΞL) == L "tmpΞL must have length L"
    @assert length(workspace.tmpΞG) == L "tmpΞG must have length L"
    @assert size(workspace.tmpΣL) == size(Gtt_template) "tmpΣL must match the Green-function block size"
    @assert size(workspace.tmpΣG) == size(Gtt_template) "tmpΣG must match the Green-function block size"
end

# ── Kernel and self-energy ─────────────────────────────────────────────────────

@inline kbe_storage_tt_rm(gf, t, t′) = @view gf.data[:, :, :, t, t′]

function fill_dispersion_kernel_q!(Ξq_tt, τ, ωq, g2q, nBq; greater::Bool)
    @inbounds for q in eachindex(Ξq_tt)
        nq = nBq[q]
        pref    = greater ? (nq + 1) : nq
        pref_tr = greater ? nq : (nq + 1)
        Ξq_tt[q] = -1im * g2q[q] * (pref * exp(-1im*ωq[q]*τ) + pref_tr * exp(1im*ωq[q]*τ))
    end
    return Ξq_tt
end

function apply_momentum_convolution!(Σtt, Ξq_tt, Gtt, kmq_idx)
    fill!(Σtt, 0)
    nd = ndims(Σtt)
    L  = size(Σtt, nd)
    @inbounds for k in 1:L
        Σk = selectdim(Σtt, nd, k)
        for q in 1:L
            Gkmq = selectdim(Gtt, nd, kmq_idx[k, q])
            Σk .+= Ξq_tt[q] .* Gkmq
        end
        Σk .*= 1im / L
    end
    return Σtt
end

function SelfEnergyUpdate!(model::ModelElectronBath, data::DataElectronBath,
                            times::Vector{Float64}, _, _, t::Int, t′::Int)
    (; GL, GG, ΣL_F, ΣG_F, workspace) = data
    (; bath_type, wq, kmq_idx, ωq, g2q, nBq) = model
    (; tmpΞL, tmpΞG, tmpΣL, tmpΣG) = workspace

    if (n = size(GL, 4)) > size(ΣL_F, 4)
        resize!(ΣL_F, n)
        resize!(ΣG_F, n)
    end

    τ = times[t] - times[t′]

    GL_tt = kbe_storage_tt_rm(GL, t, t′)
    GG_tt = kbe_storage_tt_rm(GG, t, t′)
    ΣL_tt = kbe_storage_tt_rm(ΣL_F, t, t′)
    ΣG_tt = kbe_storage_tt_rm(ΣG_F, t, t′)

    if bath_type == :spectral_density
        ΞL_ref = Ξl(τ; model)
        ΞG_ref = Ξg(τ; model)
        @inbounds for q in eachindex(wq)
            tmpΞL[q] = wq[q] * ΞL_ref
            tmpΞG[q] = wq[q] * ΞG_ref
        end
    elseif bath_type == :dispersion
        fill_dispersion_kernel_q!(tmpΞL, τ, ωq, g2q, nBq; greater=false)
        fill_dispersion_kernel_q!(tmpΞG, τ, ωq, g2q, nBq; greater=true)
    else
        throw(ArgumentError("Unknown bath_type: $(bath_type)"))
    end

    apply_momentum_convolution!(tmpΣL, tmpΞL, GL_tt, kmq_idx)
    apply_momentum_convolution!(tmpΣG, tmpΞG, GG_tt, kmq_idx)
    copyto!(ΣL_tt, tmpΣL)
    copyto!(ΣG_tt, tmpΣG)
    return nothing
end

# ── Time integrators ───────────────────────────────────────────────────────────

function integrate1(hs::Vector, t::Int, t′::Int, A::GreenFunction, B::GreenFunction, C::GreenFunction)
    ret = zero(A[t, t′])
    @inbounds for ik in 1:size(ret, 3)
        for ℓ in eachindex(hs)
            @views ret[:, :, ik] .+= hs[ℓ] .* ((A[:, :, ik, t, ℓ] .- B[:, :, ik, t, ℓ]) * C[:, :, ik, ℓ, t′])
        end
    end
    return ret
end

function integrate2(hs::Vector, t::Int, t′::Int, A::GreenFunction, B::GreenFunction, C::GreenFunction)
    ret = zero(A[t, t′])
    @inbounds for ik in 1:size(ret, 3)
        for ℓ in eachindex(hs)
            @views ret[:, :, ik] .+= hs[ℓ] .* (A[:, :, ik, t, ℓ] * (B[:, :, ik, ℓ, t′] .- C[:, :, ik, ℓ, t′]))
        end
    end
    return ret
end

# ── Solver callbacks ───────────────────────────────────────────────────────────

function fv!(model::ModelElectronBath, data::DataElectronBath,
             out::AbstractVector{<:AbstractArray{ComplexF64,3}}, times::Vector{Float64},
             h1::Vector, h2::Vector, t::Int, t′::Int)
    (; GL, GG, ΣL_F, ΣG_F) = data
    (; ks, t1, t2, Δ) = model
    ∫dt1(A, B, C) = integrate1(h1, t, t′, A, B, C)
    ∫dt2(A, B, C) = integrate2(h2, t, t′, A, B, C)

    outL = copy(out[1])
    outG = copy(out[2])

    pulse(t) = pulse_Gaussian_sin(t; model.t0, model.ω0, model.σ, model.A)
    @inbounds for (ik, k) in enumerate(ks)
        H = H_k(k - pulse(times[t]); t1=t1, t2=t2, Δ=Δ)
        outL[:, :, ik] .= H * GL[:, :, ik, t, t′]
        outG[:, :, ik] .= H * GG[:, :, ik, t, t′]
    end
    outL .+= ∫dt1(ΣG_F, ΣL_F, GL) + ∫dt2(ΣL_F, GL, GG)
    outG .+= ∫dt1(ΣG_F, ΣL_F, GG) + ∫dt2(ΣG_F, GL, GG)
    outL .*= -1im
    outG .*= -1im

    out[1] = outL
    out[2] = outG
    return nothing
end

function fd!(model::ModelElectronBath, data::DataElectronBath,
             out::AbstractVector{<:AbstractArray{ComplexF64,3}}, times::Vector{Float64},
             h1::Vector, h2::Vector, t::Int, t′::Int)
    fv!(model, data, out, times, h1, h2, t, t)
    @inbounds for ik in 1:size(out[1], 3)
        @views out[1][:, :, ik] .-= adjoint(out[1][:, :, ik])
        @views out[2][:, :, ik] .-= adjoint(out[2][:, :, ik])
    end
    return nothing
end

# ── Filename ───────────────────────────────────────────────────────────────────

function make_name(model::ModelElectronBath; tmax)
    "L$(model.L)_t1$(model.t1)_t2$(model.t2)_Δ$(model.Δ)" *
    "_Te$(model.Te)_Tb$(model.Tb)" *
    "_$(model.bath_type)_α$(model.α)_s$(model.s)_ωc$(model.ωc)" *
    "_$(model.dispersion_type)_g_b$(model.g_b)_v_b$(model.v_b)_ωb0$(model.ωb0)" *
    "_$(model.wq_profile)_t0$(model.t0)_ω0$(model.ω0)_σ$(model.σ)_A$(model.A)" *
    "_ti$(model.ti)_to$(model.to)_tmax$(tmax)"
end

# ── Main simulation ────────────────────────────────────────────────────────────

function main(; tmax=40, kwargs...)
    println(kwargs...)
    model = ModelElectronBath(; kwargs...)

    L  = model.L
    t1 = model.t1
    t2 = model.t2
    Δ  = model.Δ
    ks = model.ks

    validate_bath_config(model)

    #### Two-time objects
    GL   = GreenFunction(zeros(ComplexF64, 2, 2, L, 1, 1), SkewHermitian)
    GG   = GreenFunction(zeros(ComplexF64, 2, 2, L, 1, 1), SkewHermitian)
    ΣL_F = GreenFunction(zeros(ComplexF64, 2, 2, L, 1, 1), SkewHermitian)
    ΣG_F = GreenFunction(zeros(ComplexF64, 2, 2, L, 1, 1), SkewHermitian)

    norb1, norb2 = size(GL.data, 1), size(GL.data, 2)
    workspace = (
        tmpΞL = zeros(ComplexF64, L),
        tmpΞG = zeros(ComplexF64, L),
        tmpΣL = zeros(ComplexF64, norb1, norb2, L),
        tmpΣG = zeros(ComplexF64, norb1, norb2, L),
    )
    validate_workspace!(workspace, GL[1,1], model)

    #### Initial conditions
    I2 = Matrix{ComplexF64}(I, 2, 2)
    for (ik, k) in enumerate(ks)
        @views GL.data[:, :, ik, 1, 1] .= g0l_kt(k, 0.0; t1, t2, Δ, model.Te)
        @views GG.data[:, :, ik, 1, 1] .= -1im*I2 .+ GL.data[:, :, ik, 1, 1]
    end

    # Σ(0,0) = 0 (adiabatic switch off at t=0)
    if model.bath_type == :spectral_density
        @inbounds for q in eachindex(model.wq)
            workspace.tmpΞL[q] = 0.0
            workspace.tmpΞG[q] = 0.0
        end
    else
        fill_dispersion_kernel_q!(workspace.tmpΞL, 0.0, model.ωq, model.g2q, model.nBq; greater=false)
        fill_dispersion_kernel_q!(workspace.tmpΞG, 0.0, model.ωq, model.g2q, model.nBq; greater=true)
        workspace.tmpΞL .*= 0.0
        workspace.tmpΞG .*= 0.0
    end
    apply_momentum_convolution!(workspace.tmpΣL, workspace.tmpΞL, GL[1,1], model.kmq_idx)
    apply_momentum_convolution!(workspace.tmpΣG, workspace.tmpΞG, GG[1,1], model.kmq_idx)
    copyto!(ΣL_F[1,1], workspace.tmpΣL)
    copyto!(ΣG_F[1,1], workspace.tmpΣG)

    data = DataElectronBath(; GL, GG, ΣL_F, ΣG_F, workspace)

    #### Time propagation
    atol = 1e-5
    rtol = 1e-4

    sol = @time kbsolve!(
        (x...) -> fv!(model, data, x...),
        (x...) -> fd!(model, data, x...),
        [data.GL, data.GG],
        (0.0, tmax);
        callback = (x...) -> SelfEnergyUpdate!(model, data, x...),
        atol = atol,
        rtol = rtol,
        stop = x -> (println("t: $(x[end])"); flush(stdout); false)
    )

    #### Save results
    file   = "Data"
    name_p = make_name(model; tmax)
    mkpath(file)

    @save "$(file)/GL_$(name_p).jld2" GL
    @save "$(file)/GG_$(name_p).jld2" GG
    @save "$(file)/ts_$(name_p).jld2" sol

    println("Saved all results.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
