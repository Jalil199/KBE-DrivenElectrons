using KadanoffBaym, LinearAlgebra, BlockArrays
using LaTeXStrings
using FFTW, Interpolations
using Tullio
using JLD2
using PyPlot

function make_momentum_weights(profile::Symbol; ks, s_q::Float64, λ_q::Float64, η::Float64)
    if profile == :uniform
        wq_raw = ones(Float64, length(ks))
    elseif profile == :power_exp
        wq_raw = abs.(ks) .^ s_q .* exp.(-abs.(ks) ./ λ_q)
    else
        throw(ArgumentError("Unknown momentum weight profile: $profile"))
    end

    # Previous normalization:
    # wq_sum = sum(wq_raw)
    # wq_sum > 0 || throw(ArgumentError("Momentum weights must sum to a positive value before normalization"))
    # return (η / λ_q) .* (wq_raw ./ wq_sum)
    λ_q > 0 || throw(ArgumentError("Momentum cutoff λ_q must be positive"))
    η ≥ 0 || throw(ArgumentError("Time cutoff η must be nonnegative"))
    return (η / λ_q) .* wq_raw
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

function make_bath_coupling2(; wq, α::Float64)
    return α .* wq
end

Base.@kwdef struct ModelElectronBath{Hk}
    L::Int = 100
    Te::Float64 = 0.1
    Tb::Float64 = 0.1
    u::Float64 = 0.0
    γ::Float64 = 1.0
    α::Float64 = 0.0001
    s::Float64 = 1.0
    ωc::Float64 = 10.0
    t0::Float64 = 50.0
    ω0::Float64 = Float64(pi)
    σ::Float64 = 2.0
    A::Float64 = 0.1
    switch_on::Bool = false
    ti::Float64 = 3.0
    to::Float64 = 20.0
    bath_type::Symbol = :spectral_density
    dispersion_type::Symbol = :linear
    boson_kernel::Symbol = :delta
    η::Float64 = 0.0
    ωA_max::Float64 = 20.0
    dωA::Float64 = 0.01
    ωb0::Float64 = 0.2
    v_b::Float64 = 1.0
    Δk = 2*pi/L
    ks = collect(range(-pi, stop=pi-Δk, length=L))
    wq_profile::Symbol = :uniform
    s_q::Float64 = 0.0
    λ_q::Float64 = 1.0
    wq::Vector{Float64} = make_momentum_weights(wq_profile; ks, s_q, λ_q, η)
    bath_qs::Vector{Float64} = copy(ks)
    ωq::Vector{Float64} = make_bath_dispersion(dispersion_type; qs=bath_qs, ωb0, v_b)
    g2q::Vector{Float64} = make_bath_coupling2(; wq, α)
    nBq::Vector{Float64} = bose.(ωq; model=(; Tb))
    # The k-grid is stored on [-π, π), so index arithmetic must include the
    # half-Brillouin-zone offset when mapping k-q back onto the same grid.
    kmq_idx::Matrix{Int} = [mod1(k - q + L ÷ 2 + 1, L) for k in 1:L, q in 1:L]
    hk::Hk = t -> ϵ_k(ks .- pulse_Gaussian_sin(t; t0, ω0, σ, A);  u, γ)
end

Base.@kwdef struct DataElectronBath{T}
    GL::T
    GG::T
    
    ΞL::T
    ΞG::T
    
    ΣL_F::T
    ΣG_F::T
    workspace::NamedTuple
end

function fermi(ϵ; model)
    (; Te) = model
    β = 1/Te
    1/(exp(β*ϵ)+1)
end

function bose(ϵ; model)
    (; Tb) = model
    β = 1/Tb
    if abs(ϵ)<1e-5
        return 0.0
    else
        return 1/(exp(β*ϵ)-1)
    end
end

function ϵ_k(k; u, γ)
    #(; u, γ) = model
    u .- 2 * γ * cos.(k)
end

# Pulse
function pulse_Gaussian_sin(t; t0, ω0, σ, A)
    #(; t0, ω0, σ, A) = model
    A * exp(-0.5 * (t-t0)^2 / σ^2) * sin(t * ω0)
end

function g0l(t; model)
    (; L) = model
    Δk = 2*pi/L
    ks = collect(range(-pi, stop=pi-Δk, length=L))
    ϵ = ϵ_k(ks .- pulse_Gaussian_sin(t; model); model)
    1im * fermi.(ϵ; model) .* exp.(-1im * ϵ * t)
end

function g0g(t; model)
    (; L) = model
    Δk = 2*pi/L
    ks = collect(range(-pi, stop=pi-Δk, length=L))
    ϵ = ϵ_k(ks .- pulse_Gaussian_sin(t; model); model)
    1im * (fermi.(ϵ; model) .- 1) .* exp.(-1im * ϵ * t)
end

function J(ω; model)
    (; α, s, ωc) = model
    sign = ω ≥ 0 ? 1 : -1
    return sign * α * ωc^(1-s) * (abs(ω))^s * exp(-sign*ω/ωc)
end

# function Ξl(t)
#     f(ω) = J(ω) * bose(ω) * exp(-1im * ω * t)
#     I, err = quadgk(f, 0.0, 200.0; atol=1e-11, rtol=1e-11)
#     return -1im/(2π) * I * ones(L)
# end

# function Ξg(t)
#     f(ω) = J.(ω) .* (bose(ω) + 1) .* exp.(-1im * ω * t)
#     I, err = quadgk(f, 0.0, 200.0; atol=1e-11, rtol=1e-11)
#     return -1im/(2π) * I * ones(L)
# end

function Ξl(t; model)
    dω = 0.01
    ωs = -100:dω:100
    -1im / (2pi) * sum(J.(ωs; model) .* bose.(ωs; model) .* exp.(-1im * ωs * t)) * dω
end

function Ξg(t; model)
    dω = 0.01
    ωs = -100:dω:100
    -1im / (2pi) * sum(J.(ωs; model) .* (bose.(ωs; model) .+ 1) .* exp.(-1im * ωs * t)) * dω
end

function stepp(t; model)
    (; ti, to) = model
    1/(1+exp(-(t-to)/ti))
end

@inline function interaction_switch(t, t′; model, apply_switch::Union{Nothing,Bool}=nothing)
    use_switch = isnothing(apply_switch) ? model.switch_on : apply_switch
    return use_switch ? stepp(t; model) * stepp(t′; model) : 1.0
end

function homogeneous_momentum_sum(Gtt)
    sumG = similar(Gtt)
    sumG .= sum(Gtt)
    return sumG
end


function fill_dispersion_kernel_q!(Ξq_tt, τ, ωq, g2q, nBq; η::Real=0.0, greater::Bool)
    damp = exp(-η * abs(τ))
    @inbounds for q in eachindex(Ξq_tt)
        nq = nBq[q]
        pref = greater ? (nq + 1) : nq
        pref_tr = greater ? nq : (nq + 1)
        Ξq_tt[q] = -1im * g2q[q] * damp * (pref * exp(-1im * ωq[q] * τ) + pref_tr * exp(1im * ωq[q] * τ))
    end
    return Ξq_tt
end

function boson_spectral_A(ω, ω0, η)
    return 2η / ((ω - ω0)^2 + η^2) - 2η / ((ω + ω0)^2 + η^2)
end

function fill_dispersion_kernel_q_spectral!(Ξq_tt, τ, ωgrid, dω, ωq, g2q; model, greater::Bool)
    @inbounds for q in eachindex(Ξq_tt)
        acc = zero(eltype(Ξq_tt))
        for ω in ωgrid
            iszero(ω) && continue
            Aω = boson_spectral_A(ω, ωq[q], model.η)
            occ = greater ? (bose(ω; model) + 1) : bose(ω; model)
            acc += (-1im) * occ * Aω * exp(-1im * ω * τ)
        end
        Ξq_tt[q] = g2q[q] * acc * dω / (2pi)
    end
    return Ξq_tt
end

function fill_homogeneous_from_q!(Ξtt, Ξq_tt)
    Ξ_ref = sum(Ξq_tt) / length(Ξq_tt)
    Ξtt .= Ξ_ref
    return Ξtt
end

function apply_momentum_convolution!(Σtt, Ξq_tt, Gtt, kmq_idx)
    fill!(Σtt, 0)
    L = length(Σtt)

    @inbounds for k in 1:L
        acc = zero(eltype(Σtt))
        for q in eachindex(Ξq_tt)
            acc += Ξq_tt[q] * Gtt[kmq_idx[k, q]]
        end
        Σtt[k] = 1im * acc / L
    end

    return Σtt
end

# Return a persistent 1D view into the underlying GreenFunction storage.
# This avoids ambiguous mutation semantics of wrapper-style indexing gf[t,t′].
@inline kbe_storage_tt(gf, t, t′) = @view gf.data[:, t, t′]

"""
Small diagnostic for storage semantics:
- `nonpersistent_slice_mutation=true` means mutating `gf[t,t′]` did not persist.
- `persistent_view_mutation=true` means mutating `kbe_storage_tt(gf,t,t′)` did persist.
"""
function verify_kbe_slice_persistence(; L::Int=4)
    gf = GreenFunction(zeros(ComplexF64, L, 1, 1), SkewHermitian)

    slice_copy = gf[1, 1]
    slice_copy[1] = 1 + 0im
    nonpersistent_slice_mutation = gf.data[1, 1, 1] == 0 + 0im

    storage_view = kbe_storage_tt(gf, 1, 1)
    storage_view[1] = 2 + 0im
    persistent_view_mutation = gf.data[1, 1, 1] == 2 + 0im

    return (; nonpersistent_slice_mutation, persistent_view_mutation)
end


function Xi_k_at_time(t; model, t′::Real=0.0, greater::Bool=false, apply_switch::Union{Nothing,Bool}=nothing)
    τ = t - t′
    switch = interaction_switch(t, t′; model, apply_switch)

    if model.bath_type == :spectral_density
        ξτ = greater ? Ξg(τ; model) : Ξl(τ; model)
        return ξτ .* switch
    elseif model.bath_type == :dispersion
        ξk = similar(model.ks, ComplexF64)
        if model.boson_kernel == :delta
            fill_dispersion_kernel_q!(ξk, τ, model.ωq, model.g2q, model.nBq; η=model.η, greater=greater)
        elseif model.boson_kernel == :spectral
            ωgrid_b = collect(-model.ωA_max:model.dωA:model.ωA_max)
            fill_dispersion_kernel_q_spectral!(ξk, τ, ωgrid_b, model.dωA, model.ωq, model.g2q; model, greater=greater)
        else
            throw(ArgumentError("Unknown boson_kernel: $(model.boson_kernel). Use :delta or :spectral."))
        end
        return ξk .* switch
    else
        throw(ArgumentError("Unknown bath_type: $(model.bath_type). Use :spectral_density or :dispersion."))
    end
end

function plot_Xi_vs_k(times; model=ModelElectronBath(), t_ref::Real=0.0, greater::Bool=false, component::Symbol=:real, apply_switch::Union{Nothing,Bool}=nothing)
    @assert component in (:real, :imag, :abs) "component must be :real, :imag, or :abs"

    plt = nothing
    for t in times
        ξk = Xi_k_at_time(t; model=model, t′=t_ref, greater=greater, apply_switch=apply_switch)
        y = component === :real ? real.(ξk) : component === :imag ? imag.(ξk) : abs.(ξk)
        label = "t=$(round(t; digits=3)), t′=$(round(t_ref; digits=3))"

        if plt === nothing
            plt = plot(model.ks, y; label=label, xlabel="k", ylabel="Ξ(k)", lw=2)
        else
            plot!(plt, model.ks, y; label=label, lw=2)
        end
    end

    return plt
end

function SelfEnergyUpdate!(model, data, times, _, _, t, t′)
    (; GL, GG, ΞL, ΞG, ΣL_F, ΣG_F, workspace) = data
    (; bath_type, boson_kernel, kmq_idx, ωq, g2q, nBq) = model
    (; tmpΞL, tmpΞG, tmpΣL, tmpΣG, itp_ΞL, itp_ΞG, ωgrid_b) = workspace

    if (n = size(GL, 3)) > size(ΣL_F, 3)
        resize!(ΞL, n)
        resize!(ΞG, n)
        resize!(ΣL_F, n)
        resize!(ΣG_F, n)
    end

    switch = interaction_switch(times[t], times[t′]; model)
    τ = times[t] - times[t′]

    ΞL_tt = kbe_storage_tt(ΞL, t, t′)
    ΞG_tt = kbe_storage_tt(ΞG, t, t′)
    GL_tt = kbe_storage_tt(GL, t, t′)
    GG_tt = kbe_storage_tt(GG, t, t′)
    ΣL_tt = kbe_storage_tt(ΣL_F, t, t′)
    ΣG_tt = kbe_storage_tt(ΣG_F, t, t′)

    if bath_type == :spectral_density
        # Interpolate Ξ(τ) from precomputed table; Ξ(-τ) = Ξ(τ)* for real J, nB.
        raw_ΞL = itp_ΞL(abs(τ))
        raw_ΞG = itp_ΞG(abs(τ))
        ΞL_ref = (τ ≥ 0 ? raw_ΞL : conj(raw_ΞL)) * switch
        ΞG_ref = (τ ≥ 0 ? raw_ΞG : conj(raw_ΞG)) * switch
        # Store homogeneous kernel for fv! (ΣR_H term).
        fill!(ΞL_tt, ΞL_ref)
        fill!(ΞG_tt, ΞG_ref)
        # Local bath: uniform Ξ per q; 1/L in apply_momentum_convolution! is the
        # electron-BZ normalization. No extra wq factor to avoid double counting.
        fill!(tmpΞL, ΞL_ref)
        fill!(tmpΞG, ΞG_ref)
    elseif bath_type == :dispersion
        if boson_kernel == :delta
            fill_dispersion_kernel_q!(tmpΞL, τ, ωq, g2q, nBq; η=model.η, greater=false)
            fill_dispersion_kernel_q!(tmpΞG, τ, ωq, g2q, nBq; η=model.η, greater=true)
        elseif boson_kernel == :spectral
            fill_dispersion_kernel_q_spectral!(tmpΞL, τ, ωgrid_b, model.dωA, ωq, g2q; model, greater=false)
            fill_dispersion_kernel_q_spectral!(tmpΞG, τ, ωgrid_b, model.dωA, ωq, g2q; model, greater=true)
        else
            throw(ArgumentError("Unknown boson_kernel: $(boson_kernel). Use :delta or :spectral."))
        end
        tmpΞL .*= switch
        tmpΞG .*= switch
        fill_homogeneous_from_q!(ΞL_tt, tmpΞL)
        fill_homogeneous_from_q!(ΞG_tt, tmpΞG)
    else
        throw(ArgumentError("Unknown bath_type: $(bath_type). Use :spectral_density or :dispersion."))
    end

    apply_momentum_convolution!(tmpΣL, tmpΞL, GL_tt, kmq_idx)
    apply_momentum_convolution!(tmpΣG, tmpΞG, GG_tt, kmq_idx)
    copyto!(ΣL_tt, tmpΣL)
    copyto!(ΣG_tt, tmpΣG)
end

# Auxiliary integrator for the first type of integral
function integrate1(hs::Vector, t1, t2, A::GreenFunction, B::GreenFunction, C::GreenFunction; tmax=t1)
    ret = zero(A[:, t1, t1])

    @inbounds for k in 1:tmax
        @views ret .+= (A[:, t1, k] .- B[:, t1, k]) .* C[:, k, t2] * hs[k]
    end

    return ret
end

# Auxiliary integrator for the second type of integral
function integrate2(hs::Vector, t1, t2, A::GreenFunction, B::GreenFunction, C::GreenFunction; tmax=t2)
    ret = zero(A[:, t1, t1])

    @inbounds for k in 1:tmax
        @views ret .+= A[:, t1, k] .* (B[:, k, t2] .- C[:, k, t2]) * hs[k]
    end

    return ret
end

# Auxiliary integrator for the third type of integral
function integrate3(hs::Vector, t1, A::GreenFunction, B::GreenFunction, C::GreenFunction; tmax=t1)
    ret = zero(A[:, t1, t1])

    @inbounds for k in 1:tmax
        @views ret .+= (A[:, t1, k] .- B[:, t1, k]) .* C[:, k, k] * hs[k]
    end

    return ret
end

function fv!(model, data, out, times, h1, h2, t, t′)
    (; GL, GG, ΞL, ΞG, ΣL_F, ΣG_F) = data
    (; hk) = model

    ∫dt1(A, B, C) = integrate1(h1, t, t′, A, B, C)
    ∫dt2(A, B, C) = integrate2(h2, t, t′, A, B, C)
    ∫dt3(A, B, C) = integrate3(h1, t, A, B, C)
    
    ΣR_H = -1im * ∫dt3(ΞG, ΞL, GL)
    
    out[1] = -1im * ((hk(times[t]) + ΣR_H).* GL[t, t′]  + ∫dt1(ΣG_F, ΣL_F, GL) + ∫dt2(ΣL_F, GL, GG))

    out[2] = -1im * ((hk(times[t]) + ΣR_H).* GG[t, t′]  + ∫dt1(ΣG_F, ΣL_F, GG) + ∫dt2(ΣG_F, GL, GG))

    return out
end

function fd!(model, data, out, times, h1, h2, t, t′)
    fv!(model, data, out, times, h1, h2, t, t)
    out .-= conj(out)
end

function make_name(model; tmax)
    "L$(model.L)_Te$(model.Te)_Tb$(model.Tb)_u$(model.u)_γ$(model.γ)" *
    "_$(model.bath_type)_α$(model.α)_s$(model.s)_ωc$(model.ωc)" *
    "_$(model.dispersion_type)_$(model.boson_kernel)_η$(model.η)_v_b$(model.v_b)_ωb0$(model.ωb0)" *
    "_$(model.wq_profile)_s_q$(model.s_q)_λ_q$(model.λ_q)_t0$(model.t0)_ω0$(model.ω0)_σ$(model.σ)_A$(model.A)_switch$(Int(model.switch_on))" *
    "_ti$(model.ti)_to$(model.to)_tmax$(tmax)"
end

function extract_occupations(GL)
    L = size(GL.data, 1)
    Nt = size(GL.data, 2)
    nk_t = zeros(Float64, L, Nt)
    @inbounds for it in 1:Nt
        nk_t[:, it] .= imag.(kbe_storage_tt(GL, it, it))
    end
    nbar_t = vec(sum(nk_t; dims=1) ./ L)
    return nk_t, nbar_t
end

function main(; tmax=10, save_mode::Symbol=:full, kwargs...)
    #### Read kwargs

    println(kwargs...)
    #### Setting the initial parameters
    model = ModelElectronBath(; kwargs...)
    

    L = model.L
    @assert model.bath_type in (:spectral_density, :dispersion) "bath_type must be :spectral_density or :dispersion"
    @assert model.dispersion_type in (:linear, :sin_lattice) "dispersion_type must be :linear or :sin_lattice"
    @assert model.boson_kernel in (:delta, :spectral) "boson_kernel must be :delta or :spectral"
    @assert model.η ≥ 0 "η must be nonnegative"
    @assert model.λ_q > 0 "λ_q must be positive"
    @assert length(model.wq) == L "wq must have length L"
    # Previous normalization check:
    # @assert isapprox(sum(model.wq), model.η / model.λ_q; atol=1e-12) "wq must satisfy sum(wq) = η / λ_q"
    @assert length(model.ωq) == L "ωq must have length L"
    @assert length(model.g2q) == L "g2q must have length L"
    u = model.u
    γ = model.γ
    ks = model.ks 
    (; Te, Tb, α, s, ωc, t0, ω0, σ, A, ti, to) = model
    #### Initial conditions ####

    
    #### Lesser and greater Green's functions
    GL  = GreenFunction(zeros(ComplexF64, L, 1, 1), SkewHermitian)
    GG  = GreenFunction(zeros(ComplexF64, L, 1, 1), SkewHermitian)
    ΞL  = GreenFunction(zeros(ComplexF64, L, 1, 1), SkewHermitian)
    ΞG  = GreenFunction(zeros(ComplexF64, L, 1, 1), SkewHermitian)
    ΣL_F = GreenFunction(zeros(ComplexF64, L, 1, 1), SkewHermitian)
    ΣG_F = GreenFunction(zeros(ComplexF64, L, 1, 1), SkewHermitian)

    #### Time integration parameters (needed before workspace for table range)
    atol = 1e-6
    rtol = 1e-5

    # Precompute Ξ^</>( τ ) on a uniform grid and build interpolants.
    # Only needed for :spectral_density; :dispersion evaluates analytically per q.
    if model.bath_type == :spectral_density
        _dω     = 0.01
        _ωs     = collect(-100.0:_dω:100.0)
        _J      = J.(_ωs; model)
        _nB     = bose.(_ωs; model)
        Nτ      = 4000
        τ_grid  = collect(range(0.0, Float64(tmax) * 1.05; length=Nτ))
        ΞL_tab  = ComplexF64[-1im/(2π) * sum(_J .* _nB        .* exp.(-1im*_ωs*τ)) * _dω for τ in τ_grid]
        ΞG_tab  = ComplexF64[-1im/(2π) * sum(_J .* (_nB .+ 1) .* exp.(-1im*_ωs*τ)) * _dω for τ in τ_grid]
        itp_ΞL  = interpolate((τ_grid,), ΞL_tab, Gridded(Linear()))
        itp_ΞG  = interpolate((τ_grid,), ΞG_tab, Gridded(Linear()))
        println("Ξ tables precomputed ($(Nτ) points, τ ∈ [0, $(round(τ_grid[end]; digits=2))])")
    else
        itp_ΞL = nothing
        itp_ΞG = nothing
    end

    workspace = (
        tmpΞL  = similar(model.ks, ComplexF64),
        tmpΞG  = similar(model.ks, ComplexF64),
        tmpΣL  = similar(model.ks, ComplexF64),
        tmpΣG  = similar(model.ks, ComplexF64),
        itp_ΞL = itp_ΞL,
        itp_ΞG = itp_ΞG,
        ωgrid_b = vcat(
            collect(-model.ωA_max:model.dωA:-model.dωA),
            collect(model.dωA:model.dωA:model.ωA_max),
        ),
    )

    #### Initial conditions lesser and greater Green's functions
    # Use persistent views (kbe_storage_tt) for IC writes.
    # gf[t,t′] returns a copy → mutations via = or .= on that copy do not persist.
    GL_11 = kbe_storage_tt(GL, 1, 1)
    GG_11 = kbe_storage_tt(GG, 1, 1)
    copyto!(GL_11, 1im .* fermi.(ϵ_k(ks; u, γ); model=(; Te)))
    copyto!(GG_11, GL_11 .- 1im)
    switch_00 = interaction_switch(0.0, 0.0; model)
    if model.bath_type == :spectral_density
        ΞL_00 = Ξl(0.0; model) * switch_00
        ΞG_00 = Ξg(0.0; model) * switch_00
        fill!(workspace.tmpΞL, ΞL_00)
        fill!(workspace.tmpΞG, ΞG_00)
    elseif model.bath_type == :dispersion
        if model.boson_kernel == :delta
            fill_dispersion_kernel_q!(workspace.tmpΞL, 0.0, model.ωq, model.g2q, model.nBq; η=model.η, greater=false)
            fill_dispersion_kernel_q!(workspace.tmpΞG, 0.0, model.ωq, model.g2q, model.nBq; η=model.η, greater=true)
        elseif model.boson_kernel == :spectral
            fill_dispersion_kernel_q_spectral!(workspace.tmpΞL, 0.0, workspace.ωgrid_b, model.dωA, model.ωq, model.g2q; model, greater=false)
            fill_dispersion_kernel_q_spectral!(workspace.tmpΞG, 0.0, workspace.ωgrid_b, model.dωA, model.ωq, model.g2q; model, greater=true)
        else
            throw(ArgumentError("Unknown boson_kernel: $(model.boson_kernel). Use :delta or :spectral."))
        end
        workspace.tmpΞL .*= switch_00
        workspace.tmpΞG .*= switch_00
    else
        throw(ArgumentError("Unknown bath_type: $(model.bath_type). Use :spectral_density or :dispersion."))
    end
    apply_momentum_convolution!(kbe_storage_tt(ΣL_F, 1, 1), workspace.tmpΞL, kbe_storage_tt(GL, 1, 1), model.kmq_idx)
    apply_momentum_convolution!(kbe_storage_tt(ΣG_F, 1, 1), workspace.tmpΞG, kbe_storage_tt(GG, 1, 1), model.kmq_idx)

    #### Setting the initial dynamical variables
    data = DataElectronBath(GL=GL, GG=GG, ΞL=ΞL, ΞG=ΞG, ΣL_F=ΣL_F, ΣG_F=ΣG_F, workspace=workspace)
    
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
   
    @assert save_mode in (:full, :occupations_only) "save_mode must be :full or :occupations_only"

    file = "Data"
    mkpath(file)
    name_p = make_name(model; tmax)

    if save_mode == :full
        @save "$(file)/GL_$(name_p).jld2" GL
        @save "$(file)/GG_$(name_p).jld2" GG
        @save "$(file)/ts_$(name_p).jld2" sol
        println("Saved full Green-function results.")
    else
        nk_t, nbar_t = extract_occupations(GL)
        ks = collect(model.ks)
        params = (; save_mode, tmax, model.L, model.Te, model.Tb, model.u, model.γ, model.α,
            model.s, model.ωc, model.t0, model.ω0, model.σ, model.A, model.switch_on,
            model.ti, model.to, model.bath_type, model.dispersion_type, model.boson_kernel,
            model.η, model.ωA_max, model.dωA, model.ωb0, model.v_b, model.wq_profile,
            model.s_q, model.λ_q)
        @save "$(file)/occ_$(name_p).jld2" nk_t nbar_t ks params
        @save "$(file)/ts_$(name_p).jld2" sol
        println("Saved occupations-only results.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
