using KadanoffBaym, LinearAlgebra, BlockArrays
using LaTeXStrings
using FFTW, Interpolations
using Tullio
using JLD2
using PyPlot

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
    ti::Float64 = 3.0
    to::Float64 = 20.0
    bath_type::Symbol = :spectral_density
    dispersion_type::Symbol = :linear
    ωb0::Float64 = 0.2
    v_b::Float64 = 1.0
    g_b::Float64 = 0.1
    Δk = 2*pi/L
    ks = collect(range(-pi, stop=pi-Δk, length=L))
    wq_profile::Symbol = :uniform
    s_q::Float64 = 0.0
    λ_q::Float64 = 1.0
    wq::Vector{Float64} = make_momentum_weights(wq_profile; ks, s_q, λ_q)
    bath_qs::Vector{Float64} = copy(ks)
    ωq::Vector{Float64} = make_bath_dispersion(dispersion_type; qs=bath_qs, ωb0, v_b)
    g2q::Vector{Float64} = make_bath_coupling2(; qs=bath_qs, g_b)
    nBq::Vector{Float64} = bose.(ωq; model=(; Tb))
    kmq_idx::Matrix{Int} = [mod1(k - q, L) for k in 1:L, q in 1:L]
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

function homogeneous_momentum_sum(Gtt)
    sumG = similar(Gtt)
    sumG .= sum(Gtt)
    return sumG
end


function fill_dispersion_kernel_q!(Ξq_tt, τ, ωq, g2q, nBq; greater::Bool)
    @inbounds for q in eachindex(Ξq_tt)
        nq = nBq[q]
        pref = greater ? (nq + 1) : nq
        pref_tr = greater ? nq : (nq + 1)
        Ξq_tt[q] = -1im * g2q[q] * (pref * exp(-1im * ωq[q] * τ) + pref_tr * exp(1im * ωq[q] * τ))
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


function Xi_k_at_time(t; model, t′::Real=0.0, greater::Bool=false, apply_switch::Bool=true)
    τ = t - t′
    switch = apply_switch ? stepp(t; model) * stepp(t′; model) : 1.0

    if model.bath_type == :spectral_density
        ξτ = greater ? Ξg(τ; model) : Ξl(τ; model)
        return ξτ .* switch
    elseif model.bath_type == :dispersion
        ξk = similar(model.ks, ComplexF64)
        fill_dispersion_kernel_q!(ξk, τ, model.ωq, model.g2q, model.nBq; greater=greater)
        return ξk .* switch
    else
        throw(ArgumentError("Unknown bath_type: $(model.bath_type). Use :spectral_density or :dispersion."))
    end
end

function plot_Xi_vs_k(times; model=ModelElectronBath(), t_ref::Real=0.0, greater::Bool=false, component::Symbol=:real, apply_switch::Bool=true)
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
    (; bath_type, kmq_idx, ωq, g2q, nBq) = model
    (; tmpΞL, tmpΞG, tmpΣL, tmpΣG, itp_ΞL, itp_ΞG) = workspace

    if (n = size(GL, 3)) > size(ΣL_F, 3)
        resize!(ΞL, n)
        resize!(ΞG, n)
        resize!(ΣL_F, n)
        resize!(ΣG_F, n)
    end

    switch = stepp(times[t]; model) * stepp(times[t′]; model)
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
        fill_dispersion_kernel_q!(tmpΞL, τ, ωq, g2q, nBq; greater=false)
        fill_dispersion_kernel_q!(tmpΞG, τ, ωq, g2q, nBq; greater=true)
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
    "_$(model.dispersion_type)_g_b$(model.g_b)_v_b$(model.v_b)_ωb0$(model.ωb0)" *
    "_$(model.wq_profile)_t0$(model.t0)_ω0$(model.ω0)_σ$(model.σ)_A$(model.A)" *
    "_ti$(model.ti)_to$(model.to)_tmax$(tmax)"
end

function main(; tmax=10, kwargs...)
    #### Read kwargs

    println(kwargs...)
    #### Setting the initial parameters
    model = ModelElectronBath(; kwargs...)
    

    L = model.L
    @assert model.bath_type in (:spectral_density, :dispersion) "bath_type must be :spectral_density or :dispersion"
    @assert model.dispersion_type in (:linear, :sin_lattice) "dispersion_type must be :linear or :sin_lattice"
    @assert length(model.wq) == L "wq must have length L"
    @assert isapprox(sum(model.wq), 1.0; atol=1e-12) "wq must satisfy sum(wq) = 1"
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
    )

    #### Initial conditions lesser and greater Green's functions
    # Use persistent views (kbe_storage_tt) for IC writes.
    # gf[t,t′] returns a copy → mutations via = or .= on that copy do not persist.
    GL_11 = kbe_storage_tt(GL, 1, 1)
    GG_11 = kbe_storage_tt(GG, 1, 1)
    copyto!(GL_11, 1im .* fermi.(ϵ_k(ks; u, γ); model=(; Te)))
    copyto!(GG_11, GL_11 .- 1im)
    # Σ(0,0) = 0 because the adiabatic switch stepp(0) ≈ 0 kills the bath at t=0.
    fill!(workspace.tmpΞL, zero(ComplexF64))
    fill!(workspace.tmpΞG, zero(ComplexF64))
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
   
    file = "Data"
    mkpath(file)
    name_p = make_name(model; tmax)
    
    @save "$(file)/GL_$(name_p).jld2" GL
    @save "$(file)/GG_$(name_p).jld2" GG
    @save "$(file)/ts_$(name_p).jld2" sol

    println("Saved all results.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
