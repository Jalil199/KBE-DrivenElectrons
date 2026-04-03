using KadanoffBaym, LinearAlgebra, BlockArrays
using LaTeXStrings
using FFTW, Interpolations
using Tullio
using JLD2

Base.@kwdef struct ModelElectronBath{Hk}
    L::Int = 100
    T::Float64 = 0.1
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
    Δk = 2*pi/L
    ks = collect(range(-pi, stop=pi-Δk, length=L))
    wq::Vector{Float64} = fill(1 / L, L)
    kmq_idx::Matrix{Int} = [mod1(k - q, L) for k in 1:L, q in 1:L]
    hk::Hk = t -> ϵ_k(ks .- pulse_Gaussian_sin(t; t0, ω0, σ, A);  u, γ)
end

Base.@kwdef struct DataElectronBath{T}
    GL::T
    GG::T
    
    ΞL::T
    ΞG::T
    ΞL_q::T
    ΞG_q::T
    
    ΣL_F::T
    ΣG_F::T
end

function fermi(ϵ; model)
    (; T) = model
    β = 1/T
    1/(exp(β*ϵ)+1)
end

function bose(ϵ; model)
    (; T) = model
    β = 1/T
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
    (; L) = model
    dω = 0.01
    ωs = -100:dω:100
    -1im / (2pi) * sum(J.(ωs; model) .* bose.(ωs; model) .* exp.(-1im * ωs * t)) * dω * ones(L)
end

function Ξg(t; model)
    (; L) = model
    dω = 0.01
    ωs = -100:dω:100
    -1im / (2pi) * sum(J.(ωs; model) .* (bose.(ωs; model) .+ 1) .* exp.(-1im * ωs * t)) * dω * ones(L)
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

function weighted_kernel_q_from_homogeneous!(Ξq, Ξ, wq, t, t′)
    Ξq_tt = Ξq[t, t′]
    Ξ_ref = Ξ[t, t′][1]

    @inbounds for q in eachindex(wq)
        Ξq_tt[q] = wq[q] * Ξ_ref
    end
end

function apply_momentum_convolution!(Σtt, Ξq_tt, Gtt, kmq_idx)
    fill!(Σtt, 0)
    L = length(Σtt)

    @inbounds for k in 1:L
        acc = zero(eltype(Σtt))
        for q in eachindex(Ξq_tt)
            acc += Ξq_tt[q] * Gtt[kmq_idx[k, q]]
        end
        Σtt[k] = 1im * acc
    end

    return Σtt
end

function SelfEnergyUpdate!(model, data, times, _, _, t, t′)
    (; GL, GG, ΞL, ΞG, ΞL_q, ΞG_q, ΣL_F, ΣG_F) = data
    (; wq, kmq_idx) = model
    
    if (n = size(GL, 3)) > size(ΣL_F, 3)
        resize!(ΞL, n)
        resize!(ΞG, n)
        resize!(ΞL_q, n)
        resize!(ΞG_q, n)
        resize!(ΣL_F, n)
        resize!(ΣG_F, n)
    end

    ΞL[t,t′] = Ξl(times[t] - times[t′]; model) * stepp.(times[t]; model) * stepp.(times[t′]; model)
    ΞG[t,t′] = Ξg(times[t] - times[t′]; model) * stepp.(times[t]; model) * stepp.(times[t′]; model)
    weighted_kernel_q_from_homogeneous!(ΞL_q, ΞL, wq, t, t′)
    weighted_kernel_q_from_homogeneous!(ΞG_q, ΞG, wq, t, t′)
    apply_momentum_convolution!(ΣL_F[t,t′], ΞL_q[t,t′], GL[t,t′], kmq_idx)
    apply_momentum_convolution!(ΣG_F[t,t′], ΞG_q[t,t′], GG[t,t′], kmq_idx)
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

function main(; kwargs...)
    #### Read kwargs 
    
    println(kwargs...)
    #### Setting the initial parameters
    model = ModelElectronBath(;kwargs...)
    

    L = model.L
    @assert length(model.wq) == L "wq must have length L"
    @assert isapprox(sum(model.wq), 1.0; atol=1e-12) "wq must satisfy sum(wq) = 1"
    u = model.u
    γ = model.γ
    ks = model.ks 
    (; T, α, s, ωc, t0, ω0, σ, A, ti, to) = model
    #### Initial conditions ####

    
    #### Lesser and greater Green's functions
    GL = GreenFunction(zeros(ComplexF64, L, 1, 1), SkewHermitian)
    GG = GreenFunction(zeros(ComplexF64, L, 1, 1), SkewHermitian)
    ΞL = GreenFunction(zeros(ComplexF64, L, 1, 1), SkewHermitian)
    ΞG = GreenFunction(zeros(ComplexF64, L, 1, 1), SkewHermitian)    
    ΞL_q = GreenFunction(zeros(ComplexF64, L, 1, 1), SkewHermitian)
    ΞG_q = GreenFunction(zeros(ComplexF64, L, 1, 1), SkewHermitian)
    ΣL_F = GreenFunction(zeros(ComplexF64, L, 1, 1), SkewHermitian)
    ΣG_F = GreenFunction(zeros(ComplexF64, L, 1, 1), SkewHermitian)
      
    #### Initial conditions lesser and greater Green's functions
    GL[1, 1] = 1im * fermi.(ϵ_k(ks;  u, γ); model)
    GG[1, 1] = GL[1, 1] .- 1im
    ΞL[1,1] = Ξl(0; model) * 0.0
    ΞG[1,1] = Ξg(0; model) * 0.0
    weighted_kernel_q_from_homogeneous!(ΞL_q, ΞL, model.wq, 1, 1)
    weighted_kernel_q_from_homogeneous!(ΞG_q, ΞG, model.wq, 1, 1)
    apply_momentum_convolution!(ΣL_F[1,1], ΞL_q[1,1], GL[1,1], model.kmq_idx)
    apply_momentum_convolution!(ΣG_F[1,1], ΞG_q[1,1], GG[1,1], model.kmq_idx)
    
    #### Setting the initial dynamical variables
    data = DataElectronBath(GL=GL, GG=GG, ΞL=ΞL, ΞG=ΞG, ΞL_q=ΞL_q, ΞG_q=ΞG_q, ΣL_F=ΣL_F, ΣG_F=ΣG_F)
  
    #### Setting the time integration
    tmax = 10
    atol = 1e-6
    rtol = 1e-5
    
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
    name_p = "T$(T)_α$(α)_s$(s)_ωc$(ωc)_t0$(t0)_ω0$(ω0)_σ$(σ)_A$(A)_ti$(ti)_to$(to)"
    
    @save "$(file)/GL_$(name_p).jld2" GL
    @save "$(file)/GG_$(name_p).jld2" GG
    @save "$(file)/ts_$(name_p).jld2" sol

    println("Saved all results.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
