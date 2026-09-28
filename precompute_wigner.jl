using FFTW, Interpolations
using JLD2
using Tullio
using KadanoffBaym, LinearAlgebra
using Statistics

# ── Window functions ────────────────────────────────────────────────────────
function gauss_window(N::Integer)
    Δ = (0:N-1) .- Int(round(N/2))
    σ = (N-1)/8
    return @. exp(-0.5*(Δ^2) / σ^2)
end

# ── Override KadanoffBaym ft/wigner_transform ────────────────────────────────
import KadanoffBaym: ft
import KadanoffBaym: wigner_transform

function ft(xs, ys; inverse::Bool=false, window=ones(length(ys)))
    @assert issorted(xs)
    L  = length(xs)
    dx = xs[2] - xs[1]
    x̂s = fftfreq(L, 2π/dx)
    ℯⁱᵠ = (inverse ? 1/(2π) : 1.0) * dx *
           exp.((inverse ? -1.0 : 1.0) * im * xs[1] .* x̂s)
    ŷs = ℯⁱᵠ .* (inverse ? fft : bfft)(ys .* window)
    return fftshift(x̂s), fftshift(ŷs)
end

function wigner_transform(x::AbstractMatrix; ts=1:size(x,1), fourier=true,
                           window=ones(size(x,1)))
    Nt = size(x,1)
    @assert length(ts) == Nt
    @assert let d = diff(ts); all(z -> z ≈ d[1], d) end "`ts` not equidistant"
    x_W = zero(x)
    for T in 1:Nt
        τ_max = minimum([T-1, Nt-T, Nt÷2-1])
        τs = (-τ_max):τ_max
        is = 1 .+ rem.(Nt .+ τs, Nt)
        for (i, τᵢ) in zip(is, τs)
            x_W[i, T] = x[T+τᵢ, T-τᵢ]
        end
        τ = Nt÷2
        if T <= Nt-τ && T >= τ+1
            x_W[τ+1, T] = 0.5*(x[T+τ, T-τ] + x[T-τ, T+τ])
        end
    end
    x_W = fftshift(x_W, 1)
    τs = ts - reverse(ts)
    τs = τs .- (isodd(Nt) ? 0.0 : 0.5(τs[2]-τs[1]))
    !fourier && return x_W, (τs, ts)
    ωs = ft(τs, τs; window=window)[1]
    x_Ŵ = mapslices(x -> ft(τs, x; inverse=false)[2], x_W; dims=1)
    return x_Ŵ, (ωs, ts)
end

# ── Auxiliary functions ──────────────────────────────────────────────────────
@inline function pulse_Gaussian_sin(t::Float64; t0::Float64, ω0::Float64,
                                     σ::Float64, A::Float64)
    A * exp(-0.5*(t-t0)^2/σ^2) * sin(t*ω0)
end

@inline function wigner_transform_itp(x, ts::Vector; fourier=true,
        ts_lin=range(first(ts), last(ts); length=length(ts)),
        window=gauss_window(length(ts)))
    itp = interpolate((ts, ts), x, Gridded(Linear()))
    return wigner_transform([itp(t1,t2) for t1 in ts_lin, t2 in ts_lin];
                             ts=ts_lin, fourier=fourier, window=window)
end

# ── Gauge transform ──────────────────────────────────────────────────────────
@inline function gauge_transform(G_data, ts, L)
    T    = length(ts)
    ks   = collect(range(-pi, stop=pi, length=L))
    G_itp    = interpolate((ks, ts, ts), G_data, Gridded(Linear()))
    G_itp_ex = extrapolate(G_itp, (Periodic(), Throw(), Throw()))
    pulse(t) = pulse_Gaussian_sin(t; t0=20.0, ω0=2.2, σ=2.0, A=0.5)
    A_vec = [pulse(t) for t in ts]
    F = zeros(eltype(A_vec), T)
    @inbounds for i in 2:T
        Δt = ts[i]-ts[i-1]
        F[i] = F[i-1] + 0.5*(A_vec[i]+A_vec[i-1])*Δt
    end
    Aavg = zeros(eltype(A_vec), T, T)
    @inbounds for i in 1:T
        Aavg[i,i] = A_vec[i]
        for j in 1:T
            i != j && (Aavg[i,j] = (F[i]-F[j])/(ts[i]-ts[j]))
        end
    end
    G_out = similar(G_data)
    @inbounds for (ki,k) in enumerate(ks)
        for i in 1:T, j in 1:T
            G_out[ki,i,j] = G_itp_ex(k+Aavg[i,j], ts[i], ts[j])
        end
    end
    return G_out
end

# ── get_wigner ───────────────────────────────────────────────────────────────
function get_wigner(GL_data, GG_data, ts_data; L=40, gauge=false)
    ks  = collect(range(-pi, stop=pi, length=L))
    ts  = ts_data.t
    size_dims = length(size(GL_data))
    if size_dims <= 4
        GL_g = gauge ? gauge_transform(GL_data[:,:,:], ts, L) : GL_data[:,:,:]
        GG_g = gauge ? gauge_transform(GG_data[:,:,:], ts, L) : GG_data[:,:,:]
    else
        base_GL = GL_data[1,1,:,:,:] .+ GL_data[2,2,:,:,:]
        base_GG = GG_data[1,1,:,:,:] .+ GG_data[2,2,:,:,:]
        GL_g = gauge ? gauge_transform(base_GL, ts, L) : base_GL
        GG_g = gauge ? gauge_transform(base_GG, ts, L) : base_GG
    end
    dim_tavg = size(GL_g, 1) == L ? size(GL_g, 2) : size(GL_g)[end]
    # dim_tavg is the number of time steps (Nt)
    dim_tavg = size(GL_g, 2)   # GL_g is L×Nt×Nt

    GL_filt = copy(GL_g)
    GG_filt = copy(GG_g)

    GL_W = zeros(ComplexF64, L, dim_tavg, dim_tavg)
    A_W  = zeros(ComplexF64, L, dim_tavg, dim_tavg)
    ωs   = Float64[]
    tavg = Float64[]
    for i in 1:L
        GL_fft, (ωs_i, tavg_i) = wigner_transform_itp(GL_filt[i,:,:], ts)
        GR_fft, _               = wigner_transform_itp((GG_filt-GL_filt)[i,:,:], ts)
        GL_W[i,:,:] = GL_fft
        A_W[i,:,:]  = @. -imag(GR_fft)/π
        if i == 1
            ωs   = collect(ωs_i)
            tavg = collect(tavg_i)
        end
    end

    @tullio sum_w[k,t] := A_W[k,t1,t]
    norm = mean(abs.(sum_w))
    A_W  ./= norm
    GL_W ./= (norm*π)

    return A_W, GL_W, tavg, ωs, ks
end

# ── Compute and save all six datasets ───────────────────────────────────────
const DATA_SRC  = "/home/jalil2/Documents/KB-Equations/Data"
const DATA_DEST = "/home/jalil2/Desktop/KBE-DrivenElectrons/Data/wigner_precomputed.jld2"

function load_set(name)
    GL = load("$DATA_SRC/GL_$name.jld2")["GL"]
    GG = load("$DATA_SRC/GG_$name.jld2")["GG"]
    ts = load("$DATA_SRC/ts_$name.jld2")["sol"]
    return GL, GG, ts
end

println("=== closed (Rice Mele gapped) ===")
GL_d, GG_d, ts_d = load_set("closed")
_, GL_11, _, ωs_11, _ = get_wigner(GL_d, GG_d, ts_d; L=80, gauge=false)

println("=== open (Rice Mele gapped + bath) ===")
GL_d, GG_d, ts_d = load_set("open")
_, GL_21, _, ωs_21, _ = get_wigner(GL_d, GG_d, ts_d; L=80, gauge=false)

println("=== driven (Rice Mele gapped + pulse) ===")
GL_d, GG_d, ts_d = load_set("driven")
_, GL_31, _, ωs_31, _ = get_wigner(GL_d, GG_d, ts_d; L=80, gauge=true)

println("=== closed_chain (1D chain gapless) ===")
GL_d, GG_d, ts_d = load_set("closed_chain")
_, GL_12, _, ωs_12, _ = get_wigner(GL_d, GG_d, ts_d; L=100, gauge=false)

println("=== open_chain (1D chain gapless + bath) ===")
GL_d, GG_d, ts_d = load_set("open_chain")
_, GL_22, _, ωs_22, _ = get_wigner(GL_d, GG_d, ts_d; L=100, gauge=false)

println("=== driven_chain (1D chain gapless + pulse) ===")
GL_d, GG_d, ts_d = load_set("driven_chain")
_, GL_32, _, ωs_32, _ = get_wigner(GL_d, GG_d, ts_d; L=100, gauge=true)

println("Saving to $DATA_DEST ...")
jldsave(DATA_DEST;
    GL_11, ωs_11,
    GL_21, ωs_21,
    GL_31, ωs_31,
    GL_12, ωs_12,
    GL_22, ωs_22,
    GL_32, ωs_32)
println("Done.")
