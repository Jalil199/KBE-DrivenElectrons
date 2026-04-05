# Add the Julia-artifact ffmpeg to PATH before matplotlib is loaded
let ffmpeg_bin = strip(read(`find /home/jalil2/.conda -name "ffmpeg" -type f`, String))
    ffmpeg_dir = dirname(ffmpeg_bin)
    ENV["PATH"] = ffmpeg_dir * ":" * ENV["PATH"]
end

using KadanoffBaym, FFTW, Interpolations, JLD2, PyPlot

# ── Parameters (must match the simulation) ────────────────────────────────────
const DATA_NAME = "test_gi"
const T_BATH    = 0.1       # temperature
const T0_PULSE  = 50.0      # pulse centre
const σ_PULSE   = 2.0       # pulse width
const ω_WINDOW  = 4.4       # frequency axis half-width for display
const ANIM_FILE = "distribution_animation.gif"

fermi(ω) = 1 / (exp(ω / T_BATH) + 1)

# ── Load data ─────────────────────────────────────────────────────────────────
GL  = load("Data/GL_$(DATA_NAME).jld2", "GL")
GG  = load("Data/GG_$(DATA_NAME).jld2", "GG")
ts  = load("Data/ts_$(DATA_NAME).jld2", "sol").t

L   = size(GL, 1)
println("L=$L  Nt=$(length(ts))  t ∈ [$(round(first(ts),digits=1)), $(round(last(ts),digits=1))]")

# ── Gaussian window in relative time (reduces FFT edge artefacts) ─────────────
Δ       = ts .- ts'
W       = @. exp(-2 * Δ^2 / last(ts)^2)

GL_raw  = GL.data   # (L, Nt, Nt)
GG_raw  = GG.data

GL_filt = GL_raw .* reshape(W, 1, size(W)...)
GG_filt = GG_raw .* reshape(W, 1, size(W)...)

# ── Wigner transform ──────────────────────────────────────────────────────────
function wigner_itp(x, ts)
    ts_lin = range(first(ts), last(ts); length=length(ts))
    itp    = interpolate((ts, ts), x, Gridded(Linear()))
    wigner_transform([itp(t1,t2) for t1 in ts_lin, t2 in ts_lin]; ts=ts_lin, fourier=true)
end

println("Computing Wigner transforms for $L momenta …")
# First k to initialise arrays, then accumulate
GLw1, (ωs, tavg) = wigner_itp(GL_filt[1,:,:], ts)
GRw1, _          = wigner_itp((GG_filt .- GL_filt)[1,:,:], ts)
GL_sum = GLw1 ./ L
GR_sum = GRw1 ./ L

for k in 2:L
    k % 20 == 0 && println("  k = $k / $L")
    GLw, _ = wigner_itp(GL_filt[k,:,:], ts)
    GRw, _ = wigner_itp((GG_filt .- GL_filt)[k,:,:], ts)
    GL_sum .+= GLw ./ L
    GR_sum .+= GRw ./ L
end

# ── Distribution function F(ω, t_avg) = Im G^< / Im(G^>−G^<) ─────────────────
# In equilibrium this equals the Fermi-Dirac distribution n_F(ω).
Nω    = length(ωs)
Ntavg = length(tavg)
δ     = 1e-6

F = fill(NaN, Nω, Ntavg)
for it in 1:Ntavg
    A_col = -imag.(GR_sum[:, it]) ./ π          # spectral function A = −Im G^R / π
    GL_im =  imag.(GL_sum[:, it]) ./ π
    mask  = abs.(A_col) .> δ
    F[mask, it] .= GL_im[mask] ./ A_col[mask]
end

# ── Select frames around the pulse ────────────────────────────────────────────
t_start = T0_PULSE - 3σ_PULSE
t_stop  = T0_PULSE + 3σ_PULSE
frames  = findall(t -> t_start ≤ t ≤ t_stop, tavg)

if isempty(frames)
    @warn "Simulation does not reach the pulse (tmax=$(round(last(ts),digits=1)) < t0=$T0_PULSE). " *
          "Showing full available time range. Re-run main with tmax ≳ $(T0_PULSE + 3σ_PULSE)."
    frames = 1:Ntavg
end

println("Animating $(length(frames)) frames  t_avg ∈ " *
        "[$(round(tavg[first(frames)],digits=1)), $(round(tavg[last(frames)],digits=1))]")

# ── Build animation ───────────────────────────────────────────────────────────
rc("text", usetex=true)
rc("font", family="serif")
rc("text.latex", preamble=raw"\usepackage{amsmath}")
rc("font", size=18)
rc("axes", labelsize=22, titlesize=22)
rc("xtick", labelsize=18)
rc("ytick", labelsize=18)
rc("legend", fontsize=16)

fig, ax = subplots(figsize=(8, 6))
ax.set_xlim(-ω_WINDOW, ω_WINDOW)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel(raw"$\omega(\gamma/\hbar)$")
ax.set_ylabel(raw"$F\mathrm{(\omega,t_{\text{avg}})}$")
ax.plot(ωs, fermi.(ωs), ls="--", color="black", lw=1.5, label=raw"$n_F(\omega)$", zorder=1)
line, = ax.plot(Float64[], Float64[], color="tab:blue", lw=2,
                label=raw"$F(\omega,t_{\text{avg}})$", zorder=2)
ttl = ax.set_title("")
ax.axvline( 2, color="yellow", lw=1.2)   # band edge markers at ±2γ
ax.axvline(-2, color="yellow", lw=1.2)
ax.legend(frameon=false)
tight_layout()

anim_mod = PyPlot.PyCall.pyimport("matplotlib.animation")


function update(i)
    it = frames[i + 1]   # FuncAnimation passes 0-based index
    Fslice = copy(F[:, it])
    line.set_data(ωs, Fslice)
    ttl.set_text("\$t_{\\text{avg}} = $(round(tavg[it]; digits=2))\$")
    return (line, ttl)
end

anim = anim_mod.FuncAnimation(fig, update; frames=length(frames), interval=80, blit=false)
anim.save(ANIM_FILE; writer="pillow", dpi=130)
println("Saved $ANIM_FILE")
