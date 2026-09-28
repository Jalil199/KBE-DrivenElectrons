using JLD2
using PyPlot

const ROOT = @__DIR__
const DATA_DIR = joinpath(ROOT, "Data")
const OUT_DIR = joinpath(ROOT, "distributions")
mkpath(OUT_DIR)

function nearest_index(ts, target)
    return argmin(abs.(ts .- target))
end

function extract_field(name::AbstractString, key::AbstractString)
    m = match(Regex(key * "([^_]+)"), name)
    return m === nothing ? missing : m.captures[1]
end

files = sort(filter(f -> startswith(basename(f), "GL_L100_Te1.0_Tb0.1_u0.0_γ1.0_dispersion_α") &&
    occursin("_linear_spectral_", basename(f)) &&
    endswith(basename(f), "_tmax60.jld2"),
    readdir(DATA_DIR; join=true)))

isempty(files) && error("No completed chain GL files found for Tb=0.1 in $DATA_DIR")

plt.rc("font", family="serif", size=15)
plt.rc("axes", linewidth=1.8)
plt.rc("xtick.major", width=1.8, size=7)
plt.rc("ytick.major", width=1.8, size=7)
plt.rc("xtick", direction="in")
plt.rc("ytick", direction="in")

alpha_order = ["0.2", "0.4", "0.6", "0.8", "1.0", "1.2", "1.4", "1.6", "1.8", "2.0"]
alpha_to_panel = Dict(a => i for (i, a) in enumerate(alpha_order))

lambda_colors = Dict("0.2" => "#1f77b4", "0.5" => "#d1495b", "1.0" => "#2a9d8f")
eta_styles = Dict("0.05" => "-", "0.5" => "--")

fig, axs = subplots(2, 5; figsize=(22.0, 8.8), sharex=true, sharey=true)
axs = vec(axs)

Lref = 100
ks_ref = collect(range(-π, stop=π - 2π / Lref, length=Lref))
nk_init_ref = 1.0 ./ (exp.((-2 .* cos.(ks_ref)) ./ 1.0) .+ 1.0)
for ax in axs
    ax.plot(ks_ref, nk_init_ref; color="0.70", linewidth=2.8, alpha=0.75, zorder=0)
end

for f in files
    base = basename(f)
    α = string(extract_field(base, "α"))
    η = string(extract_field(base, "η"))
    λ = string(extract_field(base, "λ_q"))

    haskey(alpha_to_panel, α) || continue
    ax = axs[alpha_to_panel[α]]

    GL_obj = load(f, "GL")
    GL = hasproperty(GL_obj, :data) ? GL_obj.data : GL_obj
    ts_obj = load(replace(f, "GL_" => "ts_"), "sol")
    ts = hasproperty(ts_obj, :t) ? ts_obj.t : ts_obj
    it = nearest_index(ts, 60.0)
    nk = imag.(GL[:, it, it])
    L = length(nk)
    ks = collect(range(-π, stop=π - 2π / L, length=L))

    label = raw"$\eta = " * η * raw",\ p_c = " * λ * raw"/a$"
    ax.plot(ks, nk; color=lambda_colors[λ], linestyle=eta_styles[η], linewidth=2.2, label=label, zorder=2)
end

for (i, ax) in enumerate(axs)
    ax.set_xlim(-π, π)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([-π, -π/2, 0, π/2, π])
    ax.set_xticklabels([raw"$-\pi$", raw"$-\pi/2$", raw"$0$", raw"$\pi/2$", raw"$\pi$"])
    if i > 5
        ax.set_xlabel(raw"$k$")
    end
    if i == 1 || i == 6
        ax.set_ylabel(raw"$n_k(t=60)$")
    end
    ax.text(0.05, 0.92, raw"$\alpha = " * alpha_order[i] * raw"$"; transform=ax.transAxes, fontsize=16)
    ax.legend(loc="lower center", fontsize=10, frameon=false, ncol=1, handlelength=2.5)
end

fig.text(0.5, 0.015, raw"Chain, $T_b = 0.1$, spectral kernel"; ha="center", va="bottom", fontsize=16)
fig.subplots_adjust(wspace=0.08, hspace=0.12, bottom=0.10, left=0.06, right=0.99, top=0.97)

outfile = joinpath(OUT_DIR, "nk_t60_chain_tb01_scan.png")
fig.savefig(outfile; dpi=220, bbox_inches="tight")
println("Saved $(relpath(outfile, ROOT))")
