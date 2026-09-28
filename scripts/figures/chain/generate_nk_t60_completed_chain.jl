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

if isempty(files)
    error("No completed chain GL files found in $DATA_DIR")
end

plt.rc("font", family="serif", size=22)
plt.rc("axes", linewidth=2.0)
plt.rc("xtick.major", width=2.0, size=8)
plt.rc("ytick.major", width=2.0, size=8)
plt.rc("xtick", direction="in")
plt.rc("ytick", direction="in")

fig, axs = subplots(1, 2; figsize=(15.5, 5.6), sharey=true)

alpha_order = ["0.25", "1.0"]
lambda_colors = Dict("0.2" => "#1f77b4", "0.5" => "#d1495b", "1.0" => "#2a9d8f")
eta_styles = Dict("0.05" => "-", "0.5" => "--")

for (ia, alpha_str) in enumerate(alpha_order)
    ax = axs[ia]
    for f in files
        base = basename(f)
        α = string(extract_field(base, "α"))
        η = string(extract_field(base, "η"))
        λ = string(extract_field(base, "λ_q"))
        α == alpha_str || continue

        GL_obj = load(f, "GL")
        GL = hasproperty(GL_obj, :data) ? GL_obj.data : GL_obj
        ts_obj = load(replace(f, "GL_" => "ts_"), "sol")
        ts = hasproperty(ts_obj, :t) ? ts_obj.t : ts_obj
        it = nearest_index(ts, 60.0)
        nk = imag.(GL[:, it, it])
        L = length(nk)
        ks = collect(range(-π, stop=π - 2π / L, length=L))

        label = "\$\\eta = $(η),\\ p_c = $(λ)/a\$"
        ax.plot(ks, nk; color=lambda_colors[λ], linestyle=eta_styles[η], linewidth=2.6, label=label)
    end

    ax.set_xlim(-π, π)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([-π, -π/2, 0, π/2, π])
    ax.set_xticklabels([raw"$-\pi$", raw"$-\pi/2$", raw"$0$", raw"$\pi/2$", raw"$\pi$"])
    ax.set_xlabel(raw"$k$")
    ax.text(0.05, 0.92, "\$\\alpha = $(alpha_str)\$"; transform=ax.transAxes, fontsize=24)
    if ia == 1
        ax.set_ylabel(raw"$n_k(t=60)$")
    end
    ax.legend(loc="lower center", fontsize=14, frameon=false, ncol=1, handlelength=2.8)
end

fig.subplots_adjust(wspace=0.1, bottom=0.15, left=0.08, right=0.98, top=0.97)

outfile = joinpath(OUT_DIR, "nk_t60_completed_chain_scan.png")
fig.savefig(outfile; dpi=220, bbox_inches="tight")
println("Saved $(relpath(outfile, ROOT))")
