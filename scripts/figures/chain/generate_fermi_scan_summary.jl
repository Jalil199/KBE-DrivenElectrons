using JLD2
using PyPlot
using LaTeXStrings
using Statistics
using Printf

const HAS_LATEX = Sys.which("latex") !== nothing
rc("text", usetex=HAS_LATEX)
rc("font", family="serif")
if HAS_LATEX
    rc("text.latex", preamble=raw"\usepackage{amsmath}")
end
rc("font", size=15)
rc("axes", labelsize=16, titlesize=16)
rc("xtick", labelsize=13)
rc("ytick", labelsize=13)
rc("legend", fontsize=11)

const OUTPUT_DIR = "distributions"
mkpath(OUTPUT_DIR)

fermi(ϵ, T, μ) = 1 / (exp((ϵ - μ) / T) + 1)

function analyze_occ_file(path)
    d = load(path)
    nk = d["nk_t"][:, end]
    ks = d["ks"]
    p = d["params"]
    ϵs = p.u .- 2p.γ .* cos.(ks)

    Tgrid = collect(0.02:0.02:1.50)
    μgrid = collect(-3.0:0.05:3.0)
    best_rmse = Inf
    best_T = NaN
    best_μ = NaN
    for T in Tgrid, μ in μgrid
        ff = fermi.(ϵs, T, μ)
        rmse = sqrt(mean((nk .- ff).^2))
        if rmse < best_rmse
            best_rmse = rmse
            best_T = T
            best_μ = μ
        end
    end

    f_init = fermi.(ϵs, p.Te, 0.0)
    f_bath = fermi.(ϵs, p.Tb, 0.0)
    rmse_init = sqrt(mean((nk .- f_init).^2))
    rmse_bath = sqrt(mean((nk .- f_bath).^2))

    return (
        α = p.α,
        Tb = p.Tb,
        λ = p.λ_q,
        Tbest = best_T,
        μbest = best_μ,
        rmse_best = best_rmse,
        rmse_init = rmse_init,
        rmse_bath = rmse_bath,
        delta = rmse_bath - rmse_init,
    )
end

function build_grid(rows, λ_target, field, αs, Tbs)
    grid = fill(NaN, length(Tbs), length(αs))
    for (iy, Tb) in enumerate(Tbs), (ix, α) in enumerate(αs)
        idx = findfirst(r -> isapprox(r.λ, λ_target; atol=1e-9) &&
                              isapprox(r.α, α; atol=1e-9) &&
                              isapprox(r.Tb, Tb; atol=1e-9), rows)
        if idx !== nothing
            grid[iy, ix] = getfield(rows[idx], field)
        end
    end
    grid
end

function annotate_heatmap!(ax, xs, ys, grid; fmt="%.2f", color="black", fontsize=9)
    formatter = Printf.Format(fmt)
    for (iy, y) in enumerate(ys), (ix, x) in enumerate(xs)
        val = grid[iy, ix]
        isnan(val) && continue
        ax.text(x, y, Printf.format(formatter, val); ha="center", va="center", color=color, fontsize=fontsize)
    end
end

files = sort(filter(f -> occursin(r"^occ_L100_.*linear_spectral_η0.5.*\.jld2$", basename(f)),
    readdir("Data"; join=true)))

isempty(files) && error("No se encontraron archivos occ_L100 del scan spectral η=0.5.")

rows = map(analyze_occ_file, files)
sort!(rows, by = r -> (r.λ, r.Tb, r.α))

αs = sort(unique(r.α for r in rows))
Tbs = sort(unique(r.Tb for r in rows))
λs = sort(unique(r.λ for r in rows))

length(λs) == 2 || error("Se esperaban exactamente dos λ_q en el scan.")

T_grid_1 = build_grid(rows, λs[1], :Tbest, αs, Tbs)
T_grid_2 = build_grid(rows, λs[2], :Tbest, αs, Tbs)
Δ_grid_1 = build_grid(rows, λs[1], :delta, αs, Tbs)
Δ_grid_2 = build_grid(rows, λs[2], :delta, αs, Tbs)

vT_min = minimum(vcat(vec(T_grid_1), vec(T_grid_2)))
vT_max = maximum(vcat(vec(T_grid_1), vec(T_grid_2)))
vΔ = maximum(abs, vcat(vec(Δ_grid_1), vec(Δ_grid_2)))

fig, axs = subplots(2, 2; figsize=(10.8, 8.8), sharex=true, sharey=true)

extent = [minimum(αs), maximum(αs), minimum(Tbs), maximum(Tbs)]
cmap_T = plt.get_cmap("magma")
cmap_Δ = plt.get_cmap("RdBu_r")

imT1 = axs[1, 1].imshow(T_grid_1; origin="lower", aspect="auto", extent=extent,
    interpolation="nearest", cmap=cmap_T, vmin=vT_min, vmax=vT_max)
imT2 = axs[1, 2].imshow(T_grid_2; origin="lower", aspect="auto", extent=extent,
    interpolation="nearest", cmap=cmap_T, vmin=vT_min, vmax=vT_max)
imΔ1 = axs[2, 1].imshow(Δ_grid_1; origin="lower", aspect="auto", extent=extent,
    interpolation="nearest", cmap=cmap_Δ, vmin=-vΔ, vmax=vΔ)
imΔ2 = axs[2, 2].imshow(Δ_grid_2; origin="lower", aspect="auto", extent=extent,
    interpolation="nearest", cmap=cmap_Δ, vmin=-vΔ, vmax=vΔ)

for ax in axs
    ax.set_xticks(αs)
    ax.set_yticks(Tbs)
    ax.set_xlim(minimum(αs), maximum(αs))
    ax.set_ylim(minimum(Tbs), maximum(Tbs))
    ax.tick_params(axis="both", which="both", direction="out", length=4, width=1)
    for spine in ax.spines.values()
        spine.set_linewidth(1.0)
    end
end

for ax in axs[2, :]
    ax.set_xlabel(raw"$\alpha$")
end
for ax in axs[:, 1]
    ax.set_ylabel(raw"$T_B$")
end

axs[1, 1].text(0.04, 0.92, "\$\\lambda_q = $(λs[1])\$"; transform=axs[1, 1].transAxes, color="white", fontsize=11)
axs[1, 2].text(0.04, 0.92, "\$\\lambda_q = $(λs[2])\$"; transform=axs[1, 2].transAxes, color="white", fontsize=11)
axs[2, 1].text(0.04, 0.92, "\$\\lambda_q = $(λs[1])\$"; transform=axs[2, 1].transAxes, color="black", fontsize=11)
axs[2, 2].text(0.04, 0.92, "\$\\lambda_q = $(λs[2])\$"; transform=axs[2, 2].transAxes, color="black", fontsize=11)

annotate_heatmap!(axs[1, 1], αs, Tbs, T_grid_1; fmt="%.2f", color="white", fontsize=8)
annotate_heatmap!(axs[1, 2], αs, Tbs, T_grid_2; fmt="%.2f", color="white", fontsize=8)
annotate_heatmap!(axs[2, 1], αs, Tbs, Δ_grid_1; fmt="%.2f", color="black", fontsize=8)
annotate_heatmap!(axs[2, 2], αs, Tbs, Δ_grid_2; fmt="%.2f", color="black", fontsize=8)

caxT = fig.add_axes([0.12, 0.94, 0.35, 0.018])
cbT = fig.colorbar(imT1, cax=caxT, orientation="horizontal")
cbT.ax.xaxis.set_ticks_position("top")
cbT.ax.xaxis.set_label_position("top")
cbT.set_label(raw"$T_{\mathrm{eff}}$ from best Fermi fit")

caxΔ = fig.add_axes([0.57, 0.94, 0.31, 0.018])
cbΔ = fig.colorbar(imΔ1, cax=caxΔ, orientation="horizontal")
cbΔ.ax.xaxis.set_ticks_position("top")
cbΔ.ax.xaxis.set_label_position("top")
cbΔ.set_label(raw"$\Delta = \mathrm{RMSE}_{\mathrm{bath}} - \mathrm{RMSE}_{\mathrm{init}}$")

fig.text(0.5, 0.905, "Spectral scan, " * (HAS_LATEX ? raw"$\eta = 0.5$" : "η = 0.5"); ha="center", va="top", fontsize=13)
fig.text(0.5, 0.02,
    "Top: best-fit effective Fermi temperature. Bottom: positive Δ means closer to initial Fermi than to bath Fermi.";
    ha="center", va="bottom", fontsize=11)

fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.90])

out_path = joinpath(OUTPUT_DIR, "fermi_scan_summary_spectral_eta0.5.png")
fig.savefig(out_path; dpi=300, bbox_inches="tight")
close(fig)

println("Saved $(out_path)")
