using JLD2
using PyPlot
using LaTeXStrings

const HAS_LATEX = Sys.which("latex") !== nothing
rc("text", usetex=HAS_LATEX)
rc("font", family="serif")
if HAS_LATEX
    rc("text.latex", preamble=raw"\usepackage{amsmath}")
end
rc("font", size=18)
rc("axes", labelsize=22, titlesize=20)
rc("xtick", labelsize=18)
rc("ytick", labelsize=18)
rc("legend", fontsize=12)

const TARGET_TIMES = [0.0, 20.0, 60.0]
const ALPHA_FILTER = length(ARGS) >= 1 ? ARGS[1] : nothing
const OUTPUT_DIR = length(ARGS) >= 2 ? ARGS[2] : "occupations_rice"

nearest_time_index(ts, t_target) = argmin(abs.(ts .- t_target))

function extract_field(name::AbstractString, key::AbstractString)
    m = match(Regex("$(key)([^_]+)"), name)
    return m === nothing ? missing : m.captures[1]
end

function extract_kernel(name::AbstractString)
    m = match(r"_(linear|sin_lattice)_(delta|spectral)_η", name)
    return m === nothing ? missing : m.captures[2]
end

function pretty_dataset_label(name::AbstractString)
    kernel = extract_kernel(name)
    η = extract_field(name, "η")
    λ_q = extract_field(name, "λ_q")
    return "kernel = $(kernel),  η = $(η),  λ_q = $(λ_q)"
end

mkpath(OUTPUT_DIR)

function matches_rice_sweep(path::AbstractString)
    base = basename(path)
    return occursin("GL_L80_", base) &&
           occursin("_tmax60", base) &&
           occursin("_switch0_", base) &&
           occursin("_linear_", base) &&
           (ALPHA_FILTER === nothing || occursin("α$(ALPHA_FILTER)_", base))
end

gl_files = sort(filter(matches_rice_sweep, readdir("Data"; join=true)))

println("Found $(length(gl_files)) Rice-Mele sweep files")

for gl_path in gl_files
    base = basename(gl_path)
    name = replace(base, r"^GL_" => "")
    name = replace(name, r"\.jld2$" => "")
    ts_path = joinpath("Data", "ts_$(name).jld2")

    GL = load(gl_path, "GL")
    ts = load(ts_path, "sol").t

    L = size(GL.data, 3)
    Δk = 2π / L
    ks = collect(range(-π, stop=π - Δk, length=L))

    indices = [nearest_time_index(ts, t) for t in TARGET_TIMES]
    actual_times = ts[indices]

    occ_A = [imag.(GL.data[1, 1, :, it, it]) for it in indices]
    occ_B = [imag.(GL.data[2, 2, :, it, it]) for it in indices]
    occ_tot = [imag.(GL.data[1, 1, :, it, it] .+ GL.data[2, 2, :, it, it]) for it in indices]

    fig, ax = subplots(figsize=(9, 6.5))

    time_styles = ["-", "-.", "--"]
    labels_A = ["A, t=$(round(t, digits=2))" for t in actual_times]
    labels_B = ["B, t=$(round(t, digits=2))" for t in actual_times]
    labels_T = ["A+B, t=$(round(t, digits=2))" for t in actual_times]

    for i in eachindex(indices)
        ax.plot(ks, occ_A[i]; color="blue", linewidth=2, linestyle=time_styles[i], label=labels_A[i])
        ax.plot(ks, occ_B[i]; color="red", linewidth=2, linestyle=time_styles[i], label=labels_B[i])
        ax.plot(ks, occ_tot[i]; color="green", linewidth=2, linestyle=time_styles[i], label=labels_T[i])
    end

    ax.set_xlabel(raw"$k$")
    ax.set_ylabel(raw"$n_{\mathrm{occ}}(k,t)$")
    ax.set_xlim(-π, π)
    ax.set_xticks([-π, -π / 2, 0, π / 2, π])
    ax.set_xticklabels([L"-\pi", L"-\pi/2", L"0", L"\pi/2", L"\pi"])
    ax.legend(frameon=false, loc="upper right", ncol=1)

    textbox = pretty_dataset_label(name) * "\n" *
              "times used = [" * join(string.(round.(actual_times; digits=2)), ", ") * "]"
    ax.text(
        0.03, 0.97, textbox;
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox=Dict("boxstyle" => "round", "facecolor" => "white", "alpha" => 0.9, "edgecolor" => "0.7"),
    )

    fig.tight_layout()
    out_name = "occupation_rice_" * name * ".png"
    out_path = joinpath(OUTPUT_DIR, out_name)
    fig.savefig(out_path; dpi=300, bbox_inches="tight")
    close(fig)

    println("Saved $(out_path)")
end
