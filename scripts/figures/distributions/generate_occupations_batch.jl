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
rc("legend", fontsize=16)

const TARGET_TIMES = [0.0, 20.0, 60.0]
const ALPHA_FILTER = length(ARGS) >= 1 ? ARGS[1] : nothing
const OUTPUT_DIR = length(ARGS) >= 2 ? ARGS[2] : "occupations"

nearest_time_index(ts, t_target) = argmin(abs.(ts .- t_target))

function extract_field(name::AbstractString, key::AbstractString)
    m = match(Regex("$(key)([^_]+)"), name)
    return m === nothing ? missing : m.captures[1]
end

function pretty_dataset_label(name::AbstractString)
    kernel = extract_field(name, "linear_")
    η = extract_field(name, "η")
    λ_q = extract_field(name, "λ_q")
    return "kernel = $(kernel),  η = $(η),  λ_q = $(λ_q)"
end

mkpath(OUTPUT_DIR)

function matches_chain_sweep(path::AbstractString)
    base = basename(path)
    return occursin("GL_L100_", base) &&
           occursin("_tmax60", base) &&
           occursin("_switch0_", base) &&
           occursin("_linear_", base) &&
           (ALPHA_FILTER === nothing || occursin("α$(ALPHA_FILTER)_", base))
end

gl_files = sort(filter(matches_chain_sweep, readdir("Data"; join=true)))

println("Found $(length(gl_files)) sweep files")

for gl_path in gl_files
    base = basename(gl_path)
    name = replace(base, r"^GL_" => "")
    name = replace(name, r"\.jld2$" => "")
    ts_path = joinpath("Data", "ts_$(name).jld2")

    GL = load(gl_path, "GL")
    ts = load(ts_path, "sol").t

    L = size(GL.data, 1)
    Δk = 2π / L
    ks = collect(range(-π, stop=π - Δk, length=L))

    indices = [nearest_time_index(ts, t) for t in TARGET_TIMES]
    actual_times = ts[indices]
    nk_curves = [imag.(GL.data[:, it, it]) for it in indices]

    fig, ax = subplots(figsize=(8, 6))
    colors = ["red", "blue", "gold"]
    styles = ["-", "-.", "--"]

    for (i, nk) in enumerate(nk_curves)
        ax.plot(
            ks,
            nk;
            color=colors[i],
            linestyle=styles[i],
            linewidth=2,
            label="\$t = $(round(actual_times[i], digits=2))\$",
        )
    end

    ax.set_xlabel(raw"$k$")
    ax.set_ylabel(raw"$n_k(t)$")
    ax.set_xlim(-π, π)
    ax.set_xticks([-π, -π / 2, 0, π / 2, π])
    ax.set_xticklabels([L"-\pi", L"-\pi/2", L"0", L"\pi/2", L"\pi"])
    ax.legend(frameon=false, loc="best")

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
    out_name = replace(name, "GL_" => "")
    out_name = "occupation_" * replace(out_name, ".jld2" => "") * ".png"
    out_path = joinpath(OUTPUT_DIR, out_name)
    fig.savefig(out_path; dpi=300, bbox_inches="tight")
    close(fig)

    println("Saved $(out_path)")
end
