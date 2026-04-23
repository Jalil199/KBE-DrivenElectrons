using Distributed

# Parallel runs need a writable Julia depot and distinct output filenames.
# In restricted environments, a safe launch pattern is:
#   JULIA_DEPOT_PATH=/tmp/julia-depot MPLCONFIGDIR=/tmp/mpl JULIA_WORKERS=2 julia --project=. run_parallel.jl
# The sweep should vary at least one parameter per job so two workers do not try
# to write the same Data/GL_*.jld2, Data/GG_*.jld2, or Data/ts_*.jld2 files.
n_workers = parse(Int, get(ENV, "JULIA_WORKERS", "10"))
addprocs(n_workers; exeflags="--project=$(Base.active_project())")
println("Workers: ", workers())

@everywhere include("./main.jl")

# ── Parameter sweep ────────────────────────────────────────────────────────────
# Put a vector of values for every parameter you want to vary.
# Single-element vectors keep that parameter fixed.
# All combinations (Cartesian product) are run in parallel.

tmax  = 60
force = true   # set true to rerun even if output files already exist

sweep = (
    # Electron / lattice parameters
    L               = [100],
    Te              = [1.0],
    Tb              = [0.1],
    u               = [0.0],
    γ               = [1.0],

    # Bath spectral-weight parameters
    α               = [0.25,1.0],
    s               = [1.0],
    ωc              = [10.0],

    # Drive parameters
    t0              = [50.0],
    ω0              = [Float64(pi)],
    σ               = [2.0],
    A               = [0.0],
    switch_on       = [false],

    # Smooth switch-on parameters
    ti              = [3.0],
    to              = [20.0],

    # Bath implementation
    bath_type       = [:dispersion],  # other option: :spectral_density
    dispersion_type = [:linear],      # other option: :sin_lattice
    boson_kernel    = [:spectral],       # other option: :spectral
    η               = [0.05,0.5],
    ωA_max          = [20.0],
    dωA             = [0.01],
    ωb0             = [0.1],
    v_b             = [0.2],

    # Momentum-weight profile
    wq_profile      = [:power_exp],   # other option: :uniform
    s_q             = [1.0],
    λ_q             = [0.2, 0.5, 1.0],
)

# ── Build parameter sets ───────────────────────────────────────────────────────
keys_s  = keys(sweep)
vals_s  = values(sweep)
param_sets = vec([
    NamedTuple{keys_s}(combo)
    for combo in Iterators.product(vals_s...)
])

println("Total runs: $(length(param_sets))")

# ── Run in parallel ────────────────────────────────────────────────────────────
results = pmap(param_sets) do p
    name_p = make_name(ModelElectronBath(; p...); tmax)
    if !force && isfile("Data/GL_$(name_p).jld2")
        println("Skipping $p (already exists)")
        return (; status=:skipped, p)
    end
    try
        println("Starting $p on worker $(myid())")
        main(; p..., tmax)
        (; status=:ok, p)
    catch e
        @error "Failed for $p" exception=e
        (; status=:error, p, msg=sprint(showerror, e))
    end
end

# ── Summary ────────────────────────────────────────────────────────────────────
failed   = filter(r -> r.status == :error,   results)
skipped  = filter(r -> r.status == :skipped, results)
ok       = filter(r -> r.status == :ok,      results)
println("\n=== $(length(ok)) succeeded, $(length(skipped)) skipped, $(length(failed)) failed ($(length(results)) total) ===")
for r in failed
    println("FAILED: ", r.p, "\n  ", r.msg)
end
