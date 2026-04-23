using Distributed

n_workers = parse(Int, get(ENV, "JULIA_WORKERS", "8"))
addprocs(n_workers; exeflags="--project=$(Base.active_project())")
println("Workers: ", workers())

@everywhere include("./main-rice_mele.jl")

# ── Parameter sweep ────────────────────────────────────────────────────────────
# Put a vector of values for every parameter you want to vary.
# Single-element vectors keep that parameter fixed.
# All combinations (Cartesian product) are run in parallel.

tmax  = 60
force = true   # set true to rerun even if output files already exist

sweep = (
    # Rice-Mele lattice parameters
    L               = [80],
    t1              = [-1.0],
    t2              = [-0.8],
    Δ               = [2.0],

    # Temperatures: choose Te > Tb to study thermalization into a colder bath
    Te              = [1.0],
    Tb              = [0.1],

    # Bath spectral-weight parameters
    α               = [0.25,1.0],
    s               = [1.0],
    ωc              = [3.0],

    # Drive parameters: set A = 0.0 to study thermalization without light
    t0              = [20.0],
    ω0              = [2.2],
    σ               = [2.0],
    A               = [0.0],
    switch_on       = [false],

    # Smooth switch-on parameters
    ti              = [0.5],
    to              = [5.0],

    # Bath implementation
    bath_type       = [:dispersion],   # other option: :spectral_density
    dispersion_type = [:linear],  # other option: :linear
    boson_kernel    = [:spectral],        # other option: :spectral
    η               = [0.05,0.5],
    ωA_max          = [20.0],
    dωA             = [0.01],
    ωb0             = [0.1],
    v_b             = [0.2],

    # Momentum-weight profile
    wq_profile      = [:power_exp],      # other option: :power_exp
    s_q             = [1.0],
    λ_q             = [0.2,0.5,1.0],
)

# ── Build parameter sets ───────────────────────────────────────────────────────
keys_s     = keys(sweep)
vals_s     = values(sweep)
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
failed  = filter(r -> r.status == :error,   results)
skipped = filter(r -> r.status == :skipped, results)
ok      = filter(r -> r.status == :ok,      results)
println("\n=== $(length(ok)) succeeded, $(length(skipped)) skipped, $(length(failed)) failed ($(length(results)) total) ===")
for r in failed
    println("FAILED: ", r.p, "\n  ", r.msg)
end
