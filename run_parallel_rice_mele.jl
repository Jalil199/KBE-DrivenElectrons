using Distributed

n_workers = parse(Int, get(ENV, "JULIA_WORKERS", "4"))
addprocs(n_workers)
println("Workers: ", workers())

@everywhere include("./main-rice_mele.jl")

# ── Parameter sweep ────────────────────────────────────────────────────────────
# Put a vector of values for every parameter you want to vary.
# Single-element vectors keep that parameter fixed.
# All combinations (Cartesian product) are run in parallel.

tmax  = 40
force = false   # set true to rerun even if output files already exist

sweep = (
    L               = [80],
    bath_type       = [:dispersion],
    dispersion_type = [:sin_lattice],
    boson_kernel    = [:delta],
    η               = [0.0],
    wq_profile      = [:power_exp],
    s_q             = [1.0],
    λ_q             = [1.0],
    Te              = [0.2, 0.5, 1.0],
    Tb              = [0.4, 0.8],
    α               = [0.3],
    t1              = [-1.0],
    t2              = [-0.6],
    Δ               = [2.0],
    A               = [0.5],
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
