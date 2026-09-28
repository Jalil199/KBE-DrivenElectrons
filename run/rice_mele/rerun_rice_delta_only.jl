using Distributed

n_workers = parse(Int, get(ENV, "JULIA_WORKERS", "4"))
addprocs(n_workers; exeflags="--project=$(Base.active_project())")
println("Workers: ", workers())

@everywhere include("./main-rice_mele.jl")

tmax  = 60
force = true

sweep = (
    L               = [80],
    t1              = [-1.0],
    t2              = [-0.8],
    Δ               = [2.0],
    Te              = [1.0],
    Tb              = [0.5],
    α               = [0.5],
    s               = [1.0],
    ωc              = [3.0],
    t0              = [20.0],
    ω0              = [2.2],
    σ               = [2.0],
    A               = [0.0],
    switch_on       = [false],
    ti              = [0.5],
    to              = [5.0],
    bath_type       = [:dispersion],
    dispersion_type = [:linear],
    boson_kernel    = [:delta],
    η               = [0.05, 0.5],
    ωA_max          = [20.0],
    dωA             = [0.01],
    ωb0             = [0.1],
    v_b             = [0.2],
    wq_profile      = [:power_exp],
    s_q             = [1.0],
    λ_q             = [1.0, 0.5],
)

keys_s = keys(sweep)
vals_s = values(sweep)
param_sets = vec([
    NamedTuple{keys_s}(combo)
    for combo in Iterators.product(vals_s...)
])

println("Total runs: $(length(param_sets))")

results = pmap(param_sets) do p
    try
        println("Starting $p on worker $(myid())")
        main(; p..., tmax)
        (; status=:ok, p)
    catch e
        @error "Failed for $p" exception=e
        (; status=:error, p, msg=sprint(showerror, e))
    end
end

failed = filter(r -> r.status == :error, results)
ok = filter(r -> r.status == :ok, results)
println("\n=== $(length(ok)) succeeded, $(length(failed)) failed ($(length(results)) total) ===")
for r in failed
    println("FAILED: ", r.p, "\n  ", r.msg)
end
