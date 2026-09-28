using Distributed

n_workers = parse(Int, get(ENV, "JULIA_WORKERS", "8"))
addprocs(n_workers; exeflags="--project=$(Base.active_project())")
println("Workers: ", workers())

@everywhere include("./main.jl")

tmax  = 60
force = true

sweep = (
    L               = [100],
    Te              = [1.0],
    Tb              = [0.05],
    u               = [0.0],
    γ               = [1.0],
    α               = [0.25],
    s               = [1.0],
    ωc              = [10.0],
    t0              = [50.0],
    ω0              = [Float64(pi)],
    σ               = [2.0],
    A               = [0.0],
    switch_on       = [false],
    ti              = [3.0],
    to              = [20.0],
    bath_type       = [:dispersion],
    dispersion_type = [:linear],
    boson_kernel    = [:delta, :spectral],
    η               = [0.05, 0.5],
    ωA_max          = [20.0],
    dωA             = [0.01],
    ωb0             = [0.1],
    v_b             = [0.2],
    wq_profile      = [:power_exp],
    s_q             = [1.0],
    λ_q             = [1.0, 0.5],
)

keys_s  = keys(sweep)
vals_s  = values(sweep)
param_sets = vec([
    NamedTuple{keys_s}(combo)
    for combo in Iterators.product(vals_s...)
])

println("Total runs: $(length(param_sets))")

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

failed   = filter(r -> r.status == :error,   results)
skipped  = filter(r -> r.status == :skipped, results)
ok       = filter(r -> r.status == :ok,      results)
println("\n=== $(length(ok)) succeeded, $(length(skipped)) skipped, $(length(failed)) failed ($(length(results)) total) ===")
for r in failed
    println("FAILED: ", r.p, "\n  ", r.msg)
end
