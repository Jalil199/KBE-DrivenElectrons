using Distributed

n_workers = parse(Int, get(ENV, "JULIA_WORKERS", "3"))
addprocs(n_workers; exeflags="--project=$(Base.active_project())")
println("Workers: ", workers())

@everywhere include("./main-rice_mele.jl")

tmax = 30
force = true

base = (
    L               = 80,
    t1              = -1.0,
    t2              = -0.8,
    Δ               = 2.0,
    Te              = 1.0,
    Tb              = 0.5,
    α               = 1.0,
    s               = 1.0,
    ωc              = 3.0,
    t0              = 20.0,
    ω0              = 2.2,
    σ               = 2.0,
    A               = 0.0,
    switch_on       = false,
    ti              = 0.5,
    to              = 5.0,
    bath_type       = :dispersion,
    dispersion_type = :linear,
    boson_kernel    = :spectral,
    η               = 0.05,
    ωA_max          = 20.0,
    dωA             = 0.01,
    wq_profile      = :power_exp,
    s_q             = 1.0,
    λ_q             = 1.0,
)

baths = [
    (; label = :current,   ωb0 = 0.1, v_b = 0.2),
    (; label = :res_gap,   ωb0 = 4.0, v_b = 0.0),
    (; label = :res_upper, ωb0 = 5.0, v_b = 0.0),
]

param_sets = [merge(base, (; ωb0=b.ωb0, v_b=b.v_b)) for b in baths]

println("Total runs: $(length(param_sets))")
for (b, p) in zip(baths, param_sets)
    println("Case $(b.label): ωb0=$(p.ωb0), v_b=$(p.v_b)")
end

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

failed  = filter(r -> r.status == :error,   results)
skipped = filter(r -> r.status == :skipped, results)
ok      = filter(r -> r.status == :ok,      results)
println("\n=== $(length(ok)) succeeded, $(length(skipped)) skipped, $(length(failed)) failed ($(length(results)) total) ===")
for r in failed
    println("FAILED: ", r.p, "\n  ", r.msg)
end
