using Distributed

n_workers = parse(Int, get(ENV, "JULIA_WORKERS", "6"))
addprocs(n_workers; exeflags="--project=$(Base.active_project())")
println("Workers: ", workers())

const _MAIN = normpath(joinpath(@__DIR__, "../../main-rice_mele.jl"))
@everywhere include($_MAIN)

tmax  = 30
force = false

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
    ωA_max          = 20.0,
    dωA             = 0.01,
    v_b             = 0.0,
    wq_profile      = :power_exp,
    s_q             = 1.0,
    ωb0             = 0.1,   # placeholder, overridden below
)

# Gap minimum ≈ 2.04 — sweep crosses the threshold
ωb0_vals = [0.5, 1.0, 1.5, 2.0, 2.04, 2.2, 2.5, 2.8, 3.0]

bath_configs = [
    (; λ_q=3.0,  η=0.05),
    (; λ_q=10.0, η=0.05),
    (; λ_q=10.0, η=0.2),
]

param_sets = NamedTuple[]
for bc in bath_configs, ωb0 in ωb0_vals
    push!(param_sets, merge(base, (; ωb0, bc.λ_q, bc.η)))
end

println("Total combinations: $(length(param_sets))")
for p in param_sets
    println("  ωb0=$(p.ωb0)  λ_q=$(p.λ_q)  η=$(p.η)")
end

mkpath("logs")

results = pmap(param_sets) do p
    name_p  = make_name(ModelElectronBath(; p...); tmax)
    label   = "wb$(p.ωb0)_lq$(p.λ_q)_eta$(p.η)"
    logfile = "logs/rice_gap_threshold_$(label).log"

    if !force && isfile("Data_rice/GL_$(name_p).jld2")
        println("Skipping $(label) (already exists)")
        return (; status=:skipped, label, p)
    end

    try
        open(logfile, "w") do io
            redirect_stdout(io) do
                redirect_stderr(io) do
                    println("Starting $(label) on worker $(myid())")
                    flush(stdout)
                    main(; p..., tmax)
                    println("Finished $(label)")
                    flush(stdout)
                end
            end
        end
        (; status=:ok, label, p)
    catch e
        open(logfile, "a") do io
            println(io, "FAILED $(label): ", sprint(showerror, e))
            showerror(io, e, catch_backtrace())
        end
        (; status=:error, label, p, msg=sprint(showerror, e))
    end
end

failed  = filter(r -> r.status == :error,   results)
skipped = filter(r -> r.status == :skipped, results)
ok      = filter(r -> r.status == :ok,      results)
println("\n=== $(length(ok)) ok, $(length(skipped)) skipped, $(length(failed)) failed ($(length(results)) total) ===")
for r in failed
    println("FAILED $(r.label): ", r.msg)
end
