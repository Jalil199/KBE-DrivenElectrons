using Distributed

n_workers = parse(Int, get(ENV, "JULIA_WORKERS", "15"))
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
    ωA_max          = 20.0,
    dωA             = 0.01,
    wq_profile      = :power_exp,
    s_q             = 1.0,
    v_b             = 0.0,
)

param_sets = NamedTuple[]

for ωb0 in (2.2, 2.5, 2.8, 3.0)
    for λ_q in (1.0, 3.0, 10.0)
        push!(param_sets, merge(base, (; η=0.05, ωb0, λ_q, label=Symbol("w$(ωb0)_l$(λ_q)_eta0.05"))))
    end
end

for η in (0.02, 0.1, 0.2)
    push!(param_sets, merge(base, (; η, ωb0=2.5, λ_q=10.0, label=Symbol("w2.5_l10.0_eta$(η)"))))
end

println("Total runs: $(length(param_sets))")
for p in param_sets
    println("Case $(p.label): ωb0=$(p.ωb0), λ_q=$(p.λ_q), η=$(p.η), v_b=$(p.v_b)")
end

mkpath("logs")

results = pmap(param_sets) do p
    label = p.label
    run_params = Base.structdiff(p, NamedTuple{(:label,)})
    name_p = make_name(ModelElectronBath(; run_params...); tmax)
    logfile = "logs/rice_resonance_cutoff_t30_$(label).log"

    if !force && isfile("Data_rice/GL_$(name_p).jld2")
        open(logfile, "a") do io
            println(io, "Skipping $(label): already exists")
        end
        return (; status=:skipped, label, p=run_params)
    end

    try
        open(logfile, "w") do io
            redirect_stdout(io) do
                redirect_stderr(io) do
                    println("Starting $(label) on worker $(myid())")
                    println("Parameters: ", run_params)
                    flush(stdout)
                    main(; run_params..., tmax)
                    println("Finished $(label)")
                    flush(stdout)
                end
            end
        end
        (; status=:ok, label, p=run_params)
    catch e
        open(logfile, "a") do io
            println(io, "FAILED $(label): ", sprint(showerror, e))
            showerror(io, e, catch_backtrace())
            println(io)
        end
        (; status=:error, label, p=run_params, msg=sprint(showerror, e))
    end
end

failed  = filter(r -> r.status == :error,   results)
skipped = filter(r -> r.status == :skipped, results)
ok      = filter(r -> r.status == :ok,      results)
println("\n=== $(length(ok)) succeeded, $(length(skipped)) skipped, $(length(failed)) failed ($(length(results)) total) ===")
for r in failed
    println("FAILED $(r.label): ", r.msg)
end
