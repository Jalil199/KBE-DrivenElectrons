using Distributed

n_workers = parse(Int, get(ENV, "JULIA_WORKERS", "4"))
addprocs(n_workers; exeflags="--project=$(Base.active_project())")
println("Workers: ", workers())

const _MAIN = normpath(joinpath(@__DIR__, "../../main-rice_mele.jl"))
@everywhere include($_MAIN)

tmax  = 60
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
    η               = 0.5,
    ωA_max          = 20.0,
    dωA             = 0.01,
    v_b             = 0.0,
    wq_profile      = :power_exp,
    s_q             = 1.0,
    λ_q             = 10.0,
)

ωb0_vals   = collect(0.2:0.2:4.0)
init_types = [:thermal, :upper_full]

param_sets = NamedTuple[]
for init in init_types, ωb0 in ωb0_vals
    push!(param_sets, merge(base, (; ωb0, init_type=init)))
end

println("Total runs: $(length(param_sets))")
for p in param_sets
    println("  init=$(p.init_type)  ωb0=$(p.ωb0)")
end

mkpath("logs")

results = pmap(param_sets) do p
    label   = "flat_eta05_lq10_wb$(p.ωb0)_$(p.init_type)"
    logfile = "logs/$(label).log"
    name_p  = make_name(ModelElectronBath(; p...); tmax)

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
