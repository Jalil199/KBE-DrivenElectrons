using Distributed

n_workers = parse(Int, get(ENV, "JULIA_WORKERS", "36"))
addprocs(n_workers; exeflags="--project=$(Base.active_project())")
println("Workers: ", workers())

@everywhere include("./main-rice_mele.jl")

tmax = 60
force = false

base = (
    L                = 80,
    t1               = -1.0,
    t2               = -0.8,
    Δ                = 2.0,
    Te               = 1.0,
    Tb               = 0.1,
    s                = 1.0,
    ωc               = 3.0,
    t0               = 20.0,
    ω0               = 2.2,
    σ                = 2.0,
    A                = 0.0,
    switch_on        = false,
    ti               = 0.5,
    to               = 5.0,
    bath_type        = :dispersion,
    dispersion_type  = :linear,
    boson_kernel     = :spectral,
    ωA_max           = 20.0,
    dωA              = 0.01,
    ωb0              = 0.1,
    v_b              = 0.2,
    wq_profile       = :power_exp,
    s_q              = 1.0,
    use_bath2        = false,
)

param_sets = NamedTuple[]
seen = Set{Tuple{Symbol,Float64,Float64,Float64}}()

function add_case!(param_sets, seen; init_type, α, η, λ_q)
    key = (Symbol(init_type), Float64(α), Float64(η), Float64(λ_q))
    key in seen && return
    push!(seen, key)
    label = Symbol("$(init_type)_a$(α)_eta$(η)_lq$(λ_q)_bath1only")
    push!(param_sets, merge(base, (; init_type=Symbol(init_type), α=Float64(α), η=Float64(η), λ_q=Float64(λ_q), label)))
end

for init_type in (:thermal, :upper_full), α in (1.0, 2.0, 4.0), η in (0.05, 0.5), λ_q in (0.2, 0.5, 1.0)
    add_case!(param_sets, seen; init_type, α, η, λ_q)
end

println("Total unique runs: $(length(param_sets))")
for p in param_sets
    println(
        "Case $(p.label): init=$(p.init_type), bath1=(α=$(p.α), η=$(p.η), λ_q=$(p.λ_q)), Tb=$(p.Tb), ",
        "use_bath2=$(p.use_bath2), tmax=$(tmax)",
    )
end

mkpath("logs")

results = pmap(param_sets) do p
    label = p.label
    run_params = Base.structdiff(p, NamedTuple{(:label,)})
    name_p = make_name(ModelElectronBath(; run_params...); tmax)
    logfile = "logs/rice_bath1_only_control_t60_$(label).log"

    if !force && isfile("Data_rice/GL_$(name_p).jld2")
        open(logfile, "w") do io
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

