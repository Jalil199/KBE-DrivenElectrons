using Distributed

n_workers = parse(Int, get(ENV, "JULIA_WORKERS", "39"))
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
    α                = 0.0,
    s                = 1.0,
    ωc               = 3.0,
    η                = 0.05,
    t0               = 20.0,
    ω0               = 2.2,
    σ                = 2.0,
    A                = 0.0,
    switch_on        = false,
    init_type        = :thermal,
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
    λ_q              = 1.0,
    use_bath2        = true,
    α2               = 0.5,
    ωb0_2            = 2.5,
    v_b2             = 0.0,
    η2               = 0.1,
    dispersion_type2 = :linear,
    boson_kernel2    = :spectral,
    wq_profile2      = :power_exp,
    s_q2             = 0.0,
    λ_q2             = 10.0,
)

param_sets = NamedTuple[]
seen = Set{Tuple{Float64,Float64,Float64,Float64}}()

function add_case!(param_sets, seen; scan, ωb0_2, η2=0.1, λ_q2=10.0, α2=0.5)
    key = (Float64(ωb0_2), Float64(η2), Float64(λ_q2), Float64(α2))
    key in seen && return
    if scan != :resonance && η2 == 0.1 && λ_q2 == 10.0 && α2 == 0.5
        # Covered by the concurrently running flat-bath resonance scan.
        return
    end
    push!(seen, key)
    label = Symbol("$(scan)_wb2$(ωb0_2)_eta2$(η2)_lq2$(λ_q2)_a2$(α2)")
    push!(param_sets, merge(base, (; ωb0_2=Float64(ωb0_2), η2=Float64(η2), λ_q2=Float64(λ_q2), α2=Float64(α2), label)))
end

for ωb0_2 in (2.0, 2.5, 3.0), η2 in (0.02, 0.05, 0.1, 0.2, 0.5)
    add_case!(param_sets, seen; scan=:eta2, ωb0_2, η2)
end

for ωb0_2 in (2.0, 2.5, 3.0), λ_q2 in (0.5, 1.0, 3.0, 10.0)
    add_case!(param_sets, seen; scan=:lq2, ωb0_2, λ_q2)
end

for ωb0_2 in (2.0, 2.5, 3.0), α2 in (0.1, 0.25, 0.5, 1.0)
    add_case!(param_sets, seen; scan=:a2, ωb0_2, α2)
end

println("Total unique runs: $(length(param_sets))")
for p in param_sets
    println(
        "Case $(p.label): α=$(p.α), η=$(p.η), λ_q=$(p.λ_q), Tb=$(p.Tb), ",
        "bath2=(α2=$(p.α2), ωb0_2=$(p.ωb0_2), η2=$(p.η2), λ_q2=$(p.λ_q2)), tmax=$(tmax)",
    )
end

mkpath("logs")

results = pmap(param_sets) do p
    label = p.label
    run_params = Base.structdiff(p, NamedTuple{(:label,)})
    name_p = make_name(ModelElectronBath(; run_params...); tmax)
    logfile = "logs/rice_flat_bath_paramscan_t60_$(label).log"

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
