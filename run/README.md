# Run Scripts

Batch launchers are grouped by model:

- `chain/`: single-band chain sweeps and reruns (`main.jl`).
- `rice_mele/`: Rice-Mele sweeps and reruns (`main-rice_mele.jl`).
- `archived/`: old launchers kept only for reference.

Every `.jl` launcher changes directory to the repository root before adding
workers, so it can be started from anywhere and outputs still land in
`Data/`, `Data_rice/` and `logs/`:

```bash
JULIA_WORKERS=40 nohup julia --project=. run/rice_mele/run_parallel_rice_mele_two_bath_lq_fine_scan_t60.jl \
    > logs/run_two_bath_lq_fine.log 2>&1 &
```

The `run_rice_*.sh` wrappers in `rice_mele/` set `JULIA_WORKERS` and launch the
matching `run_parallel_rice_mele_*.jl`.

## Rice-Mele campaign (Apr–May 2026)

| Launcher | What it scans |
|---|---|
| `run_parallel_rice_mele_gap_scan_t30.jl` | ωb0 across the gap, tmax=30 |
| `run_parallel_rice_mele_resonance_cutoff_scan_t30.jl` | resonance / cutoff, tmax=30 |
| `run_parallel_rice_mele_resonant_test.jl` | resonant bath tests (ωb0=4, 5) |
| `run_parallel_rice_gap_threshold_scan.jl` | gap-threshold scan (3 bath configs) |
| `run_parallel_rice_flat_eta05_lq10.jl` | flat bath, η=0.5, thermal vs `upper_full` |
| `run_parallel_rice_mele_flat_bath_*_t60.jl` | flat-bath parameter / resonance scans |
| `run_parallel_rice_mele_best_flat_bath_with_bath1_scan_t60.jl` | best second bath + varying bath 1 |
| `run_parallel_rice_mele_bath1_only_control_scan_t60.jl` | same cases without the second bath (control) |
| `run_parallel_rice_mele_two_bath_thermal_scan_t60.jl` | two baths, thermal initial state |
| `run_parallel_rice_mele_two_bath_lq_fine_scan_t60.jl` | fine λ_q scan (0.2–1.5), two baths |
