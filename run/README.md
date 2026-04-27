# Run Scripts

Batch launchers are grouped by model:

- `chain/`: single-band chain sweeps and reruns.
- `rice_mele/`: Rice-Mele sweeps and reruns.
- `archived/`: old launchers kept only for reference.

Run these scripts from the repository root unless the script explicitly changes
directory internally. The currently running resonant Rice-Mele test remains in
the repository root until that background job finishes.
