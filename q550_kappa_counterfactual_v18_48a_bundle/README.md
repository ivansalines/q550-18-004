# Q550 v18.48a — kappa0 counterfactual, original RHS

Uses the original v18.47 scripts and source state, and checks the SHA-256 of the pinned original server dependencies. Compares the original kappa0 = 1/6 against kappa0 = 35*pi*n/(12*Q), with n=10, Q=550, keeping all other RHS code and numerical conditions unchanged.

## Run

Extract the original q550_kick203_hessian_kernel_compensation_v18_47_bundle alongside this directory. From this directory run:

```bash
bash run_on_server.sh 2>&1 | tee v18_48a_server.log
```

If the original v18.47 folder lives elsewhere, set `V47_ROOT=/absolute/or/relative/path` before running. BASE_ROOT defaults to `/root/elab1/hysteresis` and must contain the original SHA-pinned Q550 source files as in v18.47.

## Scientific boundary

This audit compares a **single authentic original checkpoint state**, not the 36 replay states, and does not reintegrate a perturbed Q550 orbit. Mixed Hessian directional approximations use plain coordinate directions, not the dynamically transported source/normal directions of v18.47. They answer whether the fixed-state local RHS Hessian probe changes when kappa0 changes. They do NOT establish whether the full v18.47 material compensation persists under a new trajectory. They do not validate the autonomous mathematical prescription as a physical law or supply an exact Hessian.

Outputs contain the original and modified 11D RHS, pairwise Hessian approximations at h=0.004, 0.008, 0.012, relative changes, signed products and cosine. Audit explicitly aborts on missing original pinned files and hash mismatches.
