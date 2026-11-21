# Q550 v18.48c — Hessian Stencil Convergence & Compensation Defect Audit

## Scope

A genuinely numerical audit using the same Q550 original RHS and SHA-pinned dependencies selected by v18.47, plus the authentic 12-case / 36-snapshot replay of v18.48b. It tests the original `mixed_compensated` and independent `polarization_rhs` stencils at both `kappa0 = 1/6` and `kappa_geom = 35*pi*10/(12*550)`.

Unlike v18.48b, the mixed-minus-polarization defect is computed **separately** for each state and each kappa, including the restarted trajectories. It is never copied from the fixed-state mode. The 8 step sizes in normalized chart coordinates are 0.024, 0.016, 0.012, 0.008, 0.006, 0.004, 0.003, 0.002. This broad scan is diagnostic: the smallest h values may be dominated by roundoff.

## Run on server

Extract the ZIP beside the existing unzipped v18.47 parent in `/root/elab1/hysteresis` (and with the SHA-pinned files previously mapped there):

```bash
cd /root/elab1/hysteresis/q550_kappa_hessian_stencil_v18_48c_bundle
bash run_on_server.sh 2>&1 | tee v18_48c_server.log
```

Uses **python3.13 by default**, reflecting the working environment correction in the v18.48b sendback. Override with `Q550_PYTHON=/path/to/python`. Pass `--h` values or `--rtol`, `--atol`, `--max-step` after the runner to override defaults. No new silent remapping of dependencies. An original parent/dependency mismatch raises an error.

## Output

- `results/stencil_defects.csv`: raw case, snapshot, step, mode, reference/geometric defects and normalized differences, including signed focus coordinates 0,2,8.
- `results/stencil_convergence.csv`: medians across the 36 snapshots for each step and mode.
- `results/case_integrations.csv`: time horizons and solver steps.
- `results/report.json`: authority, verified dependency hashes, methodology and numerical limits.
- `sendback.zip`: complete scripts, outputs, corrected runner, README and SHA-256 manifest.

## Method and scientific boundaries

A common original phase/gauge and 11D frame is retained; the *state* is re-integrated for the geometric kappa but the *direction* is not transported via a new tangent cocycle. Compare changes in Hessian mixed kernels to the defects `mixed - polarized`, using both self-normalized and Hessian-scale-normalized magnitudes. The two kernels use the same binary64 RHS; agreement is not an exact Hessian certificate. No claim of structural persistence until stable step-size plateau and numerical floor assessment. No new analytic structural RHS decomposition or rigorous curvature identity is asserted.

## Provenance

- Source: user's authentic v18.48b server sendback, including the `python3.13` correction.
- Runtime dynamic dependency: extracted v18.47 scripts (not bundled or rewritten), SHA-verified original Q550 support inputs and replay genealogy.
- Reconstructed adaptation: `scripts/run_audit.py` is a documented functional modification of the v18.48b audit. Actual Python script is included, not only a report.
