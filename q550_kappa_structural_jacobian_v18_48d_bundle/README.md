# Q550 v18.48d — Pinned-Jacobian Hessian Pathway & Numerical Floor Audit

## What this audits
This is an original-Q550 **numerical cross-check**, not an exact symbolic structural Hessian. The true RHS source `test_support_q550_support_ode_interval_residual_pilot_v1.py` is SHA-pinned **on the user's server**, but absent from this delivery. Instead of pretending to reconstruct its algebra, we use the pre-existing original Jacobian from `test_support_q550_exact_nonlinear_local_tube_v2_1.py`. Its **directional first derivative** computes the Hessian contraction by an independent route, reducing catastrophic O(h^-2) second-difference RHS cancellation to an O(h^-1) Jacobian differentiation sensitivity. Both remain binary64 and may share implementation biases.

Original v18.47 snapshot genealogy and SHA verification are preserved using the unmodified parent scripts; original 11D directions and chart gauge are held fixed when testing alternate kappa. Cases: 12, snapshots 36; 2 kappas; 10 steps `h=0.048,0.032,0.024,0.016,0.012,0.008,0.006,0.004,0.003,0.002`; 24 very short integrations if restart enabled. No claim of full material co-moving transport.

## Run
Extract beside `q550_kick203_hessian_kernel_compensation_v18_47_bundle` in `/root/elab1/hysteresis`, retaining original authenticated files and earlier symlink mapping; then:

```bash
cd /root/elab1/hysteresis/q550_kappa_structural_jacobian_v18_48d_bundle
bash run_on_server.sh 2>&1 | tee v18_48d_server.log
```

Uses `python3.13` by default (`Q550_PYTHON` override). Uses verified original paths (`Q550_BASE_ROOT`, `Q550_V47_ROOT` override). Output: `results/stencil_defects.csv`, `results/stencil_convergence.csv`, `results/case_integrations.csv`, `results/report.json`, and complete `sendback.zip`.

## Interpretation gates
The `jacobian_kappa_relative` column is the most relevant alternative estimate of kappa effect; plot it by h alongside `mixed_relative`. The `jacobian_ref_relative`/`jacobian_geo_relative` columns give disagreement between pinned-Jacobian and RHS-based paths. Require a stable h-window and evaluate absolute norms before interpreting compensation. Do not equate any differential identity with a physical invariant. Full analytic RHS decomposition and interval/high precision validation remain separate future work requiring the authenticated RHS sources.

## Provenance
`run_audit.py` is an adaptation of executed v18.48c; `pack_sendback.py` and the verified python3.13 runner are retained. Full execution sources, environment/version, hash manifest and outputs are preserved in sendback. Archived originals are imported but never modified. The package includes Python script source, not `.pyc` artifacts.
