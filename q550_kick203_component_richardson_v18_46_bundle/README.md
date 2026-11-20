# Q550 v18.46 — Component-Resolved Precision & Richardson Consistency Audit

## Purpose

Study the *same 36 original Q550 local material snapshots* as v18.45, holding the original ODE, kappa0, source candidates, 12 accepted seeds, support limits, phase, normalization chart, and integration tolerances fixed. The primary objects are the 10 components of the numerical contraction `(L_F B)^T g` and the numerical disagreement between two independent differentiation paths:

1. `mixed_rhs_hessian_contraction`: fourth-order *mixed* finite differences of the original **RHS** projected with the original material gradient; and
2. `dot_B_transpose_g_from_jacobian`: fourth-order directional finite differences of the original pinned **Jacobian**.

Neither is an analytic/symbolic Hessian; agreement between them can result from common systematic errors. A numerical minimum of their difference is **not** a certified accurate radius. In particular, this audit does NOT certify an intrinsic material metric, stiffness, a nonlinear invariant surface, curvature or physical torsion. The geometrical kappa0 proposal is NOT activated.

## Experimental design

- Authenticate SHA-pinned **original v18.45 sendback** (156 original members), its nested **v18.44** original archive, and all historical ancestors through the `standard`/`tight` checkpoint tiers. All 36 snapshots are replayed from original accepted v18.36 seed history, followed by strict replay regressions against the original v18.45 RHS and Jacobian vectors (all radii shared by the experiments).
- Execute **25 original integrations**, using the unchanged pinned server ODE. No 400-period orbit rerun. All 36 snapshots must remain in the calibrated support.
- RHS radii: `0.001, 0.0015, 0.002, 0.003, 0.004, 0.006, 0.008, 0.012, 0.016, 0.024, 0.032, 0.048`.
- Jacobian scaled targets: `0.000125, 0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016`.
- Evaluate the ratio-two **4:2:1** ladders: **8 RHS and 6 Jacobian** ladders per snapshot. For each, record the observed vector difference order, direction consistency of successive differences, pair-to-pair Richardson disagreement, and comparisons against **three separate numerical Jacobian consensuses**.
- Record all ten components (focus 0, 2, 8): signed source discrepancies and extrapolation outcomes. Use absolute/component-normalized quantities, retaining signs so cancellation is visible. Test a **fixed**, predeclared radius `0.008` and Richardson ladder `0.008 → 0.004 → 0.002` separately from the **post hoc** best-per-snapshot radii v18.45.
- Explicitly flag unresolvable tiny differences. Empirical order close to four does **not** establish the true Hessian because a constant bias survives Richardson.

## Run on original server

```bash
cd /root/elab1/hysteresis
unzip q550_kick203_component_richardson_v18_46_bundle.zip
cd q550_kick203_component_richardson_v18_46_bundle
bash scripts/run_on_server.sh 2>&1 | tee v18_46_server.log
```

The successful original run creates in `/root/elab1/hysteresis`:

`q550_kick203_component_richardson_v18_46_sendback.zip`

For a **preflight without new Q550 numerical integration**:

```bash
bash scripts/run_preflight_v18_46.sh
```

The preflight is explicitly marked `PREFLIGHT_ONLY_...` and cannot produce the scientific sendback. The packaging script fails closed unless all new original Q550 results, proof and hashes exist.

## Planned outputs after original execution

- `results/original_v18_46_proof.json` (original ODE/hash/run provenance and limitations)
- `results/original_v18_46_snapshot_summary.csv` (36)
- `results/original_v18_46_rhs_radii.csv` (432)
- `results/original_v18_46_jac_radii.csv` (288)
- `results/original_v18_46_rhs_richardson_ladders.csv` (288)
- `results/original_v18_46_jac_richardson_ladders.csv` (216)
- `results/original_v18_46_component_ladders.csv` (2,880)
- `results/original_v18_46_solver_steps.csv` (24)
- four original-Q550-only plots, README, executable Python source, tests, requirements, sha256 manifest, source-checkpoint genealogy and raw parent archive.

## Interpretive and numerical limits

Fourth-order Richardson factor `15` holds asymptotically for a resolved fourth-order leading truncation term, not automatically on an arbitrary finite radius or near cancellation. The nominal order from two successive differences is a diagnostic. In particular, **if the Jacobian pathway has its own radius-dependent bias, all cross-path minimum claims remain reference-dependent**. `median` Jacobian reference is not ground truth. Component numbering is normalized material coordinate numbering (0-based), not physically canonical modes. The temporal phase is unwrapped; equal reduced phase is not equated with equal material state.

Software uses Python 3.13 preferred (Python 3 fallback), NumPy, SciPy, Matplotlib. Commentary, README and code comments intentionally in English. Historical nested archives are kept unmodified and checked by SHA256.
