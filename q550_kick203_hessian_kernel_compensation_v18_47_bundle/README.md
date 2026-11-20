# Q550 v18.47 — Hessian Kernel Structure & Compensated Evaluation Audit

## Scientific question

Original v18.45 found a minimum of **cross-method disagreement** near normalized radius 0.008, but original v18.46 showed that **nominal fourth-order Richardson consistency failed** (3/288 RHS ladders, 0/216 Jacobian ladders). Those are observations about numerical methods, not evidence of physical stiffness, a curvature law, a material invariant or an exact Hessian.

This audit holds the original SHA-pinned Q550 ODE, original kappa0, all original candidate surfaces and their 12 source seeds and support regions **unchanged**. It replays **25 original integrations** and the same **36 phase-tracked snapshots**. Original v18.46 (166 SHA256-authenticated files) is embedded unmodified. All older ancestors are automatically recovered from the nested genealogy and verified, including standard/tight checkpoint tiers.

## Kernels (same scalar Hessian contraction in the normalized chart)

Let `z=(y_material-center)/scale`, `theta` unwrapped, `v=(F_material/scale,F_theta)`, `u=v/||v||`, and `g` the normalized gradient of the same frozen diagnostic candidate C. The required vector is `q_j = g · D_z² F_z[u,e_j] ||v||`.

1. **Historic mixed RHS**: 4x4 centered mixed first differences (`h=.004,.008,.012`), exactly regressed against original v18.46.
2. **Compensated mixed RHS**: identical 4x4 nodes but Python `math.fsum` for the inner `g·F_z` and outer weighted sum; transpose the displacement addition order for a floating-point symmetry/associativity diagnostic.
3. **Polarized RHS**: the *different* algebraic identity `H[u,e]=(H[u+e,u+e]-H[u-e,u-e])/4` using five-point second directional differences of the original RHS; scalar reductions are compensated. Same three radii.
4. **Acceleration-gradient identity**: differentiate `g·(B(z)v(z))` with a centered four-point first-derivative stencil and subtract `g·B(z)D_z v(z)`; `B = D_z F_z` derives from the original pinned Jacobian. Radii `.001,.002,.004`. This pathway **uses both original F and J**; it is not statistically or physically independent of the Jacobian reference.

Every kernel evaluates original binary64 functions. Compensated *summation* cannot undo lost significant digits inside them, and no symbolic Hessian or multiprecision model is available. Agreement is diagnostic only: systematic common bias can make methods agree yet wrong.

All ten components are exported; **0, 2, 8** are tagged for scrutiny rather than privileged physical axes. Signed residuals, error shares, cross-path disagreement and compensated-vs-historical deltas are preserved. Equal reduced phase is never treated as equal material state.

## Server

```bash
cd /root/elab1/hysteresis
unzip q550_kick203_hessian_kernel_compensation_v18_47_bundle.zip
cd q550_kick203_hessian_kernel_compensation_v18_47_bundle
bash scripts/run_on_server.sh 2>&1 | tee v18_47_server.log
```

This generates `/root/elab1/hysteresis/q550_kick203_hessian_kernel_compensation_v18_47_sendback.zip` only after the original Q550 run. Preflight alone:

```bash
bash scripts/run_preflight_v18_47.sh
```

## Original outputs when server run succeeds

- `results/original_v18_47_proof.json` — 25 Q550 integrations, original dependency validation, controls and scientific limits.
- `results/original_v18_47_kernel_vectors.csv` — 432 complete kernel vectors.
- `results/original_v18_47_component_budget.csv` — 2160 signed component records.
- `results/original_v18_47_snapshot_summary.csv` — 36 snapshot-level cross-kernel diagnostics.
- `results/original_v18_47_solver_steps.csv` — 24 local original integration records.
- `results/original_v18_47_cross_kernel.png`, `results/original_v18_47_component_signature.png` — plots from the original outputs only.
- Python generator, math kernels, synthetic tests, source provenance, pinned parent ZIP, dependency notes and complete SHA256 manifest in the bundled genealogy.

## Reproducibility / limitations

Run with Python 3.13 preferred (Python 3 fallback), NumPy, SciPy, Matplotlib. Original server must hold unchanged SHA-pinned Q550 dependencies under `q550_state_impulse_response_v13.DEFAULT_BASE` (historical parent convention). The Python scripts include all reproducible transformations, inputs and results: no preflight data is mislabeled as an original ODE computation. The diagnostic Euclidean chart is *not* a certified material metric. No kappa0 change is applied; the independent geometric packing proposal remains outside this audit.
