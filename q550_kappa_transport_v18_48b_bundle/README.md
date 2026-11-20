# Q550 v18.48b — Transported Hessian Compensation under Kappa Perturbation

Numerical original-model counterfactual (not an exact Hessian certificate).

## Run
Unzip adjacent to the **extracted original v18.47 bundle** in `/root/elab1/hysteresis`, then:

```bash
cd /root/elab1/hysteresis/q550_kappa_transport_v18_48b_bundle
bash run_on_server.sh 2>&1 | tee v18_48b_server.log
```

Send back `sendback.zip` from this directory. Override only if needed: `Q550_V47_ROOT` (directory of the *extracted* v18.47 bundle) and `Q550_BASE_ROOT` (historical reference file root). Original pinned dependency SHA-256 checks are mandatory and performed by the inherited original `model()` plus explicit verification. No file paths are silently remapped and no original files are modified.

## Method / interpretation
- Recreate 12 original case seeds (archived attempt genealogy verified), original anchors, original 36 snapshot times (12 cases × 3 fractions), original numerical Hessian directions as in v18.47.
- Compare `k0=1/6` to `k0=35*pi*10/(12*550)` with baseline state/directions unchanged for pure parameter sensitivity.
- With `--restart` (enabled by `run_on_server.sh`), re-integrate geometric-kappa RHS from *identical initial seeds*, evaluate at identical elapsed times, and compare with the **common baseline direction**. This is a trajectory-restart test, **not** a separately reoptimized 11D co-moving frame or a complete variational transport of the Hessian.
- Repeat compensated mixed RHS kernel and polarization diagnostics at `h=0.004,0.008,0.012`, preserve focus components (0,2,8) and 0+8 relative defect fraction.
- Original reference integrations are intentionally rerun and the runtime may be substantial due to stringent DOP853 max_step.
- All changes are numerical. A near-unit cosine is not proof of global transport invariance; a compensation defect relative to polarization is not an exact or interval-verified Hessian identity. Observed persistence is restricted to the tested states/directions.

## Files
`results/transported_kappa_sensitivity.csv`, `results/case_integrations.csv`, `results/report.json`, `sendback.zip`, `SHA256SUMS_OUTPUT.json`. The scripts (including packaging code) are included in the sendback. Python: 3.10+, numpy, scipy. The parent archive is not copied, to avoid duplicating a large ancestor, but parent script identities and runtime dependency hashes are recorded.
