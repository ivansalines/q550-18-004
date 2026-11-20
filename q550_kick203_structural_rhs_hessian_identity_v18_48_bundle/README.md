# v18.48 — Structural RHS Decomposition & Hessian Identity Audit

## Status
Prepared numerical audit harness. No authenticated v18.47 parent bundle was accessible during construction. No original Q550 integration or structural RHS decomposition has been claimed or performed. **Do not interpret the supplied synthetic test as Q550 evidence.**

## Mathematical checks
For an independently defined split `F(x)=sum_i F_i(x)`, check:
1. pointwise `F-sum F_i` over original 11D states;
2. `D²F[u,v]-sum_i D²F_i[u,v]` over step ladder;
3. mixed-derivative symmetry `D²F[u,v]=D²F[v,u]`;
4. norm of each signed Hessian contribution and cancellation ratio `sum_i ||H_i||/||sum_i H_i||`;
5. optional independent analytic bilinear evaluation through adapter functions.

**Caution:** finite differences of identical code paths are not independent symbolic proof. For an identity audit, supply an independently derived analytic Hessian. Verify continuity with actual v18.47 source and trace its hash. A finite-difference residual may be dominated by cancellation at small `h`.

## Integration contract
Supply a Python adapter file defining `rhs(x)` and `components: dict[str, callable]`; both take and return 11-component arrays. The components must correspond to genuine, physically meaningful original RHS terms, *not* arbitrary numerical splitting. Optional `analytic_bilinear(x,u,v)` and `analytic_component_bilinears(x,u,v)` provide stronger independent checks. No implicit discovery or guess of original Q550 module names is performed.

An NPZ input needs `states` of shape `(N,11)`; optional `directions_u`,`directions_v` of the same shape. Prefer states and perturbation directions selected from the authenticated kick-203 trajectory and v18.47 Hessian audit.

## Commands
Smoke test (synthetic only):
```bash
python3 scripts/make_smoke_inputs.py
python3 scripts/audit_v18_48.py --adapter scripts/example_adapter.py --states inputs/synthetic_smoke.npz --output outputs/smoke.json
```
Server, from extracted bundle root:
```bash
export Q550_V1848_ADAPTER=/absolute/path/to/authenticated_q550_adapter.py
export Q550_V1848_STATES=/absolute/path/to/authenticated_kick203_states.npz
bash scripts/run_on_server.sh 2>&1 | tee v18_48_server.log
```

## Decision conditions
- Never classify structural identity as verified unless original-model components are independently authenticated.
- Check the `h` ladder for an actual convergence region (avoid mistaking rounding floor for evidence).
- Large cancellation ratio warrants attention; it does not demonstrate a real physical singularity.
- Do not infer absence or presence of a material source from only projected observables.
- Missing parent v18.47 implies genealogy status `PARENT_NOT_ATTACHED` until its SHA256 and contents are verified.

## Runtime
Python >= 3.10, NumPy. Script reads data, does not integrate an orbit. Code and docs English as per research convention.
