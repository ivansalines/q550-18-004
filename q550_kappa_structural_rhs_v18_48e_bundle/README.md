# Q550 v18.48e — Structural RHS Decomposition & Exact-Formula Hessian Identity Audit

## Scientific motivation

Original v18.48d showed an empirical Jacobian-derived sensitivity plateau where the RHS finite-difference Hessian was poorly resolved. We have now inspected the **real SHA256-pinned original `support_rhs_float`** and derived the full Hessian of its real-valued formulas. The source and scientific state conventions are preserved. Results of the original-server audit are not yet available until the server runner executes.

## Structural identity (proven algebraically from original source)

Internal state order is:

`(y1,v1,theta1,omega1,y2,v2,theta2,omega2,z,vz,Theta)`.

Let `x=sigmoid(z)`, `rho_i=exp(y_i)`, `delta=theta1+theta2-phi0-Theta`, `A=exp(y2-y1)`, `B=exp(y1-y2)`. The original support ODE is `(2*pi/x)*model_derivative_float`. Its effective coupling is `kappa_eff=kappa0*x`. Therefore

`F_support(y,kappa) = F_base(y) + kappa * K(y)`

with exactly the four nonzero coupling rows of `K`:

- `K[1] = -2*pi*A*cos(delta)`
- `K[3] = +2*pi*A*sin(delta)`
- `K[5] = -2*pi*B*cos(delta)`
- `K[7] = +2*pi*B*sin(delta)`

The other seven coupling rows are zero. In particular **all first and second derivatives of the coupling part with respect to logistic coordinate z (index 8) vanish**. The full original RHS *does* depend on z, and its base Hessian need not vanish in these entries.

In real arithmetic, at *fixed identical state* and with the same positive checkpoint slope `s`:

`D2F(y,k_geo) - D2F(y,k_ref) = s*(k_geo-k_ref)*D2K(y)`.

Similarly `dF = s*(k_geo-k_ref)*K(y)` and `dJ = s*(k_geo-k_ref)*DK(y)`. The `k_ref=1/6` and independent proposed `k_geo=35*pi*10/(12*550)` are compared without altering the model. This is a sensitivity identity, not a physical validation of the geometric phase rule.

### Closed Hessian of K

Let `a=(-1,0,0,0,+1,0,0,0,0,0,0)` and `d=(0,0,+1,0,0,0,+1,0,0,0,-1)`; for reciprocal B use `-a`. For `R=e^{a·y}` and phase delta,

`H(R cos delta) = R * [(a⊗a-d⊗d)cos(delta) - (a⊗d+d⊗a)sin(delta)]`

`H(R sin delta) = R * [(a⊗a-d⊗d)sin(delta) + (a⊗d+d⊗a)cos(delta)]`.

Rows 1, 5 carry minus sign; rows 3, 7 plus sign; multiply by `2*pi`. This is implemented without finite-difference differentiation in `structural_jet.py`.

### Complete RHS/Jacobian/Hessian

`generate_structural_math.py` transcribes *the entire* pinned real RHS with the documented original `dU`, `dV_barrier`, and `ModelParameters`. SymPy computes derivatives of the kappa-independent term, generates `generated_base_jet.py` using common-subexpression elimination, and is **not required on the server** (only when regenerating). `structural_jet.py` combines that with the explicit K derivatives to give `F`, `DF`, and `D²F` for any κ and fixed slope.

These are exact derivatives of the specified analytic formulas, **but their evaluation is IEEE binary64**; no interval proof or floating-point error enclosure is claimed. Bytewise SHA pins of 3 original sources are included under `reference/`; their path and content are checked again on the server. A verbatim v18.48d runner is included for the measurement genealogy.

## Before Q550 server run: genuine original source equivalence test

From this bundle:

```bash
python3.13 -m unittest discover -s tests -v
```

The test suite compares analytic generated RHS and Jacobian with **the actual original pinned source files**, on independent off-trajectory sample states, at both kappas. There are also independent directional Hessian checks. This is a functional regression, not a Q550 experimental result.

## Run on Q550 server

Unzip into `/root/elab1/hysteresis`, next to the *extracted* original v18.47 bundle, and run:

```bash
cd /root/elab1/hysteresis/q550_kappa_structural_rhs_v18_48e_bundle
bash run_on_server.sh 2>&1 | tee v18_48e_server.log
```

Default interpreter `python3.13` (override with `Q550_PYTHON`). Existing original source dependencies in `/root/elab1/hysteresis` must already be SHA-pinned and path-resolved, including `q_550_orbit.npz` and `inputs/presence_metrology_events.csv`. Use the verified symlinks made during v18.48a if still present; do not duplicate, patch or overwrite source files. `Q550_BASE_ROOT` and `Q550_V47_ROOT` optionally override base and v18.47 locations.

The script replays the 12 original case seeds, three phase-tracked snapshots per case, and original frozen diagnostic gradient. It authenticates all historical dependencies. Each 11x11x11 Hessian is analytic, and its kappa difference is checked against the closed `D²K` formula. The original Jacobian finite difference is independently compared at 6 radii to document the numerical floor. The output includes **true state vectors and gradients** so the scientific genealogy can be replayed.

## Output and scientific limits

- `results/report.json`: original-source hashes and max RMS errors, exact-identity diagnostics, convergence/floor scan.
- `results/structural_snapshots.csv`: 36 fixed-state snapshots with signed focus columns and chart-kernel sensitivity, base/coupling cancellation ratio.
- `results/jacobian_floor.csv`: per-snapshot cross-check against the historical pinned Jacobian finite-difference construction.
- `results/floor_summary.csv`: median discrepancy versus h.
- `results/structural_snapshot_states.npz`: authentic 11D states, source gradients, frozen chart scale/center, reference flow and labels.
- `results/case_integrations.csv`: replay metadata.
- `sendback.zip`: all actual scripts, source copies, inputs generated from the original seeds, results, SHA manifest and optional run log.

No restart with geometric-kappa trajectory and no reoptimized observer/frame occurs here. Conclusions concern **fixed states and fixed original diagnostic chart**. A large relative change in a nearly zero kernel requires absolute magnitudes; `cancellation_ratio` is merely a diagnostic numerical ratio, not a proof of physical mechanical balance. Full material memory and hysteresis persistence are open questions.
