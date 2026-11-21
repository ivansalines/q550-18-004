# Q550 v18.48g — Phase-Matched Branch Continuation & Cancellation Transfer Audit

## Scientific question

Is the migration of high projected-Hessian cancellation between `phaseblind xi=3.5 branch=0` and `xi=4.0 branch=1` consistent with continuity of material posture, or is it partly an observer/branch-label correspondence effect?

**Status: controlled, exploratory phase-gauge-matched homotopy, NOT a certified material branch continuation.** The authenticated data contain only three isolated `xi` values (3.5, 4, 4.5), two branches and two observation families. New intermediate states must be explicitly labelled constructed states. Mathematical smoothness of an interpolation cannot establish physical continuity of the underlying family of solutions.

## Original-source genealogy

- Exact 36 original Q550 11D states and 10D observation gradients from the v18.48e server sendback are embedded, with their `PARENT_SHA256.json` pins unchanged.
- The executed v18.48f snapshot ledger, branch pairs and report are embedded under `inputs/`, with a separate `V18_48F_SHA256.json` pin.
- `scripts/run_audit_v18_48f.py`, `scripts/structural_jet.py`, `scripts/generated_base_jet.py` and `scripts/generate_structural_math.py` are frozen functional ancestors, not rewritten physical equations.
- Three original physical source programs are frozen under `reference/` and their hashes must also match the actual server files under `Q550_BASE_ROOT`. No silent dependence on file names alone.
- Python 3.13, NumPy, SciPy, mpmath; no SymPy needed during server execution.

## Exact phase-gauge symmetry (not a shared Poincaré section)

Original internal state order: `(y1,v1,theta1,omega1,y2,v2,theta2,omega2,z,vz,Theta)`.
Original RHS depends on angles through `delta=theta1+theta2-phi0-Theta`. Therefore a simultaneous constant translation `(theta1,Theta) -> (theta1+c,Theta+c)` preserves the relative phase and the original RHS and its Jacobian/Hessian. Additional *integer* `2pi` lifts of individual angles are exact periodic symmetries. The script tests this against the original RHS and independently generated analytic Hessian. This phase match is an equivalent representation of each anchor, **not** proof that separate `xi` cases hit a common physical event surface.

For every neighboring `xi` pair, both candidate endpoint assignments are considered: **identity** `0->0,1->1` and **swap** `0->1,1->0`. Matching costs are computed using (i) frozen chart-normalized distances, (ii) relative-phase proximity, (iii) observer cosine and (iv) a declared combined heuristic. Disagreement between the criteria is a reported ambiguity, not silently resolved as a fact. A lowest-cost pairing is used for *local transport calculation only*, while **both pairings appear in the static control ledger**.

## Two independent coordinate interpolations

- Internal interpolation: linear in `y` after phase symmetry matching, initial historical observer gradient linearly interpolated in the fixed original 10D chart.
- Positive-radius control: interpolate actual positive `rho1=e^y1` and `rho2=e^y2`, then map back with `log`, keeping other coordinates as above.

Every intermediate seed at `3.5<xi<4.0` or `4.0<xi<4.5` is a **constructed homotopy state**; its high cancellation, or an apparent crossing of branch ratios, is not an observed state on a true Q550 solution branch. Endpoints must reproduce the original v18.48f kernel ratios within tolerance after valid phase-gauge moves.

## Genuine local replay for each constructed seed

For the selected pairing, the script independently integrates the original 11D RHS **and** the full 11D cotangent variational equation for each of the two `kappa0` values. The original support-time windows `T` are linearly interpolated between endpoints and remain extremely short (`~1e-6`). The observer is the interpolated frozen original gradient, **not** a newly optimized physical source gradient. The projected 10D kernel uses the original chart and exact analytic base + coupling Hessians, with signed opposition cosine and both component norms.

A nonzero value of `log(C_track0/C_track1)` switching sign along a specified homotopy is reported with its bracket, never as a certified physical branch-switch event. A different assignment can remove the sign change simply by relabelling endpoints; that is why both assignments are essential.

## Server command

Extract in `/root/elab1/hysteresis`, alongside the original SHA-matching mathematical sources:

```bash
cd /root/elab1/hysteresis
unzip q550_kappa_phase_matched_v18_48g_bundle.zip
cd q550_kappa_phase_matched_v18_48g_bundle
bash run_on_server.sh 2>&1 | tee v18_48g_server.log
```

`Q550_PYTHON` defaults to `python3.13`, and `Q550_BASE_ROOT` to `/root/elab1/hysteresis`. No manually created orbit or events symlink is necessary for this stage, as frozen original snapshot data and SHA genealogy are included.

Optional: `--alpha-points 13` or `--max-step 1e-7`. Default is **9 alpha samples**, 2 families × 2 xi intervals × 2 tracks × 9 seeds × 2 kappas = **144 genuine short original RHS + cotangent integrations**, evaluated at 3 local times each. Both identity and swapped branch pairings are evaluated as *instantaneous* analytic controls, while only the heuristically selected assignment is transported, not counted as authentic integrations.

## Outputs

- `results/branch_assignment_costs.csv`: both match hypotheses; state/phase/observer/combined costs.
- `results/interpolation_controls.csv`: all candidate pairings, internal-vs-radius controls, cancellation vectors' norms and opposition cosines at synthetic seeds.
- `results/phase_matched_transport.csv`: actual short ODE + adjoint replay from the selected constructed seeds for both kappa values.
- `results/transported_arrays.npz`: full 11D states/covectors, signed 10D base and coupling kernels and resultants.
- `results/report.json`: gauge invariance checks, endpoint regression checks, matching ambiguity, crossings of *constructed* paths and boundary of evidence.
- `results/phaseblind_branch_matching_hypotheses.png`: optional comparison plot.
- `sendback.zip`, `SHA256SUMS_OUTPUT.json`: complete source-script/input/output genealogy.

## Decision rules

1. Never infer true branch continuity from a smooth interpolation alone.
2. Require exact phase-gauge-invariance and end-anchor reproduction checks before reading intermediate data.
3. If state and observer assignments prefer different pairings, classify identity of branches **AMBIGUOUS**.
4. Crossing of the constructed ratio only shows consistency of that interpolation hypothesis; an alternative branch assignment yielding no crossing defeats any label-independent transfer claim.
5. Any physical continuation proof requires new validated Q550 solutions for intermediate xi, a continuation boundary-value problem or a certified common event section, and a transported physical observer/frame.
6. Preserve all generator scripts and computed outputs in every sendback.

## Completed local preflight (not the user-server run)

With `--alpha-points 9` on the frozen authenticated inputs: 5 unit tests pass; 144 actual short Q550 ODE+cotangent integrations and 432 temporal observations complete. The RHS gauge-invariance relative error reaches at most `2.37e-13`, Hessian gauge-invariance error `1.20e-13`, and the maximum original-endpoint ratio discrepancy vs v18.48f is `3.51e-12`.

For **phaseblind, 3.5→4.0**, keeping branch labels fixed makes the *constructed* log cancellation-ratio difference pass through zero at about `xi=3.8703` (both internal and positive-radius interpolations), while the swapped correspondence has **no crossing** in this interval. Relative-phase proximity favours swapped endpoints; observer proximity favours unchanged labels. Therefore the available anchors do **not** establish a label-independent physical cancellation transfer. See `preflight/report.json`, `preflight/interpolation_controls.csv` and the phaseblind matching figure.
