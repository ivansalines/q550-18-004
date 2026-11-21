# Q550 v18.48f — Analytic Hessian Cancellation Transport & Branch Localization

## Scientific question

Does the pronounced projected-kernel cancellation in v18.48e (notably `phaseblind`, `xi=3.5`, branch `0`) persist when changing the original `kappa0=1/6` to the autonomous comparison `kappa0=35*pi*10/(12*550)` and independently evolving both 11D states **and a transported observation covector**? Which of the apparent change is from the Hessian itself, from a changed state, from the local flow used in the diagnostic, and from the moving dual observer?

## Scope and provenance

- This is a genuine new numerical ODE integration: **24 new 11D + 11D cotangent integrations**, 2 per each of the original 12 cases.
- Initial states and the source-gradient covectors are the **36 authenticated snapshots** from original-server v18.48e; each case restarts at its `fraction=0.125` snapshot. The two kappa flows share exactly the same initial state and the same initial covector.
- Historical fraction `0.5` and `1.0` Q550 original states are held out as replay checks. This is **not** a full material history reconstruction; the tiny `T` in these cases ranges from `4.44e-7` to `1.23e-6` in original independent-case time.
- The script verifies the SHA-256 for every embedded parent input and for all three frozen original mathematical sources **both inside the bundle and at the source location on the server**. The parent report carries original six dependency pins and source genealogy. The original server files are never edited.
- Generated symbolic Hessians `scripts/generated_base_jet.py` and `scripts/structural_jet.py` are inherited byte-for-byte from v18.48e, accompanied by the original symbolic generator and historical original-source copies.
- The original v18.47 and v18.48e folders are not required in the same directory: the relevant **actual v18.48e states and lineage metadata are embedded inside the new bundle**. The three SHA-matching original Q550 source files must still be present in `Q550_BASE_ROOT` (default `/root/elab1/hysteresis`).

## Analytic decomposition

For `y=(y1,v1,theta1,omega1,y2,v2,theta2,omega2,z,vz,Theta)` and checkpoint scalar `slope`, the authenticated support RHS is

`F(y,kappa) = slope * (F_base(y) + kappa * K(y))`.

`K` has only four nonzero rows `1,3,5,7`; its Hessian has **zero entries whenever any input direction has coordinate `z` (index 8)**. The full Hessian in general does NOT vanish in those directions. The kappa sensitivity at fixed state is exactly (as a formula in real arithmetic)

`H(y,k_geo) - H(y,k_ref) = slope*(k_geo-k_ref)*D2K(y)`.

Each analytic Hessian is evaluated in binary64; no interval enclosure is claimed.

The projected kernel uses exactly the original frozen 10D chart and the original v18.47 diagnostic definition. For two projected vectors `B,C`, define the cancellation indicator:

`Cancellation = (||B|| + ||C||) / ||B+C||`.

Always report the underlying norms and signed cosine as well. A large ratio could arise from a small denominator; it is **not** by itself a metric-independent invariant or evidence of physical hysteresis.

## Moving observation frame

For each original 10D diagnostic gradient `g0`, embed it in the eleven-dimensional chart cotangent as `g11(0)=[g0,0]`, with the historical first-ten chart scales and the eleventh weight 1. In each evolving trajectory, transport the covector according to the **full 11D dual variational equation**:

`d(g11)/dt = -J_chart(y(t),kappa)^T g11`,

`J_chart = diag(w)^(-1) * J_internal * diag(w)`.

The kernel observable still uses the first ten coordinates of the transported covector, consistent with the original projected chart kernel; the eleventh dual component is preserved during integration and saved in NPZ. This is a mathematically defined moving dual covector, **not** a reoptimized original source-gradient observer. `historical_observer_comparison.csv` quantifies the distinction at the three historical fractions. The frozen-initial-observer control appears as two additional ledger columns.

### Exact telescoping budget at each time

Evaluate the kernel as a multilinear `L(H, f, g)` of the analytic Hessian, local flow and 10D projected covector. At a given time, with subscript `r` and `g` for reference and geometric kappa:

1. `DIRECT`: `L(H_g(y_r)-H_r(y_r), f_r, g_r)`; analytic coupling change at the same state.
2. `STATE`: `L(H_g(y_g)-H_g(y_r), f_r, g_r)`; changed state at geometric kappa.
3. `FLOW`: `L(H_g(y_g), f_g-f_r, g_r)`; changed instantaneous flow.
4. `DUAL`: `L(H_g(y_g), f_g, g_g-g_r)`; transported covector change.

These terms sum exactly algebraically to the observed geometric-minus-reference kernel. The reported telescoping residual is a bookkeeping consistency check, **not independent experimental confirmation**. All terms are saved as signed ten-component arrays in `transported_states_and_terms.npz`; CSV holds their norms and selected columns.

## Server execution (Python 3.13)

Unzip at `/root/elab1/hysteresis` and run:

```bash
cd /root/elab1/hysteresis
unzip q550_kappa_analytic_transport_v18_48f_bundle.zip
cd q550_kappa_analytic_transport_v18_48f_bundle
bash run_on_server.sh 2>&1 | tee v18_48f_server.log
```

`Q550_PYTHON` defaults to `python3.13` (the environment with `numpy`, `scipy`, `mpmath`). `Q550_BASE_ROOT` defaults to `/root/elab1/hysteresis`. Historical source links to `q_550_orbit.npz` and `inputs/presence_metrology_events.csv` are **not required for this new run**, because the original states and SHA-anchored genealogy are embedded.

Optional arguments passed through to `scripts/run_audit.py`: `--samples 33`, `--max-step 2.5e-7`, `--rtol 3e-12`, `--atol 3e-13`, `--horizon-factor 1`. Setting `--horizon-factor >1` extrapolates past the historical endpoint; classify any extra time as exploratory, **not authenticated trajectory support**.

To make optional figures after the server run:

```bash
python3.13 scripts/plot_transport.py
python3.13 scripts/pack_sendback.py
```

## Outputs

- `results/report.json` — all counts, SHA genealogy, branch comparisons, replay and telescoping checks, limitations.
- `results/transported_kernel_ledger.csv` — reference and geometric analytic base/coupling norms, ratios, opposition cosines, time and four-term sensitivity budget.
- `results/branch_localization.csv` — maxima and medians for all 12 original cases.
- `results/branch_pairs.csv` — explicit branch-0/branch-1 comparisons for `phasecov` and `phaseblind` at `xi=3.5,4,4.5`.
- `results/historical_observer_comparison.csv` — advected dual covector versus independently evaluated historical original source gradient.
- `results/transported_states_and_terms.npz` — the actual 11D state and 11D dual for both kappas, ten-component signed kernel and each of four telescope terms.
- `results/*.png` — optional plots generated by included script.
- `SHA256SUMS_OUTPUT.json`, `sendback.zip` — portable genealogical archive including original read-only script references, generator, input pins, all outputs, and execution script/log where present.

## Local smoke / scientific boundary

The included test suite checks the frozen SHA-pinned v18.48e input dataset, direct exact coupling identity, expected z zeros, additive telescope and branch IDs. A pre-server local reference execution using those exact inputs produced 12 cases, 24 integrations and 408 sampled time points (including historical fractions) with at most `1.29e-12` absolute replay difference at historical endpoints. This is a **local computational preflight**, not a completed new run on the user's server.

In that local preflight, `phaseblind xi=3.5 branch=0` reaches reference and geometric cancellation ratios about 103.44 and 102.22, respectively, while branch 1 peaks near 1.85. These are ratios for a **projected 10D observable** over a very short support window; no long-time memory, bifurcation, mechanical 11D invariant, or stability transition is inferred. An observed change in a projected kernel may also arise from change of flow and observer; report the signed decomposition before causal claims.
