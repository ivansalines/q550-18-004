# Q550 v18.48h — Physical Branch Continuation & Common-Section Observer Audit

## Purpose, evidence boundary, and essential correction

This bundle answers a narrower, testable question than v18.48g: **what does the original Q550 IVP actually do when seeded at the historical xi=3.5 branch-0 and branch-1 states and followed continuously to the reference xi=4.0 and 4.5 clock sections?** No intermediate state is interpolated, and no branch label is changed along an IVP. This is a physical **initial-value continuation**, NOT an established family of BVP-connected solutions or a certified branch-isomorphism theorem.

The independently projected/calibrated seeds at xi=4.0 and xi=4.5 are not guaranteed to be forward images of seeds generated independently at xi=3.5. The new genuine IVPs must be compared with those independent historical anchors and must NOT be forcibly glued to them.

## Source genealogy

* Frozen authentic original 11D model scripts (`reference/`); their SHA-256 values must match the corresponding untouched files in `$Q550_BASE_ROOT`, via `scripts/run_audit_v18_48f.py`.
* Original v18.48e 36-server-snapshot NPZ, CSV, historical small-T case list, report, and preserved parent hash manifest in `inputs/`.
* Complete **original v18.48g user-server sendback ZIP**, with frozen SHA pin, is included inside `inputs/`, and every member is checked against its internal output manifest. This corrects any ambiguity about archive root hierarchy.
* Frozen functional mathematical scripts are included under `scripts/`; the physical model has not been altered.
* Every source, script, input, result, figure and report is included in the output sendback with a newly computed SHA-256 manifest.

## Common section

The support-time RHS has `dTheta/dxi = 2*pi*slope`. Each xi=3.5 original branch first snapshot (historically at 0.125*T, with T ~ 1e-6) is transformed by the exact gauge `(theta1,Theta) -> (theta1+c,Theta+c)` and periodic angular lifts so its **starting Theta=0**, without altering the physical RHS. Then the **original RHS** (or the same RHS with the autonomous kappa) advances continuously to support-time offsets `0..1` in 21 samples including reference offsets 0.5 and 1.0. Therefore all paths are evaluated on exactly the same relative lifted-Theta sections; no estimated zero-crossing or artificial event time is needed. The original 0.125*T seed-offset is retained as a caveat; this is not an interval-certified event section.

**Two kappa cases:** historical `kappa=1/6`; autonomous `kappa=35*pi*10/(12*550)`.

**Branch identifiers:** the branch label belongs to the xi=3.5 *initial seed* and is preserved on the complete IVP, even when the output resembles the opposite historical branch at some later reference anchor.

## Observers / critical structural null

The 11D chart-cotangent equation is integrated from the xi=3.5 initial observer `dg/dxi = -J_chart.T*g` using the authentic original Jacobian. Three types of observer are **explicitly separated**:

1. Historical fitted gradient at xi=3.5, then cotangent-transported — **not refitted** to intermediate states.
2. Physical instantaneous material-activity scalar observers `A_sin=rho1*rho2*sin(delta)` and `A_cos=rho1*rho2*cos(delta)`. Their **state-recomputed gradients** and their independently cotangent-transported initial gradients are shown separately. Their instantaneous gradients have **exactly zero coupling Hessian kernel** for this projected kernel, because the gradients have no indices 1,3,5,7 (the only nonzero coupling RHS output indices). This is a structural null/control, NOT evidence of cancellation or of absence of dynamic coupling.
3. **State-recomputed Lie-derivative activity observer**, `dA/dxi=grad(A).F`, with gradient `H_A F+J_F.T grad A`. This physically defined observer responds in the acceleration coordinates. Its gradient is compared with its own cotangent-transported seed value. They are not interchangeable.

The projected signed Hessian terms retain the exact v18.48e analytic `H_base + kappa H_coupling` and the frozen historical 10D chart. The ratio `(||K_base||+||K_coupling||)/||K_base+K_coupling||` is an observable-dependent *local cancellation diagnostic*, NOT an invariant mechanical torsion, energy, or hysteresis certificate.

## Exact command on the server

```bash
cd /root/elab1/hysteresis
unzip q550_kappa_physical_section_v18_48h_bundle.zip
cd q550_kappa_physical_section_v18_48h_bundle
bash run_on_server.sh 2>&1 | tee v18_48h_server.log
```

The runner defaults to `python3.13`. Nothing must be moved into `inputs/` manually. It verifies the original reference code against actual `$Q550_BASE_ROOT` files (default `/root/elab1/hysteresis`), and validates the embedded v18.48g parent zip and v18.48e data. The script accepts `--sections 41` for denser common clock sections (choose odd number so xi4.0 is included), `--max-step 0.0075` for a refined integration, or a different `--base-root` if needed. It generates `results/`, optional figure, `SHA256SUMS_OUTPUT.json`, and `sendback.zip`.

## Output content

* `physical_section_cancellation.csv`: original IVP sampled at common clock sections, using all nine observer constructions (historical, instantaneous and Lie material observables; cotangent and freshly computed) for both kappas.
* `physical_paths.npz`: all genuine propagated 11D states + 5 transported chart cotangents, allowing independent downstream audits without information loss.
* `original_anchor_matching.csv`, `best_anchor_matches.csv`: explicit independent-seed comparisons using frozen chart-normalized distances, a separate unit-scaled diagnostic and modulo-relative-phase difference.
* `same_seed_identity_branch_balance.csv`, `branch_balance_summary.json`: signed log cancellation balances for the *same actual seed identities*, with sign-change brackets on sampled sections. Reported crossing coordinates use only linear interpolation to locate a bracket and are not certified event roots.
* `report.json`: completeness, source pins, clock residuals, observer definitions, matching decisions, exact limitations.
* `phaseblind_physical_section_cancellation.png`: optional visualization, with no claim of intrinsic observer invariance.

## Completed local preflight using authentic original source files

* 4 local unit tests pass; 8 actual full Q550 IVPs (2 families × 2 xi3.5 branches × 2 kappas), 21 clock sections, 1512 observational rows (9 observer types).
* Max lifted-clock identity error approximately `1.45e-15` absolute.
* `phaseblind`, seed branch 0, original-kappa **historically fitted transported observer**: cancellation ratio `103.436` at 3.5, `1.154` at 4.0, and approximately `1.000` at 4.5. The v18.48g artificial correspondence of the high-cancellation ridge therefore is **NOT** reproduced on this specific genuine initial-value trajectory.
* The historical independent xi=4 and xi=4.5 branch seed states are **NOT** matched by this forward propagation within the declared stringent frozen-chart threshold. The original chart contains very small velocity scales, so absolute magnitudes of those chart distances should not be read as intrinsic physical separation; the separate unit-scaled and modulo-phase diagnostics are also retained.
* The instantaneous (not Lie-derived) activity observers are coupling-null by mathematical structure. Their cancellation ratio of 1 is not a physically meaningful loss of an underlying coupling.
* A Lie-derivative activity observer has nonzero projected coupling and gives observer-dependent cancellation behavior; in this preflight phaseblind seed0 at xi3.5, the `sin` Lie observer ratio is about 1.285 versus 103.436 for the historical fitted covector.

**Scientific classification:** `GENUINE_Q550_INITIAL_VALUE_SECTION_CONTINUATION_NOT_BRANCH_BVP_CERTIFICATE`. This is a verified *numerical physical initial-value continuation*, not interval enclosures or an existence proof for a connected family of independently seeded branch solutions. A true BVP or continuation using a source defining the xi-dependent seed conditions is still needed for that stronger claim.
