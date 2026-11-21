# Q550 v18.48l — Cancellation Near-Null Subspace & Observer-Family Continuation Audit

**Status:** authenticated frozen-state, analytic-Hessian observer-geometry diagnostic. The output is **not** an interval proof, a physical adjoint-transport result, or evidence for an intrinsic torsional flip.

## Why this audit

v18.48k found unusually thin observer-space cancellation crests in two different physical Q550 trajectories: `phaseblind ξ=3.575 branch 0` and `phasecov ξ=4.25 branch 0`. The smallest singular values of the resultant linear operator approached the double-precision floor. v18.48l tests whether this reflects source-bearing near-null directions or a scale-dependent numerical artefact, and tracks such directions along authentic common-phase samples rather than interpolating branches.

## Analytic operators

The v18.48e analytic support-RHS jet obeys `H(y,κ₀)=H_base(y)+κ₀ H_coupling(y)`. For each authentic state and original/alternative κ₀, it contracts both contributions with the **same original model field and transport direction**. In the historical 10D frozen observer chart,

```
B(g) = Lb @ g
C(g) = Lc @ g
T    = Lb + Lc
A    = [Lb; Lc]
R(g) = (||B(g)|| + ||C(g)||)/||T(g)||
```

`Lb` and `Lc` are each 10×10. `A` is 20×10. A near-null of `T` is significant only when `A@g` remains nonzero. All projected quantities depend on the chosen observer contraction. Affine changes of chart preserve the projected ratio **only if covectors and the induced cometrics are transformed consistently**, as checked in v18.48j/k.

## What is actually implemented

1. Manifest and nested genealogy verification of original v18.48k server sendback, including its SHA-pinned v18.48j/i/h lineage.
2. Rebuilding `Lb,Lc` from the previously authenticated symbolic Hessian at **all 328 actual Q550 states**, without new integrations.
3. Raw float64 ten-value singular-spectrum summary of `T` and `A`, projected historical observer ratios, and source response of the smallest-singular vector.
4. Source-active SVD analysis for five declared *relative* cutoffs `1e-4,1e-6,1e-8,1e-10,1e-12` (with source whitening); numerical ranks, source-normalized residuals and ratios.
5. Sign-consistent projective continuation of the full-null candidate and source-active representatives along each 41-section physical path, with angular separations and rank changes. No physical observer equation is solved.
6. Independent **mpmath second-order automatic differentiation**, directly from the 11D physical model, to recompute `Lb,Lc` for the four crests using 45/70/90 decimal digits; mpmath SVD is applied directly to T and A, not their squared Gramians. Distinct original/alternative κ₀ input float values remain faithful to the actual server genealogy.

### Preliminary result from this package's own preflight

At the `phaseblind ξ=3.575 original` crest the minimum singular of T was approximately `1.90e-45`, `1.94e-70`, and `1.40e-90` at 45, 70, and 90 digits respectively, while the minimum singular of A remained about `1.53e-7`. At `phasecov ξ=4.25 original`, T's minimum displayed the same precision-following behavior while the minimum singular of A remained about `1.82e-6`. This is strong **numerical evidence consistent with** an algebraically exact source-bearing null; no exact rank proof is claimed. The continued source-active family changes with the cutoff and cannot yet be considered unique.

## Run

From `/root/elab1/hysteresis`:

```bash
unzip q550_kappa_nearnull_v18_48l_bundle.zip
cd q550_kappa_nearnull_v18_48l_bundle
bash run_on_server.sh 2>&1 | tee v18_48l_server.log
```

Default interpreter: `python3.13`. Override with `Q550_PYTHON` if required. Requires `numpy`, `scipy`, `mpmath`, `matplotlib`. No external data files, manual adapter, links, or recomputed long Q550 trajectory are needed.

Optional precision/cutoff customization:

```bash
bash run_on_server.sh --hp-dps 45,70,90 --cutoffs 1e-4,1e-6,1e-8,1e-10,1e-12
```

### Main output files

- `results/report.json` — summary, numerical bounds and exact claimed scientific status.
- `results/full_null_singular_atlas.csv` — per-state 10D SVD, historical observer alignment, source activity.
- `results/source_active_cutoff_sweep.csv` — numerical rank and source-normalized residual by threshold.
- `results/source_active_continuation.csv` — projective modal transport (not an adjoint trajectory).
- `results/high_precision_focal_checks.csv` — precision sweep and original model recomputation.
- `results/phaseblind_near_null_continuation.png`, `results/phasecov_near_null_continuation.png` — visual scans.
- `sendback.zip` and `SHA256SUMS_OUTPUT.json` — complete reproducible lineage, scripts, source and input pins.

Refer to `SCIENTIFIC_DECISION_RULES.md` for criteria and limits. Original v18.48k/i/j/h scripts and input archives are **not altered**; new script implementations are openly listed and pinned. No core results are asserted as mathematically certified without an independent exact proof.
