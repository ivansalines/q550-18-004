# v18.48l — Scientific decision rules

## Objects and conventions

At an authenticated state `y(ξ)`, let `H_base` and `H_coupling` be the real-analytic derivative formulas already validated in v18.48e. Let `f` be the full Q550 support field evaluated at that state and parameter value. The frozen historical 10D chart defines two observer-covector linear maps `B(g)=Lb@g`, `C(g)=Lc@g`. Define `T=Lb+Lc`, and stacked source `A=[Lb; Lc]`.

The cancellation quotient `R=(||B||+||C||)/||Tg||` is undefined if all three norms vanish. A very small `||Tg||` only counts as evidence for cancellation if `||Ag||` is independently nonzero.

### Distinct questions

1. **Numerical null of T:** track ten singular values of `T` in float64; near-floor singular values alone cannot resolve exact algebraic rank.
2. **Source non-vanishing:** compare the smallest singular value of `A`, and evaluate `||Lb g||` and `||Lc g||` at a high-precision near-null vector of `T`.
3. **High-precision discriminant:** recompute the physical model's RHS *independently*, using 11-variable mpmath second-order jets at 45, 70, and 90 decimal digits; singular decomposition uses `mp.svd_r`, not a squared Gram matrix. An apparent smallest singular value of T which moves with precision while the other singular values and smallest source singular remain stable is evidence *consistent with* an algebraic kernel, not a formal determinant identity or interval certificate.
4. **Effective observer subspace:** for each relative source cutoff `r`, retain only right singular vectors of `A` with `σ/σ_max > r`. Whiten this source-active subspace, then minimize `||Tg||/||Ag||`. Report source rank and sensitivity to `r` as part of every result. A rank chosen by a threshold must never be called exact.
5. **Continuation:** track projective one-dimensional modes across the 41 **already integrated** common-phase sections for each of the 8 parent Q550 paths. Sign-match consecutive vectors; report projective angles and rank changes. This is not a newly integrated adjoint or material branch continuation.
6. **Parameter contrast:** keep the original and autonomous `κ₀` pathways separate and use the authentic v18.48j snapshots for each. Input double-precision states are not magically made rigorous by high-precision re-evaluation.

## Numerical warning

The scale of the 10D normalized observer coordinates spans orders of magnitude. Full-null directions estimated by a standard float64 SVD can jump because of conditioning even when the underlying physical branch remains smooth. Compare with source-active cutoff results and high-precision focal data before drawing a geometric conclusion.

## Decision statuses

- `NUMERICAL_NULL_CANDIDATE`: small singular in T with non-vanishing source, but no analytic null-vector identity.
- `RESOLVED_SOURCE_ACTIVE_FAMILY`: generalised source-normalised minimizer stable across reported thresholds and nearby sections; check rank and angular gaps.
- `CUTOFF_SENSITIVE`: apparent family geometry changes when the active source cutoff changes.
- `NOT_A_PHYSICAL_OBSERVER`: any frozen covector rotated or singular-mode-minimised without solving an observer evolution equation.
- `EXACT_RANK_CERTIFICATE`: **not issued** by this bundle; it would need an explicit algebraic identity, exact arithmetic or a rigorous interval rank proof.

All scripts, dependencies, input hashes, and produced CSVs are included in the release/sendback genealogy. The parent v18.48k server ZIP is SHA-pinned and recursively checked against its v18.48j/i/h lineage.
