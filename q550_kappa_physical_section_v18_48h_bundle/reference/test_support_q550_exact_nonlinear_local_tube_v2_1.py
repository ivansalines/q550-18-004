#!/usr/bin/env python3
"""
Q=550 exact nonlinear orbit local self-consistent tube pilot (v2).

Purpose
-------
This is the next proof layer after the outward-rounded support-ODE residual pilot.

Let X_hat(s) be the natural cubic spline used by the already-closed frozen
linear cocycle, in the support coordinate

    s = Theta/(2*pi).

The exact nonlinear support-coordinate ODE is

    X' = H(X) = (2*pi/x) F(X).

For an exact IVP solution initialized on the spline at the left endpoint,

    X_exact(s_L) = X_hat(s_L),

write

    e = X_exact - X_hat.

Then

    e' = H(X_hat + e) - H(X_hat) - r_s,

where

    r_s = X_hat' - H(X_hat).

This pilot proves a LOCAL tube over selected single cubic-spline segments by:

  1. outward-rounded interval enclosure of r_s on each subcell;
  2. outward-rounded interval enclosure of J_H over
         X_hat(subcell) + [-rho,+rho]^11;
  3. a rigorous Euclidean logarithmic-norm upper bound using interval
     Gershgorin on sym(J_H);
  4. a scalar comparison equation
         y' <= mu_+ y + R,
     with y(s_L)=0;
  5. a self-consistency check that the assumed tube radius rho contains the
     propagated error bound throughout each subcell.

Important scope
---------------
PASS means:

  For the exact nonlinear IVP starting exactly at X_hat(s_L), the solution
  exists through the tested cubic segment and is enclosed by the certified
  Euclidean error tube around X_hat.

It does NOT yet prove:
  * one global exact orbit shadows the entire 12.67-rotation proof domain;
  * the archived numerical orbit itself is exact;
  * the previously closed cocycle margins survive the nonlinear-orbit tube.

Those are subsequent layers.

Dependencies
------------
Requires:
  * test_support_mobius_monodromy_audit.py
  * test_support_q550_support_ode_interval_residual_pilot_v1.py
  * support_feedback_q_curtain_cache/q_550_orbit.npz
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np


SCRIPT_VERSION = "2026-08-11-q550-exact-nonlinear-local-tube-v2.1-numpy-mpmath-fix"

DEFAULT_AUDIT = Path("test_support_mobius_monodromy_audit.py")
DEFAULT_RESIDUAL = Path("test_support_q550_support_ode_interval_residual_pilot_v1.py")
DEFAULT_ORBIT = Path("support_feedback_q_curtain_cache/q_550_orbit.npz")
DEFAULT_OUTPUT = Path("support_q550_exact_nonlinear_local_tube_v2")


def load_module(path: Path, name: str) -> Any:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_int_list(text: str) -> list[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or values[0] < 1:
        raise ValueError("expected positive integer subdivisions")
    return values


def lo(x) -> float:
    # Convert interval endpoint to float and push one ULP outward.
    return float(np.nextafter(float(x.a), -np.inf))


def hi(x) -> float:
    return float(np.nextafter(float(x.b), +np.inf))


def absmax(x) -> float:
    return max(abs(lo(x)), abs(hi(x)))


def width(x) -> float:
    return hi(x) - lo(x)


def contains(x, value: float, ulps: int = 8) -> bool:
    a = lo(x)
    b = hi(x)
    for _ in range(ulps):
        a = float(np.nextafter(a, -np.inf))
        b = float(np.nextafter(b, +np.inf))
    return a <= float(value) <= b


def radius_interval(base: Any, rho: float):
    # NumPy 2.x / Python 3.13 may repr(np.float64(...)) as
    # 'np.float64(...)', which mpmath.iv.mpf cannot parse as a numeric string.
    # Cast the outward-rounded endpoints back to plain Python floats first.
    r = float(abs(rho))
    r_hi = float(np.nextafter(r, np.inf))
    return mp.iv.mpf([repr(-r_hi), repr(+r_hi)])


def ddU_iv(base: Any, rho, e_reserve: float):
    alpha = base.iv_const(1.0)
    beta = base.iv_const(0.8)
    gamma = base.iv_const(0.2)
    e = base.iv_const(e_reserve)
    return (
        base.iv_const(2.0) * alpha
        - base.iv_const(6.0) * beta * rho
        + base.iv_const(12.0) * gamma * rho**2
        + base.iv_const(3.0) * e**2 / rho**4
    )


def ddV_iv(base: Any, x, barrier_B: float):
    B = base.iv_const(barrier_B)
    one = base.iv_const(1.0)
    return B * (one / x**2 + one / (one - x)**2)


def full_rhs_and_jacobian_iv(
    state: list[Any],
    *,
    kappa0: float,
    params: Any,
    base: Any,
) -> tuple[list[Any], list[list[Any]], Any, Any]:
    """
    Interval F(X), interval J_F(X), support x, and q=2*pi/x.

    Algebra follows audit.full_jacobian exactly, without the harmless
    modulo reduction of delta because sin/cos are periodic.
    """
    (
        y1, v1, theta1, omega1,
        y2, v2, theta2, omega2,
        z, vz, Theta,
    ) = state

    zero = base.iv_const(0.0)
    one = base.iv_const(1.0)
    two = base.iv_const(2.0)

    rho1 = mp.iv.exp(y1)
    rho2 = mp.iv.exp(y2)
    x = one / (one + mp.iv.exp(-z))
    g = x * (one - x)

    if lo(x) <= 0.0 or hi(x) >= 1.0:
        raise FloatingPointError("support x tube left (0,1)")
    if lo(rho1) <= 0.0 or lo(rho2) <= 0.0:
        raise FloatingPointError("rho tube reached nonpositive values")

    delta = theta1 + theta2 - base.iv_const(params.phi0) - Theta
    sd = mp.iv.sin(delta)
    cd = mp.iv.cos(delta)

    k0 = base.iv_const(kappa0)
    ke = k0 * x
    ratio21 = rho2 / rho1
    ratio12 = rho1 / rho2

    gamma_rho = base.iv_const(params.gamma_rho)
    gamma_x = base.iv_const(params.gamma_x)
    mu_x = base.iv_const(params.mu_x)
    mass_x = base.iv_const(params.m_x)

    radial1 = ke * ratio21 * cd
    angular1 = ke * ratio21 * sd
    radial2 = ke * ratio12 * cd
    angular2 = ke * ratio12 * sd

    rho1_dot = rho1 * v1
    rho2_dot = rho2 * v2

    F0 = v1
    F1 = (
        omega1**2
        - base.dU_iv(rho1, params.e_reserve) / rho1
        - radial1
        - gamma_rho * v1
        - v1**2
    )
    F2 = omega1
    F3 = angular1 - two * v1 * omega1

    F4 = v2
    F5 = (
        omega2**2
        - base.dU_iv(rho2, params.e_reserve) / rho2
        - radial2
        - gamma_rho * v2
        - v2**2
    )
    F6 = omega2
    F7 = angular2 - two * v2 * omega2

    F8 = vz
    activity = rho1 * rho2 * sd
    activity_cos = rho1 * rho2 * cd
    x_dot = g * vz
    x_ddot = (
        -base.dV_iv(x, params.barrier_B)
        + mu_x * activity
        - gamma_x * x_dot
    ) / mass_x
    F9 = x_ddot / g - (one - two * x) * vz**2
    F10 = x

    F = [F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10]

    J = [[zero for _ in range(11)] for _ in range(11)]

    J[0][1] = one

    log_u1 = ddU_iv(base, rho1, params.e_reserve) - base.dU_iv(
        rho1, params.e_reserve
    ) / rho1

    J[1][0] = -log_u1 + radial1
    J[1][1] = -gamma_rho - two * v1
    J[1][2] = angular1
    J[1][3] = two * omega1
    J[1][4] = -radial1
    J[1][6] = angular1
    J[1][8] = -k0 * g * ratio21 * cd
    J[1][10] = -angular1

    J[2][3] = one

    J[3][0] = -angular1
    J[3][1] = -two * omega1
    J[3][2] = radial1
    J[3][3] = -two * v1
    J[3][4] = angular1
    J[3][6] = radial1
    J[3][8] = k0 * g * ratio21 * sd
    J[3][10] = -radial1

    J[4][5] = one

    log_u2 = ddU_iv(base, rho2, params.e_reserve) - base.dU_iv(
        rho2, params.e_reserve
    ) / rho2

    J[5][0] = -radial2
    J[5][2] = angular2
    J[5][4] = -log_u2 + radial2
    J[5][5] = -gamma_rho - two * v2
    J[5][6] = angular2
    J[5][7] = two * omega2
    J[5][8] = -k0 * g * ratio12 * cd
    J[5][10] = -angular2

    J[6][7] = one

    J[7][0] = angular2
    J[7][2] = radial2
    J[7][4] = -angular2
    J[7][5] = -two * omega2
    J[7][6] = radial2
    J[7][7] = -two * v2
    J[7][8] = k0 * g * ratio12 * sd
    J[7][10] = -radial2

    J[8][9] = one

    activity_factor = mu_x * activity / (mass_x * g)
    phase_factor = mu_x * activity_cos / (mass_x * g)
    numerator = (
        -base.dV_iv(x, params.barrier_B)
        + mu_x * activity
        - gamma_x * g * vz
    ) / mass_x

    J[9][0] = activity_factor
    J[9][2] = phase_factor
    J[9][4] = activity_factor
    J[9][6] = phase_factor
    J[9][8] = (
        (
            -ddV_iv(base, x, params.barrier_B)
            - gamma_x * (one - two * x) * vz
        ) / mass_x
        - numerator * (one - two * x) / g
        + two * g * vz**2
    )
    J[9][9] = -gamma_x / mass_x - two * (one - two * x) * vz
    J[9][10] = -phase_factor

    J[10][8] = g

    q = (two * mp.iv.pi) / x
    return F, J, x, q


def support_jacobian_iv(
    state: list[Any],
    *,
    kappa0: float,
    params: Any,
    base: Any,
) -> tuple[list[list[Any]], Any]:
    """
    J_H for H=(2*pi/x)F.

    dq/dz = -q*(1-x), because dx/dz=x(1-x).
    Row 10 is set exactly to zero because H_10 == 2*pi.
    """
    F, JF, x, q = full_rhs_and_jacobian_iv(
        state, kappa0=kappa0, params=params, base=base
    )
    one = base.iv_const(1.0)
    zero = base.iv_const(0.0)
    dq_dz = -q * (one - x)

    JH = [[q * JF[i][j] for j in range(11)] for i in range(11)]
    for i in range(10):
        JH[i][8] = JH[i][8] + F[i] * dq_dz

    # Exact cancellation for H_10 = 2*pi.
    JH[10] = [zero for _ in range(11)]
    return JH, x


def gershgorin_mu2_upper(JH: list[list[Any]]) -> float:
    """
    Rigorous upper bound for lambda_max(sym(JH)).

    For symmetric S, lambda_max(S) <= max_i(S_ii + sum_j!=i |S_ij|).
    Each S_ij is itself an outward interval.
    """
    half = mp.iv.mpf(["0.5", "0.5"])
    best = -math.inf
    for i in range(11):
        diag = half * (JH[i][i] + JH[i][i])
        row = hi(diag)
        for j in range(11):
            if i == j:
                continue
            sij = half * (JH[i][j] + JH[j][i])
            row += absmax(sij)
        # Float summation safety.
        row = float(np.nextafter(row, np.inf))
        best = max(best, row)
    return best


def float_support_jacobian(
    state: np.ndarray,
    *,
    kappa0: float,
    params: Any,
    audit: Any,
    base: Any,
) -> np.ndarray:
    F = base.model_derivative_float(
        state, kappa0=kappa0, params=params, audit=audit
    )
    JF = np.asarray(
        audit.full_jacobian(state, kappa0=kappa0, params=params),
        dtype=float,
    )
    x = float(audit.sigmoid(float(state[8])))
    q = 2.0 * math.pi / x
    dq_dz = -q * (1.0 - x)
    JH = q * JF
    JH[:, 8] += F * dq_dz
    JH[10, :] = 0.0
    return JH


def scalar_step(y0: float, R: float, mu_nonnegative: float, h: float) -> float:
    """Comparison solution y' = mu*y + R over width h."""
    mu = max(0.0, float(mu_nonnegative))
    h = float(h)
    y0 = float(y0)
    R = float(R)
    if mu * h < 1.0e-8:
        # Stable series through expm1.
        phi = h if mu == 0.0 else math.expm1(mu * h) / mu
    else:
        phi = math.expm1(mu * h) / mu
    return math.exp(mu * h) * y0 + R * phi


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audit-script", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--residual-script", type=Path, default=DEFAULT_RESIDUAL)
    p.add_argument("--orbit", type=Path, default=DEFAULT_ORBIT)
    p.add_argument("--q", type=float, default=550.0)
    p.add_argument("--gamma-rho", type=float, default=0.3)
    p.add_argument("--tail-periods", type=int, default=70)
    p.add_argument("--steps-per-period", type=int, default=160)
    p.add_argument("--samples-per-rotation", type=int, default=720)
    p.add_argument("--initial-x-perturbation", type=float, default=1e-6)
    p.add_argument("--segment", type=int, default=50382)
    p.add_argument("--subdivision-sweep", default="128,256,512,1024")
    p.add_argument("--iv-dps", type=int, default=60)
    p.add_argument("--tube-growth-factor", type=float, default=1.20)
    p.add_argument("--tube-max-iterations", type=int, default=30)
    p.add_argument("--jacobian-sample-checks", type=int, default=1)
    p.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    mp.iv.dps = args.iv_dps

    audit = load_module(args.audit_script, "q550_local_tube_audit")
    base = load_module(args.residual_script, "q550_interval_residual_base")

    params = audit.ModelParameters(gamma_rho=args.gamma_rho)
    orbit = audit.load_orbit(args.q, args.orbit)
    tail = audit.collect_uniform_tail(
        orbit,
        tail_periods=args.tail_periods,
        steps_per_period=args.steps_per_period,
        samples_per_rotation=args.samples_per_rotation,
        initial_x_perturbation=args.initial_x_perturbation,
        params=params,
        progress_every=100,
    )
    spline = audit.build_state_spline(tail)

    knots = np.asarray(spline.x, dtype=float)
    coeffs = np.asarray(spline.c, dtype=float)
    seg = int(args.segment)
    if not (0 <= seg < knots.size - 1):
        raise IndexError(f"segment {seg} outside 0..{knots.size-2}")

    sL = float(knots[seg])
    sR = float(knots[seg + 1])
    total_width = sR - sL
    kappa0 = (55.0 / 6.0) * (10.0 / args.q)
    sweep = parse_int_list(args.subdivision_sweep)

    print(f"Script version       = {SCRIPT_VERSION}")
    print(f"Q                    = {args.q:g}")
    print(f"audit SHA256         = {sha256(args.audit_script)}")
    print(f"residual SHA256      = {sha256(args.residual_script)}")
    print(f"orbit SHA256         = {sha256(args.orbit)}")
    print(f"segment              = {seg}")
    print(f"segment domain       = [{sL:.15f}, {sR:.15f}]")
    print(f"segment width        = {total_width:.15e}")
    print(f"subdivision sweep    = {sweep}")
    print("initial exact IVP    = X_exact(sL) = X_hat(sL)")
    print("error equation       = e' = H(X_hat+e)-H(X_hat)-r_s")
    print()

    summary_rows: list[dict[str, Any]] = []
    subcell_rows: list[dict[str, Any]] = []

    overall_pass = True

    for M in sweep:
        h = total_width / M
        y = 0.0
        max_y = 0.0
        max_trial_rho = 0.0
        max_mu = -math.inf
        max_mu_positive = 0.0
        max_R = 0.0
        max_residual_component = 0.0
        max_state_width = 0.0
        min_x = 1.0
        max_x = 0.0
        all_residual_samples = True
        all_jacobian_samples = True
        all_self_consistent = True
        worst_subcell = -1
        worst_mu_subcell = -1
        worst_R_subcell = -1

        for j in range(M):
            a = sL + total_width * j / M
            b = sL + total_width * (j + 1) / M
            u0 = a - sL
            u1 = b - sL

            spline_box = [
                base.spline_component_box(coeffs[:, seg, k], u0, u1)
                for k in range(11)
            ]
            deriv_box = [
                base.spline_derivative_box(coeffs[:, seg, k], u0, u1)
                for k in range(11)
            ]
            rhs_hat_box = base.support_rhs_iv(
                spline_box, kappa0=kappa0, params=params
            )
            residual_box = [
                deriv_box[k] - rhs_hat_box[k] for k in range(11)
            ]
            comp_R = [absmax(r) for r in residual_box]
            R = float(np.nextafter(
                math.sqrt(sum(v*v for v in comp_R)), np.inf
            ))
            max_R = max(max_R, R)
            max_residual_component = max(max_residual_component, max(comp_R))
            max_state_width = max(max_state_width, *(width(x) for x in spline_box))

            # Point residual containment sanity checks.
            for q in np.linspace(a, b, 5)[1:-1]:
                state = np.asarray(spline(float(q)), dtype=float)
                deriv = np.asarray(spline(float(q), 1), dtype=float)
                target = base.support_rhs_float(
                    state, kappa0=kappa0, params=params, audit=audit
                )
                r = deriv - target
                for k in range(11):
                    if not contains(residual_box[k], float(r[k])):
                        all_residual_samples = False
                        break

            # Initial tube guess: conservative zero-mu forcing plus current y.
            trial = max(
                y + 2.0 * R * h,
                y * 1.10 + R * h * 2.0,
                1.0e-15,
            )

            accepted = False
            accepted_mu = math.nan
            accepted_yout = math.nan
            accepted_xlo = math.nan
            accepted_xhi = math.nan
            accepted_iter = -1
            accepted_jac_sample_ok = True

            for iteration in range(args.tube_max_iterations):
                delta = radius_interval(base, trial)
                tube_box = [x + delta for x in spline_box]

                JH_box, x_box = support_jacobian_iv(
                    tube_box, kappa0=kappa0, params=params, base=base
                )
                mu = gershgorin_mu2_upper(JH_box)
                mu_pos = max(0.0, mu)
                yout = scalar_step(y, R, mu_pos, h)
                needed = max(y, yout)

                jac_sample_ok = True
                if args.jacobian_sample_checks > 0:
                    for q in np.linspace(
                        a, b, args.jacobian_sample_checks + 2
                    )[1:-1]:
                        state = np.asarray(spline(float(q)), dtype=float)
                        Jfloat = float_support_jacobian(
                            state,
                            kappa0=kappa0,
                            params=params,
                            audit=audit,
                            base=base,
                        )
                        for ii in range(11):
                            for jj in range(11):
                                if not contains(JH_box[ii][jj], Jfloat[ii, jj]):
                                    jac_sample_ok = False
                                    break
                            if not jac_sample_ok:
                                break
                        if not jac_sample_ok:
                            break

                if needed <= trial and jac_sample_ok:
                    accepted = True
                    accepted_mu = mu
                    accepted_yout = yout
                    accepted_xlo = lo(x_box)
                    accepted_xhi = hi(x_box)
                    accepted_iter = iteration + 1
                    accepted_jac_sample_ok = jac_sample_ok
                    break

                trial = max(
                    trial * args.tube_growth_factor,
                    needed * args.tube_growth_factor,
                    1.0e-15,
                )

            if not accepted:
                all_self_consistent = False
                overall_pass = False
                print(
                    f"M={M} subcell={j}: SELF-CONSISTENCY FAILED "
                    f"after {args.tube_max_iterations} iterations",
                    flush=True,
                )
                break

            all_jacobian_samples = (
                all_jacobian_samples and accepted_jac_sample_ok
            )
            y = float(np.nextafter(accepted_yout, np.inf))
            max_y = max(max_y, y)
            max_trial_rho = max(max_trial_rho, trial)
            max_mu = max(max_mu, accepted_mu)
            max_mu_positive = max(max_mu_positive, max(0.0, accepted_mu))
            min_x = min(min_x, accepted_xlo)
            max_x = max(max_x, accepted_xhi)

            if y >= max_y:
                worst_subcell = j
            if accepted_mu >= max_mu:
                worst_mu_subcell = j
            if R >= max_R:
                worst_R_subcell = j

            subcell_rows.append({
                "subdivisions": M,
                "subcell_index": j,
                "left": a,
                "right": b,
                "residual_norm_upper": R,
                "mu2_gershgorin_upper": accepted_mu,
                "mu_positive": max(0.0, accepted_mu),
                "incoming_error_radius": 0.0 if j == 0 else math.nan,
                "outgoing_error_radius": y,
                "accepted_tube_radius": trial,
                "self_consistency_margin": trial - max(y, 0.0),
                "tube_iterations": accepted_iter,
                "x_lower": accepted_xlo,
                "x_upper": accepted_xhi,
                "jacobian_sample_containment_passed": accepted_jac_sample_ok,
            })

        segment_pass = bool(
            all_self_consistent
            and all_residual_samples
            and all_jacobian_samples
        )
        overall_pass = overall_pass and segment_pass

        summary = {
            "segment_index": seg,
            "subdivisions": M,
            "segment_width": total_width,
            "final_error_radius": y,
            "maximum_error_radius": max_y,
            "maximum_accepted_tube_radius": max_trial_rho,
            "maximum_residual_norm_upper": max_R,
            "maximum_residual_component_upper": max_residual_component,
            "maximum_mu2_gershgorin_upper": max_mu,
            "maximum_mu_positive": max_mu_positive,
            "maximum_spline_state_box_width": max_state_width,
            "minimum_support_x_in_tube": min_x,
            "maximum_support_x_in_tube": max_x,
            "residual_sample_containment_passed": all_residual_samples,
            "jacobian_sample_containment_passed": all_jacobian_samples,
            "self_consistency_passed": all_self_consistent,
            "segment_pass": segment_pass,
        }
        summary_rows.append(summary)

        print(
            f"M={M:4d} "
            f"Rmax={max_R:.6e} "
            f"muMax={max_mu:.6e} "
            f"rhoFinal={y:.6e} "
            f"rhoTubeMax={max_trial_rho:.6e} "
            f"x=[{min_x:.9f},{max_x:.9f}] "
            f"resOK={all_residual_samples} "
            f"jacOK={all_jacobian_samples} "
            f"selfOK={all_self_consistent} "
            f"pass={segment_pass}",
            flush=True,
        )

    largest = max(sweep)
    largest_row = [r for r in summary_rows if r["subdivisions"] == largest][0]

    if overall_pass:
        classification = (
            "LOCAL EXACT-NONLINEAR ORBIT TUBE PILOT PASSED: for the tested cubic "
            "segment, the exact support-coordinate IVP initialized at the spline "
            "left endpoint is rigorously enclosed in a self-consistent Euclidean "
            "tube around the spline. NEXT: multi-segment propagation / reset strategy."
        )
    else:
        classification = (
            "LOCAL EXACT-NONLINEAR ORBIT TUBE PILOT OPEN: at least one residual, "
            "Jacobian containment, or self-consistency check failed."
        )

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(Path(str(prefix) + "_summary.csv"), summary_rows)
    write_csv(Path(str(prefix) + "_subcells.csv"), subcell_rows)

    payload = {
        "script_version": SCRIPT_VERSION,
        "scope": "local exact nonlinear support-coordinate IVP tube pilot",
        "q": args.q,
        "audit_sha256": sha256(args.audit_script),
        "residual_script_sha256": sha256(args.residual_script),
        "orbit_sha256": sha256(args.orbit),
        "segment_index": seg,
        "segment_domain": [sL, sR],
        "segment_width": total_width,
        "initial_condition": "X_exact(sL)=X_hat(sL), error radius 0",
        "error_equation": "e'=H(X_hat+e)-H(X_hat)-r_s",
        "subdivision_sweep": sweep,
        "interval_dps": args.iv_dps,
        "summaries": summary_rows,
        "largest_subdivision_result": largest_row,
        "classification": classification,
    }
    Path(str(prefix) + ".json").write_text(json.dumps(payload, indent=2) + "\n")

    with Path(str(prefix) + "_report.txt").open("w") as f:
        f.write(f"Script version: {SCRIPT_VERSION}\n\n")
        f.write("Q=550 LOCAL EXACT NONLINEAR ORBIT TUBE PILOT\n")
        f.write("============================================\n\n")
        f.write(f"segment index       = {seg}\n")
        f.write(f"segment domain      = [{sL:.15f}, {sR:.15f}]\n")
        f.write(f"segment width       = {total_width:.15e}\n")
        f.write("initial condition    = X_exact(sL) = X_hat(sL)\n")
        f.write("error equation       = e'=H(X_hat+e)-H(X_hat)-r_s\n\n")
        f.write("Method\n------\n")
        f.write(
            "Outward interval residual + outward interval J_H + interval "
            "Gershgorin upper bound for mu_2(J_H) + scalar comparison equation "
            "with subcellwise self-consistent Euclidean tubes.\n\n"
        )
        f.write("Summaries\n---------\n")
        for r in summary_rows:
            f.write(
                f"M={int(r['subdivisions']):4d} "
                f"Rmax={r['maximum_residual_norm_upper']:.9e} "
                f"muMax={r['maximum_mu2_gershgorin_upper']:.9e} "
                f"rhoFinal={r['final_error_radius']:.9e} "
                f"rhoTubeMax={r['maximum_accepted_tube_radius']:.9e} "
                f"resOK={r['residual_sample_containment_passed']} "
                f"jacOK={r['jacobian_sample_containment_passed']} "
                f"selfOK={r['self_consistency_passed']} "
                f"pass={r['segment_pass']}\n"
            )
        f.write("\n")
        f.write(f"classification = {classification}\n\n")
        f.write("Proof boundary\n--------------\n")
        f.write(
            "PASS is local: it validates the exact IVP initialized on the spline at "
            "the left endpoint of this one cubic segment. A full-orbit validation "
            "still requires propagation across many segments without unjustified "
            "resets, or a rigorous shadowing/periodic-orbit boundary argument.\n"
        )

    print("=" * 78)
    print("classification = " + classification)
    print(
        f"largest M={largest}: final exact-orbit error radius "
        f"<= {largest_row['final_error_radius']:.6e}"
    )
    print(f"report          = {prefix}_report.txt")
    print(f"JSON            = {prefix}.json")
    print(f"summary CSV     = {prefix}_summary.csv")
    print(f"subcells CSV    = {prefix}_subcells.csv")


if __name__ == "__main__":
    main()
