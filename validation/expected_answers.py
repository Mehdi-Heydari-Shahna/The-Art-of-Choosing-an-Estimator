"""Fresh validation cases for revision 2 and their expected answers, established WITHOUT the package.

    python -I validation/expected_answers.py        # writes validation/cases.json and validation/expected.json

This file does not import heros.  Every expected answer follows from a closed form, a short recursion written here, or
a construction stated in the reason.  The cases were fixed and committed (git tag validation-frozen) before the frozen
software (tag revision-2-frozen) was run on them; experiments/run_validation.py then runs each request through the
public interface and compares.

Answers:  "release" (with the value the claim must not exceed), "ask: <field>" (exactly one missing fact), or
"decline: <kind>" (no estimator can meet it / no verified design / the data contradict the request).
"""
import json
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def servo(b):
    return {"family": "validation_axis", "kind": "second_order",
            "states": [{"name": "axis position", "unit": "rad"}, {"name": "axis speed", "unit": "rad/s"}],
            "inputs": [{"name": "motor torque", "unit": "N m"}], "outputs": [{"name": "encoder", "unit": "rad"}],
            "linear": {"clock": "continuous", "A": [[0.0, 1.0], [0.0, 0.0]], "B": [[0.0], [b]], "C": [[1.0, 0.0]], "Ts_native": None},
            "second_order": {"f": "b*u", "f0": "b*u", "params": {"b": b}, "param_ranges": {}, "output": "x1"}, "flags": []}


def scalar(a):
    return {"family": "validation_scalar", "kind": "linear", "states": [{"name": "level", "unit": "m"}], "inputs": [{"name": "valve", "unit": "-"}],
            "outputs": [{"name": "level gauge", "unit": "m"}],
            "linear": {"clock": "discrete", "A": [[a]], "B": [[0.0]], "C": [[1.0]], "Ts_native": 1.0}, "second_order": None, "flags": []}


# ------------------------------------------------------------------------------------------------ independent formulas
def scalar_filtered_variances(a, Q, R, P0, n):
    """Optimal (Kalman) filtered variances P_{k|k}, k = 0..n, of x+ = a x + w, y = x + v (update, then predict)."""
    out, P = [], P0
    for _ in range(n + 1):
        Pf = P * R / (P + R)
        out.append(Pf)
        P = a * a * Pf + Q
    return out


def scalar_steady_filtered(a, Q, R):
    """Positive root of P^2 + (R (1 - a^2) - Q) P - Q R = 0 (prior), then P R / (P + R)."""
    bq = R * (1 - a * a) - Q
    P = 0.5 * (-bq + math.sqrt(bq * bq + 4 * Q * R))
    return P * R / (P + R)


def observer_ultimate(T, eps, D, V, n=None):
    """lim sup of the worst-case velocity error of the pole-matched sampled observer (gain M = [1 - p^2, (1 - p)^2 / T]):
    the sum over j of D * int_0^T |r_j . [tau, 1]| dtau + V |r_j . Ad M| + V |M_2|, with r_j = [0 1](I - M C) Phi^j,
    evaluated term by term in float64 until the terms are below 1e-18 of the sum."""
    p = math.exp(-T / eps)
    M = np.array([1 - p * p, (1 - p) ** 2 / T])
    Ad = np.array([[1.0, T], [0.0, 1.0]])
    IMC = np.array([[1 - M[0], 0.0], [-M[1], 1.0]])
    Phi = Ad @ IMC
    r = np.array([0.0, 1.0]) @ IMC
    AdM = Ad @ M
    total = V * abs(M[1])
    for j in range(2_000_000):
        r1, r2 = r
        f0, fT = r2, r1 * T + r2                     # the weight at tau = 0 and tau = T
        if f0 * fT >= 0:
            integ = 0.5 * T * (abs(f0) + abs(fT))
        else:
            integ = 0.5 * T * (f0 * f0 + fT * fT) / (abs(f0) + abs(fT))
        term = D * integ + V * abs(r @ AdM)
        total += term
        if j > 50 and term < 1e-18 * total:
            break
        r = r @ Phi
    return total


def best_observer_ultimate(T, D, V):
    grid = np.exp(np.linspace(math.log(0.4 * T), math.log(400 * T), 160))
    vals = [(observer_ultimate(T, e, D, V), e) for e in grid]
    return min(vals)


def kalman_steady_velocity_variance(T, D, V):
    """Steady-state filtered velocity variance of the Kalman filter for the double integrator with acceleration noise of
    variance D^2/3 held per sample and measurement noise of variance V^2/3 (uniform noises), by Riccati iteration."""
    A = np.array([[1.0, T], [0.0, 1.0]]); G = np.array([[T * T / 2], [T]]); C = np.array([[1.0, 0.0]])
    Q = (D * D / 3) * (G @ G.T); R = V * V / 3
    P = np.eye(2)
    for _ in range(200000):
        S = float((C @ P @ C.T)[0, 0]) + R
        K = P @ C.T / S
        Pf = P - K @ C @ P
        Pn = A @ Pf @ A.T + Q
        if np.max(np.abs(Pn - P)) <= 1e-16 * np.max(np.abs(P)):
            P = Pn
            break
        P = Pn
    S = float((C @ P @ C.T)[0, 0]) + R
    Pf = P - (P @ C.T / S) @ C @ P
    return float(Pf[1, 1]), Q.tolist(), [[R]]


# ------------------------------------------------------------------------------------------------ cases
def main():
    cases, expected = [], {}

    def add(cid, request, spec, answer, value, reason, data=None):
        cases.append({"id": cid, "request": request, "spec": spec, **({"data": data} if data else {})})
        expected[cid] = {"answer": answer, "value": value, "reason": reason}

    # V01 release: Kalman filter, finite record
    a, Q, R, P0, n = 1.0, 0.5, 2.0, 10.0, 5
    pf = scalar_filtered_variances(a, Q, R, P0, n)
    add("V01", "level gauge, random-walk level, Q = 0.5, R = 2, P0 = 10, record of 5 s at 1 s; smallest mean-square error",
        {"plant": scalar(a), "target": 0, "Ts": 1.0, "Q": [[Q]], "Q_kind": "discrete_cov", "R": [[R]], "P0": [[P0]], "x0_hat": [0.0],
         "distribution": "gaussian", "white": True, "t_end": 5.0, "objective": {"kind": "mse_optimal", "horizon": "finite"}},
        "release", max(pf), f"Kalman recursion from P0 = 10: P_k|k = {', '.join(f'{v:.4g}' for v in pf)}; the largest is at t = 0")

    # V02 release: steady-state level
    a, Q, R = 0.8, 0.36, 1.0
    pss = scalar_steady_filtered(a, Q, R)
    add("V02", "AR(1) level, a = 0.8, Q = 0.36, R = 1; steady-state mean-square error at most 0.5",
        {"plant": scalar(a), "target": 0, "Ts": 1.0, "Q": [[Q]], "Q_kind": "discrete_cov", "R": [[R]], "distribution": "gaussian", "white": True,
         "objective": {"kind": "mse_bound", "horizon": "steady_state", "mse_target": 0.5}},
        "release", pss, f"scalar Riccati equation: prior P = 0.6, filtered P R / (P + R) = {pss:.6g} <= 0.5")

    # V03 release: guaranteed ultimate speed error, observer
    T, D, V, tol = 0.001, 4.0, 1e-4, 0.06
    w, e = best_observer_ultimate(T, D, V)
    add("V03", "axis, encoder error <= 1e-4 rad, unmodelled acceleration <= 4 rad/s^2, 1 kHz; speed error eventually <= 0.06 rad/s, guaranteed",
        {"plant": servo(2.0), "target": 1, "Ts": T, "V": V, "D": D, "E": [1e-3, 2.0], "objective": {"kind": "hard_bound", "tolerance": tol}},
        "release", tol, f"a pole-matched observer with eps = {e * 1e3:.3g} ms has exact ultimate worst case {w:.4g} <= {tol} (term-by-term sum)")

    # V04 release: total and additive disturbance bounds together
    T, V, tol = 0.001, 1e-4, 0.045
    w, e = best_observer_ultimate(T, 1.5, V)
    add("V04", "same axis, total disturbance <= 6 rad/s^2 and, with the exact model b u, additive disturbance <= 1.5 rad/s^2; speed error eventually <= 0.045",
        {"plant": servo(2.0), "target": 1, "Ts": T, "V": V, "D": 6.0, "d_bound": 1.5, "E": [1e-3, 2.0], "objective": {"kind": "hard_bound", "tolerance": tol}},
        "release", tol, f"with the exact model the acceleration error is the additive disturbance (<= 1.5); eps = {e * 1e3:.3g} ms gives {w:.4g} <= {tol}, "
                        f"whereas with 6 alone no estimator could (2 sqrt(6 V) = {2 * math.sqrt(6 * V):.4g})")

    # V05 ask: measurement-error bound missing
    add("V05", "axis, unmodelled acceleration <= 4, initial errors given; speed error <= 0.1 rad/s after 0.05 s, guaranteed; encoder error bound not given",
        {"plant": servo(2.0), "target": 1, "Ts": 0.001, "D": 4.0, "E": [1e-3, 2.0], "objective": {"kind": "hard_bound", "tolerance": 0.1, "deadline": 0.05}},
        "ask: deterministic.V", None, "every gain passes the measurement error to the speed estimate with a nonzero weight, so no finite bound holds without it")

    # V06 ask: measurement-noise variance missing
    add("V06", "level gauge, random walk, Q = 0.5, P0 = 10; smallest mean-square error over 5 s; sensor noise variance not given",
        {"plant": scalar(1.0), "target": 0, "Ts": 1.0, "Q": [[0.5]], "Q_kind": "discrete_cov", "P0": [[10.0]], "x0_hat": [0.0], "distribution": "gaussian",
         "white": True, "t_end": 5.0, "objective": {"kind": "mse_optimal", "horizon": "finite"}},
        "ask: stochastic.R", None, "the optimal gain P/(P + R) and the reported variance depend on R")

    # V07 ask: initial-error bound missing with a deadline
    add("V07", "axis, encoder error <= 1e-4, acceleration <= 4; speed error <= 0.1 rad/s from 0.02 s on; no initial-error bound",
        {"plant": servo(2.0), "target": 1, "Ts": 0.001, "V": 1e-4, "D": 4.0, "objective": {"kind": "hard_bound", "tolerance": 0.1, "deadline": 0.02}},
        "ask: initial.E", None, "at t = 0.02 s the initial velocity error still enters the error with a positive weight, so an unbounded initial error defeats any finite limit")

    # V08 decline: two-signal limit
    T, D, V, tol = 0.001, 4.0, 1e-4, 0.035
    add("V08", "axis, encoder error <= 1e-4, acceleration <= 4 (arbitrary); speed error eventually <= 0.035 rad/s, guaranteed",
        {"plant": servo(2.0), "target": 1, "Ts": T, "V": V, "D": D, "E": [1e-3, 2.0], "objective": {"kind": "hard_bound", "tolerance": tol}},
        "decline: impossible", 2 * math.sqrt(D * V), f"two admissible signals with identical readings differ in speed by 4 sqrt(DV); 2 sqrt(DV) = {2 * math.sqrt(D * V):.4g} > {tol}")

    # V09 decline: transient cap at the first sample
    add("V09", "axis as V03; speed error <= 0.2 rad/s after 0.05 s and never above 0.5 rad/s; initial speed uncertainty 2 rad/s",
        {"plant": servo(2.0), "target": 1, "Ts": 0.001, "V": 1e-4, "D": 4.0, "E": [1e-3, 2.0],
         "objective": {"kind": "hard_bound", "tolerance": 0.2, "deadline": 0.05, "peak_cap": 0.5}},
        "decline: impossible", 2.0, "at t = 0 only one position reading exists; it says nothing about the speed, which may be anywhere in [-2, 2] rad/s, so some admissible speed is 2 rad/s from any estimate")

    # V10 decline: mean-square error below the optimum
    a, Q, R = 0.8, 0.36, 1.0
    pss = scalar_steady_filtered(a, Q, R)
    add("V10", "AR(1) level as V02; steady-state mean-square error at most 0.3",
        {"plant": scalar(a), "target": 0, "Ts": 1.0, "Q": [[Q]], "Q_kind": "discrete_cov", "R": [[R]], "distribution": "gaussian", "white": True,
         "objective": {"kind": "mse_bound", "horizon": "steady_state", "mse_target": 0.3}},
        "decline: impossible", pss, f"the steady-state Kalman filter attains the minimum over all estimators (Gaussian), {pss:.6g} > 0.3")

    # V11 decline: no verified design for a probability level
    T, D, V = 0.001, 6.0, 1e-4
    var, Qd, Rd = kalman_steady_velocity_variance(T, D, V)
    P0 = [[(1e-3) ** 2 / 3, 0.0], [0.0, 4.0 / 3]]
    add("V11", "axis, uniform noises (acceleration within +-6, encoder within +-1e-4); speed error within 0.02 rad/s with probability 0.9",
        {"plant": servo(2.0), "target": 1, "Ts": T, "V": V, "D": D, "E": [1e-3, 2.0], "Q": Qd, "Q_kind": "discrete_cov", "R": Rd, "P0": P0,
         "distribution": "bounded", "white": True, "objective": {"kind": "probability", "horizon": "steady_state", "tolerance": 0.02, "prob": 0.9}},
        "decline: no verified design", 1 - var / 0.02 ** 2,
        f"the smallest error variance of any linear estimator is {var:.4g}; Chebyshev then proves only {1 - var / 0.02 ** 2:.3f} < 0.9, and the worst-case "
        f"limit 2 sqrt(DV) = {2 * math.sqrt(D * V):.4g} > 0.02 rules out a probability-one bound; neither proves the level unattainable")

    # V12 decline: the data were recorded at another period
    rng = np.random.default_rng(12)
    x = np.array([0.0, 1.0]); rows = []
    for k in range(300):
        y = x[0] + rng.uniform(-1e-4, 1e-4); rows.append([round(k * 0.002, 6), float(y), 0.0])
        acc = rng.uniform(-4, 4); x = np.array([x[0] + 0.002 * x[1] + 2e-6 * acc, x[1] + 0.002 * acc])
    add("V12", "request of V03 stated at 1 kHz; the attached record has timestamps every 2 ms",
        {"plant": servo(2.0), "target": 1, "Ts": 0.001, "V": 1e-4, "D": 4.0, "E": [1e-3, 2.0], "objective": {"kind": "hard_bound", "tolerance": 0.06}},
        "decline: data contradict the request", 0.002, "the gains, the model and the certificate are computed for 1 ms; the record is sampled every 2 ms",
        data={"t": [r[0] for r in rows], "y": [r[1] for r in rows], "u": [r[2] for r in rows]})

    (HERE / "cases.json").write_text(json.dumps({"cases": cases}, indent=1))
    (HERE / "expected.json").write_text(json.dumps({"note": "computed by validation/expected_answers.py without the package", "expected": expected}, indent=1))
    for cid, e in expected.items():
        print(cid, e["answer"], e["value"])


if __name__ == "__main__":
    main()
