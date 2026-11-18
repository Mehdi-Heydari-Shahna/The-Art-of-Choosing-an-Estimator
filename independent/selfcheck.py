"""Self-consistency checks of ref_kalman.py / ref_hgo.py against closed forms.

Run `python selfcheck.py`; prints one PASS/FAIL line per check and exits 0 iff all pass.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_hgo as H      # noqa: E402
import ref_kalman as K   # noqa: E402

FAILS = []


def check(name, err, tol):
    ok = bool(np.isfinite(err) and err <= tol)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: err = {err:.3e} (tol {tol:.1e})")
    if not ok:
        FAILS.append(name)


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def map_last(A, B, C, Qd, R, x0h, P0, y, u):
    """Mean/covariance of x_N given y_0..y_N from the whitened least-squares (MAP)
    trajectory problem, solved by QR -- an information-form check of batch_filtered."""
    n, N = A.shape[0], y.shape[0] - 1
    wh = lambda S: np.linalg.inv(np.linalg.cholesky(S))      # whitening factor
    blk = np.zeros((n, (N + 1) * n)); blk[:, :n] = wh(P0)
    rows, rhs = [blk], [wh(P0) @ x0h]
    for k in range(N + 1):
        blk = np.zeros((C.shape[0], (N + 1) * n)); blk[:, k * n:(k + 1) * n] = wh(R) @ C
        rows.append(blk); rhs.append(wh(R) @ y[k])
        if k < N:
            blk = np.zeros((n, (N + 1) * n))
            blk[:, (k + 1) * n:(k + 2) * n], blk[:, k * n:(k + 1) * n] = wh(Qd), -wh(Qd) @ A
            rows.append(blk); rhs.append(wh(Qd) @ (B @ u[k]))
    Qf, Rf = np.linalg.qr(np.vstack(rows))
    xall = np.linalg.solve(Rf, Qf.T @ np.concatenate(rhs))
    Rnn = Rf[-n:, -n:]
    return xall[-n:], np.linalg.inv(Rnn.T @ Rnn)


def check_kalman():
    one = np.eye(1)
    # (a) scalar static model y = x + v, single measurement
    for p, r, frac in ((0.5, 1.5, 0.25), (1.0, 1.0, 0.5)):
        yv = 0.7321
        m, P = K.batch_filtered(one, None, one, 0 * one, r * one, [0.0], p * one, [[yv]])
        err = max(abs(m[0, 0] - frac * yv), abs(m[0, 0] - p / (p + r) * yv),
                  abs(P[0, 0, 0] - p * r / (p + r)))
        check(f"(a) static scalar p={p}, r={r}: mean y*{frac}, var p r/(p+r)", err, 1e-14)
        m, _ = K.batch_filtered(one, None, one, 0 * one, r * one, [0.3], p * one, [[yv]])
        check(f"(a) same with x0 = 0.3", abs(m[0, 0] - (0.3 + p / (p + r) * (yv - 0.3))), 1e-14)
    # (a1) static scalar, repeated measurements: information-form closed form
    rng = np.random.default_rng(1)
    p, r, ys = 0.5, 1.5, rng.standard_normal(30)
    m, P = K.batch_filtered(one, None, one, 0 * one, r * one, [0.0], p * one, ys)
    var = 1.0 / (1.0 / p + np.arange(1, 31) / r)
    check("(a1) static scalar, 30 measurements vs closed form",
          max(rel(P[:, 0, 0], var), rel(m[:, 0], var * np.cumsum(ys) / r)), 1e-12)
    # (a2) 2-state model with input: last filtered mean vs MAP trajectory by lstsq
    T, N = 0.1, 25
    A, B, C = np.array([[1, T], [0, 1.0]]), np.array([[T * T / 2], [T]]), np.array([[1.0, 0]])
    Qd = 0.3 * np.array([[T ** 3 / 3, T ** 2 / 2], [T ** 2 / 2, T]])
    R, P0, x0h = np.array([[0.04]]), np.diag([0.5, 2.0]), np.array([0.2, -0.4])
    u, y = rng.standard_normal((N, 1)), rng.standard_normal((N + 1, 1))
    means, covs = K.batch_filtered(A, B, C, Qd, R, x0h, P0, y, u)
    xmap, cov_map = map_last(A, B, C, Qd, R, x0h, P0, y, u)
    check("(a2) batch_filtered at k = N vs weighted-least-squares MAP trajectory",
          max(rel(means[-1], xmap), rel(covs[-1], cov_map)), 1e-8)
    m_mp, c_mp = K.batch_filtered(A, B, C, Qd, R, x0h, P0, y, u, dps=40)
    check("(a3) float64 batch_filtered vs mpmath (dps=40) batch_filtered, all k",
          max(rel(means, m_mp), rel(covs, c_mp)), 1e-9)
    Au, Qu = np.array([[1.5, 0.1], [0.0, 0.9]]), 0.01 * np.eye(2)
    m_un, c_un = K.batch_filtered(Au, B, C, Qu, R, x0h, P0, y, u, dps=40)
    xmap, cov_map = map_last(Au, B, C, Qu, R, x0h, P0, y, u)
    check("(a4) unstable A (eig 1.5, N = 25): mpmath batch_filtered at k = N vs MAP by QR",
          max(rel(m_un[-1], xmap), rel(c_un[-1], cov_map)), 1e-8)
    m_fl, c_fl = K.batch_filtered(Au, B, C, Qu, R, x0h, P0, y, u)
    print(f"      (info) same case in float64: rel. error mean {rel(m_fl[-1], xmap):.1e}, cov {rel(c_fl[-1], cov_map):.1e}")
    K0 = K.batch_gain_first(A, C, Qd, R, P0)
    check("batch_gain_first vs P0 C'(C P0 C' + R)^-1 and vs batch mean",
          max(rel(K0, P0 @ C.T / (C @ P0 @ C.T + R)),
              rel(means[0], x0h + K0 @ (y[0] - C @ x0h))), 1e-13)
    # (b) scalar random walk Riccati root
    for q, r in ((0.3, 2.0), (1e-4, 1.0), (5.0, 0.01)):
        Pp, Pf, M = K.riccati_fixed_point(one, one, q * one, r * one)
        root = 0.5 * (q + np.sqrt(q * q + 4 * q * r))
        err = max(abs(Pp[0, 0] / root - 1), abs(Pf[0, 0] / (root * r / (root + r)) - 1),
                  abs(M[0, 0] / (root / (root + r)) - 1))
        check(f"(b) riccati_fixed_point scalar random walk q={q}, r={r}", err, 1e-10)
    Pp, Pf, M = K.riccati_fixed_point(A, C, Qd, R)
    _, covs_long = K.batch_filtered(A, B, C, Qd, R, x0h, P0, np.zeros((81, 1)))
    check("(b') batch filtered covariance at k = 80 vs Riccati fixed point", rel(covs_long[-1], Pf), 1e-7)
    check("lyapunov_mse_of_gain at the Riccati gain vs P_filtered",
          rel(K.lyapunov_mse_of_gain(A, C, Qd, R, M), Pf), 1e-9)
    Msub = np.array([[0.3], [0.5]])
    Plyap = K.lyapunov_mse_of_gain(A, C, Qd, R, Msub)
    for dist in ("gaussian", "uniform"):
        mse = K.fixed_gain_mse_montecarlo(A, B, C, Qd, R, Msub, x0h, P0, 150, 40000, 7, dist)
        check(f"fixed_gain_mse_montecarlo ({dist}) tail vs Lyapunov diagonal",
              rel(mse[100:].mean(axis=0), np.diag(Plyap)), 0.03)
    Pinit = (np.eye(2) - Msub @ C) @ P0 @ (np.eye(2) - Msub @ C).T + Msub @ R @ Msub.T
    check("fixed_gain_mse_montecarlo at k = 0 vs Joseph formula", rel(mse[0], np.diag(Pinit)), 0.03)
    # (c) discretisation by quadrature
    Ts, q = 0.37, 2.5
    Ac, Bc = np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0], [1.0]])
    Ad, Bd = K.zoh_quadrature(Ac, Bc, Ts)
    Qq = K.process_cov_quadrature(Ac, Bc, [[q]], Ts)
    err = max(rel(Ad, [[1, Ts], [0, 1]]), rel(Bd, [[Ts ** 2 / 2], [Ts]]),
              rel(Qq, q * np.array([[Ts ** 3 / 3, Ts ** 2 / 2], [Ts ** 2 / 2, Ts]])))
    check("(c) zoh / process-cov quadrature, double integrator", err, 1e-13)
    a = 3.0
    Ad, Bd = K.zoh_quadrature([[-a]], [[1.0]], Ts)
    Qq = K.process_cov_quadrature([[-a]], [[1.0]], [[q]], Ts)
    err = max(abs(Ad[0, 0] / np.exp(-a * Ts) - 1), abs(Bd[0, 0] * a / (1 - np.exp(-a * Ts)) - 1),
              abs(Qq[0, 0] * 2 * a / (q * (1 - np.exp(-2 * a * Ts))) - 1))
    check("(c') zoh / process-cov quadrature, scalar lag", err, 1e-13)
    w, th = 40.0, 40.0 * Ts                                   # oscillator, ||Ac|| Ts = 592
    Ad, Bd = K.zoh_quadrature([[0, 1], [-w * w, 0]], Bc, Ts)
    Qq = K.process_cov_quadrature([[0, 1], [-w * w, 0]], Bc, [[q]], Ts)
    Qex = q * np.array([[(Ts / 2 - np.sin(2 * th) / (4 * w)) / w ** 2, np.sin(th) ** 2 / (2 * w ** 2)],
                        [np.sin(th) ** 2 / (2 * w ** 2), Ts / 2 + np.sin(2 * th) / (4 * w)]])
    err = max(rel(Ad, [[np.cos(th), np.sin(th) / w], [-w * np.sin(th), np.cos(th)]]),
              rel(Bd, [[(1 - np.cos(th)) / w ** 2], [np.sin(th) / w]]), rel(Qq, Qex))
    check("(c'') zoh / process-cov quadrature, oscillator w*Ts = 14.8", err, 1e-11)


def check_hgo():
    T, M, E, D, V, N, sub = 0.01, np.array([0.5, 20.0]), np.array([0.02, 0.3]), 1.5, 2e-3, 40, 16
    rng = np.random.default_rng(3)
    for form in ("current", "predictor"):
        for c in (np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([0.7, -0.2])):
            wc = H.worst_case_by_enumeration(T, M, c, E, D, V, N, form, sub)
            x0e, v, dfn = H.worst_case_signals(T, M, c, E, D, V, N, form, sub)
            xm0, a0 = np.array([0.4, -1.3]), rng.standard_normal(N)
            out = H.simulate_observer(T, M, xm0 + x0e, xm0, v, dfn, a0, N, substeps=4 * sub)
            est = out["xp"] if form == "current" else out["xm"]
            check(f"(d) worst case reproduced by simulation, form={form}, c={c}",
                  abs(c @ (out["x"][N] - est[N]) / wc - 1), 1e-9)
    c = np.array([0.0, 1.0])
    _, _, cd = H._coefficients(T, M, c, N, "current", sub)
    check("(d') delta kernel affine within each sample interval (2nd differences)",
          float(np.max(np.abs(np.diff(cd, 2, axis=1))) / np.max(np.abs(cd))), 1e-10)
    lo8 = H.worst_case_by_enumeration(T, M, c, E, D, V, N, "current", 8)
    lo, up = H.worst_case_by_enumeration_bounds(T, M, c, E, D, V, N, "current", sub)
    ex, ex8 = (H.worst_case_exact(T, M, c, E, D, V, N, "current", s) for s in (sub, 8))
    lo128 = H.worst_case_by_enumeration(T, M, c, E, D, V, N, "current", 128)
    ordered = lo8 <= lo * (1 + 1e-12) and lo <= lo128 * (1 + 1e-12) and lo128 <= ex * (1 + 1e-12) and ex <= up * (1 + 1e-12)
    check("(d'') ordering lower(8) <= lower(16) <= lower(128) <= exact <= upper", 0.0 if ordered else np.inf, 0.0)
    check("(d'') worst_case_exact independent of sub (8 vs 16)", abs(ex8 / ex - 1), 1e-10)
    print(f"      lower(8)={lo8:.10f} lower(16)={lo:.10f} lower(128)={lo128:.10f} exact={ex:.10f} upper(16)={up:.10f}")
    worst_seen = 0.0
    for _ in range(300):                                      # random admissible signals
        sg = rng.choice([-1.0, 1.0], N * 32)
        dfn = lambda t, sg=sg: D * sg[np.clip(np.floor(np.asarray(t) / (T / 32)).astype(int), 0, sg.size - 1)]
        out = H.simulate_observer(T, M, E * rng.choice([-1.0, 1.0], 2), (0, 0),
                                  V * rng.choice([-1.0, 1.0], N + 1), dfn, None, N, substeps=32)
        worst_seen = max(worst_seen, abs(c @ (out["x"][N] - out["xp"][N])))
    check("(d''') 300 random admissible bang-bang signals stay below exact worst case",
          0.0 if worst_seen <= ex else np.inf, 0.0)
    # (e) continuous-time observer limits
    g, Dc, Vc = 10.0, 1.5, 2e-3
    t, W = H.ct_observer_worst_case(g, (0.1, 0.5), Dc, Vc, 5.0, target=1)
    lim1 = 2 * Dc / g + 2 * g * Vc / np.e
    check("(e) ct_observer_worst_case target=1 limit vs 2D/g + 2gV/e", abs(W[-1] / lim1 - 1), 1e-3)
    t, W0 = H.ct_observer_worst_case(g, (0.1, 0.5), Dc, Vc, 5.0, target=0)
    lim0 = Dc / g ** 2 + Vc * (1 + 2 * np.exp(-2.0))
    check("(e') ct_observer_worst_case target=0 limit vs D/g^2 + V(1 + 2/e^2)", abs(W0[-1] / lim0 - 1), 1e-3)
    print(f"      relative deviations: target=1 {abs(W[-1] / lim1 - 1):.2e}, target=0 {abs(W0[-1] / lim0 - 1):.2e}")
    _, Wc = H.ct_observer_worst_case(g, (0.1, 0.5), Dc, Vc, 5.0, target=1, n=2001)
    check("(e'') ct_observer_worst_case grid convergence (n = 2001 vs 20001)", abs(Wc[-1] / W[-1] - 1), 1e-9)
    check("(e''') W(0) equals E_target", abs(W[0] - 0.5) + abs(W0[0] - 0.1), 1e-15)
    # (f) two-signal lower limit
    Dl, Vl = 1.0, 0.01
    res = H.lower_limit_two_signals(Dl, Vl)
    check("(f) lower_limit_two_signals: |x''|<=D, |x|<=V, y=0, gap 4 sqrt(DV)",
          abs(res["velocity_gap"] / (4 * np.sqrt(Dl * Vl)) - 1) if res["ok"] else np.inf, 1e-8)
    print(f"      max|x''|={res['max_abs_acc']:.12f} (D={Dl}), max|x|={res['max_abs_x']:.12f} (V={Vl}), "
          f"vel+={res['vel_plus']:.12f}, vel-={res['vel_minus']:.12f}")
    Nl, hold = 30, 0.1                                        # 10 samples hold, then parabola
    kT = T * np.arange(Nl + 1)
    runs = []
    for s in (1.0, -1.0):
        vs = -s * res["x_plus"](kT - hold)
        runs.append(H.simulate_observer(T, M, (-s * Vl, 0.0), (0.0, 0.0), vs,
                                        lambda t, s=s: s * Dl * (np.asarray(t) >= hold), None, Nl, substeps=50))
    same = max(np.max(np.abs(runs[0]["xp"] - runs[1]["xp"])), np.max(np.abs(runs[0]["y"])), np.max(np.abs(runs[1]["y"])))
    gap = runs[0]["x"][Nl, 1] - runs[1]["x"][Nl, 1]
    check("(f') sampled observer: identical estimates for both signals, velocity gap 4 sqrt(DV)",
          max(same, abs(gap / res["claimed_gap"] - 1)), 1e-12)
    wl = H.worst_case_exact(T, M, [0.0, 1.0], (Vl, 0.0), Dl, Vl, Nl, "current", 8)
    check("(f'') discrete worst case (E1 = V) is >= 2 sqrt(DV)", 0.0 if wl >= 2 * np.sqrt(Dl * Vl) else np.inf, 0.0)
    print(f"      worst case {wl:.6f} vs lower limit {2 * np.sqrt(Dl * Vl):.6f}")


if __name__ == "__main__":
    check_kalman()
    check_hgo()
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(0 if not FAILS else 1)
