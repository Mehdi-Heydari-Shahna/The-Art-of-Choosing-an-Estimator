"""Numerical plant simulation (no learned component).

simulate_chain   second-order chain-form plant integrated in continuous time (RK4, `substeps` per sample) with the
                 input held over each sample; measurements at the sample instants.  An optional controller is called
                 once per sample with (k, t, y_k, estimate) -- the estimate comes from a callback so that the same
                 loop serves open-loop estimation and observer-based feedback.
simulate_linear  discrete linear stochastic model x+ = A x + B u + w, y = C x + v with Gaussian or uniform noise of
                 the stated covariances.
"""
from __future__ import annotations

import math

import numpy as np
import sympy as sp

from ..core import bounds


def chain_rhs(f_expr: str, params: dict):
    """Callable f(x1, x2, u) from a symbolic chain-form acceleration and numeric parameters."""
    names = ["x1", "x2", "u"] + sorted(params)
    e = bounds.parse(f_expr, names).subs({sp.Symbol(k, real=True): sp.Float(v) for k, v in params.items()})
    return sp.lambdify([sp.Symbol("x1", real=True), sp.Symbol("x2", real=True), sp.Symbol("u", real=True)], e, "math")


def simulate_chain(f, x0, Ts: float, N: int, v, d, controller=None, u_seq=None, substeps: int = 20, estimator=None):
    """Returns dict(t, x[N+1,2], y[N+1], u[N+1], xh[N+1,2] or None).

    f(x1,x2,u): acceleration of the true plant;  v[k]: measurement error at sample k;  d(t): additive disturbance;
    controller(k, t, y, xh) -> u_k  (or u_seq[k]);  estimator.step(k, y_k) -> xh_k and estimator.feed(k, u_k)."""
    x = np.array(x0, float)
    X = np.empty((N + 1, 2)); Y = np.empty(N + 1); U = np.zeros(N + 1)
    XH = np.full((N + 1, 2), np.nan)
    h = Ts / substeps

    def rhs(t, s, u):
        return np.array([s[1], f(s[0], s[1], u) + d(t)])

    for k in range(N + 1):
        t = k * Ts
        X[k] = x
        Y[k] = x[0] + v[k]
        xh = estimator.step(k, Y[k]) if estimator is not None else None
        if xh is not None:
            XH[k] = xh
        u = float(controller(k, t, Y[k], xh)) if controller is not None else (float(u_seq[k]) if u_seq is not None and k < len(u_seq) else 0.0)
        U[k] = u
        if estimator is not None:
            estimator.feed(k, u)
        if k == N:
            break
        for j in range(substeps):
            tt = t + j * h
            k1 = rhs(tt, x, u); k2 = rhs(tt + h / 2, x + h / 2 * k1, u)
            k3 = rhs(tt + h / 2, x + h / 2 * k2, u); k4 = rhs(tt + h, x + h * k3, u)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return {"t": np.arange(N + 1) * Ts, "x": X, "y": Y, "u": U, "xh": XH}


class OnlineHGO:
    """Step-by-step version of heros.khalil.specialist.run for use inside a feedback loop (same recursion)."""

    def __init__(self, cfg):
        from ..khalil import specialist as H
        from ..core import peak
        self.cfg = cfg
        self.M = np.asarray(cfg["M"], float)
        self.Ad, self.Gam = peak.Ad_of(cfg["Ts"]), peak.gam_of(cfg["Ts"])
        self.a0 = H._nominal(cfg)
        self.xm = None
        self.xp = None
        self.yk = None

    def step(self, k, y):
        if self.xm is None:
            init = self.cfg["init"]
            self.xm = np.array([y, init.get("x2_hat0", 0.0)]) if init["policy"] == "first_measurement" else np.array(init["x0_hat"], float)
        self.xp = self.xm + self.M * (y - self.xm[0])
        self.yk = y
        return self.xp.copy()

    def feed(self, k, u):
        self.xm = self.Ad @ self.xp + self.Gam * self.a0(self.yk, u)


class OnlineKalman:
    """Step-by-step Kalman filter with the gain sequence of heros.kalman.specialist (same recursion as run())."""

    def __init__(self, cfg, N):
        from ..kalman import specialist as K
        self.A, self.B, self.C = cfg["A"], cfg["B"], cfg["C"]
        self.M = K.gains(cfg, N)[0]
        self.xm = np.asarray(cfg["x0_hat"], float).copy()
        self.xp = None

    def step(self, k, y):
        self.xp = self.xm + self.M[k] @ (np.atleast_1d(y) - self.C @ self.xm)
        return self.xp.copy()

    def feed(self, k, u):
        self.xm = self.A @ self.xp + self.B @ np.atleast_1d(u)


def simulate_linear(A, B, C, Qd, R, x0, u, N, rng, dist="gaussian"):
    """x_{k+1} = A x_k + B u_k + w_k,  y_k = C x_k + v_k,  k = 0..N.  dist: 'gaussian' or 'uniform' (same covariances)."""
    n, ny = A.shape[0], C.shape[0]
    ev, Uq = np.linalg.eigh(0.5 * (Qd + Qd.T))
    Lq = Uq @ np.diag(np.sqrt(np.clip(ev, 0, None)))
    Lr = np.linalg.cholesky(R)

    def draw(k):
        if dist == "gaussian":
            return rng.standard_normal(k)
        return rng.uniform(-math.sqrt(3.0), math.sqrt(3.0), k)

    x = np.asarray(x0, float).copy()
    X = np.empty((N + 1, n)); Y = np.empty((N + 1, ny))
    u = np.zeros((N + 1, B.shape[1])) if u is None else np.asarray(u, float).reshape(-1, B.shape[1])
    for k in range(N + 1):
        X[k] = x
        Y[k] = C @ x + Lr @ draw(ny)
        if k < N:
            x = A @ x + B @ u[k] + Lq @ draw(n)
    return X, Y
