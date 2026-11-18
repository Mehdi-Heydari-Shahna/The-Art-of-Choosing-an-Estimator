"""Kalman specialist: discrete-time Kalman filtering of known linear models (state estimates and covariances).

Supported (method register K1-K4, see docs/SOURCE_TO_METHOD.md):
  K1  time-varying Kalman filter, finite horizon, known inputs, measurement-update / time-update form
  K2  steady-state (constant-gain) Kalman filter
  K3  zero-order-hold sampling of a continuous-time plant and Van Loan discretisation of a continuous
      process-noise spectral density (numerical adaptation; not part of Kalman's papers)
  K4  exact mean-square error of the filter (covariance recursion) and probability statements derived from it

Timing convention (one clock, used by design, execution, simulation and the independent reference):
  y_k is taken at t_k = k Ts;  measurement update -> x_{k|k};  u_k is held on [t_k, t_{k+1});
  time update -> x_{k+1|k}.  P_{k|k-1} / P_{k|k} are the prior / filtered error covariances.

Claims (never stronger than the supplied statistical information):
  distribution == "gaussian"      conditional mean; minimum mean-square error among all estimators that are
                                  measurable functions of the same measurements and inputs
  otherwise (second-order info)   minimum mean-square error among linear (affine) estimators of the same data
Not supported (precise disposition instead of a result): cross-correlated or coloured noise, irregular sampling,
uncertain model parameters, re-sampling of a discrete model, nonlinear dynamics or measurements.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from ..core import spec as S
from ..core.assess import fmt_up, fmt_down, Assessment, Need, is_double_integrator, jsonable
from ..core import peak
from . import legacy_kernels as LK

METHOD = "kalman_filter"
VERSION = "1.0.0"


class SpecIssue(Exception):
    def __init__(self, kind, message, path=None):
        super().__init__(message)
        self.kind, self.message, self.path = kind, message, path


# ---------------------------------------------------------------------------------------------------------
# model normalisation: one model and one clock
# ---------------------------------------------------------------------------------------------------------
def zoh(Ac, Bc, Ts):
    n, m = Ac.shape[0], Bc.shape[1]
    Mx = np.zeros((n + m, n + m))
    Mx[:n, :n] = Ac
    Mx[:n, n:] = Bc
    E = expm(Mx * Ts)
    return E[:n, :n], E[:n, n:]


def normalize_model(spec: dict) -> dict:
    """Discrete model (A, B, C) at the single period Ts, with the derivations that produced it."""
    lin = S.get(spec, "plant.linear")
    if not lin:
        raise SpecIssue("not_applicable", "no linear state-space model is supplied")
    C = np.atleast_2d(np.asarray(lin["C"], float))
    der = []
    if lin.get("clock") == "discrete":
        A = np.atleast_2d(np.asarray(lin["A"], float))
        B = np.atleast_2d(np.asarray(lin["B"], float)) if lin.get("B") is not None else np.zeros((A.shape[0], 1))
        tn = lin.get("Ts_native")
        Ts = S.get(spec, "time.Ts")
        if tn is None and Ts is None:
            raise SpecIssue("missing", "the sample period of the discrete model is not stated", "time.Ts")
        if tn is not None and Ts is not None and abs(Ts - tn) > 1e-12 * max(abs(tn), 1.0):
            raise SpecIssue("invalid", f"timing conflict: discrete model defined at {tn} s, request states {Ts} s", "time.Ts")
        Ts = tn if tn is not None else Ts
        der.append({"field": "model", "how": "discrete model used at its native period", "Ts": Ts})
    elif lin.get("clock") == "continuous":
        Ts = S.get(spec, "time.Ts")
        if Ts is None:
            raise SpecIssue("missing", "a continuous-time model needs a sample period", "time.Ts")
        if S.get(spec, "time.input_hold", "zoh") != "zoh":
            raise SpecIssue("not_applicable", "only a zero-order hold of the applied input is supported")
        Ac = np.atleast_2d(np.asarray(lin["A"], float))
        Bc = np.atleast_2d(np.asarray(lin["B"], float)) if lin.get("B") is not None else np.zeros((Ac.shape[0], 1))
        A, B = zoh(Ac, Bc, Ts)
        der.append({"field": "model", "how": "zero-order-hold discretisation of the continuous model (matrix exponential)",
                    "Ts": Ts, "conditions": "applied input constant on each sample interval"})
    else:
        raise SpecIssue("invalid", "plant.linear.clock must be 'discrete' or 'continuous'", "plant.linear.clock")
    n = A.shape[0]
    if A.shape != (n, n) or B.shape[0] != n or C.shape[1] != n:
        raise SpecIssue("invalid", f"inconsistent dimensions A{A.shape} B{B.shape} C{C.shape}", "plant.linear")
    return {"A": A, "B": B, "C": C, "Ts": float(Ts), "clock": lin["clock"], "derivations": der}


def process_cov(spec: dict, model: dict):
    """Discrete-step process-noise covariance Qd at the model period, with its derivation."""
    Q = S.get(spec, "stochastic.Q")
    kind = S.get(spec, "stochastic.Q_kind")
    n = model["A"].shape[0]
    if Q is None:
        raise SpecIssue("missing", "process-noise information is not stated", "stochastic.Q")
    if kind is None:
        raise SpecIssue("missing", "it is not stated whether the process noise is a continuous spectral density or a per-step covariance",
                        "stochastic.Q_kind")
    Q = np.atleast_2d(np.asarray(Q, float))
    lin = S.get(spec, "plant.linear")
    if kind == "discrete_cov":
        if Q.shape != (n, n):
            raise SpecIssue("invalid", f"per-step process covariance must be {n}x{n}", "stochastic.Q")
        return LK.sym(Q), {"field": "Qd", "how": "per-step covariance used as stated", "conditions": f"valid at period {model['Ts']} s only"}
    if kind == "continuous_psd":
        if lin.get("clock") != "continuous":
            raise SpecIssue("invalid", "a continuous spectral density was given for a discrete-time model; there are no "
                                       "continuous dynamics to integrate it through", "stochastic.Q_kind")
        G = S.get(spec, "stochastic.G")
        G = np.eye(n) if G is None else np.atleast_2d(np.asarray(G, float))
        if G.shape[0] != n or Q.shape != (G.shape[1], G.shape[1]):
            raise SpecIssue("invalid", "dimensions of the noise input matrix G and the spectral density do not match", "stochastic.Q")
        Ac = np.atleast_2d(np.asarray(lin["A"], float))
        Qd = LK.discretize_noise(Ac, G, Q, model["Ts"])
        return Qd, {"field": "Qd", "how": "Van Loan discretisation of the spectral density: Qd = int_0^Ts e^{As} G Qc G' e^{A's} ds",
                    "conditions": "white continuous-time process noise, independent of the held input"}
    raise SpecIssue("invalid", f"unknown process-noise kind {kind!r}", "stochastic.Q_kind")


def _pbh_ok(A, Cm, thresh=1.0 - 1e-9, dual=False):
    """PBH rank test at every eigenvalue on or outside the unit circle (numerical rank, relative tolerance 1e-9).
    dual=False: detectability of (A, Cm).  dual=True: stabilisability of (A, G) with Cm = G' passed in."""
    A = np.asarray(A, float)
    n = A.shape[0]
    for lam in np.linalg.eigvals(A):
        if abs(lam) >= thresh:
            At = (A.T if dual else A) - lam * np.eye(n)
            sv = np.linalg.svd(np.vstack([At, Cm]), compute_uv=False)
            if sv[n - 1] < 1e-9 * max(sv[0], 1.0):
                return False
    return True


def detectable(A, C):
    return _pbh_ok(A, np.atleast_2d(C))


def stabilizable_noise(A, Q):
    """Stabilisability of (A, Q^(1/2)) by the PBH test, after a diagonal similarity scaling by the noise standard
    deviations (a change of state units, under which stabilisability is invariant), so that a state excited by a
    noise many orders of magnitude below another's is not judged unexcited by the relative rank tolerance (second
    recheck A: a q/r = 1e-20 random walk next to a unit-variance state)."""
    A, Q = np.asarray(A, float), LK.sym(np.asarray(Q, float))
    d = np.sqrt(np.clip(np.diag(Q), 0.0, None))
    s = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 1.0)
    A, Q = (A * s[:, None]) / s[None, :], Q * s[:, None] * s[None, :]
    ev, U = np.linalg.eigh(LK.sym(Q))
    Gq = U @ np.diag(np.sqrt(np.clip(ev, 0, None)))
    return _pbh_ok(A, Gq.T, dual=True)


def observable(A, C):
    n = A.shape[0]
    O = np.vstack([np.atleast_2d(C) @ np.linalg.matrix_power(A, k) for k in range(n)])
    sv = np.linalg.svd(O, compute_uv=False)
    return bool(sv[-1] > 1e-9 * max(sv[0], 1.0)), O


_SS_CACHE: dict = {}


def _refine_riccati_mp(A, C, Qd, R, P, dps=50, maxit=12):
    """Newton (Kleinman-Hewer) refinement of the stabilising filter Riccati solution in `dps`-digit arithmetic, started
    from a stabilising P: each step takes the gain of P and solves the fixed-gain covariance equation exactly (by
    vectorisation).  Returns (P, Pf, M) in float64, Pf in Joseph form computed in the same arithmetic."""
    import mpmath as mp
    n = A.shape[0]
    with mp.workdps(dps):
        Am, Cm, Qm, Rm = (mp.matrix(np.asarray(X, float).tolist()) for X in (A, C, Qd, R))
        Pm, I = mp.matrix(np.asarray(P, float).tolist()), mp.eye(n)
        for _ in range(maxit):
            Mm = Pm * Cm.T * mp.inverse(Cm * Pm * Cm.T + Rm)
            Acl = Am * (I - Mm * Cm)
            W = Am * (Mm * Rm * Mm.T) * Am.T + Qm
            K = mp.matrix(n * n, n * n)
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        for l in range(n):
                            K[j * n + i, l * n + k] = (1 if (i == k and j == l) else 0) - Acl[i, k] * Acl[j, l]
            x = mp.lu_solve(K, mp.matrix([W[i, j] for j in range(n) for i in range(n)]))
            Pn = mp.matrix(n, n)
            for j in range(n):
                for i in range(n):
                    Pn[i, j] = x[j * n + i]
            Pn = (Pn + Pn.T) / 2
            change = max(abs(Pn[i, j] - Pm[i, j]) / mp.sqrt(abs(Pn[i, i] * Pn[j, j]) + mp.mpf(10) ** (-300))
                         for i in range(n) for j in range(n))
            Pm = Pn
            if change <= mp.mpf(10) ** (-(dps - 12)):
                break
        Mm = Pm * Cm.T * mp.inverse(Cm * Pm * Cm.T + Rm)
        J = I - Mm * Cm
        Pf = J * Pm * J.T + Mm * Rm * Mm.T
        tof = lambda X: np.array([[float(X[i, j]) for j in range(X.cols)] for i in range(X.rows)])
        return LK.sym(tof(Pm)), LK.sym(tof(Pf)), tof(Mm)


def steady_state(A, C, Qd, R):
    """Stabilising solution of the filter Riccati equation: prior covariance P, filtered Pf, gain M.

    The library solution is verified by one Newton (fixed-gain covariance) step in floating point; if that step moves
    any entry by more than 1e-10 of its own scale, the solution is refined by Newton steps in 50-digit arithmetic
    (second recheck A, New-1: badly conditioned models, e.g. a precise encoder with rank-1 acceleration noise, where
    the library solution was off by up to 8e-4 relative in the filtered velocity variance)."""
    A, C, Qd, R = (np.atleast_2d(np.asarray(X, float)) for X in (A, C, Qd, R))
    key = (A.shape, C.shape, A.tobytes(), C.tobytes(), Qd.tobytes(), R.tobytes())
    hit = _SS_CACHE.get(key)
    if hit is not None:
        return tuple(x.copy() for x in hit)
    P = solve_discrete_are(A.T, C.T, Qd, R)
    G = C @ P @ C.T + R
    M = np.linalg.solve(G, C @ P).T
    I_MC = np.eye(A.shape[0]) - M @ C
    Pf = LK.sym(I_MC @ P @ I_MC.T + M @ R @ M.T)
    P = LK.sym(P)
    from scipy.linalg import solve_discrete_lyapunov
    import warnings
    Acl = A @ I_MC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # an ill-conditioned check only sends the case to the refinement
        P1 = solve_discrete_lyapunov(Acl, A @ M @ R @ M.T @ A.T + Qd)
    dg = np.maximum(np.abs(np.diag(P)), 1e-300)
    moved = np.max(np.abs(P1 - P) / np.sqrt(np.outer(dg, dg)))
    if not np.isfinite(moved) or moved > 1e-10:
        P, Pf, M = _refine_riccati_mp(A, C, Qd, R, P)
    out = (P, Pf, M)
    if len(_SS_CACHE) > 4096:
        _SS_CACHE.clear()
    _SS_CACHE[key] = tuple(x.copy() for x in out)
    return out


# ---------------------------------------------------------------------------------------------------------
# design and run-time path
# ---------------------------------------------------------------------------------------------------------
def design(spec: dict, n_steps: int | None = None) -> dict:
    """Build the executable filter configuration from a complete specification (raises SpecIssue otherwise)."""
    for flag, msg in (("stochastic.cross_cov", "cross-correlated process and measurement noise"),):
        cc = S.get(spec, flag)
        if cc is not None and np.any(np.asarray(cc, float) != 0):
            raise SpecIssue("not_applicable", f"{msg} is outside the supported method register")
    if S.get(spec, "stochastic.white") is False:
        raise SpecIssue("not_applicable", "temporally correlated (coloured) noise is outside the supported method register")
    if S.get(spec, "time.irregular"):
        raise SpecIssue("not_applicable", "irregular sampling is outside the supported method register")
    model = normalize_model(spec)
    A, B, C, Ts = model["A"], model["B"], model["C"], model["Ts"]
    n, ny = A.shape[0], C.shape[0]
    Qd, dq = process_cov(spec, model)
    R = S.get(spec, "stochastic.R")
    if R is None:
        raise SpecIssue("missing", "measurement-noise covariance is not stated", "stochastic.R")
    R = np.atleast_2d(np.asarray(R, float))
    if R.shape != (ny, ny):
        raise SpecIssue("invalid", f"measurement-noise covariance must be {ny}x{ny}", "stochastic.R")
    if np.linalg.eigvalsh(LK.sym(R)).min() <= 0:
        raise SpecIssue("invalid", "measurement-noise covariance must be positive definite", "stochastic.R")
    horizon = S.get(spec, "objective.horizon") or "finite"
    ders = model["derivations"] + [dq]
    cfg = {"method": METHOD, "version": VERSION, "Ts": Ts, "A": A, "B": B, "C": C, "Qd": Qd, "R": R,
           "horizon": horizon, "derivations": ders, "target": S.get(spec, "target.state")}
    x0 = S.get(spec, "initial.x0_hat")
    cfg["x0_hat"] = np.zeros(n) if x0 is None else np.asarray(x0, float)
    cfg["x0_provenance"] = "zero (no initial estimate supplied; prior mean taken as zero)" if x0 is None else "supplied"
    if horizon == "steady_state":
        if not detectable(A, C):
            raise SpecIssue("not_applicable", "(A, C) is not detectable: a steady-state filter does not exist")
        if not stabilizable_noise(A, Qd):
            raise SpecIssue("not_applicable", "(A, Qd^1/2) is not stabilisable on the unit circle: the steady-state filter is not unique")
        P, Pf, M = steady_state(A, C, Qd, R)
        cfg.update(mode="steady_state", P_ss=P, Pf_ss=Pf, M_ss=M)
        P0 = S.get(spec, "initial.P0")
        cfg["P0"] = P if P0 is None else LK.sym(np.atleast_2d(np.asarray(P0, float)))
        cfg["P0_provenance"] = "not needed for the steady-state gain" if P0 is None else "supplied (not used by the constant gain)"
    else:
        P0 = S.get(spec, "initial.P0")
        if P0 is None:
            raise SpecIssue("missing", "initial error covariance is not stated", "initial.P0")
        P0 = np.atleast_2d(np.asarray(P0, float))
        if P0.shape != (n, n) or np.linalg.eigvalsh(LK.sym(P0)).min() < 0:
            raise SpecIssue("invalid", "initial covariance must be symmetric positive semidefinite of the state dimension", "initial.P0")
        cfg.update(mode="time_varying", P0=LK.sym(P0), P0_provenance="supplied")
    if n_steps is None:
        te = S.get(spec, "time.t_end")
        from ..core import timing as TM
        n_steps = TM.sample_window(Ts, None, te)[1] if te is not None else None   # last sample at or before t_end; none: every k (see assess)
    cfg["N"] = None if n_steps is None else int(n_steps)
    return cfg


def gains(cfg: dict, N: int | None = None):
    """Gain and covariance sequences for N+1 measurement updates (k = 0..N)."""
    N = (cfg["N"] if cfg.get("N") is not None else 200) if N is None else N
    if cfg["mode"] == "steady_state":
        M = [cfg["M_ss"]] * (N + 1)
        Pp = [cfg["P_ss"]] * (N + 2)
        Pf = [cfg["Pf_ss"]] * (N + 1)
        return M, Pp, Pf
    return LK.kalman_gains(cfg["A"], cfg["C"], cfg["Qd"], cfg["R"], cfg["P0"], N + 1)


def run(cfg: dict, y, u=None) -> dict:
    """PUBLIC RUN-TIME PATH: process actual measurements y_0..y_N and applied inputs u_0..u_{N-1}.

    Returns filtered estimates x_{k|k}, their covariances, one-step predictions, gains, innovations and the
    normalised innovation squared (NIS).  No learned component is involved."""
    y = np.atleast_2d(np.asarray(y, float))
    if y.shape[0] == 1 and cfg["C"].shape[0] != 1:
        y = y.T
    if y.ndim == 2 and y.shape[1] != cfg["C"].shape[0]:
        y = y.reshape(-1, cfg["C"].shape[0])
    N = y.shape[0] - 1
    m = cfg["B"].shape[1]
    if u is None:
        u = np.zeros((N + 1, m))
    u = np.asarray(u, float).reshape(-1, m)
    if u.shape[0] < N:
        raise ValueError(f"need at least {N} applied-input samples for {N + 1} measurements, got {u.shape[0]}")
    if u.shape[0] == N:
        u = np.vstack([u, np.zeros((1, m))])
    M, Pp, Pf = gains(cfg, N)
    x0 = np.array(cfg["x0_hat"], float)
    n = cfg["A"].shape[0]
    basis = ("Riccati recursion of the time-varying gain from the stated initial covariance" if cfg["mode"] == "time_varying" else None)
    if cfg.get("init_policy") == "first_measurement":
        # position estimate = first measurement: the first update is the gain [1, 0, ...]', so the estimate, the gain that is
        # exported and the covariance all describe the same start-up (revision 2, finding 3)
        M = [np.eye(n)[:, :1]] + list(M[1:])
    if cfg["mode"] == "steady_state" or cfg.get("init_policy") == "first_measurement":
        P0c = np.atleast_2d(np.asarray(cfg["P0"], float))
        Pp, Pf = LK.fixed_gain_cov(cfg["A"], cfg["C"], cfg["Qd"], cfg["R"], P0c, list(M[: N + 1]))
        stated = cfg.get("P0_provenance", "").startswith("supplied")
        basis = ("error covariance of the released gain sequence propagated from the stated initial covariance (exact for this estimate "
                 "at every sample, including start-up)" if stated else
                 "no initial covariance is stated: the propagation starts from the steady-state prior covariance, so the values describe "
                 "the estimate only once the start-up transient has decayed")
    xf, xp = LK.run_filter(cfg["A"], cfg["B"], cfg["C"], M, list(y), list(u[: N + 1]), x0)
    C, R = cfg["C"], cfg["R"]
    innov = np.array([y[k] - C @ xp[k] for k in range(N + 1)])
    nis = np.array([float(innov[k] @ np.linalg.solve(C @ Pp[k] @ C.T + R, innov[k])) for k in range(N + 1)])
    out = {"t": np.arange(N + 1) * cfg["Ts"], "x_filt": np.array(xf), "P_filt": np.array(Pf[: N + 1]),
           "x_pred": np.array(xp[: N + 1]), "P_pred": np.array(Pp[: N + 1]), "gain": np.array(M[: N + 1]),
           "innovation": innov, "nis": nis, "covariance_basis": basis,
           "uncertainty_type": "error covariance (second-order statistic); not a deterministic error bound"}
    hb = cfg.get("hard_bound")
    if hb:
        # a hard-bound release: the uncertainty that the claim refers to is the deterministic envelope, exported as such
        W = peak.worst_case_lti(cfg["Ts"], np.asarray(cfg["M_ss"], float).reshape(2), np.eye(2)[hb["target"]], hb["E"], hb["D"], hb["V"], N,
                                init="first_measurement")
        out["envelope"] = W["W"]
        out["envelope_kind"] = "exact worst case at each sample (Proposition 1, first-reading initialisation), evaluated in floating point"
        out["uncertainty_type"] = ("deterministic error envelope at the sample instants (the claim); the covariance is reported for reference "
                                   "and is not a bound")
    return out


def window_variance(A, C, Qd, R, P0, tgt, k_dead=0, n_end=None, gains=None, rtol=1e-13, kmax=400000):
    """Largest filtered error variance of state component `tgt` over the samples k = k_dead .. n_end.

    gains=None: the optimal time-varying gain (Riccati recursion from the prior covariance P0 at k = 0);
    gains = array (G, n, ny): G constant gains evaluated together.  n_end=None means every k >= k_dead.  Both
    covariance maps are monotone in the covariance, so once the prior covariance has decreased in one step (in the
    positive-semidefinite order) it never increases again, and once it has increased it increases up to its limit:
    the recursion stops at that point (adding the limit in the second case) or when it has converged.
    Returns (vmax (G,), v_at_deadline (G,), steps, all_resolved)."""
    n = A.shape[0]
    I = np.eye(n)
    P = np.atleast_2d(np.asarray(P0, float))
    G = 1
    if gains is not None:
        Mb = np.asarray(gains, float).reshape(-1, n, C.shape[0])
        G = Mb.shape[0]
        IMC = I[None] - Mb @ C[None]
        MRM = Mb @ R[None] @ np.transpose(Mb, (0, 2, 1))
        P = np.broadcast_to(P, (G, n, n)).copy()
    else:
        P = P[None].copy()
    vmax = np.full(G, -np.inf)
    vd = None
    done = np.zeros(G, bool)
    k = 0
    while True:
        if gains is None:
            Pk = P[0]
            Sg = C @ Pk @ C.T + R
            M = Pk @ C.T @ np.linalg.inv(Sg)
            IM = I - M @ C
            Pf = (IM @ Pk @ IM.T + M @ R @ M.T)[None]
            Pn = (A @ Pf[0] @ A.T + Qd)[None]
        else:
            Pf = IMC @ P @ np.transpose(IMC, (0, 2, 1)) + MRM
            Pn = A[None] @ Pf @ A.T[None] + Qd[None]
        v = Pf[:, tgt, tgt]
        if not np.all(np.isfinite(v)):
            return np.full(G, np.inf), None, k, False
        if k >= k_dead:
            vmax = np.where(done, vmax, np.maximum(vmax, v))
        if k == k_dead:
            vd = v.copy()
        if n_end is not None and k >= n_end:
            return vmax, vd, k, True
        if n_end is None and k >= k_dead:
            for g in np.nonzero(~done)[0]:
                # The tests are made on the increment scaled by the variances, D^-1/2 (P_next - P) D^-1/2, so that a state
                # with a small variance is judged on its own scale (review A recheck, N1); the congruence preserves the
                # positive-semidefinite order on which the monotonicity argument rests.
                dP = Pn[g] - P[g]
                dg = np.maximum(np.maximum(np.diag(Pn[g]), np.diag(P[g])), 1e-300)
                sc = 1.0 / np.sqrt(dg)
                Sd = dP * sc[:, None] * sc[None, :]
                ev = np.linalg.eigvalsh(0.5 * (Sd + Sd.T))

                def _limit():
                    if gains is None:
                        return float(steady_state(A, C, Qd, R)[1][tgt, tgt]) if detectable(A, C) and stabilizable_noise(A, Qd) else np.inf
                    Pl = steady_cov_of_constant_gain(A, C, Qd, R, Mb[g])
                    return np.inf if Pl is None else float(Pl[tgt, tgt])
                if np.max(np.abs(Sd)) <= rtol:
                    lim = _limit()                       # converged: the limit is included as well (conservative)
                    vmax[g] = max(vmax[g], lim) if np.isfinite(lim) else vmax[g]
                    done[g] = True
                elif ev.max() <= 1e-14:
                    # non-increasing from now on: the maximum is reached; the limit is included as well, which costs nothing
                    # when the sequence really decreases and covers very slow modes whose rise is below the test (second recheck A)
                    lim = _limit()
                    vmax[g] = max(vmax[g], lim) if np.isfinite(lim) else vmax[g]
                    done[g] = True
                elif ev.min() >= -1e-14:
                    vmax[g] = max(vmax[g], _limit())     # non-decreasing from now on: the supremum is the limit
                    done[g] = True
            if done.all():
                return vmax, vd, k, bool(np.all(np.isfinite(vmax)))
        if k >= kmax:
            return vmax, vd, k, False
        P = Pn
        k += 1


def error_cov_of_gain(cfg: dict, Ms, N: int):
    """Exact prior / filtered error covariances of an arbitrary gain sequence on the same model (affine in Q, R, P0)."""
    return LK.fixed_gain_cov(cfg["A"], cfg["C"], cfg["Qd"], cfg["R"], cfg["P0"], Ms)


def steady_cov_of_constant_gain(A, C, Qd, R, M, tol=1e-13, itmax=200000):
    """Steady-state filtered error covariance of a constant-gain estimator xh+ = xh- + M(y - C xh-), by iteration.
    Returns None if the error map is not strictly stable."""
    n = A.shape[0]
    I_MC = np.eye(n) - M @ C
    Phi = A @ I_MC
    if max(abs(np.linalg.eigvals(Phi))) >= 1 - 1e-12:
        return None
    from scipy.linalg import solve_discrete_lyapunov
    Wn = A @ M @ R @ M.T @ A.T + Qd
    Pp = solve_discrete_lyapunov(Phi, Wn)          # prior covariance:  Pp = Phi Pp Phi' + A M R M' A' + Qd
    return LK.sym(I_MC @ Pp @ I_MC.T + M @ R @ M.T)


# ---------------------------------------------------------------------------------------------------------
# assessment against the requested objective
# ---------------------------------------------------------------------------------------------------------
def _claim_scope(spec):
    dist = S.get(spec, "stochastic.distribution")
    if dist == "gaussian":
        return "mmse", ("conditional mean of the state given the measurements and inputs; minimum mean-square error among all "
                        "estimators measurable with respect to the same information (jointly Gaussian initial state and noises)")
    return "lmmse", ("minimum mean-square error among linear (affine) estimators of the same measurements and inputs "
                     "(only means and covariances are assumed; no claim against nonlinear estimators)")


def assess(spec: dict) -> Assessment:
    a = Assessment(method=METHOD, status="not_applicable")
    obj = S.get(spec, "objective.kind")
    if S.get(spec, "plant.kind") not in ("linear", "second_order") or not S.get(spec, "plant.linear"):
        a.reasons.append("the Kalman filter needs a known linear state-space model; none is supplied for this plant")
        return a
    for fl in S.get(spec, "plant.flags") or []:
        a.reasons.append(f"declared feature outside the register: {fl}")
    if a.reasons:
        return a
    unc = uncertain_parameters(spec)
    if unc and obj != "hard_bound":
        # the stated linear model is the nominal one; the true plant differs from it within the stated ranges (revision 2,
        # finding 4).  The recursion and its covariance are exact only for a known model, so no statistical claim is made.
        a.reasons.append("the true plant parameters differ from the nominal model or are known only as ranges ("
                         + ", ".join((f"{k} = {r[0]:.6g}" if r[0] == r[1] else f"{k} in [{r[0]:.6g}, {r[1]:.6g}]") for k, r in unc.items())
                         + "); the Kalman filter's error covariance and its optimality hold only for a known model, so no statistical claim is "
                           "made for the nominal one")
        return a
    needs = []
    try:
        cfg = design(spec)
    except SpecIssue as e:
        if e.kind == "missing" and obj == "hard_bound" and e.path != "time.Ts":
            a.reasons.append("no second-order noise model is supplied, so no Kalman gain is defined (" + e.message + "); supplying one would allow "
                             "its steady-state gain to be tested against the same hard bound")
            return a
        if e.kind == "missing":
            # collect every missing statistical field, not only the first
            for path, why, q in (
                ("time.Ts", "the discrete model and the noise conversion depend on the sample period", "What is the sample period?"),
                ("stochastic.Q", "the filter gain and the reported covariance depend on the process noise", "What is the process-noise covariance (per step) or spectral density?"),
                ("stochastic.Q_kind", "a spectral density and a per-step covariance differ by a factor of order Ts", "Is the process-noise value a continuous spectral density or a per-step covariance?"),
                ("stochastic.R", "the filter gain and the reported covariance depend on the measurement noise", "What is the measurement-noise variance per sample?"),
                ("initial.P0", "the transient gains and covariances depend on the initial error covariance", "What is the covariance of the initial estimate error?"),
            ):
                if path == "initial.P0" and (S.get(spec, "objective.horizon") == "steady_state"):
                    continue
                if path == "stochastic.Q_kind" and S.get(spec, "stochastic.Q") is None:
                    continue
                if path == "time.Ts":
                    lin = S.get(spec, "plant.linear")
                    if S.get(spec, "time.Ts") is not None or (lin.get("clock") == "discrete" and lin.get("Ts_native") is not None):
                        continue
                elif S.get(spec, path) is not None:
                    continue
                needs.append(Need(path, why, q))
            a.status, a.needs = "needs", needs
            a.reasons.append(e.message)
            return a
        a.status = "not_applicable" if e.kind == "not_applicable" else "fails"
        a.reasons.append(e.message)
        if e.kind == "invalid":
            a.status = "fails"
            a.limits.append("invalid input for this method: " + e.message)
        return a
    A, C, Qd, R, Ts = cfg["A"], cfg["C"], cfg["Qd"], cfg["R"], cfg["Ts"]
    n = A.shape[0]
    tgt = S.get(spec, "target.state")
    obs_ok, _ = observable(A, C)
    a.checks.append({"name": "observability of (A, C)", "passed": bool(obs_ok)})
    a.checks.append({"name": "detectability of (A, C)", "passed": bool(detectable(A, C))})
    if tgt is None:
        a.status = "needs"
        a.needs.append(Need("target.state", "the reported error measure refers to one state component", "Which quantity should be estimated?"))
        return a
    if not obs_ok:
        # is the target direction observable?  e_t must lie in the row space of the observability matrix
        _, O = observable(A, C)
        et = np.zeros(n); et[tgt] = 1.0
        res = et - np.linalg.pinv(O) @ (O @ et)
        if np.linalg.norm(res) > 1e-8:
            a.status = "not_applicable"
            a.reasons.append("the requested quantity is not observable from the declared measurements (rank test on the observability matrix); "
                             "measurements cannot reduce its uncertainty")
            return a
    scope, statement = _claim_scope(spec)
    if S.get(spec, "stochastic.distribution") is None:
        a.harmless.append({"path": "stochastic.distribution",
                           "why": "the filter recursion is the same for every distribution with the stated means and covariances; "
                                  "only the claim changes, and the weaker linear-estimator claim is used"})
    # which statement can be made: asymptotic (steady-state request without deadline) or at finite times from P0
    dl = S.get(spec, "objective.deadline")
    te = S.get(spec, "time.t_end")
    from ..core import timing as TM
    k_dead, n_end = TM.sample_window(Ts, dl, te)
    asymptotic = cfg["horizon"] == "steady_state" and dl is None
    try:
        var_ss = float(steady_state(A, C, Qd, R)[1][tgt, tgt]) if detectable(A, C) and stabilizable_noise(A, Qd) else float("nan")
    except Exception:
        var_ss = float("nan")
    if cfg["mode"] == "steady_state":
        var_ss = float(cfg["Pf_ss"][tgt, tgt])
    a.metrics.update(mse_steady=var_ss, Ts=Ts)
    stat_claim = obj in ("mse_optimal", "mse_bound", "probability")
    if asymptotic or not stat_claim:
        if S.get(spec, "initial.P0") is None and stat_claim:
            a.harmless.append({"path": "initial.P0",
                               "why": "only a steady-state (k -> infinity) statement is requested; the steady-state gain and its limiting error "
                                      "covariance do not depend on the initial covariance when (A, C) is detectable and (A, Qd^1/2) is "
                                      "stabilisable (both verified)"})
        a.metrics["mse_post_deadline_max"] = var_ss
        window = "in steady state (the limit as k -> infinity)"
    else:
        P0 = S.get(spec, "initial.P0")
        if P0 is None:
            a.status = "needs"
            a.needs.append(Need("initial.P0", "an error statement at finite times (a deadline or a finite horizon) depends on the initial error covariance",
                                "What is the covariance of the initial estimate error (or of the initial state)?"))
            return a
        gain = None if cfg["mode"] == "time_varying" else cfg["M_ss"][None]
        vmax, vd, steps, conv = window_variance(A, C, Qd, R, P0, tgt, k_dead, n_end, gains=gain)
        if not conv:
            a.status = "fails"
            a.limits.append("the error covariance did not converge within the iteration limit; no statement for every later sample is made")
            return a
        a.metrics["mse_post_deadline_max"] = float(vmax[0])
        a.metrics["window_steps"] = int(steps)
        window = (f"at every sample from t = {k_dead * Ts:.6g} s to the last sample of the record, t = {n_end * Ts:.6g} s" if n_end is not None
                  else f"at every sample from t = {k_dead * Ts:.6g} s on (covariance iterated to convergence; the limit is included)")
        if cfg["mode"] == "steady_state":
            # the constant gain is optimal only in the limit; the optimal (time-varying) value bounds what any estimator can do
            a.metrics["optimal_window_max"] = float(window_variance(A, C, Qd, R, P0, tgt, k_dead, n_end)[0][0])
    a.metrics["window"] = window
    a.config = cfg
    conds = ["known linear model at one sample period", "zero-mean white process and measurement noise, mutually uncorrelated and uncorrelated with the initial error",
             "stated covariances are the true ones"]
    if S.get(spec, "initial.x0_hat") is None:
        if cfg["mode"] == "steady_state":
            conds.append("no initial estimate was supplied (zero is used); its effect decays and is not part of this steady-state claim")
        else:
            conds.append("no initial estimate was supplied, so zero is used as the prior mean: the claim holds only if the initial state has "
                         "zero mean (accepted default, shown in the configuration)")
    if obj == "mse_optimal":
        a.status = "qualifies"
        v = a.metrics["mse_post_deadline_max"]
        if cfg["mode"] == "steady_state":
            stmt = ("constant (steady-state) Kalman gain: in the limit k -> infinity its error covariance equals that of the optimal filter, whose estimate is the "
                    + statement + "; at finite times the time-varying filter is better (horizon 'finite')")
        else:
            stmt = statement + " (time-varying gain, optimal at every sample given the stated initial covariance)"
        stmt += f"; error variance of the target {window}: at most {fmt_up(v)}"
        if cfg["mode"] == "steady_state":
            stmt += ("; the exported covariance at each sample is that of the constant gain started from the stated initial covariance"
                     if S.get(spec, "initial.P0") is not None else
                     "; no initial covariance is stated, so the exported covariance is the steady-state value and describes the estimate "
                     "only after the start-up transient")
        a.claim = {"type": scope, "statement": stmt, "value": v, "conditions": conds, "uncertainty": "error covariance P_{k|k}"}
    elif obj == "mse_bound":
        target = S.get(spec, "objective.mse_target")
        if target is None:
            a.status = "needs"
            a.needs.append(Need("objective.mse_target", "the request asks for a bounded mean-square error", "What mean-square error is acceptable?"))
            return a
        worst = a.metrics["mse_post_deadline_max"]
        if worst <= target:
            a.status = "qualifies"
            a.claim = {"type": "mse_bound", "statement": f"mean-square error of the target {window} is at most {fmt_up(worst)} <= {target:.6g}",
                       "value": worst, "conditions": conds, "uncertainty": "error covariance P_{k|k}"}
        else:
            a.status = "fails"
            opt = a.metrics.get("optimal_window_max", worst)      # optimal (time-varying) value over the same samples
            if opt > target:
                cls = "any estimator" if scope == "mmse" else "any linear estimator"
                a.impossibility = {"argument": f"the optimal (time-varying) Kalman filter attains the minimum mean-square error over {cls} of the same data under the "
                                               f"stated model ({scope}) at every sample; its error variance {window} reaches {opt:.6g} > {target:.6g}, so {cls} "
                                               "exceeds the requested level at that sample as well",
                                   "class": cls, "bound": opt}
                a.limits.append(f"requested mean-square error {target:.6g} is below the optimum {opt:.6g} of the stated model")
            else:
                a.limits.append(f"the constant (steady-state) gain reaches {worst:.6g} > {target:.6g} {window}; a time-varying gain could meet "
                                "the request (horizon 'finite')")
    elif obj == "probability":
        tol, pr = S.get(spec, "objective.tolerance"), S.get(spec, "objective.prob")
        for path, v, q in (("objective.tolerance", tol, "What error radius should hold?"), ("objective.prob", pr, "With what probability should the error stay within the radius?")):
            if v is None:
                a.needs.append(Need(path, "a probability statement needs a radius and a level", q))
        if a.needs:
            a.status = "needs"
            return a
        var = a.metrics["mse_post_deadline_max"]
        if scope == "mmse":
            level = math.erf(tol / math.sqrt(2.0 * var)) if var > 0 else 1.0
            how = "Gaussian error with variance P: P(|e| <= r) = erf(r / sqrt(2 P))"
        else:
            level = max(0.0, 1.0 - var / (tol * tol))
            how = "Chebyshev inequality with error variance P (distribution-free): P(|e| <= r) >= 1 - P / r^2"
        a.metrics["prob_level"] = level
        if level >= pr:
            a.status = "qualifies"
            a.claim = {"type": "probability", "statement": f"{window}, |error| <= {tol:.6g} with probability at least {fmt_down(level)} ({how}); "
                                                           "the statement is per time instant, not simultaneous over the window",
                       "value": level, "conditions": conds, "uncertainty": "probability level at a fixed radius"}
        else:
            a.status = "fails"
            a.limits.append(f"the provable level {level:.6f} is below the requested {pr:.6f} ({how})")
            if scope == "lmmse":
                a.limits.append("a failed Chebyshev test is not an impossibility: knowledge of the distribution could give a higher level")
    elif obj == "hard_bound":
        a.limits.append("a covariance is not a deterministic bound; a hard error limit for the Kalman gain is evaluated only on the "
                        "sampled double-integrator class, where the exact worst case of any constant linear gain is available")
        if not is_double_integrator(A, C, Ts) or tgt not in (0, 1):
            a.status = "not_applicable"
            a.reasons.append("no deterministic error certificate for a Kalman gain on this model class is in the register")
            return a
        if not (detectable(A, C) and stabilizable_noise(A, Qd)):
            a.status = "not_applicable"
            a.reasons.append("no steady-state gain exists for the stated noise model")
            return a
        Vb, Db, E = S.get(spec, "deterministic.V"), hard_disturbance(spec), S.get(spec, "initial.E")
        Db, Ddet = (Db if Db is not None else (None, None))
        tol = S.get(spec, "objective.tolerance")
        for path, v, why, q in (
            ("deterministic.V", Vb, "a guaranteed error limit needs a hard bound on the measurement error", "What is the largest possible measurement error (hard bound)?"),
            ("deterministic.D", Db, "a guaranteed error limit needs a hard bound on the unmodelled acceleration", "What is the largest possible unmodelled acceleration (hard bound)?"),
            ("objective.tolerance", tol, "the request asks for a guaranteed error limit", "What error limit must be guaranteed?")):
            if v is None:
                a.needs.append(Need(path, why, q))
        if E is None and (dl is not None or S.get(spec, "objective.peak_cap") is not None):
            a.needs.append(Need("initial.E", "with a deadline or a transient cap the guaranteed error depends on the initial error",
                                "What are hard bounds on the initial position and velocity estimate errors?"))
        if a.needs:
            a.status = "needs"
            return a
        Mss = steady_state(A, C, Qd, R)[2].reshape(2)
        c = np.eye(2)[tgt]
        # same initialisation and premise as the observer: position estimate = first measurement, so e1(0) = -v(0) exactly;
        # velocity estimate = x0_hat[1] (zero if none) with |x2(0) - xh2(0)| <= E2
        Eb = [Vb, float(E[1])] if E is not None else [0.0, 0.0]
        k_hi = n_end if dl is not None else None
        cert = peak.certify_lti(Ts, Mss, c, Eb, Db, Vb, k_deadline=k_dead if dl is not None else 0, init="first_measurement", k_end=k_hi)
        a.metrics["worst_case"] = cert
        a.config = dict(cfg, mode="steady_state", M_ss=Mss.reshape(2, 1), P_ss=steady_state(A, C, Qd, R)[0], Pf_ss=steady_state(A, C, Qd, R)[1],
                        init_policy="first_measurement",
                        hard_bound={"V": float(Vb), "D": float(Db), "E": [float(Vb), float(Eb[1])], "target": int(tgt)})
        if not cert["stable"]:
            a.status = "fails"; a.limits.append("contraction certificate not obtained for the steady-state gain")
            return a
        bound = cert["post_deadline_upper"] if dl is not None else cert["ultimate_upper"]
        cap = S.get(spec, "objective.peak_cap")
        ok = bound <= tol and (cap is None or cert["peak_upper"] <= cap)
        a.metrics["hard_bound_value"] = bound
        if ok:
            a.status = "qualifies"
            a.claim = {"type": "hard_bound",
                       "statement": TM.claim_statement(bound, cert["peak_upper"] if cap is not None else None, Ts, k_dead, k_hi, dl)
                                    + " for the steady-state Kalman gain, for every admissible initial error, measurement error and disturbance "
                                      "(exact worst case, interval-certified)",
                       "value": bound, "conditions": ["sampled double-integrator error model", f"|measurement error| <= {Vb:.12g}",
                                                      f"|x2' - nominal model| <= {Db:.12g} on each sample interval (arbitrary within this bound"
                                                      + ("" if not Ddet else "; " + Ddet) + ")",
                                                      "initial estimate: position = first measurement (so e1(0) = -v(0)), velocity = supplied initial estimate "
                                                      "(zero if none), with |x2(0) - xh2(0)| <= E2" if E is not None else "ultimate bound: the initial error does not enter",
                                                      "applied input held constant on each sample interval",
                                                      "constant (steady-state) gain; the time-varying transient gain is not covered"],
                       "uncertainty": "deterministic error envelope"}
            if cap is not None:
                a.claim["peak_value"] = cert["peak_upper"]
        else:
            a.status = "fails"
            a.limits.append(f"the steady-state Kalman gain has certified worst case {bound:.6g} > {tol:.6g}"
                            + ("" if cap is None or cert["peak_upper"] <= cap else f" or a transient above the cap {cap:.6g}")
                            + "; other gains are not excluded by this test")
    else:
        a.status = "needs"
        a.needs.append(Need("objective.kind", "the applicable claim depends on what success means",
                            "How is success defined: smallest mean-square error, a mean-square-error level, a probability level, or a guaranteed error limit?"))
    return a


def uncertain_parameters(spec: dict) -> dict:
    """Parameters of the stated plant whose true value is known only as a non-degenerate range, or is stated exactly but
    differs from the nominal value of the model (a range of one point; review case 02)."""
    so = S.get(spec, "plant.second_order") or {}
    nom = so.get("params") or {}
    out = {}
    for k, r in (so.get("param_ranges") or {}).items():
        if r is None:
            continue
        if r[1] > r[0] or (k in nom and abs(float(nom[k]) - float(r[0])) > 1e-12 * max(1.0, abs(float(r[0])))):
            out[k] = list(r)
    return out


def hard_disturbance(spec: dict):
    """(D, note) for the hard-bound evaluation of a Kalman gain on the double integrator, or None.

    The supplied total bound is used; when the second-order block states the same nominal model as the linear one
    (f0 = b u with the linear input gain b) the bound derived from the stated model is used as well and the smaller one
    is taken, exactly as for the observer (revision 2, finding 4)."""
    D = S.get(spec, "deterministic.D")
    so = S.get(spec, "plant.second_order") or {}
    lin = S.get(spec, "plant.linear") or {}
    try:
        import sympy as sp
        b = float(np.asarray(lin.get("B"), float).reshape(-1)[-1])
        f0 = sp.sympify(str(so.get("f0")), locals={k: sp.Float(v) for k, v in (so.get("params") or {}).items()})
        same = so.get("f0") is not None and sp.simplify(f0 - sp.Float(b) * sp.Symbol("u")) == 0
    except Exception:  # noqa
        same = False
    if same:
        from ..khalil import specialist as H
        try:
            Dt, det = H.total_disturbance(spec)
            note = None if D is not None and abs(Dt - float(D)) <= 0 else det.get("how")
            return float(Dt), note
        except Exception:  # noqa
            pass
    return (float(D), None) if D is not None else None


def export_config(cfg: dict) -> dict:
    return jsonable({k: v for k, v in cfg.items()})


METHOD_CARD = {
    "id": "K", "name": "Kalman filter (discrete time, known linear model)",
    "inputs": ["A, B, C (or continuous A, B with Ts and zero-order hold)", "Qd or (Qc, G)", "R", "x0_hat, P0 (or steady state)", "y_0..y_N, u_0..u_N-1"],
    "outputs": ["x_{k|k}", "P_{k|k}", "x_{k+1|k}", "P_{k+1|k}", "gains", "innovations", "NIS"],
}
