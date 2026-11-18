"""Exact worst-case (peak) analysis of linear estimators on the sampled double-integrator error model.

Setting.  Plant in chain form  x1' = x2,  x2' = a(t),  measured at t_k = k T:  y_k = x1(t_k) + v_k.
A linear estimator that knows a nominal acceleration a0_k on [t_k, t_{k+1}) runs

    update   xh+_k   = xh-_k + M_k (y_k - C xh-_k),         C = [1 0]
    predict  xh-_k+1 = Ad xh+_k + Gam a0_k,                 Ad = [[1, T], [0, 1]],  Gam = [T^2/2, T]'.

With the total disturbance  delta(t) = a(t) - a0_k  (t in [t_k, t_{k+1})) the errors obey EXACTLY

    e+_k   = (I - M_k C) e-_k - M_k v_k,
    e-_k+1 = Ad e+_k + w_k,       w_k = int_0^T [T - s, 1]' delta(t_k + s) ds.

Admissible uncertainty (independent boxes):
    |e-_0,1| <= E1,  |e-_0,2| <= E2,  |v_k| <= V,  |delta(t)| <= D   (delta measurable, otherwise arbitrary).

Because the error is linear in (e-_0, v, delta) and the admissible set is a product of symmetric boxes, the
largest value of |c' e_N| over the admissible set is the sum of the absolute contributions,

    W_N = |rho_N| . E + sum_{k<N} ( D * int_0^T |rho_{N-1-k} . [tau, 1]'| dtau + V |rho_{N-1-k} Ad M_k| ) + V |c' M_N| [current form],

with row vectors rho_j = c~' Phi^j for constant gains (Phi = Ad (I - M C),  c~' = c'(I - M C) for the current
estimate xh+, c~' = c' for the predictor estimate xh-).  The value is attained by sign-aligned inputs, so W_N is
the exact pointwise worst case, not merely a bound (elementary; stated as Proposition 1 of the project).

Initialisation from the first reading.  When the position estimate is set to the first measurement, xh-_0 = (y_0, xh2_0),
the first update leaves the estimate unchanged and e_0 = (-v_0, e_0,2): the initial position error IS the first measurement
error, so the two are not independent boxes.  The same sign-alignment argument applies to the independent unknowns
(e_0,2, v_0, ..., v_N, delta), which gives, with q_N = c' for N = 0 and q_N = rho_{N-1} Ad for N >= 1 (current form),

    W_N = |q_N| . (V, E2) + sum_{k<N} D * int_0^T |rho_{N-1-k} . [tau, 1]'| dtau + sum_{1<=k<N} V |rho_{N-1-k} Ad M| + [N>=1] V |c' M|,

again attained by admissible signals (`init="first_measurement"`).  It equals the formula above for the gain sequence
M_0 = [1, 0]', M_k = M (k >= 1), and it never exceeds the independent-box formula with E1 = V, which is therefore a valid
but conservative bound for this initialisation (it double-counts v_0).

Two evaluations are provided:
  * float64  (search, plotting, time-varying gains) -- a floating-point evaluation of the exact formula;
  * mpmath interval arithmetic with outward rounding (`certify_lti`) -- the only evidence used for a released
    universal guarantee.  The gain M and the period T are the binary floats that the estimator actually uses,
    which are exact rationals, so the enclosure refers to the implemented observer.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from mpmath import iv, mp

C_ROW = np.array([1.0, 0.0])


def Ad_of(T: float) -> np.ndarray:
    return np.array([[1.0, T], [0.0, 1.0]])


def gam_of(T: float) -> np.ndarray:
    return np.array([0.5 * T * T, T])


# ----------------------------------------------------------------------------------------------------------
# float64 evaluation
# ----------------------------------------------------------------------------------------------------------
def _gamma_f(r1, r2, T):
    """int_0^T |r1*tau + r2| dtau, exact piecewise formula (vectorised)."""
    v0 = r2
    vT = r1 * T + r2
    same = (v0 * vT) >= 0
    a0, aT = np.abs(v0), np.abs(vT)
    den = np.where(a0 + aT > 0, a0 + aT, 1.0)
    return np.where(same, 0.5 * T * (a0 + aT), 0.5 * T * (a0 * a0 + aT * aT) / den)


def phi_of(T: float, M) -> np.ndarray:
    M = np.asarray(M, float).reshape(2)
    return Ad_of(T) @ (np.eye(2) - np.outer(M, C_ROW))


INIT_POLICIES = ("box", "first_measurement")


def _check_init(init, form):
    if init not in INIT_POLICIES:
        raise ValueError(f"init must be one of {INIT_POLICIES}")
    if init == "first_measurement" and form != "current":
        raise ValueError("initialisation from the first reading is defined for the current (updated) estimate only")


def worst_case_lti(T: float, M, c, E, D: float, V: float, N: int, form: str = "current", init: str = "box") -> dict:
    """W_0..W_N for a constant gain M.  Returns the envelope and its three parts (initial, disturbance, noise).

    init="box": independent boxes |e-_0,i| <= E_i (Proposition 1).  init="first_measurement": xh1(0) = y_0, so e_0 = (-v_0, e_0,2)
    and E[0] is not used (the exact worst case of that initialisation, see the module docstring)."""
    _check_init(init, form)
    M = np.asarray(M, float).reshape(2)
    c = np.asarray(c, float).reshape(2)
    Ad = Ad_of(T)
    ImMC = np.eye(2) - np.outer(M, C_ROW)
    Phi = Ad @ ImMC
    ct = c @ ImMC if form == "current" else c.copy()
    rows = np.empty((N + 1, 2))
    r = ct.copy()
    for j in range(N + 1):
        rows[j] = r
        r = r @ Phi
    AdM = Ad @ M
    tD = D * _gamma_f(rows[:, 0], rows[:, 1], T)
    tV = V * np.abs(rows @ AdM)
    SD = np.concatenate([[0.0], np.cumsum(tD)])[: N + 1]
    SV = np.concatenate([[0.0], np.cumsum(tV)])[: N + 1]
    const = V * abs(float(c @ M)) if form == "current" else 0.0
    if init == "box":
        init_part = np.abs(rows[:, 0]) * E[0] + np.abs(rows[:, 1]) * E[1]
        noise = SV + const
    else:
        q = np.empty((N + 1, 2))
        q[0] = c
        if N >= 1:
            q[1:] = rows[:-1] @ Ad
        init_part = np.abs(q[:, 0]) * V + np.abs(q[:, 1]) * E[1]
        # v_0 enters only through e_0: drop its update term (the last term of each running sum) and the k = N term at N = 0
        SVf = np.zeros(N + 1)
        if N >= 2:
            SVf[2:] = np.cumsum(tV)[: N - 1]
        noise = SVf + np.where(np.arange(N + 1) >= 1, const, 0.0)
    return {"W": init_part + SD + noise, "init": init_part, "dist": SD, "noise": noise, "rows": rows, "init_policy": init}


def worst_case_ltv(T: float, Ms: Sequence, c, E, D: float, V: float, form: str = "current", init: str = "box") -> np.ndarray:
    """W_0..W_N for a gain sequence M_0..M_N (float64, O(N^2)).  Used for time-varying Kalman gains.
    init="first_measurement" replaces the first update by the gain [1, 0]' (xh1(0) = y_0), which is exact for that
    initialisation; E[0] then has coefficient zero."""
    _check_init(init, form)
    Ms = [np.asarray(m, float).reshape(2) for m in Ms]
    if init == "first_measurement":
        Ms = [np.array([1.0, 0.0])] + Ms[1:]
    c = np.asarray(c, float).reshape(2)
    Ad = Ad_of(T)
    N = len(Ms) - 1
    W = np.empty(N + 1)
    I2 = np.eye(2)
    for n in range(N + 1):
        if form == "current":
            r = c @ (I2 - np.outer(Ms[n], C_ROW))
            tot = V * abs(float(c @ Ms[n]))
        else:
            r = c.copy()
            tot = 0.0
        for k in range(n - 1, -1, -1):
            tot += D * float(_gamma_f(r[0], r[1], T)) + V * abs(float(r @ Ad @ Ms[k]))
            r = r @ Ad @ (I2 - np.outer(Ms[k], C_ROW))
        W[n] = tot + abs(r[0]) * E[0] + abs(r[1]) * E[1]
    return W


def worst_inputs_lti(T: float, M, c, E, D: float, V: float, N: int, form: str = "current", init: str = "box") -> dict:
    """Admissible initial error, noise sequence and disturbance that attain W_N at step N (tightness witness).

    The disturbance on [t_k, t_{k+1}) is +-D with at most one switch at tau* (returned per step as
    (sign_before, sign_after, s_switch) with s measured from t_k).  For init="first_measurement" the returned e0 is
    (unused, e_0,2): the initial position error is -v_0, with v_0 chosen in v."""
    _check_init(init, form)
    M = np.asarray(M, float).reshape(2)
    c = np.asarray(c, float).reshape(2)
    Ad = Ad_of(T)
    ImMC = np.eye(2) - np.outer(M, C_ROW)
    Phi = Ad @ ImMC
    ct = c @ ImMC if form == "current" else c.copy()
    rows = [ct]
    for _ in range(N):
        rows.append(rows[-1] @ Phi)
    sg = lambda z: 1.0 if z >= 0 else -1.0
    e0 = np.array([E[0] * sg(rows[N][0]), E[1] * sg(rows[N][1])])
    v = np.zeros(N + 1)
    dist = []
    AdM = Ad @ M
    for k in range(N):
        r = rows[N - 1 - k]
        v[k] = -V * sg(float(r @ AdM))            # contribution  -rho Ad M v_k
        # contribution  int_0^T (r1*(T-s) + r2) delta(t_k+s) ds ; integrand sign changes at s* = T + r2/r1
        f0, fT = r[0] * T + r[1], r[1]            # value of r.[T-s,1] at s = 0 and s = T
        if f0 * fT >= 0:
            dist.append((sg(f0 + fT), sg(f0 + fT), T))
        else:
            dist.append((sg(f0), sg(fT), T * abs(f0) / (abs(f0) + abs(fT))))
    if form == "current":
        v[N] = -V * sg(float(c @ M))
    if init == "first_measurement":
        q = c if N == 0 else rows[N - 1] @ Ad
        v[0] = -V * sg(float(q[0]))            # e_0,1 = -v_0 has coefficient q_1
        e0 = np.array([0.0, E[1] * sg(float(q[1]))])
    return {"e0": e0, "v": v, "dist": dist, "init_policy": init}


# ----------------------------------------------------------------------------------------------------------
# interval-arithmetic certificate
# ----------------------------------------------------------------------------------------------------------
def _ivm(x):
    """Exact interval for a binary float / int (a float is an exact rational)."""
    return iv.mpf(x)


def _abs_iv(x):
    return abs(x)


def _gamma_iv(r1, r2, T):
    """Upper-safe enclosure of int_0^T |r1*tau + r2| dtau."""
    v0 = r2
    vT = r1 * T + r2
    a0, aT = abs(v0), abs(vT)
    pos0, neg0 = v0.a >= 0, v0.b <= 0
    posT, negT = vT.a >= 0, vT.b <= 0
    if (pos0 and posT) or (neg0 and negT):
        return T * (a0 + aT) / 2
    if (pos0 and negT) or (neg0 and posT):
        den = a0 + aT
        if den.a > 0:
            return T * (a0 * a0 + aT * aT) / (2 * den)
    # sign not decided by the enclosure: |affine| is convex, so the trapezoid of the end values is an upper bound
    hull_hi = (T * (a0 + aT) / 2).b
    return iv.mpf([0, hull_hi])


def _up(x) -> float:
    """Upper endpoint of an interval as a float, rounded up."""
    f = float(x.b)
    if mp.mpf(f) < x.b:
        f = math.nextafter(f, math.inf)
    return f


def _dn(x) -> float:
    f = float(x.a)
    if mp.mpf(f) > x.a:
        f = math.nextafter(f, -math.inf)
    return f


def certify_lti(T: float, M, c, E, D, V, k_deadline: int = 0, form: str = "current",
                n_extra: int | None = None, prec: int | None = None, init: str = "box", k_end: int | None = None) -> dict:
    """Certified upper bounds for the exact worst case of |c' e_N| of a constant-gain estimator.

    E = (E1, E2), D, V may be floats or mpmath intervals (their upper endpoints are what matters; all
    coefficients are nonnegative, so the worst case is monotone in each bound).  init="first_measurement": the
    position estimate starts at the first reading (E1 is not used; see the module docstring).  k_end: last sample of a
    finite requirement window; the window bounds then use the samples k_deadline..k_end only.

    Returns (all rounded outward):
      stable                 a contraction certificate ||Phi^m||_inf <= q < 1 was verified
      peak_upper             >= sup_{0 <= N <= k_end} W_N            (k_end = None: every N >= 0)
      post_deadline_upper    >= sup_{k_deadline <= N <= k_end} W_N   (k_end = None: every N >= k_deadline)
      ultimate_lower/upper   enclosure of lim sup W_N
      W_at_deadline_upper    >= W_{k_deadline}
      m, q, N_max, prec      what was computed
    A result with stable=False carries no bound (the sufficient contraction test failed or the observer is unstable).
    """
    _check_init(init, form)
    Mf = np.asarray(M, float).reshape(2)
    cf = np.asarray(c, float).reshape(2)
    Phif = phi_of(T, Mf)
    rho = max(abs(np.linalg.eigvals(Phif)))
    out = {"stable": False, "spectral_radius_float": float(rho), "form": form, "T": T, "M": [float(Mf[0]), float(Mf[1])],
           "k_deadline": int(k_deadline), "k_end": None if k_end is None else int(k_end), "init": init}
    if k_end is not None and k_end < k_deadline:
        out["reason"] = "the requirement window contains no sample"
        return out
    if not np.all(np.isfinite(Phif)) or rho >= 1.0 - 1e-12:
        out["reason"] = "spectral radius of the error map is not below one"
        return out
    # horizon: contraction index m (float search), then enough steps for the tail to be negligible
    P = np.eye(2)
    m = None
    for j in range(1, 200001):
        P = P @ Phif
        if np.abs(P).sum(axis=1).max() < 0.5:
            m = j
            break
    if m is None:
        out["reason"] = "no contraction index found within 200000 steps"
        return out
    if n_extra is None:
        n_extra = 45 * m                     # ||Phi^(45 m)|| < 2^-45
    N_max = int(k_deadline) + n_extra
    # precision: enough to dominate the wrapping effect of interval matrix products
    rho_abs = max(abs(np.linalg.eigvals(np.abs(Phif))))
    if prec is None:
        prec = int(120 + 1.2 * (N_max + m) * max(0.0, math.log2(max(rho_abs, 1.0))))
        prec = min(max(prec, 160), 60000)
    old = iv.prec
    iv.prec = prec
    try:
        Tm = _ivm(T)
        m1, m2 = _ivm(float(Mf[0])), _ivm(float(Mf[1]))
        c1, c2 = _ivm(float(cf[0])), _ivm(float(cf[1]))
        to_iv = lambda z: z if hasattr(z, "a") else _ivm(float(z))
        E1, E2, Di, Vi = to_iv(E[0]), to_iv(E[1]), to_iv(D), to_iv(V)
        fm = init == "first_measurement"
        if fm:
            E1 = Vi                            # e_0,1 = -v_0: used only by the tail bound below (box value with E1 = V dominates)
        # I - M C = [[1-m1, 0], [-m2, 1]] ;  Phi = Ad (I - M C)
        p11, p12 = (1 - m1) - Tm * m2, Tm
        p21, p22 = -m2, iv.mpf(1)
        if form == "current":
            r1, r2 = c1 * (1 - m1) - c2 * m2, c2
            const = Vi * abs(c1 * m1 + c2 * m2)
        else:
            r1, r2 = c1, c2
            const = iv.mpf(0)
        adm1, adm2 = m1 + Tm * m2, m2          # Ad M
        S = iv.mpf(0)
        peak = iv.mpf(0)
        post = iv.mpf(0)
        W_dead = None
        norms = []
        last = N_max if k_end is None else min(N_max, int(k_end))
        total = N_max + m
        prev = None                            # (rho_{j-1}, its noise term) for the first-reading initialisation
        for j in range(total + 1):
            t_noise = Vi * abs(r1 * adm1 + r2 * adm2)
            if j <= last:
                if not fm:
                    W = abs(r1) * E1 + abs(r2) * E2 + S + const
                elif j == 0:
                    W = abs(c1) * Vi + abs(c2) * E2
                else:
                    q1, q2 = prev[0], prev[0] * Tm + prev[1]          # rho_{j-1} Ad
                    W = abs(q1) * Vi + abs(q2) * E2 + (S - prev[2]) + const
                if W.b > peak.b:
                    peak = iv.mpf(W.b)
                if j >= k_deadline and W.b > post.b:
                    post = iv.mpf(W.b)
                if j == k_deadline:
                    W_dead = W
            if j > N_max:
                norms.append(abs(r1) + abs(r2))
            # term j and step
            t = Di * _gamma_iv(r1, r2, Tm) + t_noise
            prev = (r1, r2, t_noise)
            if j <= N_max:
                S = S + t
            r1, r2 = r1 * p11 + r2 * p21, r1 * p12 + r2 * p22
        S_next = S                            # = S_{N_max+1}
        # contraction certificate ||Phi^m||_inf <= q  (interval matrix power)
        a11, a12, a21, a22 = iv.mpf(1), iv.mpf(0), iv.mpf(0), iv.mpf(1)
        for _ in range(m):
            a11, a12, a21, a22 = (a11 * p11 + a12 * p21, a11 * p12 + a12 * p22,
                                  a21 * p11 + a22 * p21, a21 * p12 + a22 * p22)
        q1, q2 = abs(a11) + abs(a12), abs(a21) + abs(a22)
        q = q1 if q1.b >= q2.b else q2
        width_ok = all((x.b - x.a) < iv.mpf(10) ** (-12) * (1 + abs(x.b)) for x in (S_next, q))
        if not (q.b < 1) or not width_ok:
            out["reason"] = "contraction or enclosure-width check failed at precision %d" % prec
            out["q_upper"] = float(q.b)
            return out
        R_max = norms[0]
        R_sum = iv.mpf(0)
        for nr in norms:
            R_sum = R_sum + nr
            if nr.b > R_max.b:
                R_max = nr
        Emax = E1 if E1.b >= E2.b else E2
        kappa = Di * (Tm * Tm / 2 if (Tm * Tm / 2).b >= Tm.b else Tm) + Vi * (abs(adm1) if abs(adm1).b >= abs(adm2).b else abs(adm2))
        tail_sum = kappa * R_sum / (1 - q)
        tail = Emax * R_max + S_next + tail_sum + const
        ult_hi = S_next + tail_sum + const
        ult_lo = S_next + const
        if k_end is not None and k_end <= N_max:
            peak_u, post_u = _up(peak), _up(post)          # finite window: every sample of it was evaluated
        else:
            peak_u = max(_up(peak), _up(tail))
            post_u = max(_up(post), _up(tail))
        out.update(stable=True, m=int(m), q_upper=_up(q), N_max=int(N_max), prec=int(prec),
                   peak_upper=peak_u, post_deadline_upper=post_u,
                   ultimate_lower=_dn(ult_lo), ultimate_upper=_up(ult_hi),
                   W_at_deadline_upper=_up(W_dead), tail_upper=_up(tail))
        return out
    finally:
        iv.prec = old


# ----------------------------------------------------------------------------------------------------------
# continuous-time idealisation (repeated observer pole), closed form
# ----------------------------------------------------------------------------------------------------------
def ct_envelope(g: float, E, D: float, V: float, t, target: int = 1):
    """Exact pointwise worst case of the continuous-time observer error with gains (2g, g^2), g = 1/eps:

        e1' = e2 - 2g (e1 + v),   e2' = delta - g^2 (e1 + v),   |delta| <= D, |v| <= V, |e_i(0)| <= E_i.

    target=1: |e2(t)| (velocity);  target=0: |e1(t)| (position).  tau = g t.
    Impulse responses (repeated pole -g):
        Phi(t) = e^-tau [[1 - tau, t], [-g tau, 1 + tau]]            (g*tau = g^2 t)
        velocity:  k_d = (1+tau)e^-tau,       k_v = -g^2 (1 - tau) e^-tau ... L1 norms in closed form
        position:  k_d = t e^-tau,            k_v = -g (2 - tau) e^-tau
    """
    t = np.asarray(t, float)
    tau = g * t
    ex = np.exp(-tau)
    if target == 1:
        Kd = (2 - (2 + tau) * ex) / g
        Kv = g * np.where(tau <= 1, tau * ex, 2 / math.e - tau * ex)
        return g * tau * ex * E[0] + (1 + tau) * ex * E[1] + D * Kd + V * Kv
    # position error
    Kd = (1 - (1 + tau) * ex) / (g * g)
    # int_0^tau |2 - s| e^-s ds  =  1 - (tau-1)... computed piecewise:  F(s) = (s - 1) e^-s is an antiderivative of (2 - s) e^-s
    F = lambda s: (s - 1) * np.exp(-s)
    Kv = np.where(tau <= 2, F(tau) - F(0.0), (F(2.0) - F(0.0)) + (F(2.0) - F(tau)))
    return np.abs(1 - tau) * ex * E[0] + t * ex * E[1] + D * Kd + V * Kv


def ct_ultimate(g: float, D: float, V: float, target: int = 1) -> float:
    """lim_{t->inf} of ct_envelope: 2D/g + 2gV/e (velocity);  D/g^2 + (1 + 2/e^2) V (position)."""
    if target == 1:
        return 2 * D / g + 2 * g * V / math.e
    return D / (g * g) + (1 + 2 * math.exp(-2.0)) * V


def ct_post_deadline_sup(g: float, E, D: float, V: float, t_dead: float, target: int = 1, n: int = 4000) -> float:
    """Numerical sup_{t >= t_dead} of the closed-form envelope (grid in tau up to 60 plus the limit).
    A numerical evaluation with grid resolution 60/n in tau; used for the continuous-time reference curves only."""
    taus = g * t_dead + np.concatenate([[0.0], np.exp(np.linspace(math.log(1e-4), math.log(60.0), n))])
    return float(max(ct_envelope(g, E, D, V, taus / g, target).max(), ct_ultimate(g, D, V, target)))


def lower_limit_velocity(D: float, V: float) -> float:
    """2 sqrt(D V): no causal estimator of x2 can guarantee a smaller worst-case error at any time
    t >= 4 sqrt(V/D) under |x2'| <= D and |v| <= V (two indistinguishable parabolic signals; see the article sidebar).
    Attributed to Seeber and Haimovich (Automatica 2023) for the differentiation setting."""
    return 2.0 * math.sqrt(D * V)
