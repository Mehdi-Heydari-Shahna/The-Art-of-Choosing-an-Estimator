"""Independent reference computations for sampled / continuous second-order observers
with bounded uncertainty (worst-case, i.e. peak-to-peak, analysis).

Plant:  x1' = x2,  x2' = a(t),  measured y_k = x1(kT) + v_k at t_k = kT.
Sampled observer (knows a nominal acceleration a0_k on [t_k, t_{k+1}))::

    update :  xp_k     = xm_k + M (y_k - xm_k[0])               (M is a 2-vector)
    predict:  xm_{k+1} = Ad xp_k + Gam a0_k,  Ad = [[1, T], [0, 1]],  Gam = [T^2/2, T]

delta(t) = a(t) - a0_k on [t_k, t_{k+1}).  Admissible set: |x_i(0) - xm_0[i]| <= E_i,
|v_k| <= V, |delta(t)| <= D (measurable).  "current" estimate = xp_k (uses y_k),
"predictor" estimate = xm_k (uses y_0..y_{k-1}).  Errors are TRUE minus ESTIMATE.

All discrete-time worst cases are obtained from simulations in original coordinates
(true state and estimate propagated separately); no error recursion or closed-form
gain formula is used.
"""
import math

import numpy as np
from scipy.linalg import expm


def _eval_delta(delta_fn, t):
    """Evaluate delta_fn on the array t (vectorised call if possible, else point-wise)."""
    if delta_fn is None:
        return np.zeros_like(t)
    try:
        d = np.asarray(delta_fn(t), dtype=float)
        if d.shape == t.shape:
            return d
        if d.ndim == 0:
            return np.full(t.shape, float(d))
    except Exception:
        pass
    return np.array([float(delta_fn(float(s))) for s in t])


def simulate_observer(T, M, x0, xm0, v, delta_fn, a0_seq, N, substeps=200, init="box"):
    """Simulate plant and sampled observer over k = 0..N in original coordinates.

    x0, xm0 : true initial state and initial predictor state xm_0 (2-vectors).
    init    : "box" uses xm0 as given; "first_measurement" replaces its position entry by the first reading y_0 (as the
              released observer does when no initial position estimate is used).
    v       : measurement disturbances v_0..v_N (None = zeros).
    delta_fn: callable t -> delta(t) (vectorised or scalar; None = zero), sampled at the
              midpoints (i + 1/2) T/substeps and held over each sub-step.
    a0_seq  : nominal accelerations a0_0..a0_{N-1} (None = zeros; extra entries ignored).
    The true plant uses a(t) = a0_k + delta(t) and is integrated exactly for that
    piecewise-constant acceleration (per sample: x2 += h*sum(a_i),
    x1 += T*x2 + h^2*sum(a_i*(S - i - 1/2)), h = T/S, the closed-form sum of S exact
    sub-steps).  Returns dict with t (N+1,), x, xp, xm (each (N+1, 2)) and y (N+1,).
    """
    m1, m2 = (float(q) for q in np.asarray(M, dtype=float).ravel())
    S, N = int(substeps), int(N)
    h = T / S
    v = np.zeros(N + 1) if v is None else np.asarray(v, dtype=float).ravel()
    a0 = np.zeros(N) if a0_seq is None else np.asarray(a0_seq, dtype=float).ravel()[:N]
    if v.size < N + 1 or a0.size < N:
        raise ValueError("need v_0..v_N and a0_0..a0_{N-1}")
    tm = (np.arange(N * S) + 0.5) * h
    acc = _eval_delta(delta_fn, tm).reshape(N, S) + a0[:, None]
    sum_a = acc.sum(axis=1)
    sum_aw = acc @ (S - np.arange(S) - 0.5)
    x, xp, xm, y = (np.empty((N + 1, 2)), np.empty((N + 1, 2)), np.empty((N + 1, 2)),
                    np.empty(N + 1))
    x1, x2 = (float(q) for q in np.asarray(x0, dtype=float).ravel())
    h1, h2 = (float(q) for q in np.asarray(xm0, dtype=float).ravel())
    for k in range(N + 1):
        yk = x1 + float(v[k])
        if k == 0 and init == "first_measurement":
            h1 = yk
        r = yk - h1
        p1, p2 = h1 + m1 * r, h2 + m2 * r
        x[k], xm[k], xp[k], y[k] = (x1, x2), (h1, h2), (p1, p2), yk
        if k < N:
            x1, x2 = x1 + T * x2 + h * h * float(sum_aw[k]), x2 + h * float(sum_a[k])
            ak = float(a0[k])
            h1, h2 = p1 + T * p2 + 0.5 * T * T * ak, p2 + T * ak
    return {"t": T * np.arange(N + 1), "x": x, "xp": xp, "xm": xm, "y": y}


def _coefficients(T, M, c, N, form, sub, init="box"):
    """Numerical coefficients of the linear map (e_0, v, delta) -> c.(x_N - estimate_N).

    One simulate_observer run per unit input (xm_0 = 0, a0 = 0, substeps = sub).
    Returns ce (2,), cv (N+1,), cd (N, sub); cd[k, j] is the response to delta = 1 on
    [kT + jT/sub, kT + (j+1)T/sub) and 0 elsewhere.
    """
    if form not in ("current", "predictor"):
        raise ValueError("form must be 'current' or 'predictor'")
    c = np.asarray(c, dtype=float).ravel()
    key, h, zv = ("xp" if form == "current" else "xm"), T / sub, np.zeros(N + 1)

    def final(x0, v, dfn):
        out = simulate_observer(T, M, x0, (0.0, 0.0), v, dfn, None, N, substeps=sub, init=init)
        return float(c @ (out["x"][N] - out[key][N]))

    base = final((0.0, 0.0), zv, None)                       # zero by linearity
    ce = np.array([final((1.0, 0.0), zv, None), final((0.0, 1.0), zv, None)]) - base
    cv = np.empty(N + 1)
    for k in range(N + 1):
        vk = zv.copy()
        vk[k] = 1.0
        cv[k] = final((0.0, 0.0), vk, None) - base
    cd = np.empty(N * sub)
    for i in range(N * sub):
        cd[i] = final((0.0, 0.0), zv, lambda t, i=i: (np.floor(t / h) == i) * 1.0) - base
    return ce, cv, cd.reshape(N, sub)


def _pair(E):
    E = np.asarray(E, dtype=float).ravel()
    return np.full(2, E[0]) if E.size == 1 else E[:2]


def worst_case_by_enumeration(T, M, c, E, D, V, N, form="current", sub=64, init="box"):
    """max |c . (x(t_N) - estimate_N)| over the admissible set, delta restricted to be
    constant on each of `sub` equal sub-intervals of every sample interval.

    = |ce|.E + V sum|cv| + D sum|cd| with the coefficients of `_coefficients`.
    This is a LOWER bound of the worst case over measurable delta (see
    worst_case_by_enumeration_bounds / worst_case_exact).  E = (E_1, E_2).
    """
    ce, cv, cd = _coefficients(T, M, c, N, form, sub, init)
    return float(np.abs(ce) @ _pair(E) + V * np.abs(cv).sum() + D * np.abs(cd).sum())


def _defects(cd):
    """Per sample interval: (bound, exact) of int|g| - sum_j |int_j g| in coefficient units.

    Uses the structural fact that the kernel g(tau) (response to an acceleration impulse at
    tau) is AFFINE in tau inside each sample interval (the impulse reaches the next sample
    as the state jump [t_{k+1} - tau, 1], which the rest of the loop maps linearly).  Then
    cd[k, j] = h g(mid_j) is affine in j, g has at most one zero per sample interval, and
    for the sub-interval containing it  int|g| - |int g| = |s| min(r, 1 - r)^2 <= |s|/4,
    s = cd[k, j+1] - cd[k, j], r = relative position of the zero in that sub-interval.
    """
    N, sub = cd.shape
    if sub < 2:
        raise ValueError("need sub >= 2")
    s = (cd[:, -1] - cd[:, 0]) / (sub - 1)                   # slope per sub-interval
    g_start, g_end = cd[:, 0] - 0.5 * s, cd[:, -1] + 0.5 * s
    crossing = g_start * g_end < 0.0
    u = np.where(crossing, -g_start / np.where(s == 0.0, 1.0, s), 0.0)
    r = u - np.floor(u)
    exact = np.where(crossing, np.abs(s) * np.minimum(r, 1.0 - r) ** 2, 0.0)
    return np.where(crossing, 0.25 * np.abs(s), 0.0), exact


def worst_case_by_enumeration_bounds(T, M, c, E, D, V, N, form="current", sub=64, init="box"):
    """(lower, upper) bracket of the true worst case over measurable delta.

    lower = worst_case_by_enumeration(..., sub).
    upper = lower + D * sum over sample intervals whose (affine) kernel changes sign of
            |cd[k, 1] - cd[k, 0]| / 4,  rigorous in exact arithmetic (see _defects);
            upper - lower = O(D N T^2 / sub^2).
    """
    ce, cv, cd = _coefficients(T, M, c, N, form, sub, init)
    lower = float(np.abs(ce) @ _pair(E) + V * np.abs(cv).sum() + D * np.abs(cd).sum())
    return lower, lower + D * float(_defects(cd)[0].sum())


def worst_case_exact(T, M, c, E, D, V, N, form="current", sub=64, init="box"):
    """Worst case over measurable delta: lower bound plus the exactly evaluated defect of
    the single sign-change sub-interval per sample interval (affine kernel, see _defects).
    Independent of `sub` up to rounding (sub >= 2)."""
    ce, cv, cd = _coefficients(T, M, c, N, form, sub, init)
    lower = float(np.abs(ce) @ _pair(E) + V * np.abs(cv).sum() + D * np.abs(cd).sum())
    return lower + D * float(_defects(cd)[1].sum())


def worst_case_signals(T, M, c, E, D, V, N, form="current", sub=64, init="box"):
    """(x0err, v, delta_fn) attaining the value of worst_case_by_enumeration (same sub).

    x0err = E * sign(ce) is the initial error x(0) - xm_0, v_k = V sign(cv_k),
    delta_fn(t) = D sign(cd[k, j]) on sub-interval j of sample interval k (0 outside
    [0, N T)); delta_fn is vectorised.  Reproduce with
    simulate_observer(T, M, xm0 + x0err, xm0, v, delta_fn, a0_seq, N, substeps=m*sub)
    for any integer m >= 1 (substeps MUST be a multiple of sub, so the default 200 does
    not match the default sub = 64); then c.(x_N - estimate_N) equals the lower bound.
    """
    ce, cv, cd = _coefficients(T, M, c, N, form, sub, init)
    sgn, h = np.sign(cd).ravel(), T / sub

    def delta_fn(t):
        idx = np.floor(np.asarray(t, dtype=float) / h).astype(int)
        inside = (idx >= 0) & (idx < sgn.size)
        return D * np.where(inside, sgn[np.clip(idx, 0, sgn.size - 1)], 0.0)

    return _pair(E) * np.sign(ce), V * np.sign(cv), delta_fn


def _cum_abs_integral(f, dt):
    """Cumulative int_0^t |f| on cells of width dt from samples f at spacing dt/2.

    Simpson per cell; in cells where the three samples change sign the quadratic
    interpolant is split at its roots and |.| integrated piecewise (keeps 4th order).
    """
    f0, fm, f1 = f[0:-1:2], f[1::2], f[2::2]
    cell = dt * np.abs(f0 + 4.0 * fm + f1) / 6.0
    lo, hi = np.minimum(np.minimum(f0, fm), f1), np.maximum(np.maximum(f0, fm), f1)
    for i in np.nonzero((lo < 0.0) & (hi > 0.0))[0]:
        b, cq = -3.0 * f0[i] + 4.0 * fm[i] - f1[i], 2.0 * f0[i] - 4.0 * fm[i] + 2.0 * f1[i]
        roots = np.roots([cq, b, f0[i]])
        roots = np.sort(roots[np.isreal(roots)].real)
        brk = np.concatenate([[0.0], roots[(roots > 0.0) & (roots < 1.0)], [1.0]])
        prim = f0[i] * brk + b * brk ** 2 / 2.0 + cq * brk ** 3 / 3.0
        cell[i] = dt * np.abs(np.diff(prim)).sum()
    return np.concatenate([[0.0], np.cumsum(cell)])


def ct_observer_worst_case(g, E, D, V, t_end, target=1, n=20001):
    """Pointwise worst case W(t) of |e_target(t)| for the continuous-time observer.

    xh1' = xh2 + 2 g (y - xh1),  xh2' = a0 + g^2 (y - xh1),  y = x1 + v,  e = x - xh:
    e' = F e + [0, 1]' delta + [-2g, -g^2]' v,  F = [[-2g, 1], [-g^2, 0]].
    W(t) = sum_i |Phi(t)[target, i]| E_i + D int_0^t |(Phi(s) b_d)[target]| ds
           + V int_0^t |(Phi(s) b_v)[target]| ds,   Phi(s) = scipy.linalg.expm(F s),
    over |e_i(0)| <= E_i, |delta| <= D, |v| <= V (each t has its own worst signal).
    target = 1: velocity error (second component); target = 0: position error.
    Returns (t, W) on t = linspace(0, t_end, n); the integrals use expm at spacing
    dt/2 and the 4th-order rule of _cum_abs_integral.
    """
    F = np.array([[-2.0 * g, 1.0], [-g * g, 0.0]])
    t = np.linspace(0.0, t_end, int(n))
    ts = np.linspace(0.0, t_end, 2 * int(n) - 1)
    Phi = np.array([expm(F * s) for s in ts])[:, int(target), :]
    dt = t[1] - t[0]
    resp_d = Phi @ np.array([0.0, 1.0])
    resp_v = Phi @ np.array([-2.0 * g, -g * g])
    W = (np.abs(Phi[::2]) @ _pair(E) + D * _cum_abs_integral(resp_d, dt)
         + V * _cum_abs_integral(resp_v, dt))
    return t, W


def lower_limit_two_signals(D, V, n_check=4001):
    """Two admissible signals with identical (zero) measurements and velocities differing
    by 4 sqrt(D V) at the common time t_star = 2 sqrt(V / D).

    x_plus(t) = -V for t <= 0 and -V + D t^2 / 2 for 0 <= t <= t_star (so x_plus(t_star)
    = +V with velocity 2 sqrt(D V)); x_minus = -x_plus.  With a0 = 0, delta = x'' and
    measurement disturbances v_plus = -x_plus, v_minus = -x_minus both give y = 0 while
    |delta| <= D and |v| <= V.  DOMAIN: t <= t_star only (the functions return nan
    beyond): the signals are admissible on (-inf, t_star], which is what a CAUSAL
    estimate at t_star sees; they cannot be continued past t_star with |x| <= V.
    Any causal estimator returns one velocity estimate for both, so its error is at least
    2 sqrt(D V) for one of them.
    Returns dict: x_plus, x_minus, v_plus, v_minus (vectorised callables), t_star,
    claimed_gap = 4 sqrt(D V), and the numerical verification on n_check points of
    [-t_star, t_star]: max_abs_acc (second differences), max_abs_x, max_abs_y,
    vel_plus / vel_minus (one-sided 3-point difference at t_star), velocity_gap, ok.
    """
    t_star = 2.0 * np.sqrt(V / D)

    def x_plus(t):
        t = np.asarray(t, dtype=float)
        out = np.where(t <= 0.0, -V, -V + 0.5 * D * np.clip(t, 0.0, None) ** 2)
        return np.where(t <= t_star * (1.0 + 1e-12), out, np.nan)

    x_minus = lambda t: -x_plus(t)
    v_plus, v_minus = (lambda t: -x_plus(t)), (lambda t: x_plus(t))
    tg = np.linspace(-t_star, t_star, int(n_check))
    h = tg[1] - tg[0]
    res = {"x_plus": x_plus, "x_minus": x_minus, "v_plus": v_plus, "v_minus": v_minus,
           "t_star": t_star, "claimed_gap": 4.0 * np.sqrt(D * V)}
    acc, amp, meas, vel = 0.0, 0.0, 0.0, []
    for xf, vf in ((x_plus, v_plus), (x_minus, v_minus)):
        xs = xf(tg)
        acc = max(acc, np.max(np.abs(xs[2:] - 2.0 * xs[1:-1] + xs[:-2])) / h ** 2)
        amp = max(amp, np.max(np.abs(xs)), np.max(np.abs(vf(tg))))
        meas = max(meas, np.max(np.abs(xs + vf(tg))))
        vel.append((3.0 * xs[-1] - 4.0 * xs[-2] + xs[-3]) / (2.0 * h))
    res.update(max_abs_acc=acc, max_abs_x=amp, max_abs_y=meas, vel_plus=vel[0],
               vel_minus=vel[1], velocity_gap=vel[0] - vel[1])
    res["ok"] = bool(acc <= D * (1 + 1e-6) and amp <= V * (1 + 1e-12) and meas == 0.0
                     and abs(res["velocity_gap"] - res["claimed_gap"])
                     <= 1e-8 * res["claimed_gap"])
    return res


# ---------------------------------------------------------------------------------------------------------------------
# first-sample limit under stated travel and speed ranges (review cases 03a, 03b of revision 2), written separately from
# heros.core.limits: its own interval evaluation (sympy.lambdify with mpmath interval functions), uniform subdivision of
# the boxes instead of adaptive bisection, a radius grid in powers of two and a closed-form speed search.
# ---------------------------------------------------------------------------------------------------------------------
def _iv_tools():
    import sympy as sp
    from mpmath import iv

    def tanh(x):
        e = iv.exp(2 * x)
        return (e - 1) / (e + 1)

    # binary-float constants (exact) and interval functions; "math" supplies nothing that the dictionary does not override
    mods = [{"sin": iv.sin, "cos": iv.cos, "exp": iv.exp, "tanh": tanh, "sqrt": iv.sqrt, "Abs": abs}, "math"]
    return sp, iv, mods


def _iv_sup(fun, ranges, n=6):
    """Upper bound of sup fun over the box `ranges` (list of (lo, hi)) by uniform subdivision into n cells per axis."""
    from mpmath import iv
    import itertools
    axes = []
    for lo, hi in ranges:
        if hi > lo:
            edges = [lo + (hi - lo) * j / n for j in range(n + 1)]
            axes.append([(edges[j], edges[j + 1]) for j in range(n)])
        else:
            axes.append([(lo, hi)])
    best = -math.inf
    for cell in itertools.product(*axes):
        val = fun(*[iv.mpf([a, b]) for a, b in cell])
        if hasattr(val, "b"):
            from mpmath import mpf
            hi_ = float(val.b)
            if mpf(hi_) < val.b:                 # round the upper end up, never down
                hi_ = math.nextafter(hi_, math.inf)
        else:
            hi_ = float(val)                     # an exact constant
        best = max(best, hi_)
    return best


def first_sample_velocity_limit(spec, lo, hi):
    """Inner approximation [v_b, v_a] of the initial velocities in [lo, hi] from which an admissible trajectory stays
    inside every stated travel and speed range for the whole window: two trajectories start at the same position, share
    the input during the first sample interval, then each brakes with its own constant input inside the stated input range
    and rests (zero acceleration with the input then held and an admissible disturbance); or (energy construction) both
    hold one input with zero disturbance and an energy bound keeps them inside the travel range.  Without a stated travel
    or speed range the whole interval [lo, hi] is admissible.  Returns (v_b, v_a) or None."""
    sp, iv, mods = _iv_tools()
    pl_ = spec.get("plant") or {}
    so = pl_.get("second_order") or {}
    det = spec.get("deterministic") or {}
    dom = det.get("domain") or {}
    if dom.get("x1") is None and dom.get("x2") is None:
        return lo, hi
    tm = spec.get("time") or {}
    Ts, t_end = tm.get("Ts"), tm.get("t_end")
    if Ts is None:
        return None
    D, db = det.get("D"), det.get("d_bound")
    params = dict(so.get("params") or {})
    ranges = {k: v for k, v in (so.get("param_ranges") or {}).items() if v is not None}
    theta = {}
    for k in set(params) | set(ranges):
        v = params.get(k)
        if k in ranges and (v is None or not (ranges[k][0] <= v <= ranges[k][1])):
            v = 0.5 * (ranges[k][0] + ranges[k][1])
        theta[k] = v
    x1, x2, u, y = sp.symbols("x1 x2 u y", real=True)
    loc = {k: sp.Symbol(k, real=True) for k in list(theta) + ["x1", "x2", "u", "y"]}
    loc.update({"sin": sp.sin, "cos": sp.cos, "tanh": sp.tanh, "exp": sp.exp, "sqrt": sp.sqrt, "Abs": sp.Abs})
    cls = []
    if so.get("f") and (db is not None or D is None):
        e = sp.sympify(so["f"], locals=loc).subs({loc[k]: v for k, v in theta.items()})
        cls.append((e, float(db) if db is not None else 0.0))
    if D is not None or not so.get("f"):
        e = sp.sympify(str(so.get("f0") if so.get("f0") is not None else "0"), locals=loc).subs({loc[k]: v for k, v in params.items()}).subs(y, x1)
        if x2 in e.free_symbols:
            return None
        cls.append((e, float(D) if D is not None else 0.0))
    funs = [(sp.lambdify((x1, x2, u), e, modules=mods), sp.lambdify((x1, x2, u), -e, modules=mods), b) for e, b in cls]
    diff = None
    if len(cls) == 2:
        dd = cls[0][0] - cls[1][0]
        diff = (sp.lambdify((x1, x2, u), dd, modules=mods), sp.lambdify((x1, x2, u), -dd, modules=mods))
    L, H = (dom["x1"][0], dom["x1"][1]) if dom.get("x1") is not None else (-math.inf, math.inf)
    x0, E = (spec.get("initial") or {}).get("x0_hat"), (spec.get("initial") or {}).get("E")
    pol = (spec.get("initial") or {}).get("policy") or ("given" if x0 is not None else "first_measurement")
    pa, pb = L, H
    if pol == "given" and x0 is not None and E is not None:
        pa, pb = max(pa, x0[0] - E[0]), min(pb, x0[0] + E[0])
    if pb < pa:
        return None
    p = 0.5 * (pa + pb) if (math.isfinite(pa) and math.isfinite(pb)) else min(max(x0[0] if x0 is not None else 0.0, pa), pb)
    ur = dom.get("u")
    if ur is not None:
        u0 = 0.0 if ur[0] <= 0.0 <= ur[1] else 0.5 * (ur[0] + ur[1])
        own = [ur[0] + (ur[1] - ur[0]) * j / 4 for j in range(5)]
    else:
        u0, own = 0.0, [0.0]

    def sup_e(j, sgn, box, uu):
        f, g, _ = funs[j]
        fn = (lambda a, b: f(a, b, uu)) if sgn > 0 else (lambda a, b: g(a, b, uu))
        return _iv_sup(fn, box)

    def auth(box, sgn, uu):
        A = Ab = math.inf
        for j, (_, _, b) in enumerate(funs):
            A = min(A, b - sup_e(j, sgn, box, uu))
            Ab = min(Ab, b - sup_e(j, -sgn, box, uu))
        if diff is not None:
            if max(_iv_sup(lambda a, c: diff[0](a, c, uu), box), _iv_sup(lambda a, c: diff[1](a, c, uu), box)) > funs[0][2] + funs[1][2]:
                return -math.inf, -math.inf
        return A, Ab

    def rest(box, inputs):
        rb = [box[0], (0.0, 0.0)]
        for uu in inputs:
            if all(max(sup_e(j, 1, rb, uu), sup_e(j, -1, rb, uu)) <= b for j, (_, _, b) in enumerate(funs)):
                return True
        return False

    x2lim = dom.get("x2")

    def largest(sgn, room, vmax):
        if vmax <= 0 or (math.isfinite(room) and room <= 0):
            return 0.0
        # the path may pass through every speed between zero and the start and rests at zero, so a stated speed range
        # has to admit zero on the side of the motion
        if x2lim is not None and ((sgn > 0 and x2lim[0] > 0) or (sgn < 0 and x2lim[1] < 0)):
            return 0.0
        R0 = room if math.isfinite(room) else 1e3 * max(1.0, abs(p))
        best = 0.0
        for k in range(0, 28 if math.isfinite(room) else 52):
            r = R0 * 2.0 ** (-k)
            box = [(p, p + r) if sgn > 0 else (p - r, p), (0.0, vmax) if sgn > 0 else (-vmax, 0.0)]
            A1, A1b = auth(box, sgn, u0)
            if A1 == -math.inf:
                continue
            # second phase: one constant input from Ts on.  An input stays in effect until the next sample instant, so a
            # trajectory that stops between samples rests only if its own braking input admits zero acceleration there;
            # an input that cannot hold it at rest may brake only when the window ends before the stop.
            opts = [(A1, A1b, rest(box, [u0]))]
            for uu in own:
                a, ab = auth(box, sgn, uu)
                if a != -math.inf:
                    opts.append((a, ab, rest(box, [uu])))
            r_u0 = opts[0][2]
            holding = [o for o in opts if o[2]]
            stop_brake = max(holding, key=lambda o: o[0]) if holding else None
            run_brake = max(opts, key=lambda o: o[0])

            def feasible(v):
                T = None if t_end is None else float(t_end)
                e1 = Ts if T is None else min(Ts, T)
                if A1 > 0 and v <= A1 * e1:
                    return v * v / (2 * A1) <= r and v <= vmax and bool(r_u0)
                if A1 <= 0 and A1b < 0:
                    return False
                xa, va = v * e1 - 0.5 * A1 * e1 * e1, v - A1 * e1
                if max(v, va) > vmax:
                    return False
                if T is not None and T <= Ts:
                    return xa <= r
                rem = None if T is None else T - Ts
                for brk, may_stop in ((stop_brake, True), (run_brake, False)):
                    if brk is None:
                        continue
                    A2, A2b = brk[0], brk[1]
                    if A2 > 0:
                        if rem is None or va <= A2 * rem:
                            if may_stop and xa + va * va / (2 * A2) <= r:
                                return True
                            continue
                        if xa + va * rem - 0.5 * A2 * rem * rem <= r:
                            return True
                        continue
                    if rem is None or A2b < 0:
                        continue
                    if xa + va * rem - 0.5 * A2 * rem * rem <= r and va - A2 * rem <= vmax:
                        return True
                return False

            a_, b_ = 0.0, vmax
            if feasible(vmax):
                a_ = vmax
            else:
                for _ in range(45):
                    m = 0.5 * (a_ + b_)
                    a_, b_ = (m, b_) if feasible(m) else (a_, m)
            best = max(best, a_)
            if best >= vmax:
                break
        return best

    wu = largest(+1, H - p, max(hi, 0.0)) if hi > 0 else 0.0
    wd = largest(-1, p - L, max(-lo, 0.0)) if lo < 0 else 0.0
    va, vb = min(hi, wu), max(lo, -wd)
    # second construction (energy): one input and zero disturbance for the whole window
    if so.get("f") and (db is not None or D is None) and math.isfinite(L) and math.isfinite(H):
        f_true = cls[0][0]
        f_nom = None
        if D is not None:
            f_nom = sp.sympify(str(so.get("f0") if so.get("f0") is not None else "0"), locals=loc).subs({loc[k]: v for k, v in params.items()})
        w = _energy_velocity(sp, mods, f_true, f_nom, None if D is None else float(D), p, L, H, dom.get("x2"), ur, max(abs(lo), abs(hi)))
        ea, eb = min(hi, w), max(lo, -w)
        if w > 0 and ea >= eb and (va < vb or ea - eb > va - vb):
            va, vb = ea, eb
    return (vb, va) if va >= vb else None


def _energy_velocity(sp, mods, f_true, f_nom, D, p, L, H, x2r, ur, vcap):
    """Largest initial speed w (<= vcap) from x1 = p for which the true model, holding one input u_c with zero
    disturbance, never leaves [L, H] (nor the speed range): along a trajectory E = x2^2/2 + Phi(x1), Phi(x1) =
    -int_p^x1 f(z, 0, u_c) dz, changes at the rate x2^2 (f(x1, x2, u_c) - f(x1, 0, u_c)) / x2, which is <= 0 where that
    quotient is <= 0; a start with E < Phi(L), Phi(H) cannot reach L or H.  Interval bounds by mpmath (uniform cells)."""
    x1, x2, u, y, z = sp.symbols("x1 x2 u y z", real=True)
    inputs = []
    try:
        inputs += [float(r) for r in sp.solve(sp.Eq(f_true.subs({x1: p, x2: 0}), 0), u) if r.is_real]
    except Exception:  # noqa
        pass
    inputs.append(0.0)
    if ur is not None:
        inputs.append(0.5 * (ur[0] + ur[1]))
        inputs = [q for q in inputs if ur[0] <= q <= ur[1]]
    best = 0.0
    for uc in dict.fromkeys(round(q, 12) for q in inputs):
        fu = f_true.subs(u, uc)
        at_rest = fu.subs(x2, 0)
        damp = sp.cancel(sp.together((fu - at_rest) / x2))
        if sp.fraction(damp)[1].has(x2):
            continue
        try:
            Phi = -sp.integrate(at_rest.subs(x1, z), (z, p, x1))
        except Exception:  # noqa
            continue
        if Phi.has(sp.Integral):
            continue
        neg = sp.lambdify((x1,), -Phi, modules=mods)
        barrier = min(-_iv_sup(neg, [(L, L)]), -_iv_sup(neg, [(H, H)]))
        pmin = min(0.0, -_iv_sup(neg, [(L, H)], n=64))
        if not barrier > 0:
            continue
        w = min(vcap, math.sqrt(2 * barrier) * (1 - 1e-9))
        if x2r is not None:
            lim = min(-x2r[0], x2r[1])
            if lim <= 0:
                continue
            w = min(w, math.sqrt(max(0.0, lim * lim + 2 * pmin)) * (1 - 1e-9))
        if w <= best:
            continue
        smax = math.sqrt(w * w - 2 * pmin)
        if _iv_sup(sp.lambdify((x1, x2), damp, modules=mods), [(L, H), (-smax, smax)], n=16) > 0:
            continue
        if f_nom is not None:
            mis = fu - f_nom.subs(u, uc).subs(x1, y)
            box = [(L, H), (-smax, smax), (L, H)]
            if max(_iv_sup(sp.lambdify((x1, x2, y), mis, modules=mods), box, n=8),
                   _iv_sup(sp.lambdify((x1, x2, y), -mis, modules=mods), box, n=8)) > D:
                continue
        best = w
    return best


def two_signal_travel_room(V, D, t_sample, t_end):
    """Half-width of the travel the judge's two-signal construction needs about its centre: the signals reach +-V at the
    sample with speed 2 sqrt(D V) and then brake with D (another 2V), unless the window ends first."""
    v = 2.0 * math.sqrt(D * V)
    if t_end is None:
        return 3.0 * V
    rem = max(0.0, float(t_end) - float(t_sample))
    return V + (v * v / (2 * D) if v <= D * rem else v * rem - 0.5 * D * rem * rem)
