"""Khalil specialist: second-order, single-output high-gain observers (HGO).

Supported class (method register H1-H4, see docs/SOURCE_TO_METHOD.md):
    x1' = x2,   x2' = f(x1, x2, u) + d(t),   y = x1 + v          (chain form, output = first state)
  H1  continuous-time HGO   xh1' = xh2 + (a1/eps)(y - xh1),  xh2' = f0(y, u) + (a2/eps^2)(y - xh1)
      with the repeated-pole pattern (a1, a2) = (2, 1) and a nominal model f0 that does not use the estimate
      [structure, peaking and the eps trade-off: Esfandiari & Khalil 1992; Khalil & Praly 2014; Khalil 2017]
  H2  exact worst-case envelope of the continuous-time error for bounded disturbance, measurement error and
      initial error (closed form; steady-state part = L1 impulse-response bound of Vasiljevic & Khalil 2008)
  H3  sampled implementation in measurement-update / time-update form with pole-matched gains
      M = [1 - p^2, (1 - p)^2 / Ts],  p = exp(-Ts/eps)        (numerical adaptation of this project)
  H4  exact worst-case envelope of the SAMPLED implementation at the sample instants (heros.core.peak; elementary,
      proved in the article; interval-certified).  This is the guarantee that is released.

What the guarantee covers: the error of the sampled estimate at the sample instants, for every admissible
initial error, measurement error and disturbance, as long as the trajectory stays in the stated operating domain.
What it does not cover: inter-sample error, closed-loop stability, statistics of the error, plants outside the class.
"""
from __future__ import annotations

import math

import numpy as np
from mpmath import iv

from ..core import spec as S
from ..core import peak, bounds
from ..core.assess import fmt_up, fmt_down, Assessment, Need, is_double_integrator, jsonable

METHOD = "high_gain_observer"
VERSION = "1.0.0"
PATTERN = (2.0, 1.0)          # (alpha1, alpha2): repeated observer pole at -1/eps


# ---------------------------------------------------------------------------------------------------------
# gains
# ---------------------------------------------------------------------------------------------------------
def gain_sampled(Ts: float, eps: float) -> np.ndarray:
    """Pole-matched gain of the sampled observer: both eigenvalues of Ad (I - M C) at p = exp(-Ts/eps)."""
    p = math.exp(-Ts / eps)
    return np.array([1.0 - p * p, (1.0 - p) ** 2 / Ts])


def gain_euler_predictor(Ts: float, eps: float) -> np.ndarray:
    """Forward-Euler mapping of the continuous gains, predictor form  xh_{k+1} = Ad xh_k + Gam a0 + L (y_k - xh1_k),
    L = Ts [2/eps, 1/eps^2]; returned as the equivalent update gain M = Ad^{-1} L (use with form='predictor').
    The error map has a double eigenvalue 1 - Ts/eps (stable iff Ts/eps < 2)."""
    L = np.array([2.0 * Ts / eps, Ts / (eps * eps)])
    return np.array([L[0] - Ts * L[1], L[1]])


def rows_double_pole(Ts, M, c, N, form="current"):
    """Closed form rho_j = c~ (p^j I + j p^(j-1) (Phi - p I)) for the pole-matched gain (float search only)."""
    Phi = peak.phi_of(Ts, M)
    p = 0.5 * np.trace(Phi)
    Nil = Phi - p * np.eye(2)
    ct = c @ (np.eye(2) - np.outer(M, peak.C_ROW)) if form == "current" else np.asarray(c, float)
    j = np.arange(N + 1, dtype=float)
    pj = p ** j
    pj1 = np.where(j > 0, j * p ** np.maximum(j - 1, 0), 0.0)
    return pj[:, None] * ct[None, :] + pj1[:, None] * (ct @ Nil)[None, :]


def _float_metrics(Ts, eps, c, E, D, V, k_dead, form="current", init="box", k_end=None):
    """Float evaluation of (post-deadline sup, peak, ultimate) for the pole-matched design; with k_end, the window
    sup and peak use the samples up to k_end only.  init as in heros.core.peak."""
    M = gain_sampled(Ts, eps)
    p = math.exp(-Ts / eps)
    n_tail = int(min(max(40.0 / max(-math.log(p), 1e-12), 50), 400000))
    N = k_dead + n_tail
    rows = rows_double_pole(Ts, M, c, N, form)
    AdM = peak.Ad_of(Ts) @ M
    tD = D * peak._gamma_f(rows[:, 0], rows[:, 1], Ts)
    tV = V * np.abs(rows @ AdM)
    Scum = np.concatenate([[0.0], np.cumsum(tD + tV)])[: N + 1]
    const = V * abs(float(c @ M)) if form == "current" else 0.0
    if init == "first_measurement":
        q = np.vstack([np.asarray(c, float)[None, :], rows[:-1] @ peak.Ad_of(Ts)])
        drop = np.concatenate([[0.0], tV[: N]])                       # the update term of v_0 (index N - 1 at step N)
        W = np.abs(q[:, 0]) * V + np.abs(q[:, 1]) * E[1] + Scum - drop + np.where(np.arange(N + 1) >= 1, const, 0.0)
    else:
        W = np.abs(rows[:, 0]) * E[0] + np.abs(rows[:, 1]) * E[1] + Scum + const
    hi = N if k_end is None else min(N, int(k_end))
    return float(W[k_dead:hi + 1].max()), float(W[: hi + 1].max()), float(Scum[-1] + const)


# ---------------------------------------------------------------------------------------------------------
# total-disturbance bound and its domain evidence
# ---------------------------------------------------------------------------------------------------------
def total_disturbance(spec: dict):
    """Return (D, detail) with |x2'(t) - a0_k| <= D on each sample interval, or raise SpecMissing/SpecNA.

    The bound uses every supplied statement together: a supplied total bound D (deterministic.D) and the bound derived by
    interval arithmetic from the symbolic model,
        D_model = sup_domain |f - f0| + d_bound + Ly (V + |x2|max Ts),
    where the last term accounts for the nominal model being evaluated at the noisy sample y_k and held over the
    interval (Ly = sup |d f0 / d y|).  Both are upper bounds of the same quantity when the statements hold, so the smaller
    one is used (revision 2, finding 4); a supplied D alone is used as stated."""
    D = S.get(spec, "deterministic.D")
    so = S.get(spec, "plant.second_order") or {}
    if D is not None:
        sup = {"how": "supplied total-disturbance bound used as stated", "supplied_D": float(D),
               "conditions": "D bounds |x2'(t) - f0(y_k, u_k)| on every sample interval for all admissible trajectories"}
        if so.get("f") and so.get("f0") is not None and S.get(spec, "deterministic.d_bound") is not None:
            try:
                Dm, det = _model_disturbance(spec)
            except (SpecMissing, SpecNA) as ex:
                sup["model_bound"] = "not available: " + str(ex)
                return float(D), sup
            if Dm < float(D):
                return Dm, dict(det, how=f"bound derived from the stated model ({det['how']}), smaller than the supplied total bound "
                                         f"D = {float(D):.12g}; both statements are used together", supplied_D=float(D))
            sup["model_bound"] = Dm
            sup["how"] += f" (the bound derived from the stated model, {Dm:.6g}, is not smaller)"
        return float(D), sup
    return _model_disturbance(spec)


def _model_disturbance(spec: dict):
    """Interval-arithmetic bound of |x2' - f0(y_k, u_k)| from the symbolic model (raises SpecMissing / SpecNA)."""
    so = S.get(spec, "plant.second_order") or {}
    if not so.get("f") or so.get("f0") is None:
        raise SpecMissing("deterministic.D", "neither a total-disturbance bound nor a symbolic model to derive it from is supplied")
    Vb, Ts = S.get(spec, "deterministic.V"), S.get(spec, "time.Ts")
    db = S.get(spec, "deterministic.d_bound")
    dom = S.get(spec, "deterministic.domain") or {}
    params = dict(so.get("params") or {})
    pr = {k: tuple(v) for k, v in (so.get("param_ranges") or {}).items()}
    names = ["x1", "x2", "u"] + sorted(set(params) | set(pr))
    try:
        f = bounds.parse(so["f"], names)
        f0 = bounds.parse(str(so["f0"]), names + ["y"])
    except bounds.BoundError as ex:
        raise SpecNA(str(ex))
    import sympy as sp
    ysym, x1 = sp.Symbol("y", real=True), sp.Symbol("x1", real=True)
    f0 = f0.subs(ysym, x1)
    if sp.Symbol("x2", real=True) in f0.free_symbols:
        raise SpecNA("the nominal observer model depends on the unmeasured state; estimate-dependent nominal models are outside the register")
    # true-plant parameters range over param_ranges; the observer uses the nominal values (suffix-free names in f0 are nominal)
    nomsub = {sp.Symbol(k, real=True): sp.Float(v) for k, v in params.items()}
    f0n = f0.subs(nomsub)
    # exactly known true parameters: without a stated range the true plant uses the nominal value; a range of one point
    # states the true value, which may differ from the nominal one the observer uses (review case 02)
    exact = {sp.Symbol(k, real=True): sp.Float(v) for k, v in params.items() if k not in pr}
    exact.update({sp.Symbol(k, real=True): sp.Float(r[0]) for k, r in pr.items() if r[0] == r[1]})
    diff = bounds.collect_for_bounds(sp.simplify(f.subs(exact) - f0n), ["x1", "x2", "u"])
    if diff == 0 and sp.diff(f0n, x1) == 0:
        # the observer model is exact: the total disturbance is the additive one (asked for as D)
        if db is None:
            raise SpecMissing("deterministic.D", "a hard bound on the unmodelled acceleration is not stated")
        return float(db), {"how": "exact nominal model: the total disturbance equals the additive disturbance bound", "additive": float(db),
                           "conditions": "global bound (no domain restriction used)", "domain_used": []}
    box, used_domain = {}, []
    for k in sorted(set(params) | set(pr)):
        box[k] = pr.get(k, (params[k], params[k]))
    free = {str(s) for s in diff.free_symbols}
    Ly_expr = sp.diff(f0n, x1)
    need_x2_for_hold = Ly_expr != 0
    for v in ("x1", "x2", "u"):
        if v in free or (v == "x2" and need_x2_for_hold) or (v in {str(s) for s in Ly_expr.free_symbols}):
            rng = dom.get(v)
            if rng is None:
                raise SpecMissing("deterministic.domain", f"the mismatch bound depends on {v}, whose operating range is not stated")
            box[v] = (float(rng[0]), float(rng[1])); used_domain.append(v)
    if db is None:
        raise SpecMissing("deterministic.d_bound", "a hard bound on the additive disturbance is not stated")
    ev = S.get(spec, "deterministic.domain_evidence") or {}
    if used_domain and ev.get("kind") in (None, "none"):
        raise SpecMissing("deterministic.domain_evidence",
                          "the mismatch bound holds only on the operating domain (" + ", ".join(used_domain) +
                          "); evidence that trajectories remain there is required before a guarantee is claimed")
    ly_box = {k: box[k] for k in (str(s) for s in Ly_expr.free_symbols)} if Ly_expr != 0 else {}
    if "x1" in ly_box:
        # f0 is evaluated at y_k = x1(t_k) + v_k, which can lie up to V outside the operating range of x1
        if Vb is None:
            raise SpecMissing("deterministic.V", "the nominal model is evaluated at the noisy measurement; its bound is needed")
        # a coordinator checking several admissible completions may ask for a wider range (the largest admissible V), which
        # keeps the bound affine in V across them (second recheck A); it is never narrower than the completion's own V
        Vw = max(float(Vb), float(S.get(spec, "deterministic.V_widen") or 0.0))
        ly_box["x1"] = (box["x1"][0] - Vw, box["x1"][1] + Vw)
    try:
        Dm = bounds.sup_abs(diff, box, budget=128)
        Ly = bounds.sup_abs(Ly_expr, ly_box, budget=64) if Ly_expr != 0 else 0.0
    except bounds.BoundError as ex:
        raise SpecNA(str(ex))
    hold = 0.0
    if Ly > 0:
        if Vb is None:
            raise SpecMissing("deterministic.V", "the nominal model is evaluated at the noisy measurement; its bound is needed")
        if Ts is None:
            raise SpecMissing("time.Ts", "the nominal model is held over a sample interval; the period is needed")
        w = max(abs(box["x2"][0]), abs(box["x2"][1]))
        hold = bounds.up(iv.mpf(Ly) * (iv.mpf(Vb) + iv.mpf(w) * iv.mpf(Ts)))
    Dtot = bounds.up(iv.mpf(Dm) + iv.mpf(float(db)) + iv.mpf(hold))
    return Dtot, {"how": "derived by outward-rounded interval arithmetic: sup|f - f0| + d_bound + Ly (V + |x2|max Ts), Ly over the x1 range widened by V",
                  "model_mismatch": Dm, "additive": float(db), "sample_hold_and_noise": hold, "Ly": Ly,
                  "domain_used": used_domain, "domain_evidence": ev,
                  "conditions": ("valid while the trajectory remains in the operating domain " + str({v: list(box[v]) for v in used_domain}))
                  if used_domain else "global bound (no domain restriction used)"}


class SpecMissing(Exception):
    def __init__(self, path, why):
        super().__init__(why); self.path, self.why = path, why


class SpecNA(Exception):
    pass


_QUESTIONS = {
    "deterministic.V": "What is the largest possible measurement error (a hard bound, not an RMS value)?",
    "deterministic.D": "What is the largest possible unmodelled acceleration, including model mismatch (hard bound)?",
    "deterministic.d_bound": "What is the largest possible additive disturbance acceleration (hard bound)?",
    "deterministic.domain": "Over what ranges of position, velocity and input does the system operate?",
    "deterministic.domain_evidence": "What ensures the trajectory stays inside the stated operating ranges (mechanical limit, saturation argument, or an assumption you accept)?",
    "time.Ts": "What is the exact observer update period?",
    "initial.E": "What are hard bounds on the initial position and velocity estimate errors?",
    "objective.tolerance": "What error limit must be guaranteed?",
}


# ---------------------------------------------------------------------------------------------------------
# design
# ---------------------------------------------------------------------------------------------------------
def free_disturbance_bound(spec: dict):
    """Largest D_f such that, in the stated class, the acceleration error can be ANY measurable signal bounded by D_f while
    the plant rests at an admissible equilibrium; None if the stated class does not contain such signals.

    A true model f that depends on the state fixes part of the acceleration, so the two signals of the argument would have
    different state-dependent parts: no bound (review A, recheck m4).  For a state-free f (input and parameters only) both
    signals use the same input and the same parameter values, so their accelerations differ only through the additive
    disturbance: every stated bound on it counts, the supplied total bound D and the additive bound d_bound alike, and the
    smaller one is used (revision 2, finding 4).  Without a true model the class is |x2' - f0(y_k, u_k)| <= D."""
    so = S.get(spec, "plant.second_order") or {}
    D, db = S.get(spec, "deterministic.D"), S.get(spec, "deterministic.d_bound")
    if so.get("f"):
        try:
            params = sorted(set(so.get("params") or {}) | set(so.get("param_ranges") or {}))
            f = bounds.parse(so["f"], ["x1", "x2", "u"] + params)
        except Exception:  # noqa
            return None
        if {str(v) for v in f.free_symbols} & {"x1", "x2"}:
            return None
        cands = [float(x) for x in (D, db) if x is not None]
    else:
        cands = [float(D)] if D is not None else []
    return min(cands) if cands else None


def _rest_equilibrium(spec: dict, V: float, room: float | None = None):
    """An admissible rest state (x1 = p, x2 = 0, u = u_eq) at which the nominal model, and the true model if one is stated,
    give zero acceleration, inside the stated travel, speed and input ranges and the initial-error box, with the travel
    range containing p +- room (default V).  Returns (p, u_eq, None) or (None, None, reason)."""
    room = V if room is None else room
    import sympy as sp
    so = S.get(spec, "plant.second_order") or {}
    dom = S.get(spec, "deterministic.domain") or {}
    params = dict(so.get("params") or {})
    pr = so.get("param_ranges") or {}
    for k, v in params.items():
        if k in pr and not (pr[k][0] <= v <= pr[k][1]):
            return None, None, f"the nominal value of {k} lies outside its stated range"
    x1r, ur = dom.get("x1"), dom.get("u")
    x0, E = S.get(spec, "initial.x0_hat"), S.get(spec, "initial.E")
    pol = S.get(spec, "initial.policy") or ("given" if x0 is not None else "first_measurement")
    if x1r is not None and x1r[1] - x1r[0] < 2 * room:
        return None, None, (f"the stated travel range is narrower than the {2 * room:.6g} that the two signals of the construction "
                            "need, including their braking after the sample")
    lo, hi = (x1r[0] + room, x1r[1] - room) if x1r is not None else (-math.inf, math.inf)
    p = 0.0 if x1r is None else 0.5 * (x1r[0] + x1r[1])
    if pol == "given" and x0 is not None:
        p = min(max(float(x0[0]), lo), hi)
        if E is not None and abs(p - float(x0[0])) > float(E[0]):
            return None, None, "no rest position inside the travel range is admitted by the initial position-error bound"
    x2h = float(x0[1]) if x0 is not None else 0.0
    if E is not None and abs(x2h) > float(E[1]):
        return None, None, "a resting initial state is not admitted by the initial velocity-error bound"
    try:
        names = ["x1", "x2", "u", "y"] + sorted(set(params) | set(pr))
        nom = {sp.Symbol(k, real=True): sp.Float(v) for k, v in params.items()}
        f0 = bounds.parse(str(so.get("f0") if so.get("f0") is not None else "0"), names).subs(nom)
        f0 = f0.subs({sp.Symbol("y", real=True): sp.Float(p), sp.Symbol("x1", real=True): sp.Float(p), sp.Symbol("x2", real=True): 0})
        exprs = [f0]
        if so.get("f"):
            f = bounds.parse(so["f"], names).subs(nom).subs({sp.Symbol("x1", real=True): sp.Float(p), sp.Symbol("x2", real=True): 0})
            exprs.append(f)
        u = sp.Symbol("u", real=True)
        cands = [0.0]
        for e in exprs:
            if u in e.free_symbols:
                cands += [float(r) for r in sp.solve(sp.Eq(e, 0), u) if r.is_real]
        for ue in cands:
            if ur is not None and not (ur[0] <= ue <= ur[1]):
                continue
            if all(abs(float(e.subs(u, ue))) <= 1e-12 * max(1.0, abs(ue)) for e in exprs):
                return p, ue, None
    except Exception as ex:  # noqa
        return None, None, f"the equilibrium of the stated model could not be determined ({type(ex).__name__})"
    return None, None, "no admissible input holds the plant at rest"


def two_signal_limit(spec: dict, V: float, Ts: float, k_dead: int, n_end, tol=None, cap=None, deadline=None):
    """The lower limit of Seeber and Haimovich for the velocity, applied only where its construction is admissible.

    Construction: both signals rest at (p, 0, u_eq).  Before the claimed sample t_k they separate symmetrically,
    x = p +- w(t) with |w| <= V and |w''| <= D_f: w moves to -V and stops, then accelerates for 2 sqrt(V/D_f) and reaches
    +V with speed 2 sqrt(D_f V) at t_k (4 sqrt(V/D_f) in all).  Their measurements coincide (the noise absorbs 2w, at
    most 2V) and their velocities differ by 4 sqrt(D_f V), so one of them has a velocity error >= 2 sqrt(D_f V) for every
    causal estimator.  After t_k both must stay admissible for the rest of the window: they brake with the free
    acceleration and come to rest, which takes another 2V of travel on each side (3V from p in all; less if the window
    ends first; more where a position-dependent nominal model leaves less free deceleration).
    The claim needs (revision 2, findings 2, 4, 5, and the review cases 03): a worst-case requirement (not a probability
    level); a free disturbance bound D_f; an admissible rest state with the travel range containing the whole excursion;
    the speed range containing +-2 sqrt(D_f V); and a sample of the requirement window at or after 4 sqrt(V/D_f) (the
    tolerance from the deadline on, the transient cap from t = 0 on).
    Returns a dict (bound, argument, sample) or None, and the reason when None."""
    Df = free_disturbance_bound(spec)
    if not Df or V <= 0:
        return None, "the stated class does not let the acceleration error be an arbitrary bounded signal"
    low = 2.0 * math.sqrt(Df * V)
    tstar = 4.0 * math.sqrt(V / Df)
    vstar = 2.0 * math.sqrt(Df * V)
    x2r = (S.get(spec, "deterministic.domain") or {}).get("x2")
    if x2r is not None and not (x2r[0] <= -low and low <= x2r[1]):
        return None, f"the stated speed range {list(x2r)} does not contain the velocities +-{low:.6g} of the construction"
    from ..core import timing as TM
    best = None
    for level, start, what in ((tol, (k_dead * Ts if deadline is not None else None), "tolerance"), (cap, 0.0, "transient cap")):
        if level is None or level >= low:
            continue
        if start is None:                           # ultimate (lim sup) requirement: late samples belong to it
            k = TM.first_sample_at_or_after(Ts, tstar)
        else:
            k = TM.first_sample_at_or_after(Ts, max(tstar, start))
            if n_end is not None and k > n_end:
                continue                            # no sample of the window lies late enough for the construction
        if best is None or k < best[1]:
            best = (level, k, what)
    if best is None:
        return None, "no requirement below the limit applies at a sample at or after 4 sqrt(V/D_f) inside the requested window"
    level, k, what = best
    # travel needed after the sample: braking from vstar with the free deceleration, until rest or the end of the window
    t_end = S.get(spec, "time.t_end")
    rem = None if t_end is None else max(0.0, float(t_end) - k * Ts)
    so = S.get(spec, "plant.second_order") or {}
    room, A3, p, ueq = 3.0 * V, Df, None, None
    for _ in range(4):
        brake = vstar * vstar / (2 * A3) if (rem is None or vstar <= A3 * rem) else vstar * rem - 0.5 * A3 * rem * rem
        room = V + brake
        p, ueq, why = _rest_equilibrium(spec, V, room)
        if why:
            return None, why
        A3n = _free_deceleration(spec, so, Df, p, ueq, room + V)
        if A3n is None or A3n <= 0:
            if rem is not None and rem == 0.0:
                break
            return None, "the nominal model leaves no free deceleration near the rest position to stop the signals after the sample"
        if A3n >= A3 * (1 - 1e-12):
            break
        A3 = A3n
    arg = (f"no causal estimator of the velocity can keep the worst-case error below 2 sqrt(D V) = {low:.6g} at the sample t = {k * Ts:.6g} s, "
           f"which the requested {what} {level:.6g} covers: two admissible signals rest at x1 = {p:.6g}, u = {ueq:.6g} and separate "
           f"symmetrically within +-{V:.6g} (accelerations at most {Df:.6g}) over the 4 sqrt(V/D) = {tstar:.6g} s before that sample; their "
           f"measurements coincide within |v| <= {V:.6g}, their velocities differ by 4 sqrt(D V) at the sample, and afterwards they brake to "
           f"rest within +-{room:.6g} of the rest position (the stated travel, speed and input ranges and the initial-error bounds admit both)")
    return {"argument": arg, "class": "all causal estimators using the same measurements", "bound": low, "sample_time": k * Ts,
            "D_free": Df, "rest": {"x1": p, "u": ueq}, "travel_room": room}, None


def _free_deceleration(spec, so, Df, p, ueq, radius):
    """Free deceleration available after the sample near the rest position: D_f, reduced by the largest nominal
    acceleration f0(y, u_eq) for readings within `radius` of p when the class is bounded through the nominal model."""
    import sympy as sp
    D = S.get(spec, "deterministic.D")
    if so.get("f") and D is None:
        return Df                                       # true state-free model: the additive disturbance is free
    params = dict(so.get("params") or {})
    names = ["x1", "x2", "u", "y"] + sorted(set(params) | set(so.get("param_ranges") or {}))
    try:
        f0 = bounds.parse(str(so.get("f0") if so.get("f0") is not None else "0"), names)
        f0 = f0.subs({sp.Symbol(k, real=True): sp.Float(v) for k, v in params.items()}).subs(sp.Symbol("u", real=True), sp.Float(ueq))
        f0 = f0.subs(sp.Symbol("x1", real=True), sp.Symbol("y", real=True))
        fr = {str(x) for x in f0.free_symbols}
        if not fr:
            return Df - abs(float(f0))
        if fr - {"y"}:
            return None
        return Df - bounds.sup_abs(f0, {"y": (p - radius, p + radius)}, budget=32)
    except Exception:  # noqa
        return None


def search_eps(Ts, c, E, D, V, k_dead, tol, cap=None, eps_range=None, ultimate_only=False, init="box", k_end=None):
    """Float search for the eps that minimises the requested bound subject to the transient cap.
    Returns (eps, value, peak) or (None, best_value, None) if no eps in the range satisfies the cap."""
    lo, hi = eps_range if eps_range else (0.3 * Ts, 3000.0 * Ts)
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), 90))

    def J(eps):
        post, pk, ult = _float_metrics(Ts, eps, c, E, D, V, k_dead, init=init, k_end=k_end)
        return (ult if ultimate_only else post), pk

    vals = [J(e) for e in grid]
    feas = [i for i, (v, pk) in enumerate(vals) if cap is None or pk <= cap]
    if not feas:
        return None, min(v for v, _ in vals), None
    i = min(feas, key=lambda j: vals[j][0])
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    gr = (math.sqrt(5) - 1) / 2
    x1, x2 = b - gr * (b - a), a + gr * (b - a)
    f1, f2 = J(x1), J(x2)
    pen = lambda fv: fv[0] if (cap is None or fv[1] <= cap) else math.inf
    for _ in range(40):
        if pen(f1) <= pen(f2):
            b, x2, f2 = x2, x1, f1
            x1 = b - gr * (b - a); f1 = J(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + gr * (b - a); f2 = J(x2)
    cand = [(grid[i], vals[i]), (x1, f1), (x2, f2)]
    eps, (v, pk) = min(cand, key=lambda t: pen(t[1]))
    return float(eps), float(v), float(pk)


def design(spec: dict, eps: float | None = None) -> dict:
    """Configuration of the sampled observer for a complete specification (raises SpecMissing / SpecNA)."""
    so = S.get(spec, "plant.second_order")
    if S.get(spec, "plant.kind") != "second_order" or not so:
        raise SpecNA("the plant is not given in the second-order chain form x1' = x2, x2' = f + d, y = x1")
    if so.get("output", "x1") != "x1":
        raise SpecNA("the output is not the first chain state; a transformation to observable coordinates (which, with input "
                     "feedthrough, needs the input derivative and must be nonsingular over the whole parameter set) is not verified by this release")
    flags = S.get(spec, "plant.flags") or []
    if flags:
        raise SpecNA("declared feature outside the second-order class: " + ", ".join(flags))
    Ts = S.get(spec, "time.Ts")
    if Ts is None:
        rng = (spec.get("ranges") or {}).get("time.Ts")
        raise SpecMissing("time.Ts", "the sampled observer and its certificate are computed for one exact update period"
                          + (f"; the stated range [{rng[0]}, {rng[1]}] s does not determine it, and a period mismatch adds an error term "
                             "proportional to the velocity that the certificate does not cover" if rng else ""))
    Vb = S.get(spec, "deterministic.V")
    if Vb is None:
        raise SpecMissing("deterministic.V", "a guaranteed error limit needs a hard bound on the measurement error")
    D, ddet = total_disturbance(spec)
    tgt = S.get(spec, "target.state")
    if tgt not in (0, 1):
        raise SpecMissing("target.state", "the estimated component (position or velocity) is not stated")
    pol = S.get(spec, "initial.policy")
    x0 = S.get(spec, "initial.x0_hat")
    if pol is None:
        pol = "given" if x0 is not None else "first_measurement"
    E = S.get(spec, "initial.E")
    init = {"policy": pol}
    if pol == "first_measurement":
        init["x2_hat0"] = float(x0[1]) if x0 is not None else 0.0
        init["E1_effective"] = float(Vb)
        init["premise"] = ("xh1(0) is set to the first measurement, so e1(0) = -v(0) exactly (the same measurement error, not an independent "
                           "one); a supplied position-error bound is not used" + ("" if E is None else f" (supplied E1 = {E[0]})")
                           + "; xh2(0) = " + (f"{float(x0[1]):.6g}" if x0 is not None else "0") + " with |x2(0) - xh2(0)| <= E2")
        E_eff = [float(Vb), None if E is None else float(E[1])]
    else:
        if x0 is None:
            raise SpecMissing("initial.x0_hat", "initialisation from a supplied estimate was requested but no estimate is given")
        init["x0_hat"] = [float(x0[0]), float(x0[1])]
        init["premise"] = "the supplied initial estimate is used; |e(0)| is bounded componentwise by the supplied E"
        E_eff = [None if E is None else float(E[0]), None if E is None else float(E[1])]
    return {"method": METHOD, "version": VERSION, "pattern": list(PATTERN), "Ts": float(Ts), "V": float(Vb), "D": float(D),
            "D_detail": ddet, "target": int(tgt), "init": init, "E_eff": E_eff, "eps": eps,
            "domain": {v: (S.get(spec, "deterministic.domain") or {}).get(v) for v in ddet.get("domain_used", [])},
            "f0": so.get("f0"), "params": so.get("params") or {}, "form": "current"}


def finalize(cfg: dict, eps: float) -> dict:
    cfg = dict(cfg)
    cfg["eps"] = float(eps)
    cfg["M"] = gain_sampled(cfg["Ts"], eps).tolist()
    cfg["pole"] = math.exp(-cfg["Ts"] / eps)
    cfg["continuous_gains"] = [PATTERN[0] / eps, PATTERN[1] / (eps * eps)]
    return cfg


def uses_input(cfg) -> bool:
    """True if the nominal acceleration of the released observer depends on the applied input u."""
    import sympy as sp
    expr = cfg.get("f0")
    if expr in (None, "", "0", 0):
        return False
    names = ["y", "x1", "u"] + sorted(cfg.get("params") or {})
    e = bounds.parse(str(expr), names).subs({sp.Symbol(k, real=True): sp.Float(v) for k, v in (cfg.get("params") or {}).items()})
    return sp.Symbol("u", real=True) in e.free_symbols


def _nominal(cfg):
    """Callable nominal acceleration a0(y, u) from the stored expression (numerical evaluation at run time)."""
    import sympy as sp
    expr = cfg.get("f0")
    if expr in (None, "", "0", 0):
        return lambda y, u: 0.0
    names = ["y", "x1", "u"] + sorted(cfg.get("params") or {})
    e = bounds.parse(str(expr), names)
    e = e.subs({sp.Symbol(k, real=True): sp.Float(v) for k, v in (cfg.get("params") or {}).items()})
    e = e.subs(sp.Symbol("x1", real=True), sp.Symbol("y", real=True))
    fn = sp.lambdify([sp.Symbol("y", real=True), sp.Symbol("u", real=True)], e, "math")
    return lambda y, u: float(fn(y, u))


def run(cfg: dict, y, u=None, envelope_E=None) -> dict:
    """PUBLIC RUN-TIME PATH: sampled high-gain observer on actual measurements y_0..y_N and applied inputs u_0..u_N-1.

    Returns the estimates xh+_k at the sample instants and, when the initial-error box is known, the certified
    deterministic error envelope W_k of the estimated component (float evaluation of the exact formula)."""
    y = np.asarray(y, float).reshape(-1)
    N = len(y) - 1
    u = np.zeros(N + 1) if u is None else np.asarray(u, float).reshape(-1)
    if len(u) < N:
        raise ValueError(f"need at least {N} applied-input samples for {N + 1} measurements, got {len(u)}")
    if len(u) == N:
        u = np.append(u, 0.0)
    Ts, M = cfg["Ts"], np.asarray(cfg["M"], float)
    Ad, Gam = peak.Ad_of(Ts), peak.gam_of(Ts)
    a0 = _nominal(cfg)
    init = cfg["init"]
    xm = np.array([y[0], init.get("x2_hat0", 0.0)]) if init["policy"] == "first_measurement" else np.array(init["x0_hat"], float)
    X = np.empty((N + 1, 2)); A0 = np.empty(N + 1)
    for k in range(N + 1):
        xp = xm + M * (y[k] - xm[0])
        X[k] = xp
        A0[k] = a0(y[k], u[k])
        xm = Ad @ xp + Gam * A0[k]
    out = {"t": np.arange(N + 1) * Ts, "x_hat": X, "a0": A0,
           "uncertainty_type": "deterministic error envelope at the sample instants (not a covariance)"}
    E = envelope_E if envelope_E is not None else cfg.get("E_eff")
    if E is not None and all(e is not None for e in E):
        c = np.eye(2)[cfg["target"]]
        wc = peak.worst_case_lti(Ts, M, c, E, cfg["D"], cfg["V"], N, cfg.get("form", "current"), init=cert_init(cfg))
        out["envelope"] = wc["W"]
        out["envelope_kind"] = ("exact worst case at each sample (Proposition 1" + (", first-reading initialisation" if cert_init(cfg) == "first_measurement" else "")
                                + "), evaluated in floating point; the released bound is its interval-certified upper value")
        out["envelope_parts"] = {"initial": wc["init"], "disturbance": wc["dist"], "noise": wc["noise"]}
    out["assumption_monitor"] = _monitor(cfg, y, u, X, out.get("envelope"))
    return out


def _monitor(cfg, y, u, X, env):
    """Run-time check of the operating-domain assumption that the guarantee is conditional on.
    Reports the first sample at which a measured or estimated quantity is certainly / possibly outside the domain.
    From a certain exit on, the certificate no longer applies."""
    dom = cfg.get("domain") or {}
    first = lambda mask: (int(np.argmax(mask)) if np.any(mask) else None)
    mon = {"domain": dom, "certain_exit_index": None, "possible_exit_index": None, "details": []}
    V = cfg["V"]
    cands_c, cands_p = [], []
    if dom.get("x1"):
        lo, hi = dom["x1"]
        cands_c.append(first((y - V > hi) | (y + V < lo))); cands_p.append(first((y + V > hi) | (y - V < lo)))
    if dom.get("u"):
        lo, hi = dom["u"]
        cands_c.append(first((u[: len(y)] > hi) | (u[: len(y)] < lo)))
    if dom.get("x2") and env is not None and cfg["target"] == 1:
        lo, hi = dom["x2"]
        cands_c.append(first((X[:, 1] - env > hi) | (X[:, 1] + env < lo))); cands_p.append(first((X[:, 1] + env > hi) | (X[:, 1] - env < lo)))
    elif dom.get("x2"):
        lo, hi = dom["x2"]
        cands_p.append(first((X[:, 1] > hi) | (X[:, 1] < lo)))
    cc = [c for c in cands_c if c is not None]; cp = [c for c in cands_p + cands_c if c is not None]
    mon["certain_exit_index"] = min(cc) if cc else None
    mon["possible_exit_index"] = min(cp) if cp else None
    mon["status"] = ("domain assumption violated: the guarantee does not apply from the certain-exit sample on" if cc else
                     ("domain exit cannot be excluded from the possible-exit sample on" if cp else "no exit from the operating domain detected"))
    return mon


# ---------------------------------------------------------------------------------------------------------
# assessment
# ---------------------------------------------------------------------------------------------------------
def assess(spec: dict) -> Assessment:
    a = Assessment(method=METHOD, status="not_applicable")
    obj = S.get(spec, "objective.kind")
    if S.get(spec, "plant.kind") != "second_order" or not S.get(spec, "plant.second_order"):
        a.reasons.append("the high-gain observer in the register applies to second-order chain-form plants with one measured output")
        return a
    if S.get(spec, "time.input_hold") not in (None, "zoh"):
        a.reasons.append("the sampled observer and its disturbance bound assume the applied input is held constant on each sample interval")
        return a
    if S.get(spec, "time.irregular"):
        # the gains, the error recursion and every certificate are computed for one fixed period (review case 01a)
        a.reasons.append("irregular sampling is outside the supported method register: the gains and the certificate of the sampled "
                         "observer are computed for samples at one fixed period")
        return a
    if obj in ("mse_optimal", "mse_bound"):
        return _assess_mse(spec, a)
    if obj not in ("hard_bound", "probability"):
        a.status = "needs"
        a.needs.append(Need("objective.kind", "the applicable claim depends on what success means",
                            "How is success defined: smallest mean-square error, a mean-square-error level, a probability level, or a guaranteed error limit?"))
        return a
    tol = S.get(spec, "objective.tolerance")
    dl = S.get(spec, "objective.deadline")
    cap = S.get(spec, "objective.peak_cap")
    needs = []
    cfg = None
    try:
        cfg = design(spec)
    except SpecNA as ex:
        a.reasons.append(str(ex))
        return a
    except SpecMissing as ex:
        needs.append(Need(ex.path, ex.why, _QUESTIONS.get(ex.path, f"Please state {ex.path}.")))
    if tol is None:
        needs.append(Need("objective.tolerance", "the request asks for a guaranteed error limit", _QUESTIONS["objective.tolerance"]))
    if cfg is None or needs:
        # report every other missing hard-bound field as well (re-evaluated on each call)
        for path in ("deterministic.V",):
            if S.get(spec, path) is None and all(n.path != path for n in needs):
                needs.append(Need(path, "a guaranteed error limit needs a hard bound on the measurement error", _QUESTIONS[path]))
        a.status, a.needs = "needs", needs
        return a
    Ts, D, V = cfg["Ts"], cfg["D"], cfg["V"]
    c = np.eye(2)[cfg["target"]]
    from ..core import timing as TM
    k_dead, n_end = TM.sample_window(Ts, dl, S.get(spec, "time.t_end"))
    if dl is None:
        n_end = None                      # an ultimate (lim sup) requirement refers to late samples, not to the record length
    init = cert_init(cfg)
    E_eff = cfg["E_eff"]
    transient_claim = dl is not None or cap is not None
    missing_E = [i for i, e in enumerate(E_eff) if e is None]
    if missing_E and transient_claim:
        which = "velocity" if missing_E == [1] else "position and velocity"
        need = Need("initial.E", f"with a deadline or a transient cap the guaranteed error depends on the initial {which} error "
                                 "(the initial-error terms of the envelope have strictly positive coefficients at every finite time)",
                    _QUESTIONS["initial.E"])
        # threshold for the velocity error: largest E2 for which a certified design exists (position part known)
        if missing_E == [1] and tol is not None:
            E2_CAP = 1e6
            okf = lambda e2: _best(Ts, c, [E_eff[0], e2], D, V, k_dead, cap, init, n_end)[1] <= tol
            thr = _critical(okf, 0.0, E2_CAP)
            if thr is not None and thr >= E2_CAP:
                # no critical value below the largest value tested: say exactly that, and make no claim about larger values
                need.question = (f"What is a hard bound on the initial velocity estimate error? (A certified design was found for every bound tested, "
                                 f"up to {E2_CAP:g}; a finite bound is still required, because no design is certified for an unbounded initial error.)")
            elif thr is not None and thr > 0:
                need.threshold, need.threshold_unit = thr, "state-1 units per second"
                need.question = (f"Is the initial velocity estimate error at most {thr:.6g}? If yes, a certified design exists for the stated "
                                 f"requirement; if it can be larger, this certificate family cannot meet it.")
        a.status, a.needs = "needs", [need]
        return a
    if missing_E and not transient_claim:
        a.harmless.append({"path": "initial.E",
                           "why": "only an ultimate (lim sup) error limit is requested; the initial-error terms of the exact envelope decay to "
                                  "zero for every stable gain, so the ultimate bound does not depend on the initial error"})
    Eb = [0.0 if e is None else e for e in E_eff]
    eps, val, pk = search_eps(Ts, c, Eb, D, V, k_dead, tol, cap, ultimate_only=dl is None, init=init, k_end=n_end)
    a.metrics.update(Ts=Ts, D=D, V=V, E_eff=Eb, init=init, window=[k_dead, n_end],
                     lower_limit_velocity=peak.lower_limit_velocity(D, V) if cfg["target"] == 1 else None)
    if eps is None or val > tol:
        a.status = "fails"
        a.metrics["best_float_bound"] = val
        a.limits.append(f"no gain of the pole-matched sampled observer family meets the requirement (best bound found {val:.6g} > {tol:.6g}"
                        + ("" if eps is not None or cap is None else f", or none keeps the transient below the cap {cap:.6g}") +
                        "); this is the limit of this certificate family, not a proof that no estimator can meet it")
        if cfg["target"] == 1 and obj == "hard_bound":
            imp, why = two_signal_limit(spec, V, Ts, k_dead, n_end, tol=tol, cap=cap, deadline=dl)
            if imp:
                a.impossibility = imp
            else:
                a.metrics["two_signal_not_applied"] = why
        a.config = cfg
        return a
    cfg = finalize(cfg, eps)
    cert = peak.certify_lti(Ts, cfg["M"], c, Eb, D, V, k_deadline=k_dead, init=init, k_end=n_end)
    a.metrics["worst_case"] = cert
    a.config = cfg
    if not cert["stable"]:
        a.status = "fails"; a.limits.append("contraction certificate not obtained: " + cert.get("reason", ""))
        return a
    bound = cert["post_deadline_upper"] if dl is not None else cert["ultimate_upper"]
    ok = bound <= tol and (cap is None or cert["peak_upper"] <= cap)
    a.metrics["hard_bound_value"] = bound
    a.checks.append({"name": "interval certificate of the sampled error envelope", "passed": bool(ok),
                     "detail": {k: cert[k] for k in ("post_deadline_upper", "peak_upper", "ultimate_upper", "q_upper", "m", "N_max", "prec")}})
    if not ok:
        a.status = "fails"
        a.limits.append(f"the float search found a candidate but the interval certificate gives {bound:.6g} > {tol:.6g}")
        return a
    stmt = TM.claim_statement(bound, cert["peak_upper"] if cap is not None else None, Ts, k_dead, n_end, dl)
    a.status = "qualifies"
    conds = ["plant in second-order chain form with measured first state", f"|measurement error| <= {V:.12g}",
             f"|x2' - nominal model| <= {D:.12g} on each sample interval ({cfg['D_detail']['conditions']})",
             cfg["init"]["premise"], "applied input held constant on each sample interval",
             f"observer updated every {Ts:.6g} s with the stated binary gains; the bound is the exact worst case of this recursion in exact arithmetic "
             "(Proposition 1" + (" with the first-reading initialisation" if init == "first_measurement" else "") +
             "), certified from above by interval arithmetic; rounding of the float64 implementation is not included"]
    a.claim = {"type": "hard_bound" if obj == "hard_bound" else "probability",
               "statement": stmt + " for every admissible initial error, measurement error and disturbance"
                            + (" (hence with probability one under any distribution supported on the stated bounds)" if obj == "probability" else ""),
               "value": bound, "conditions": conds, "uncertainty": "deterministic error envelope"}
    a.limits += ["guarantee at the sample instants only; inter-sample error is not covered",
                 "no closed-loop stability claim; no statistical claim",
                 "continuous-time source theorems are not invoked for the sampled implementation: the bound is the exact worst case of the implemented recursion"]
    return a


def cert_init(cfg: dict) -> str:
    """Initialisation model of the certificate: the first reading, or independent boxes around a supplied estimate."""
    return "first_measurement" if (cfg.get("init") or {}).get("policy") == "first_measurement" else "box"


def _best(Ts, c, E, D, V, k_dead, cap, init="box", k_end=None):
    eps, val, pk = search_eps(Ts, c, E, D, V, k_dead, None, cap, init=init, k_end=k_end)
    return eps, (val if eps is not None else math.inf)


def _critical(ok, lo, hi, iters=40):
    """Largest x in [lo, hi] with ok(x) true (ok monotone decreasing); None if ok(lo) is false."""
    if not ok(lo):
        return None
    if ok(hi):
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo * (1 - 1e-6)


def _assess_mse(spec, a):
    """Mean-square-error objectives: the HGO is a fixed-gain linear estimator; on the sampled double integrator with a
    stated second-order noise model its exact error variance is computable (steady state, or over a window from the
    initial covariance), but it carries no optimality.  The same quantity as for the Kalman filter is evaluated: the
    limit for a steady-state request without deadline, otherwise the largest variance over the requested samples
    (review A2)."""
    from ..kalman import specialist as K
    obj = S.get(spec, "objective.kind")
    try:
        kcfg = K.design(dict(spec, objective=dict(spec["objective"], horizon="steady_state")))
    except Exception as ex:  # noqa
        a.reasons.append("no mean-square-error result for the high-gain observer is available here: " + str(getattr(ex, "message", ex)))
        return a
    A, C, Qd, R, Ts = kcfg["A"], kcfg["C"], kcfg["Qd"], kcfg["R"], kcfg["Ts"]
    tgt = S.get(spec, "target.state")
    if not is_double_integrator(A, C, Ts) or tgt not in (0, 1):
        a.reasons.append("no mean-square-error result for the high-gain observer on this plant is in the register")
        return a
    dl = S.get(spec, "objective.deadline")
    te = S.get(spec, "time.t_end")
    from ..core import timing as TM
    k_dead, n_end = TM.sample_window(Ts, dl, te)
    horizon = S.get(spec, "objective.horizon") or "finite"
    asymptotic = horizon == "steady_state" and dl is None
    grid = np.exp(np.linspace(math.log(0.3 * Ts), math.log(3000 * Ts), 120))
    if asymptotic:
        def var(eps):
            Pf = K.steady_cov_of_constant_gain(A, C, Qd, R, gain_sampled(Ts, eps).reshape(2, 1))
            return math.inf if Pf is None else float(Pf[tgt, tgt])
        vals = [var(e) for e in grid]
        window = "in steady state (the limit as k -> infinity)"
    else:
        P0 = S.get(spec, "initial.P0")
        if P0 is None:
            a.status = "needs"
            a.needs.append(Need("initial.P0", "an error statement at finite times (a deadline or a finite horizon) depends on the initial error covariance",
                                "What is the covariance of the initial estimate error (or of the initial state)?"))
            return a
        grid = np.exp(np.linspace(math.log(0.3 * Ts), math.log(300 * Ts), 40))
        gains = np.array([gain_sampled(Ts, e).reshape(2, 1) for e in grid])
        vmax, vd, steps, conv = K.window_variance(A, C, Qd, R, P0, tgt, k_dead, n_end, gains=gains)
        if not conv:
            a.reasons.append("the error covariance of the observer did not converge within the iteration limit")
            return a
        vals = [float(v) for v in vmax]
        window = (f"at every sample from t = {k_dead * Ts:.6g} s to the last sample of the record, t = {n_end * Ts:.6g} s" if n_end is not None
                  else f"at every sample from t = {k_dead * Ts:.6g} s on (covariance iterated to convergence; the limit is included)")
    i = int(np.argmin(vals))
    Pss = K.steady_cov_of_constant_gain(A, C, Qd, R, gain_sampled(Ts, float(grid[i])).reshape(2, 1))
    a.metrics.update(mse_steady=(math.inf if Pss is None else float(Pss[tgt, tgt])), mse_post_deadline_max=vals[i], eps_mse=float(grid[i]), Ts=Ts, window=window)
    if obj == "mse_optimal":
        a.status = "not_applicable"
        a.reasons.append("a minimum-mean-square-error claim is requested; a fixed-gain high-gain observer is one linear estimator and carries no such claim "
                         f"(its best error variance {window} on this model is {vals[i]:.6g})")
        return a
    target = S.get(spec, "objective.mse_target")
    if target is None:
        a.status = "needs"
        a.needs.append(Need("objective.mse_target", "the request asks for a bounded mean-square error", "What mean-square error is acceptable?"))
        return a
    if vals[i] <= target:
        cfg = finalize({"method": METHOD, "version": VERSION, "pattern": list(PATTERN), "Ts": Ts, "target": tgt, "form": "current",
                        "V": S.get(spec, "deterministic.V") or 0.0, "D": S.get(spec, "deterministic.D") or 0.0, "E_eff": None,
                        "init": {"policy": "prior_mean", "x0_hat": [float(v) for v in (S.get(spec, "initial.x0_hat") or [0.0, 0.0])], "x2_hat0": 0.0,
                                 "premise": "initial estimate = prior mean of the state (supplied initial estimate, zero if none)"},
                        "f0": (S.get(spec, "plant.second_order") or {}).get("f0"), "params": (S.get(spec, "plant.second_order") or {}).get("params") or {},
                        "D_detail": {"conditions": "not used for a mean-square-error claim"}}, float(grid[i]))
        a.status, a.config = "qualifies", cfg
        a.claim = {"type": "mse_bound", "statement": f"mean-square error of the target {window} is at most {fmt_up(vals[i])} <= {target:.6g} for the pole-matched gain "
                                                     f"with eps = {grid[i]:.6g} s (exact covariance of this linear estimator under the stated second-order noise model; not optimal)",
                   "value": vals[i], "conditions": ["sampled double integrator", "stated process and measurement covariances"]
                   + ([] if asymptotic else ["stated initial covariance; the observer starts from the prior mean of the state"]),
                   "uncertainty": "error variance"}
    else:
        a.status = "fails"
        a.limits.append(f"best error variance of the observer family {window} is {vals[i]:.6g}, above the requested {target:.6g}")
    return a


def export_config(cfg: dict) -> dict:
    return jsonable(cfg)


METHOD_CARD = {
    "id": "H", "name": "High-gain observer (second-order chain form, single output)",
    "inputs": ["f0(y, u)", "Ts", "V", "D or (f, parameter ranges, domain, d_bound)", "initial-error box or first-measurement initialisation",
               "y_0..y_N, u_0..u_N-1"],
    "outputs": ["xh_k", "eps and gains", "deterministic error envelope W_k", "post-deadline, peak and ultimate bounds"],
}
