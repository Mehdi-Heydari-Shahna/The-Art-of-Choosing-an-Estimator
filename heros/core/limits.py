"""Lower limits that hold for every causal estimator of the stated plant (used only to prove a request impossible).

initial_sample_limit
    At the first sample an estimator has one position reading y_0 = x1(0) + v_0 and the input u_0.  The reading carries
    no information about the velocity, so the velocity estimate at t = 0 can do no better than the middle of the set of
    initial velocities that admissible trajectories can have; its worst-case error is the half-width of that set.

    Which initial velocities are admissible depends on everything the request states about the whole window, not only
    on the initial-error bound: a travel range or a speed range that every admissible trajectory respects for the whole
    window excludes initial velocities from which the plant cannot stay inside it (review cases 03a and 03b of revision
    2: a carriage that stays inside a short travel cannot move fast).  The limit therefore uses an INNER approximation of
    the admissible initial velocities, built from explicit trajectories:
      * without a stated travel or speed range, every velocity of the initial-error box starts an admissible trajectory
        (it continues with zero disturbance; no stated restriction can be violated);
      * with such a range, two trajectories start at the same position p with velocities v_a > v_b and hold the same
        input u_0 during the first sample interval (the estimate at t = 0 has only y_0 and u_0).  Each decelerates with
        at least the authority that the stated model guarantees on a box around its path,
            A = min(d_bound - sup f(x, u), D - sup f0(y = x1, u))   (moving up; the mirror image moving down),
        first with u = u_0 and, from the second sample interval on, with its own constant input inside the stated input
        range; every velocity on its path, from zero to the start, must lie in a stated speed range; it stops inside the
        travel range and stays at rest there, which needs |f(x1, 0, u)| <= d_bound and
        |f0(x1, u)| <= D at the stopping point for the input held at that moment (an input is held until the next
        sample instant, so a braking input that cannot hold the plant at rest is used only when the window ends before
        the stop).  Every bound is an outward-rounded interval bound over the box; on a finite window the trajectory
        need only stay inside the range until the window ends.  The parameters are those of one admissible plant (the
        nominal values inside their stated ranges).  A bound that the request leaves open is taken as zero, the least
        favourable completion.
      * a second construction holds one input u_c for the whole window with zero disturbance: with the potential
        Phi(x1) = -int_p^x1 f(s, 0, u_c) ds of the true model at rest, the energy x2^2 / 2 + Phi(x1) does not increase
        where the velocity-dependent part of f only dissipates, so a trajectory whose initial energy is below Phi at
        both ends of the travel range never reaches them (a restoring force such as a spring stage); the velocities
        it admits are used when they are wider than those of the braking construction.
    The limit is the half-width of the velocities that the construction admits; a requirement that applies at t = 0
    (a transient cap, or a tolerance with deadline 0) below it cannot be met by any estimator.  When the construction
    admits no velocities far enough apart, no claim is made: the coordinator then reports what its certificates show.
    For the position, two positions 2 h apart with h = min(V, half-width of the admitted positions) give the same
    reading; with a stated travel or speed range no position claim is made.

The two-signal limit for later times is in heros.khalil.specialist.two_signal_limit.
"""
from __future__ import annotations

import functools
import json
import math

from . import spec as S
from .timing import sample_window


def _chain_form_position_output(spec: dict) -> bool:
    so = S.get(spec, "plant.second_order") or {}
    return S.get(spec, "plant.kind") == "second_order" and bool(so) and so.get("output", "x1") == "x1"


def initial_sample_limit(spec: dict):
    """Returns a dict (argument, class, bound, level, sample_time, kind) when a requirement at t = 0 is below the limit,
    else None."""
    if S.get(spec, "objective.kind") != "hard_bound" or not _chain_form_position_output(spec):
        return None
    tgt = S.get(spec, "target.state")
    tol, cap, dl, Ts = (S.get(spec, "objective.tolerance"), S.get(spec, "objective.peak_cap"), S.get(spec, "objective.deadline"),
                        S.get(spec, "time.Ts"))
    reqs = []
    if cap is not None:
        reqs.append((float(cap), "transient cap"))
    if tol is not None and dl is not None and Ts and sample_window(Ts, dl)[0] == 0:
        reqs.append((float(tol), "tolerance (deadline at the first sample)"))
    if not reqs:
        return None
    level, what = min(reqs)
    E, x0 = S.get(spec, "initial.E"), S.get(spec, "initial.x0_hat")
    dom = S.get(spec, "deterministic.domain") or {}
    restricted = dom.get("x1") is not None or dom.get("x2") is not None
    extra = {}
    if tgt == 1:
        if E is None:
            return None                                  # the initial velocity uncertainty is not stated: ask for it first
        c = float(x0[1]) if x0 is not None else 0.0
        lo, hi = c - float(E[1]), c + float(E[1])
        rng = dom.get("x2")
        if rng is not None:
            lo, hi = max(lo, float(rng[0])), min(hi, float(rng[1]))
        if hi < lo or not level < 0.5 * (hi - lo):
            return None
        if not restricted:
            h = 0.5 * (hi - lo)
            arg = (f"no estimator can keep the velocity error within the requested {what} {level:.6g} at t = 0: the only measurement then is one "
                   f"position reading, which carries no information about the velocity, and the stated initial velocity bound admits every "
                   f"velocity in [{lo:.6g}, {hi:.6g}] (no travel or speed range restricts the later motion), so for any estimate one admissible "
                   f"velocity lies at least {h:.6g} away")
        else:
            res = viable_initial_velocities(spec, lo, hi)
            if res is None:
                return None
            vlo, vhi, info = res
            h = 0.5 * (vhi - vlo)
            if not level < h:
                return None
            extra = {"construction": info}
            if info.get("route") == "energy":
                arg = (f"no estimator can keep the velocity error within the requested {what} {level:.6g} at t = 0: the only measurement then is one "
                       f"position reading, which carries no information about the velocity.  Two admissible trajectories start at x1 = {info['p']:.6g} "
                       f"with the velocities {vhi:.6g} and {vlo:.6g}, inside the stated initial velocity bound"
                       + (" and speed range" if rng is not None else "") +
                       f"; both hold the input u = {info['u0']:.6g} for the whole window with zero disturbance.  For the stated model the energy "
                       f"x2^2/2 + Phi(x1), with Phi the potential of the model at rest, does not increase (the velocity-dependent part only dissipates), "
                       f"and Phi exceeds the initial energy at both ends of the stated travel range [{dom['x1'][0]:.6g}, {dom['x1'][1]:.6g}] "
                       f"(at least {info['barrier']:.6g}), so both stay inside it"
                       + (f" with speeds below {info['speed_bound']:.6g}" if rng is not None else "") +
                       f".  For any estimate one of them lies at least {h:.6g} away")
                return dict({"argument": arg, "class": "all causal estimators using the same measurements", "bound": h, "level": level,
                             "sample_time": 0.0, "kind": "initial_sample"}, **extra)
            arg = (f"no estimator can keep the velocity error within the requested {what} {level:.6g} at t = 0: the only measurement then is one "
                   f"position reading, which carries no information about the velocity.  Two admissible trajectories start at x1 = {info['p']:.6g} "
                   f"with the velocities {vhi:.6g} and {vlo:.6g}, inside the stated initial velocity bound"
                   + (" and speed range" if rng is not None else "") +
                   f"; both hold the input u = {info['u0']:.6g} during the first sample interval, "
                   + ("and keep it" if all(x in (None, info["u0"]) for x in (info["u_up"], info["u_dn"])) else
                      "then each holds its own constant input inside the stated input range ("
                      + ", ".join(f"{x:.6g}" for x in (info["u_up"], info["u_dn"]) if x is not None) + ")")
                   + f"; their disturbances keep each within {max(info['r_up'], info['r_dn']):.6g} of the start (guaranteed decelerations "
                   + ", ".join(f"{a:.6g}" for a in (info["A_up"], info["A_dn"]) if a is not None) +
                   " on the boxes around their paths) and, where it stops inside the window, hold it at rest"
                   + (f", so both stay inside the stated travel range [{dom['x1'][0]:.6g}, {dom['x1'][1]:.6g}]" if dom.get("x1") is not None else "")
                   + f".  For any estimate one of them lies at least {h:.6g} away")
    elif tgt == 0:
        if restricted:
            return None                                  # the continuation inside stated ranges is not constructed for the position
        V = S.get(spec, "deterministic.V")
        if V is None:
            return None
        lo, hi = -math.inf, math.inf
        pol = S.get(spec, "initial.policy") or ("given" if x0 is not None else "first_measurement")
        if pol == "given" and E is not None and x0 is not None:
            lo, hi = float(x0[0]) - float(E[0]), float(x0[0]) + float(E[0])
        if hi < lo:
            return None
        h = min(float(V), 0.5 * (hi - lo))
        if not level < h:
            return None
        arg = (f"no estimator can keep the position error within the requested {what} {level:.6g} at t = 0: the first reading has an error up "
               f"to {float(V):.6g}, and two admissible positions {2 * h:.6g} apart give the same reading, so for any estimate one lies at least "
               f"{h:.6g} away")
    else:
        return None
    return dict({"argument": arg, "class": "all causal estimators using the same measurements", "bound": h, "level": level, "sample_time": 0.0,
                 "kind": "initial_sample"}, **extra)


# ---------------------------------------------------------------------------------------------------------------------
# inner approximation of the admissible initial velocities under stated travel and speed ranges
# ---------------------------------------------------------------------------------------------------------------------
def viable_initial_velocities(spec: dict, lo: float, hi: float):
    """[v_b, v_a] within [lo, hi] such that trajectories starting at one position with v_a and with v_b exist inside every
    stated range for the whole window (construction in the module docstring), or None if no such pair is found."""
    so = S.get(spec, "plant.second_order") or {}
    dom = S.get(spec, "deterministic.domain") or {}
    key = json.dumps({"so": so, "dom": dom, "D": S.get(spec, "deterministic.D"), "db": S.get(spec, "deterministic.d_bound"),
                      "x0": S.get(spec, "initial.x0_hat"), "E": S.get(spec, "initial.E"), "pol": S.get(spec, "initial.policy"),
                      "t_end": S.get(spec, "time.t_end"), "Ts": S.get(spec, "time.Ts"), "lo": lo, "hi": hi}, sort_keys=True, default=str)
    return _viable_cached(key)


@functools.lru_cache(maxsize=4096)
def _viable_cached(key: str):
    k = json.loads(key)
    try:
        return _viable(k["so"], k["dom"], k["D"], k["db"], k["x0"], k["E"], k["pol"], k["t_end"], k["Ts"], k["lo"], k["hi"])
    except Exception:  # noqa  (a model the bounds cannot handle: no claim)
        return None


def _viable(so, dom, D, db, x0, E, pol, t_end, Ts, lo, hi):
    import sympy as sp
    from . import bounds
    if Ts is None:
        return None
    params = dict(so.get("params") or {})
    pr = {kk: tuple(v) for kk, v in (so.get("param_ranges") or {}).items() if v is not None}
    pnames = sorted(set(params) | set(pr))
    names = ["x1", "x2", "u", "y"] + pnames
    X1, X2, U, Y = (sp.Symbol(n, real=True) for n in ("x1", "x2", "u", "y"))
    # one admissible plant: the nominal parameter values where they lie in their stated ranges, else the middle of the range
    th = {}
    for n in pnames:
        if n in pr:
            v = params.get(n, 0.5 * (pr[n][0] + pr[n][1]))
            th[n] = v if pr[n][0] <= v <= pr[n][1] else 0.5 * (pr[n][0] + pr[n][1])
        else:
            th[n] = params[n]
    true_sub = {sp.Symbol(n, real=True): sp.Float(v) for n, v in th.items()}
    nom_sub = {sp.Symbol(n, real=True): sp.Float(v) for n, v in params.items()}
    # stated classes of the acceleration, (expression in x1, x2, u; bound on its free part), as in free_disturbance_bound:
    #   true model f with additive disturbance |d| <= d_bound; a true model without a stated additive bound restricts
    #   nothing when a total bound D is supplied (D then bounds the free part), and counts with bound 0 (the least
    #   favourable completion) when neither bound is stated;
    #   nominal model: |x2' - f0(y_k, u_k)| <= D (D = 0 when it is not stated and no true model is given)
    classes = []
    if so.get("f") and (db is not None or D is None):
        classes.append((bounds.parse(so["f"], names).subs(true_sub), float(db) if db is not None else 0.0))
    if D is not None or not so.get("f"):
        f0 = bounds.parse(str(so.get("f0") if so.get("f0") is not None else "0"), names).subs(nom_sub).subs(Y, X1)
        if X2 in f0.free_symbols:
            return None
        classes.append((f0, float(D) if D is not None else 0.0))
    x1r, ur = dom.get("x1"), dom.get("u")
    L, H = (float(x1r[0]), float(x1r[1])) if x1r is not None else (-math.inf, math.inf)
    pl, ph = L, H
    if (pol or ("given" if x0 is not None else "first_measurement")) == "given" and x0 is not None and E is not None:
        pl, ph = max(pl, float(x0[0]) - float(E[0])), min(ph, float(x0[0]) + float(E[0]))
    if ph < pl:
        return None
    p = 0.5 * (pl + ph) if (math.isfinite(pl) and math.isfinite(ph)) else min(max(float(x0[0]) if x0 is not None else 0.0, pl), ph)
    # the input held by both trajectories during the first sample interval: an equilibrium input at the start, zero, or the
    # middle of the stated range; afterwards each trajectory uses its own input inside the stated range
    cands = []
    for e, _b in classes:
        er = e.subs({X1: sp.Float(p), X2: 0})
        if U in er.free_symbols:
            try:
                cands += [float(r) for r in sp.solve(sp.Eq(er, 0), U) if r.is_real]
            except Exception:  # noqa
                pass
    cands.append(0.0)
    own = []
    if ur is not None:
        ulo, uhi = float(ur[0]), float(ur[1])
        cands.append(0.5 * (ulo + uhi))
        cands = [c for c in cands if ulo <= c <= uhi]
        own = sorted({ulo, uhi, *[ulo + (uhi - ulo) * j / 8 for j in range(1, 8)]})
    first, seen = [], set()
    for c in cands:
        if round(c, 12) not in seen:
            seen.add(round(c, 12)); first.append(c)
    if not first:
        return None
    best = None
    # the braking construction only bounds the speed from above (it decelerates with at least the guaranteed authority),
    # so every velocity between zero and the start may occur on its path, and it rests at zero: a stated speed range must
    # contain zero on the side of the motion (review case B of revision 2b: a carriage that must keep moving forward)
    x2r = dom.get("x2")
    up_ok = x2r is None or float(x2r[0]) <= 0.0
    dn_ok = x2r is None or float(x2r[1]) >= 0.0
    for u0 in first[:3]:
        w_up = _largest(classes, p, H - p, max(hi, 0.0), t_end, Ts, +1, u0, own) if hi > 0 and up_ok else (0.0, 0.0, None, None)
        w_dn = _largest(classes, p, p - L, max(-lo, 0.0), t_end, Ts, -1, u0, own) if lo < 0 and dn_ok else (0.0, 0.0, None, None)
        va, vb = min(hi, w_up[0]), max(lo, -w_dn[0])
        if va < vb:
            continue
        h = 0.5 * (va - vb)
        if best is None or h > best[0]:
            best = (h, vb, va, {"p": p, "u0": u0, "r_up": w_up[1], "A_up": w_up[2], "u_up": w_up[3], "r_dn": w_dn[1], "A_dn": w_dn[2],
                                "u_dn": w_dn[3]})
        if h >= 0.5 * (hi - lo) * (1 - 1e-12):
            break
    # second construction: hold one input for the whole window with zero disturbance and let the energy bound keep the
    # motion inside the ranges (a restoring force with non-negative damping, e.g. a spring stage)
    has_true = bool(so.get("f")) and (db is not None or D is None)          # then classes[0] is the true model
    if has_true and math.isfinite(L) and math.isfinite(H):
        nom = classes[1] if len(classes) == 2 else None
        w, uc, info_e = _energy_speed(classes[0][0], nom, p, L, H, dom.get("x2"), ur, max(abs(lo), abs(hi)))
        if w > 0:
            va, vb = min(hi, w), max(lo, -w)
            if va >= vb and (best is None or 0.5 * (va - vb) > best[0]):
                best = (0.5 * (va - vb), vb, va, dict(info_e, p=p, u0=uc, route="energy"))
    if best is None:
        return None
    return best[1], best[2], best[3]


def _energy_speed(f, nom, p, L, H, x2r, ur, vcap):
    """Largest w <= vcap such that from x1 = p with |x2(0)| <= w the true model with one constant input u_c and zero
    disturbance stays inside the travel range [L, H] (and the speed range) for all time, or (0, None, {}).

    With Phi(x1) = -int_p^x1 f(s, 0, u_c) ds and E = x2^2 / 2 + Phi(x1):  dE/dt = x2 (f(x1, x2, u_c) - f(x1, 0, u_c)) =
    x2^2 g(x1, x2) with g = (f(x1, x2, u_c) - f(x1, 0, u_c)) / x2, so E does not increase where g <= 0.  A trajectory
    with E(0) = w^2 / 2 < min(Phi(L), Phi(H)) cannot reach either end of the travel (there E >= Phi > E(0)), and its
    speed stays below s = sqrt(w^2 - 2 min Phi).  All bounds are outward-rounded interval bounds; the zero disturbance is
    admissible in the class of the true model, and with a nominal class as well |f - f0(y_k, u_c)| <= D is checked."""
    import sympy as sp
    X1, X2, U, Y = (sp.Symbol(n, real=True) for n in ("x1", "x2", "u", "y"))
    Ssym = sp.Symbol("s_int", real=True)
    cands = []
    try:
        cands += [float(r) for r in sp.solve(sp.Eq(f.subs({X1: sp.Float(p), X2: 0}), 0), U) if r.is_real]
    except Exception:  # noqa
        pass
    cands.append(0.0)
    if ur is not None:
        cands.append(0.5 * (float(ur[0]) + float(ur[1])))
        cands = [c for c in cands if float(ur[0]) <= c <= float(ur[1])]
    best = (0.0, None, {})
    seen = set()
    for uc in cands:
        if round(uc, 12) in seen:
            continue
        seen.add(round(uc, 12))
        fz = f.subs(U, sp.Float(uc))
        try:
            g = sp.cancel(sp.together((fz - fz.subs(X2, 0)) / X2))
            if sp.fraction(g)[1].has(X2):
                continue
            phi = -sp.integrate(fz.subs(X2, 0).subs(X1, Ssym), (Ssym, sp.Float(p), X1))
            if phi.has(sp.Integral):
                continue
            barrier = min(-_sup(-phi, {"x1": (L, L)}), -_sup(-phi, {"x1": (H, H)}))
            phimin = min(0.0, -_sup(-phi, {"x1": (L, H)}, budget=32))
        except Exception:  # noqa
            continue
        if not barrier > 0:
            continue
        w = min(vcap, math.sqrt(2.0 * barrier) * (1 - 1e-9))
        if x2r is not None:
            slim = min(-float(x2r[0]), float(x2r[1]))
            if slim <= 0:
                continue
            w = min(w, math.sqrt(max(0.0, slim * slim + 2.0 * phimin)) * (1 - 1e-9))
        if w <= best[0]:
            continue
        sp_max = math.sqrt(w * w - 2.0 * phimin)
        try:
            if _sup(g, {"x1": (L, H), "x2": (-sp_max, sp_max)}, budget=32) > 0:
                continue
            if nom is not None:
                f0u = nom[0].subs(U, sp.Float(uc)).subs(X1, Y)
                mis = fz - f0u
                bx = {"x1": (L, H), "x2": (-sp_max, sp_max), "y": (L, H)}
                if max(_sup(mis, bx, budget=32), _sup(-mis, bx, budget=32)) > nom[1]:
                    continue
        except Exception:  # noqa
            continue
        best = (w, uc, {"barrier": barrier, "phi_min": phimin, "speed_bound": sp_max})
    return best


def _sup(e, box, budget=8):
    from . import bounds
    fr = {str(x) for x in e.free_symbols}
    return bounds.sup_value(e, {kk: v for kk, v in box.items() if kk in fr}, budget=budget) if fr else float(e)


def _authority(classes, box, sgn, u):
    """Guaranteed acceleration against the motion (A) and guaranteed ability not to reverse (A_back) with input u on the box,
    for every stated class at once; A < 0 means the plant may be pushed outward by up to |A|."""
    import sympy as sp
    U = sp.Symbol("u", real=True)
    A = A_back = math.inf
    for e, b in classes:
        eu = e.subs(U, sp.Float(u))
        A = min(A, b - _sup(sgn * eu, box))
        A_back = min(A_back, b - _sup(-sgn * eu, box))
    if len(classes) == 2:
        dd = (classes[0][0] - classes[1][0]).subs(U, sp.Float(u))
        if max(_sup(dd, box), _sup(-dd, box)) > classes[0][1] + classes[1][1]:
            return -math.inf, -math.inf                 # the stated classes cannot hold together here
    return A, A_back


def _can_rest(classes, box, inputs):
    """True if, at rest anywhere in the box, some input of `inputs` (one constant input) gives zero acceleration in every
    stated class with a disturbance inside its bound."""
    import sympy as sp
    U, X2 = sp.Symbol("u", real=True), sp.Symbol("x2", real=True)
    rest = dict(box, x2=(0.0, 0.0))
    for u in inputs:
        if all(max(_sup(e.subs({U: sp.Float(u), X2: 0}), rest), _sup(-e.subs({U: sp.Float(u), X2: 0}), rest)) <= b for e, b in classes):
            return True
    return False


def _excursion(v, A1, A1b, A2, A2b, Ts, T):
    """(largest displacement, largest speed, stops inside the window, stops during the first interval) of the braking
    trajectory, or None if the construction does not apply.  Phase 1, [0, Ts): common input, deceleration at least A1
    (A1 <= 0: pushed outward by at most |A1|, which needs A1b >= 0).  Phase 2: own input, deceleration at least A2."""
    end1 = Ts if T is None else min(Ts, T)
    if A1 > 0 and v <= A1 * end1:
        return v * v / (2 * A1), v, True, True
    if A1 <= 0 and A1b < 0:
        return None
    x1 = v * end1 - 0.5 * A1 * end1 * end1
    v1 = v - A1 * end1
    vpk = max(v, v1)
    if T is not None and T <= Ts:
        return x1, vpk, False, False
    rest = None if T is None else T - Ts
    if A2 > 0:
        if rest is None or v1 <= A2 * rest:
            return x1 + v1 * v1 / (2 * A2), vpk, True, False
        return x1 + v1 * rest - 0.5 * A2 * rest * rest, vpk, False, False
    if rest is None or A2b < 0:
        return None
    return x1 + v1 * rest - 0.5 * A2 * rest * rest, v1 - A2 * rest, False, False


def _largest(classes, p, room, vmax, t_end, Ts, sgn, u0, own):
    """Largest initial speed in direction sgn from position p for which the construction stays within `room` of p and
    below vmax.  Returns (w, r, A2, u_brake): the speed, the radius of the box used, the guaranteed deceleration after
    the first interval and the braking input; w = 0 when no motion is constructed.

    The input is held on each sample interval, so a trajectory that stops between two sample instants is still driven by
    its braking input until the next one: it can rest only if that input admits zero acceleration at the stopping point.
    A braking input that cannot hold the plant at rest is therefore used only when the window ends before the stop."""
    if vmax <= 0 or (math.isfinite(room) and room <= 0):
        return 0.0, 0.0, None, None
    T = None if t_end is None else float(t_end)
    R0 = room if math.isfinite(room) else 1e3 * max(1.0, abs(p))
    radii = [R0 * 4.0 ** (-k) for k in range(0, 14 if math.isfinite(room) else 26)]
    inputs = [u0] + [u for u in own if u != u0]
    best = (0.0, 0.0, None, None)
    for r in radii:
        box = dict(x1=(p, p + r) if sgn > 0 else (p - r, p), x2=(0.0, vmax) if sgn > 0 else (-vmax, 0.0))
        A1, A1b = _authority(classes, box, sgn, u0)
        if A1 == -math.inf:
            continue
        brakes = [(A1, A1b, u0)] + [(*_authority(classes, box, sgn, u), u) for u in inputs[1:]]
        brakes = [bk for bk in brakes if bk[0] != -math.inf]
        any_best = max(brakes, key=lambda bk: bk[0])
        rest_best = next((bk for bk in sorted(brakes, key=lambda bk: -bk[0]) if _can_rest(classes, box, [bk[2]])), None)
        rest_u0 = rest_best is not None and (rest_best[2] == u0 or _can_rest(classes, box, [u0]))

        def use(v):
            """The braking input after the first interval for initial speed v, or None if the construction fails."""
            for bk, rests in ((rest_best, True), (any_best, False)):
                if bk is None:
                    continue
                ex = _excursion(v, A1, A1b, bk[0], bk[1], Ts, T)
                if ex is None:
                    continue
                disp, vpk, stops, in_first = ex
                if disp > r or vpk > vmax:
                    continue
                if stops and in_first:
                    if rest_u0:
                        return bk                      # stops while u0 is held, which holds it at rest
                    continue
                if stops and not rests:
                    continue                           # stops between samples with an input that cannot hold it at rest
                return bk
            return None

        if use(0.0 if T is not None else min(vmax, 1e-12 * max(vmax, 1.0))) is None:
            if use(min(vmax, 1e-15)) is None:
                continue
        lo_v, hi_v = 0.0, vmax
        if use(vmax) is not None:
            lo_v = vmax
        else:
            for _ in range(40):
                mid = 0.5 * (lo_v + hi_v)
                if use(mid) is not None:
                    lo_v = mid
                else:
                    hi_v = mid
        if lo_v > best[0]:
            bk = use(lo_v) or any_best
            best = (lo_v, r, bk[0], bk[2])
        if lo_v >= vmax:
            break
    return best
