"""Coordinator: applicability checks, admissible-completion reasoning, criterion-aware selection, clarification.

Dispositions (exactly one per call):
  CONFIRM                 material facts read from natural language await confirmation (nothing is released)
  INVALID_INPUT           the specification contradicts itself or contains malformed values
  OUT_OF_SCOPE            no method in the register applies to this problem class (with the reason for each method)
  CLARIFY                 a consequential fact is missing; one useful question is returned with what it affects
  RELEASE                 one method (or the better of two eligible methods) with configuration, claim and conditions
  NO_VERIFIED_CANDIDATE   the request is in scope but no candidate passed its sufficient test (limit of the test stated)
  PROVED_IMPOSSIBLE       a separate valid argument shows that no estimator in a stated class can meet the request

Release rule under incomplete information: with unresolved fields restricted to ranges, a release requires ONE method
and ONE configuration whose claim holds for every admissible completion.  For hard bounds this follows from
monotonicity of the exact envelope in each bound (worst corner); for a mean-square-error level it follows from
monotonicity of a fixed gain's error covariance in the noise covariances.  Otherwise the coordinator returns a
question, with two admissible completions that lead to different justified outcomes (a disagreement witness).
Every completion, witness and threshold is checked against the declared ranges and equality constraints.
"""
from __future__ import annotations

import copy
import math
import re

import numpy as np

from ..core import spec as S
from ..core.assess import Assessment, Need, jsonable, fmt_up
from ..core.limits import initial_sample_limit
from ..core import timing as TM
from ..kalman import specialist as K
from ..khalil import specialist as H

VERSION = "1.0.0"
SPECIALISTS = {"kalman_filter": K, "high_gain_observer": H}
ASK_ORDER = ["objective.kind", "target.state", "objective.tolerance", "objective.mse_target", "objective.prob", "time.Ts",
             "deterministic.V", "deterministic.D", "deterministic.d_bound", "deterministic.domain", "deterministic.domain_evidence",
             "initial.E", "initial.x0_hat", "stochastic.R", "stochastic.Q", "stochastic.Q_kind", "initial.P0"]
FLAG_TEXT = {
    "hidden_dynamics": "dynamics between the input and the measured output that are missing from the model (relative degree above two / unmodelled states)",
    "nonlinear_measurement": "a measurement that is a nonlinear function of the state",
    "correlated_noise": "process and measurement noise that are correlated with each other",
    "coloured_noise": "temporally correlated (non-white) noise",
    "irregular_sampling": "irregular or missing samples",
    "unknown_model": "no dynamic model of the plant",
    "multi_output_nonlinear": "a nonlinear plant of order above two or with several outputs",
    "parameter_dependent_output": "an output map that depends on uncertain parameters or contains input feedthrough",
}
PATH_NAMES = {
    "deterministic.V": "the hard bound on the measurement error", "deterministic.D": "the hard bound on the unmodelled acceleration",
    "deterministic.d_bound": "the hard bound on the additive disturbance", "stochastic.R": "the measurement-noise variance per sample",
    "stochastic.Q": "the process-noise covariance", "initial.P0": "the covariance of the initial estimate error",
    "time.Ts": "the sample period", "initial.E": "the hard bounds on the initial estimate error",
}
MONOTONE_BOUNDS = ("deterministic.V", "deterministic.D", "deterministic.d_bound")


def _assess_all(spec, methods=None):
    out = {name: mod.assess(spec) for name, mod in SPECIALISTS.items() if methods is None or name in methods}
    unavailable = set(spec.get("unavailable") or [])
    for a in out.values():
        if a.status == "needs" and any(n.path in unavailable for n in a.needs):
            miss = [n.path for n in a.needs if n.path in unavailable]
            a.status = "fails"
            a.limits.append("required information is unavailable: " + ", ".join(miss))
    return out


def _criterion(spec, a: Assessment):
    obj = S.get(spec, "objective.kind")
    if obj in ("mse_optimal", "mse_bound"):
        return a.metrics.get("mse_post_deadline_max", a.metrics.get("mse_steady"))
    if obj == "hard_bound":
        return a.metrics.get("hard_bound_value")
    if obj == "probability":
        if a.claim and a.claim.get("uncertainty") == "deterministic error envelope":
            return 0.0                                    # probability-one statement
        return 1.0 - a.metrics.get("prob_level", 0.0)
    return None


def _constraints_ok(spec, completion):
    if not S.in_ranges(spec, completion):
        return False
    for c in spec.get("constraints") or []:
        vals = [completion.get(p, S.get(spec, p)) for p in c["paths"]]
        if any(v is None for v in vals):
            continue
        if abs(sum(vals) - c["sum"]) > 1e-12 * max(1.0, abs(c["sum"])):
            return False
    return True


def _complete(spec, completion):
    if not _constraints_ok(spec, completion):
        raise ValueError(f"inadmissible completion {completion}")
    s = copy.deepcopy(spec)
    for p, v in completion.items():
        S.put(s, p, v, status="completion")
    s["ranges"] = {k: r for k, r in (s.get("ranges") or {}).items() if k not in completion}
    s["constraints"] = []
    return s


def _base(spec):
    return {"coordinator_version": VERSION, "spec_hash": S.spec_hash(spec), "disposition": None, "selected": None,
            "qualifying": [], "explanations": [], "harmless": [], "question": None, "all_needs": [], "claim": None,
            "config": None, "alternatives": [], "tradeoff": None, "assessments": {}, "impossibility": None,
            "accepted_choices": [p for p, e in (spec.get("evidence") or {}).items() if e.get("status") == "accepted_choice"]}


DISPLAY = {
    "RELEASE": "Verified design released.",
    "CLARIFY": "One fact is needed before a design can be certified.",
    "CONFIRM": "Please confirm the facts read from the request; nothing is released before that.",
    "NO_VERIFIED_CANDIDATE": "No verified design found.",
    "PROVED_IMPOSSIBLE": "No estimator can meet this requirement.",
    "OUT_OF_SCOPE": "No method in the register applies to this problem.",
    "INVALID_INPUT": "The request contradicts itself or contains an invalid value.",
}


def decide(spec: dict, methods=None, ranking: str = "criterion") -> dict:
    """Decision for one specification version, with the sentence shown to the user (`display`)."""
    d = _decide(spec, methods, ranking)
    d["display"] = DISPLAY.get(d["disposition"], d["disposition"])
    return d


def _decide(spec: dict, methods=None, ranking: str = "criterion") -> dict:
    """Full evaluation of one specification version.  Nothing is cached between calls: every scenario or
    clarification leads to a complete re-evaluation, including whether an omission is still harmless.

    methods / ranking exist for the baseline policies of the evaluation only: `methods` restricts the candidate
    specialists (single-specialist baselines); ranking="priority" selects the first eligible method in the fixed order
    Kalman, HGO instead of comparing them on the requested criterion (deterministic-selector baseline)."""
    d = _base(spec)
    d["policy"] = {"methods": methods, "ranking": ranking}
    issues = S.validate(spec)
    if issues:
        d["disposition"] = "INVALID_INPUT"
        d["explanations"] = [i["message"] for i in issues]
        d["issues"] = issues
        return d
    pend = S.unconfirmed(spec)
    if pend:
        d["disposition"] = "CONFIRM"
        d["pending_confirmation"] = pend
        d["explanations"].append("material facts were read from natural language and must be confirmed before a design is released")
        return d
    kind = S.get(spec, "plant.kind")
    flags = S.get(spec, "plant.flags") or []
    if kind in (None, "other") or flags:
        d["disposition"] = "OUT_OF_SCOPE"
        if kind in (None, "other"):
            d["explanations"].append(S.get(spec, "plant.out_of_scope_reason") or "the plant is neither a known linear state-space model nor a second-order chain-form model")
        for f in flags:
            d["explanations"].append("the request involves " + FLAG_TEXT.get(f, f) + ", which no method in the register covers")
        d["unsupported_features"] = list(flags)
        return d
    # unresolved fields with admissible ranges -> completion reasoning
    ranges = {p: r for p, r in (spec.get("ranges") or {}).items() if r is not None and S.get(spec, p) is None}
    if "time.Ts" in ranges:
        r = ranges["time.Ts"]
        if r[0] == r[1]:
            spec = _complete(spec, {"time.Ts": r[0]}); ranges.pop("time.Ts")
        elif "time.Ts" in set(spec.get("unavailable") or []):
            # the engineer cannot give the exact period: nothing is verified for every admissible period (review C-13)
            d["disposition"] = "NO_VERIFIED_CANDIDATE"
            d["explanations"].append(f"the sample period is known only as the range {r[0]} to {r[1]} s and the exact value is not available; the discrete "
                                     "model, the gains and every certificate depend on the exact period, so no design is verified")
            return d
        else:
            d["disposition"] = "CLARIFY"
            q = Need("time.Ts", "the discrete model, the noise conversion, the observer gains and every certificate are computed for one exact period",
                     f"What is the exact sample/update period? (stated range: {r[0]} to {r[1]} s)",
                     witness={"completion_a": {"time.Ts": r[0]}, "completion_b": {"time.Ts": r[1]},
                              "difference": "the two periods give different discrete models and gains"})
            d["question"] = q.to_dict(); d["all_needs"] = [q.to_dict()]
            return d
    if ranges:
        return _decide_with_ranges(spec, ranges, d, methods, ranking)
    return _decide_complete(spec, d, methods, ranking)


def _decide_complete(spec, d, methods=None, ranking="criterion"):
    assessments = _assess_all(spec, methods)
    d["assessments"] = {k: a.to_dict() for k, a in assessments.items()}
    qual = [a for a in assessments.values() if a.status == "qualifies"]
    for a in assessments.values():
        for h in a.harmless:
            if h not in d["harmless"]:
                d["harmless"].append(dict(h, method=a.method))
    if qual:
        vals = [(_criterion(spec, a), a) for a in qual]
        if ranking == "priority":
            vals.sort(key=lambda t: list(SPECIALISTS).index(t[1].method))
        else:
            vals.sort(key=lambda t: (math.inf if t[0] is None else t[0]))
        best = vals[0][1]
        d["disposition"] = "RELEASE"
        d["selected"] = best.method
        d["qualifying"] = [a.method for _, a in vals]
        d["claim"] = best.claim
        d["config"] = SPECIALISTS[best.method].export_config(best.config)
        d["limits"] = best.limits
        d["checks"] = best.checks
        obj = S.get(spec, "objective.kind")
        if len(vals) > 1:
            other = vals[1][1]
            v0, v1 = vals[0][0], vals[1][0]
            d["alternatives"] = [{"method": other.method, "claim": other.claim, "criterion_value": v1,
                                  "config": SPECIALISTS[other.method].export_config(other.config)}]
            close = v0 is not None and v1 is not None and abs(v1 - v0) <= 0.01 * max(abs(v1), abs(v0), 1e-300)
            d["tradeoff"] = {"criterion": obj, "values": {best.method: v0, other.method: v1},
                             "ranking_supported": not close,
                             "note": ("both methods meet the request; the values of the requested criterion differ by less than 1 %, so the ranking "
                                      "is not meaningful and both designs are returned" if close else
                                      "both methods meet the request; the selected one has the better value of the requested criterion, "
                                      "evaluated on the same information")}
            d["explanations"].append(d["tradeoff"]["note"])
        else:
            others = [a for a in assessments.values() if a is not best]
            for o in others:
                why = "; ".join(o.reasons + o.limits) or ("needs " + ", ".join(n.path for n in o.needs))
                d["explanations"].append(f"{o.method} was not selected: {why}")
        d["explanations"].insert(0, f"{best.method} selected for objective '{obj}': {best.claim['statement']}")
        return d
    needing = [a for a in assessments.values() if a.status == "needs"]
    universal_imp = [a.impossibility for a in assessments.values() if a.status == "fails" and a.impossibility
                     and a.impossibility.get("class", "").startswith("all causal")]
    first = initial_sample_limit(spec)
    if first is not None:
        # a requirement at t = 0 below what one position reading allows: no estimator meets it, whatever else is missing
        d["disposition"] = "PROVED_IMPOSSIBLE"
        d["impossibility"] = first
        d["explanations"].append(first["argument"])
        for a in assessments.values():
            d["explanations"] += [f"{a.method}: {l}" for l in a.limits]
        return d
    if needing and not universal_imp:
        allneeds, count = {}, {}
        for a in needing:
            for n in a.needs:
                count[n.path] = count.get(n.path, 0) + 1
                allneeds.setdefault(n.path, n)
        # ask first for what every still-possible candidate needs, in a fixed order of dependence
        order = sorted(allneeds, key=lambda p: (ASK_ORDER.index(p) if p in ASK_ORDER else 99, -count[p]))
        q = allneeds[order[0]]
        d["disposition"] = "CLARIFY"
        d["question"] = dict(q.to_dict(), needed_by=[a.method for a in needing if any(n.path == q.path for n in a.needs)])
        d["all_needs"] = [allneeds[p].to_dict() for p in order]
        d["explanations"].append(f"missing fact '{q.path}': {q.why}")
        return d
    failing = [a for a in assessments.values() if a.status == "fails"]
    if failing:
        imps = [a.impossibility for a in failing if a.impossibility]
        if imps:
            d["disposition"] = "PROVED_IMPOSSIBLE"
            d["impossibility"] = imps[0]
            d["explanations"].append(imps[0]["argument"])
        else:
            d["disposition"] = "NO_VERIFIED_CANDIDATE"
        for a in failing:
            d["explanations"] += [f"{a.method}: {l}" for l in a.limits]
        for a in assessments.values():
            if a.status == "not_applicable":
                d["explanations"] += [f"{a.method} not applicable: {r}" for r in a.reasons]
        return d
    # the plant class is in scope, but no method in the register offers the requested kind of claim for it
    d["disposition"] = "NO_VERIFIED_CANDIDATE"
    if S.get(spec, "time.irregular"):
        d["explanations"].append("the request states irregular sampling; every method in the register computes its gains and its certificate "
                                 "for samples at one fixed period, so no fixed-period guarantee is given")
    else:
        d["explanations"].append("the plant is within the supported classes, but no method in the register provides the requested kind of claim for it")
    for a in assessments.values():
        d["explanations"] += [f"{a.method} not applicable: {r}" for r in a.reasons]
    return d


# fields that enter the claim of each method (per class of objective); a range on any other field cannot change it
USED_FIELDS = {("kalman_filter", "stat"): {"stochastic.R", "stochastic.Q", "initial.P0", "time.Ts"},
               ("kalman_filter", "hard"): {"deterministic.V", "deterministic.D", "initial.E", "stochastic.R", "stochastic.Q", "time.Ts"},
               ("high_gain_observer", "hard"): {"deterministic.V", "deterministic.D", "deterministic.d_bound", "initial.E", "time.Ts"},
               ("high_gain_observer", "stat"): {"stochastic.R", "stochastic.Q", "initial.P0", "time.Ts"}}


def _obj_class(method, obj):
    if obj in ("mse_optimal", "mse_bound"):
        return "stat"
    if obj == "probability":
        return "stat" if method == "kalman_filter" else "hard"
    return "hard"


def _vertices(ranges, cons):
    """Vertices of the admissible set {declared ranges} intersected with the equality constraints p1 + p2 = sum.
    Returns a list of completions, or None if the set is empty."""
    import itertools
    used = [c for c in cons if all(q in ranges for q in c["paths"])]
    if len({q for c in used for q in c["paths"]}) < 2 * len(used):
        return _vertices_general(ranges, used)          # a field appears in more than one constraint (review A, recheck m3)
    groups, done = [], set()
    for c in cons:
        p1, p2 = c["paths"]
        if p1 in ranges and p2 in ranges:
            (lo1, hi1), (lo2, hi2), tot = ranges[p1], ranges[p2], c["sum"]
            a, b = max(lo1, tot - hi2), min(hi1, tot - lo2)
            if a > b + 1e-12 * max(1.0, abs(tot)):
                return None
            pts = [a] if b - a <= 1e-15 * max(1.0, abs(a)) else [a, b]
            groups.append([{p1: x, p2: min(max(tot - x, lo2), hi2)} for x in pts])
            done |= {p1, p2}
    for pth, (lo, hi) in ranges.items():
        if pth not in done:
            groups.append([{pth: lo}, {pth: hi}] if hi > lo else [{pth: lo}])
    out = []
    for combo in itertools.product(*groups):
        d = {}
        for x in combo:
            d.update(x)
        out.append(d)
    return out


def _vertices_general(ranges, cons):
    """Vertices of {x in the box of declared ranges : x_p1 + x_p2 = sum for every constraint} when constraints share a
    field: every basic solution (r = rank of the constraint rows; n - r fields at a bound, the other r solved for) that
    lies in the box.  Returns None if the set is empty."""
    import itertools
    keys = list(ranges)
    n, m = len(keys), len(cons)
    Aeq, beq = np.zeros((m, n)), np.zeros(m)
    for i, c in enumerate(cons):
        for q in c["paths"]:
            Aeq[i, keys.index(q)] += 1.0
        beq[i] = c["sum"]
    lo = np.array([ranges[k][0] for k in keys], float)
    hi = np.array([ranges[k][1] for k in keys], float)
    r = int(np.linalg.matrix_rank(Aeq))
    scale = max(1.0, float(np.max(np.abs(np.concatenate([lo, hi, beq])))))
    pts = []
    for free in itertools.combinations(range(n), r):
        Af = Aeq[:, free]
        if np.linalg.matrix_rank(Af) < r:
            continue
        fixed = [j for j in range(n) if j not in free]
        for bnd in itertools.product(*[(lo[j], hi[j]) if hi[j] > lo[j] else (lo[j],) for j in fixed]):
            x = np.empty(n)
            x[fixed] = bnd
            rhs = beq - Aeq[:, fixed] @ x[fixed]
            sol = np.linalg.lstsq(Af, rhs, rcond=None)[0]
            if np.max(np.abs(Af @ sol - rhs), initial=0.0) > 1e-12 * scale:
                continue                                  # the bound choice is inconsistent with the constraints
            x[list(free)] = sol
            if np.all(x >= lo - 1e-12 * scale) and np.all(x <= hi + 1e-12 * scale):
                x = np.clip(x, lo, hi)
                if not any(np.max(np.abs(x - p)) <= 1e-12 * scale for p in pts):
                    pts.append(x)
    if not pts:
        return None
    return [{k: float(v) for k, v in zip(keys, p)} for p in pts]


def _verify_release(dec, spec, comp, info=None):
    """Evaluate the RELEASED configuration (fixed gain) under another admissible completion.
    Returns (passes, value) or (None, None) when no verification for this case is in the register.  If `info` (a dict)
    is given, the other certified numbers of a hard-bound check are stored in it (peak bound, deadline sample, period)."""
    from ..core import peak
    obj = S.get(spec, "objective.kind")
    cfg, meth = dec.get("config") or {}, dec.get("selected")
    try:
        s_c = _complete(spec, comp)
        dl = S.get(s_c, "objective.deadline")
        if obj in ("hard_bound", "probability") and (meth == H.METHOD or obj == "hard_bound"):
            tol, cap = S.get(s_c, "objective.tolerance"), S.get(s_c, "objective.peak_cap")
            if meth == H.METHOD:
                base = H.design(s_c)
                Ts, D, V, tgt, M = base["Ts"], base["D"], base["V"], base["target"], cfg["M"]
                E = [0.0 if e is None else e for e in base["E_eff"]]
                init = H.cert_init(base)
            else:
                hd = K.hard_disturbance(s_c)
                Ts, V, D, tgt = cfg["Ts"], S.get(s_c, "deterministic.V"), (None if hd is None else hd[0]), cfg["target"]
                Ein = S.get(s_c, "initial.E")
                E = [V, float(Ein[1])] if Ein is not None else [0.0, 0.0]
                M = np.asarray(cfg["M_ss"], float).reshape(2)
                init = "first_measurement"
            k_dead, n_end = TM.sample_window(Ts, dl, S.get(s_c, "time.t_end"))
            if dl is None:
                n_end = None
            cert = peak.certify_lti(Ts, M, np.eye(2)[tgt], E, D, V, k_deadline=k_dead, init=init, k_end=n_end)
            if not cert["stable"]:
                return False, None
            bound = cert["post_deadline_upper"] if dl is not None else cert["ultimate_upper"]
            if info is not None:
                info.update(peak=cert["peak_upper"], k_dead=k_dead, n_end=n_end, Ts=Ts, deadline=dl, cap=cap)
            return bool(bound <= tol and (cap is None or cert["peak_upper"] <= cap)), bound
        if obj == "mse_bound":
            target, tgt = S.get(s_c, "objective.mse_target"), S.get(s_c, "target.state")
            try:
                kc = K.design(s_c)                   # the completion's own model matrices
            except Exception:  # noqa
                kc = K.design(dict(s_c, objective=dict(s_c["objective"], horizon="steady_state")))
            A, C, Qd, R, Ts = kc["A"], kc["C"], kc["Qd"], kc["R"], kc["Ts"]
            te = S.get(s_c, "time.t_end")
            k_dead, n_end = TM.sample_window(Ts, dl, te)
            asym = (S.get(s_c, "objective.horizon") or "finite") == "steady_state" and dl is None
            if meth == H.METHOD or cfg.get("mode") == "steady_state":
                Mg = (np.asarray(cfg["M"], float) if meth == H.METHOD else np.asarray(cfg["M_ss"], float)).reshape(1, A.shape[0], C.shape[0])
                if asym:
                    Pf = K.steady_cov_of_constant_gain(A, C, Qd, R, Mg[0])
                    v = math.inf if Pf is None else float(Pf[tgt, tgt])
                else:
                    v = float(K.window_variance(A, C, Qd, R, S.get(s_c, "initial.P0"), tgt, k_dead, n_end, gains=Mg)[0][0])
                return bool(v <= target), v
            if n_end is not None:                      # time-varying gain sequence designed for another completion
                arr = {k: (np.asarray(v, float) if k in ("A", "B", "C", "Qd", "R", "P0", "x0_hat", "M_ss", "P_ss", "Pf_ss") and v is not None else v)
                       for k, v in cfg.items()}
                Ms, _, _ = K.gains(arr, n_end)
                Pp, Pf = K.LK.fixed_gain_cov(A, C, Qd, R, np.atleast_2d(np.asarray(S.get(s_c, "initial.P0"), float)), Ms)
                v = max(float(P[tgt, tgt]) for P in Pf[k_dead:n_end + 1])
                return bool(v <= target), v
        return None, None
    except Exception:  # noqa
        return None, None


def _decide_with_ranges(spec, ranges, d, methods=None, ranking="criterion"):
    """Unresolved fields with admissible ranges."""
    obj = S.get(spec, "objective.kind")
    mono = [p for p in ranges if p in MONOTONE_BOUNDS]
    stat = [p for p in ranges if p in ("stochastic.R", "initial.P0", "stochastic.Q")]
    other = [p for p in ranges if p not in mono and p not in stat]
    if other:
        p = other[0]
        if p in set(spec.get("unavailable") or []):
            d["disposition"] = "NO_VERIFIED_CANDIDATE"
            d["explanations"].append(f"{p} is known only as the range {ranges[p]} and no result covers a range for this field")
            return d
        d["disposition"] = "CLARIFY"
        q = Need(p, "no monotonicity or invariance result covers a range for this field", f"What is the value of {p}? (stated range {ranges[p]})")
        d["question"] = q.to_dict(); d["all_needs"] = [q.to_dict()]
        return d
    cons = spec.get("constraints") or []

    def corner(which):
        comp = {p: ranges[p][1 if which == "hi" else 0] for p in ranges}
        return comp

    verts = _vertices(ranges, cons) if cons else None
    if cons and verts is None:
        d["disposition"] = "INVALID_INPUT"
        d["explanations"].append("the declared ranges and equality constraints admit no completion")
        return d
    lo_c, hi_c = (verts[0], verts[-1]) if cons else (corner("lo"), corner("hi"))
    for comp in (lo_c, hi_c):
        if not _constraints_ok(spec, comp):
            d["disposition"] = "INVALID_INPUT"
            d["explanations"].append(f"the declared ranges and equality constraints admit no completion at {comp}")
            return d
    s_lo, s_hi = _complete(spec, lo_c), _complete(spec, hi_c)
    d_lo, d_hi = _decide_complete(s_lo, _base(s_lo), methods, ranking), _decide_complete(s_hi, _base(s_hi), methods, ranking)
    d["completion_corners"] = {"lo": lo_c, "hi": hi_c, "disposition_lo": d_lo["disposition"], "disposition_hi": d_hi["disposition"]}
    hard = obj in ("hard_bound", "probability") and not stat
    msebound = obj == "mse_bound" and not mono
    if cons and (hard or msebound):
        # equality constraints: no dominating corner.  A fixed design's worst case (exact envelope) or mean-square error
        # (fixed-gain covariance) is affine in the constrained quantities, so it is largest at a vertex of the admissible
        # set; a design is released only if it passes at every vertex (review A5, A8).
        mid = {k: float(np.mean([v[k] for v in verts])) for k in verts[0]}
        if any("deterministic.V" in v for v in verts):
            # a derived disturbance bound uses the Lipschitz constant of the nominal model over the x1 range widened by V;
            # widening by the largest admissible V at every completion makes the bound affine in (V, d_bound) and never
            # smaller than the completion's own bound, so the vertex argument below holds (second recheck A)
            vmax_V = max(v["deterministic.V"] for v in verts)
            verts = [dict(v, **{"deterministic.V_widen": vmax_V}) for v in verts]
            mid = dict(mid, **{"deterministic.V_widen": vmax_V})
        for comp in verts + [mid]:
            d_c = _decide_complete(_complete(spec, comp), _base(spec), methods, ranking)
            if d_c["disposition"] != "RELEASE":
                continue
            infos = [{} for _ in verts]
            checks = [_verify_release(d_c, spec, v, infos[i]) for i, v in enumerate(verts)]
            if all(c[0] for c in checks):
                out = dict(d_c)
                out["spec_hash"] = d["spec_hash"]
                out["completion_corners"] = d["completion_corners"]
                out["universal_over_completions"] = {
                    "ranges": ranges, "constraints": cons, "vertices": verts, "design_completion": comp,
                    "values_at_vertices": [c[1] for c in checks],
                    "argument": ("the released design is fixed; its exact worst case is affine in the constrained bounds (a derived disturbance "
                                 "bound uses the Lipschitz constant over the x1 range widened by the largest admissible V, which makes it affine "
                                 "and never smaller than the bound of each completion), and its mean-square error is affine in the noise "
                                 "covariances, so the largest value over the admissible set is attained at a vertex, and it passes at every vertex")}
                out["explanations"] = [out["universal_over_completions"]["argument"]] + out["explanations"]
                if msebound and out.get("claim"):
                    vmax = max(c[1] for c in checks)
                    out["claim"] = dict(out["claim"], type="mse_bound_robust", value=vmax,
                                        statement=f"mean-square error at most {fmt_up(vmax)} for every admissible completion (largest value at the vertices "
                                                  f"of the admissible set) <= {S.get(spec, 'objective.mse_target'):.6g}")
                if hard and out.get("claim"):
                    # the claim refers to the whole admissible set, not to the completion used for the design (recheck A, N2);
                    # the statement is rebuilt from the certified values at the vertices (second recheck A, New-2)
                    vmax = max(c[1] for c in checks if c[1] is not None)
                    cl = dict(out["claim"])
                    inf0 = infos[0]
                    pk = [i.get("peak") for i in infos if i.get("peak") is not None]
                    # the design completion's bounds are replaced by the ranges below; the domain clause of a derived bound is kept
                    conds = []
                    for c in (cl.get("conditions") or []):
                        if c.startswith("|measurement error| <= "):
                            continue
                        if c.startswith("|x2' - nominal model| <= "):
                            c = re.sub(r"^\|x2' - nominal model\| <= [0-9.eE+\-]+", "|x2' - nominal model| <= D of each admissible completion", c)
                        conds.append(c)
                    rng = "; ".join(f"{PATH_NAMES.get(q, q)} in [{ranges[q][0]:.6g}, {ranges[q][1]:.6g}]" for q in ranges)
                    cns = "; ".join(" + ".join(PATH_NAMES.get(q, q) for q in c["paths"]) + f" = {c['sum']:.6g}" for c in cons)
                    conds.append(f"the bounds are known only as ranges ({rng}) with the constraint {cns}; the stated values are the largest "
                                 "certified values of the released design over the admissible set, attained at its vertices")
                    stmt = TM.claim_statement(vmax, max(pk) if (inf0.get("cap") is not None and pk) else None, inf0["Ts"], inf0["k_dead"],
                                              inf0.get("n_end"), inf0.get("deadline"))
                    if out.get("selected") == K.METHOD:
                        stmt += " for the steady-state Kalman gain"
                    stmt += (", for every admissible initial error, measurement error and disturbance and every admissible completion of the "
                             "stated ranges and constraint")
                    if obj == "probability":
                        stmt += " (hence with probability one under any distribution supported on the stated bounds)"
                    cl.update(value=vmax, statement=stmt, conditions=conds)
                    if pk:
                        cl["peak_value"] = max(pk)
                    out["claim"] = cl
                    out["explanations"].insert(1, f"the claim states the largest values over the admissible set; the selection "
                                                  f"statement below refers to the completion used for the design ({comp})")
                return out
    if (hard or msebound) and not cons and d_hi["disposition"] == "RELEASE":
        # the hi-corner design is valid for every admissible completion (monotone envelope / monotone fixed-gain covariance)
        out = dict(d_hi)
        out["spec_hash"] = d["spec_hash"]
        out["completion_corners"] = d["completion_corners"]
        out["universal_over_completions"] = {
            "ranges": ranges,
            "argument": ("the exact error envelope of a fixed gain is nondecreasing in each of the bounds V, D, so the certificate at the upper "
                         "ends holds for every admissible value" if hard else
                         "the error covariance of a fixed gain is affine and nondecreasing in the noise covariances, so its value at the upper "
                         "ends bounds the mean-square error for every admissible value (the gain is optimal only at the upper ends)")}
        out["explanations"] = [out["universal_over_completions"]["argument"]] + out["explanations"]
        if msebound and out.get("claim"):
            out["claim"] = dict(out["claim"], type="mse_bound_robust")
        return out
    same = (d_lo["disposition"] == d_hi["disposition"] == "RELEASE" and d_lo["selected"] == d_hi["selected"]
            and _same_config(d_lo["config"], d_hi["config"]))
    if same and not (hard or msebound):
        # sampled agreement at the corners is NOT a proof of invariance; check interior points and report as a question unless a
        # proof is available (none in the register for these fields)
        pass
    # a release that is the same at both ends of every range, with ranges only on fields the selected method does not use,
    # holds for every completion: the range cannot change its claim (review A14)
    if (d_lo["disposition"] == d_hi["disposition"] == "RELEASE" and d_lo["selected"] == d_hi["selected"]
            and _same_config(d_lo["config"], d_hi["config"])
            and not (set(ranges) & USED_FIELDS.get((d_hi["selected"], _obj_class(d_hi["selected"], obj)), set()))):
        out = dict(d_hi)
        out["spec_hash"] = d["spec_hash"]; out["completion_corners"] = d["completion_corners"]
        out["harmless"] = list(out.get("harmless") or []) + [{"path": pth, "why": f"the claim of {d_hi['selected']} for this objective does not use this value"}
                                                             for pth in ranges]
        return out
    if d_lo["disposition"] in ("NO_VERIFIED_CANDIDATE", "PROVED_IMPOSSIBLE") and (hard or msebound) and not cons:
        out = dict(d_lo)
        out["spec_hash"] = d["spec_hash"]; out["completion_corners"] = d["completion_corners"]
        out["explanations"] = ["even the most favourable admissible completion fails the test"] + out["explanations"]
        return out
    # the outcome depends on the unresolved fact: ask, with an admissible disagreement witness
    path = (mono + stat)[0]
    for cand in ASK_ORDER:
        if cand in ranges:
            path = cand
            break
    witness = {"completion_a": lo_c, "completion_b": hi_c,
               "outcome_a": _outcome(d_lo), "outcome_b": _outcome(d_hi),
               "admissible": bool(_constraints_ok(spec, lo_c) and _constraints_ok(spec, hi_c))}
    if d_lo["disposition"] in ("CLARIFY", "CONFIRM", "INVALID_INPUT", "OUT_OF_SCOPE"):
        out = dict(d_lo); out["spec_hash"] = d["spec_hash"]; out["completion_corners"] = d["completion_corners"]
        return out
    thr = None
    if hard and len(ranges) == 1 and d_lo["disposition"] == "RELEASE":
        lo, hi = ranges[path]
        ok = lambda x: _decide_complete(_complete(spec, {path: x}), _base(spec), methods, ranking)["disposition"] == "RELEASE"
        a, b = lo, hi
        for _ in range(12):
            mid = 0.5 * (a + b)
            if ok(mid):
                a = mid
            else:
                b = mid
        thr = a
    why = {"hard_bound": "the guaranteed error limit", "probability": "the provable probability level",
           "mse_optimal": "the optimal gain, the estimate and the reported covariance",
           "mse_bound": "whether the mean-square-error level can be met"}.get(obj, "the result")
    nm = PATH_NAMES.get(path, path)
    if path in set(spec.get("unavailable") or []):
        # the user cannot narrow the range: no design is verified for every admissible value (review C-13)
        d["disposition"] = "NO_VERIFIED_CANDIDATE"
        d["explanations"].append(f"{nm} is known only as the range {ranges[path][0]} to {ranges[path][1]}; the admissible completions {lo_c} and "
                                 f"{hi_c} lead to different outcomes, so no design is verified for every admissible value")
        return d
    q = Need(path, f"{why} depends on this value: the admissible completions {lo_c} and {hi_c} lead to different justified outcomes",
             (f"Is {nm} at most {thr:.6g}? If yes, a certified design exists for the stated requirement; if it can be larger, no candidate is verified."
              if thr is not None else f"What is {nm}? (stated range {ranges[path][0]} to {ranges[path][1]})"),
             threshold=thr, witness=witness)
    d["disposition"] = "CLARIFY"
    d["question"] = q.to_dict(); d["all_needs"] = [q.to_dict()]
    d["explanations"].append(q.why)
    return d


def _outcome(dec):
    out = {"disposition": dec["disposition"], "selected": dec.get("selected")}
    cfg = dec.get("config") or {}
    if dec["disposition"] == "RELEASE":
        if "M_ss" in cfg:
            out["gain"] = cfg["M_ss"]
        elif "M" in cfg:
            out["gain"] = cfg["M"]
        elif cfg.get("mode") == "time_varying":
            g = K.gains({k: np.asarray(v) if isinstance(v, list) else v for k, v in cfg.items()}, 0)[0][0]
            out["first_gain"] = np.asarray(g).tolist()
        if dec.get("claim"):
            out["claim_value"] = dec["claim"].get("value")
    return jsonable(out)


def _same_config(c1, c2, rtol=1e-12):
    if c1 is None or c2 is None:
        return False
    for k in ("M", "M_ss", "eps", "Qd", "R", "P0"):
        if (k in c1) != (k in c2):
            return False
        if k in c1 and c1[k] is not None and not np.allclose(np.asarray(c1[k], float), np.asarray(c2[k], float), rtol=rtol, atol=0):
            return False
    return True


# ---------------------------------------------------------------------------------------------------------
# session: confirmation bound to the specification version, complete re-evaluation after every change
# ---------------------------------------------------------------------------------------------------------
class Session:
    def __init__(self, spec: dict):
        self.spec = copy.deepcopy(spec)
        self.history = []
        self.confirmed_hash = None

    def readback(self) -> list:
        rows = []
        for p in S.MATERIAL_PATHS:
            v = S.get(self.spec, p)
            ev = (self.spec.get("evidence") or {}).get(p, {})
            if v is None and p not in (self.spec.get("ranges") or {}):
                continue
            rows.append({"path": p, "value": v, "range": (self.spec.get("ranges") or {}).get(p), "status": ev.get("status", "supplied"),
                         "source": ev.get("source")})
        return rows

    def confirm(self, spec_hash: str):
        """Confirm the interpreted facts of exactly this specification version."""
        if spec_hash != S.spec_hash(self.spec):
            raise ValueError("stale confirmation: the specification changed after the readback was produced")
        self.spec = S.confirm(self.spec)
        self.confirmed_hash = S.spec_hash(self.spec)
        self.history.append({"event": "confirm", "spec_hash": self.confirmed_hash})

    def update(self, changes: dict, status: str = "supplied", source: str = "clarification"):
        """Apply answers / scenario changes.  Material changes invalidate an earlier confirmation of interpreted facts
        only for the changed fields (they carry their own status); the decision is always recomputed."""
        for p, v in changes.items():
            S.put(self.spec, p, v, status=status, source=source)
            (self.spec.get("ranges") or {}).pop(p, None)
        self.history.append({"event": "update", "changes": jsonable(changes), "status": status, "spec_hash": S.spec_hash(self.spec)})

    def decide(self) -> dict:
        dec = decide(self.spec)
        self.history.append({"event": "decide", "spec_hash": dec["spec_hash"], "disposition": dec["disposition"],
                             "question": (dec.get("question") or {}).get("path")})
        return dec
