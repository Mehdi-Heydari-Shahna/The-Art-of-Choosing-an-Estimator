"""Common problem specification shared by the coordinator and both specialists.

A specification is a JSON-serialisable nested dict.  Every material leaf can carry an evidence entry
(`evidence[path] = {"status": ..., "source": ..., "conditions": ...}`) with status one of

    supplied          stated explicitly by the user (structured input or confirmed readback)
    interpreted       read from natural language; NOT yet confirmed (blocks a release)
    derived           computed from other fields; `conditions` states when the derivation is valid
    accepted_choice   a design choice the user explicitly accepted
    unresolved        not known; `ranges[path] = [lo, hi]` may restrict the admissible completions

Field meanings (SI units unless a unit string is attached to the plant signal):

plant.kind            "linear" | "second_order" | "other"
plant.linear          {clock: "discrete"|"continuous", A, B, C, Ts_native}        x+ = A x + B u (+w),  y = C x (+v)
plant.second_order    {f, f0, params, param_ranges, output}   x1' = x2, x2' = f(x1,x2,u) + d,  y = x1 + v;
                      f0(y,u) is the nominal acceleration the observer uses (estimate-independent)
plant.flags           list of declared features outside the method register (hidden_dynamics, ...)
target.state          index of the state to estimate
time.Ts               observation / update period [s];  time.input_hold = "zoh";  time.t_end  [s]
initial.x0_hat        initial estimate (list) or None;  initial.policy = "given" | "first_measurement"
initial.P0            initial error covariance (matrix)          -- stochastic information
initial.E             [E1, E2] hard bounds on the initial error  -- deterministic information
stochastic.distribution   "gaussian" | "second_order" | "bounded"
stochastic.Q, Q_kind      process noise: "discrete_cov" (per-step covariance) or "continuous_psd" (spectral density, with G)
stochastic.R              measurement-noise covariance per sample
stochastic.white, cross_cov   temporal whiteness flag and process/measurement cross-covariance (None or zero)
deterministic.V           hard bound on the measurement error
deterministic.D           hard bound on the total disturbance |x2' - f0|   (or derived from f, f0, domain)
deterministic.d_bound     hard bound on the additive disturbance d
deterministic.domain      {"x1": [lo,hi]|None, "x2": [lo,hi]|None, "u": [lo,hi]|None}
deterministic.domain_evidence   {"kind": "global"|"mechanical_limit"|"invariant_set"|"assumed"|"none", "statement": str}
objective.kind            "mse_optimal" | "mse_bound" | "probability" | "hard_bound"
objective.tolerance, deadline, peak_cap, prob, mse_target, horizon ("finite"|"steady_state")
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

SCHEMA = "heros-spec/1"

MATERIAL_PATHS = [
    "time.Ts", "initial.x0_hat", "initial.P0", "initial.E", "initial.policy",
    "stochastic.distribution", "stochastic.Q", "stochastic.Q_kind", "stochastic.R", "stochastic.white",
    "stochastic.cross_cov", "deterministic.V", "deterministic.D", "deterministic.d_bound",
    "deterministic.domain", "deterministic.domain_evidence",
    "objective.kind", "objective.tolerance", "objective.deadline", "objective.peak_cap", "objective.prob",
    "objective.mse_target", "objective.horizon", "target.state",
]


def empty_spec() -> dict:
    return {
        "schema": SCHEMA,
        "plant": {"family": None, "kind": None, "states": [], "inputs": [], "outputs": [], "linear": None,
                  "second_order": None, "flags": []},
        "target": {"state": None},
        "time": {"Ts": None, "input_hold": "zoh", "t_end": None, "irregular": False},
        "initial": {"x0_hat": None, "P0": None, "E": None, "policy": None},
        "stochastic": {"distribution": None, "Q": None, "Q_kind": None, "G": None, "R": None, "white": None,
                       "cross_cov": None},
        "deterministic": {"V": None, "D": None, "d_bound": None, "domain": None, "domain_evidence": None},
        "objective": {"kind": None, "tolerance": None, "deadline": None, "peak_cap": None, "prob": None,
                      "mse_target": None, "horizon": None},
        "evidence": {},
        "ranges": {},
    }


def get(spec: dict, path: str, default=None):
    cur: Any = spec
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def put(spec: dict, path: str, value, status: str | None = None, source: str | None = None, conditions: str | None = None):
    cur = spec
    ks = path.split(".")
    for k in ks[:-1]:
        cur = cur.setdefault(k, {})
    cur[ks[-1]] = value
    if status is not None:
        ev = {"status": status}
        if source is not None:
            ev["source"] = source
        if conditions is not None:
            ev["conditions"] = conditions
        spec.setdefault("evidence", {})[path] = ev
    return spec


def status(spec: dict, path: str) -> str:
    ev = spec.get("evidence", {}).get(path)
    if ev:
        return ev["status"]
    return "unresolved" if get(spec, path) is None else "supplied"


def is_set(spec: dict, path: str) -> bool:
    return get(spec, path) is not None


def canonical(obj) -> str:
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in sorted(o.items())}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, float):
            if math.isinf(o) or math.isnan(o):
                return str(o)
            return float(o)
        if hasattr(o, "tolist"):
            return clean(o.tolist())
        if hasattr(o, "item"):
            return clean(o.item())
        return o
    return json.dumps(clean(obj), sort_keys=True, separators=(",", ":"))


def spec_hash(spec: dict) -> str:
    """Hash of the material content (plant, target, time, initial, uncertainty, objective, ranges, evidence statuses).
    A confirmation is bound to this value and becomes invalid when any material fact changes."""
    body = {k: spec.get(k) for k in ("plant", "target", "time", "initial", "stochastic", "deterministic", "objective", "ranges")}
    body["evidence_status"] = {p: e.get("status") for p, e in sorted(spec.get("evidence", {}).items())}
    return hashlib.sha256(canonical(body).encode()).hexdigest()[:16]


def unconfirmed(spec: dict) -> list:
    """Material paths whose value was interpreted from natural language and not yet confirmed."""
    return sorted(p for p, e in spec.get("evidence", {}).items() if e.get("status") == "interpreted")


def confirm(spec: dict, paths=None) -> dict:
    """Mark interpreted fields as supplied (user confirmation).  Returns a new spec."""
    s = copy.deepcopy(spec)
    for p, e in s.get("evidence", {}).items():
        if e.get("status") == "interpreted" and (paths is None or p in paths):
            e["status"] = "supplied"
            e["confirmed"] = True
    return s


# ---------------------------------------------------------------------------------------------------------
# validation: contradictions and malformed values (distinct from "missing")
# ---------------------------------------------------------------------------------------------------------
def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate(spec: dict) -> list:
    """Return a list of {"code", "path", "message"} for invalid or contradictory input (empty list = valid)."""
    issues = []

    def bad(code, path, msg):
        issues.append({"code": code, "path": path, "message": msg})

    for p in ("time.Ts", "time.t_end", "time.label_resolution", "objective.tolerance", "objective.deadline", "objective.peak_cap",
              "objective.mse_target", "deterministic.V", "deterministic.D", "deterministic.d_bound"):
        v = get(spec, p)
        if v is None:
            continue
        if not _num(v):
            bad("not_a_number", p, f"{p} must be a finite number, got {v!r}")
        elif v < 0 or (v == 0 and p in ("time.Ts", "time.t_end", "time.label_resolution", "objective.tolerance", "objective.peak_cap",
                                        "objective.mse_target")):
            bad("sign", p, f"{p} must be {'positive' if p.startswith(('time', 'objective')) else 'nonnegative'}, got {v}")
    pr = get(spec, "objective.prob")
    if pr is not None and not (_num(pr) and 0 < pr < 1):
        bad("probability", "objective.prob", f"probability level must lie strictly between 0 and 1, got {pr!r}")
    E = get(spec, "initial.E")
    if E is not None and (len(E) != 2 or not all(_num(e) and e >= 0 for e in E)):
        bad("sign", "initial.E", "initial-error bounds must be two nonnegative numbers")
    for p, r in spec.get("ranges", {}).items():
        if r is None:
            continue
        if not (isinstance(r, (list, tuple)) and len(r) == 2 and all(isinstance(x, (int, float)) for x in r)):
            bad("range_format", p, f"range for {p} must be [lo, hi]")
        elif r[0] > r[1]:
            bad("reversed_range", p, f"range for {p} is reversed: [{r[0]}, {r[1]}]")
        elif get(spec, p) is not None and _num(get(spec, p)) and not (r[0] <= get(spec, p) <= r[1]):
            bad("value_outside_range", p, f"{p} = {get(spec, p)} lies outside its declared range [{r[0]}, {r[1]}]")
    # timing
    lin = get(spec, "plant.linear")
    Ts = get(spec, "time.Ts")
    if lin and lin.get("clock") == "discrete":
        tn = lin.get("Ts_native")
        if tn is not None and Ts is not None and _num(Ts) and abs(Ts - tn) > 1e-12 * max(abs(tn), 1.0):
            bad("timing_conflict", "time.Ts",
                f"the discrete model is defined at its native period {tn} s but the request states {Ts} s; "
                "a discrete model cannot be re-sampled without the underlying continuous model")
    dl, te = get(spec, "objective.deadline"), get(spec, "time.t_end")
    if _num(dl) and _num(te) and dl > te:
        bad("timing_conflict", "objective.deadline", f"deadline {dl} s lies after the end of the stated window {te} s")
    elif _num(dl) and _num(te) and _num(Ts) and Ts > 0:
        from .timing import sample_window
        kd, ne = sample_window(Ts, dl, te)
        if kd > ne:
            bad("timing_conflict", "objective.deadline",
                f"no sample instant lies in the requested window from {dl} s to {te} s at the period {Ts} s")
    # A transient cap below the later tolerance is NOT a contradiction: the cap applies at every sample, the tolerance
    # from the deadline on, so the cap is the binding limit after the deadline as well (revision 2, finding 7).
    R = get(spec, "stochastic.R")
    if R is not None:
        try:
            import numpy as np
            Rm = np.atleast_2d(np.asarray(R, float))
            if Rm.shape[0] != Rm.shape[1] or np.linalg.eigvalsh(0.5 * (Rm + Rm.T)).min() < 0:
                bad("covariance", "stochastic.R", "measurement-noise covariance must be symmetric positive semidefinite")
        except Exception:
            bad("covariance", "stochastic.R", "measurement-noise covariance is not a numeric matrix")
    Q = get(spec, "stochastic.Q")
    if Q is not None:
        try:
            import numpy as np
            Qm = np.atleast_2d(np.asarray(Q, float))
            if Qm.shape[0] != Qm.shape[1] or np.linalg.eigvalsh(0.5 * (Qm + Qm.T)).min() < -1e-12 * max(1.0, abs(Qm).max()):
                bad("covariance", "stochastic.Q", "process-noise covariance must be symmetric positive semidefinite")
        except Exception:
            bad("covariance", "stochastic.Q", "process-noise covariance is not a numeric matrix")
    return issues


def in_ranges(spec: dict, completion: dict) -> bool:
    """True iff every completed value lies in its declared admissible range."""
    for p, v in completion.items():
        r = spec.get("ranges", {}).get(p)
        tol = 1e-12 * max(1.0, abs(r[0]), abs(r[1])) if r is not None else 0.0
        if r is not None and not (r[0] - tol <= v <= r[1] + tol):
            return False
    return True


def complete(spec: dict, completion: dict, status_: str = "completion") -> dict:
    """Return a copy with the given values filled in.  Raises if a value violates a declared range
    (every witness, scenario and proposed completion must be admissible)."""
    if not in_ranges(spec, completion):
        raise ValueError(f"inadmissible completion {completion} for ranges {spec.get('ranges')}")
    s = copy.deepcopy(spec)
    for p, v in completion.items():
        put(s, p, v, status=status_)
        s.get("ranges", {}).pop(p, None)
    return s
