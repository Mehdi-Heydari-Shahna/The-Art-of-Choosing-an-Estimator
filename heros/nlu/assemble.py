"""Deterministic assembly of a specification from a reading of the three answers and the attached model.

The reader (rules or a trained model) proposes a role for every numeric mention and a few request-level labels.
Everything after that is deterministic: unit checks, role-to-field mapping, detection of repeated and conflicting
statements, and provenance.  Every field taken from the text is marked `interpreted` and must be confirmed
before the coordinator releases a design.
"""
from __future__ import annotations

import copy
import math
import re

import numpy as np

from ..core import spec as S
from .quantities import find_quantities, Quantity

ALLOWED = {
    "time": ["other", "Ts", "deadline", "window"], "freq": ["other", "fs"],
    "pos": ["other", "meas_bound", "meas_std", "init_pos_bound", "init_pos_std", "tolerance", "peak_cap", "rms_target"],
    "pos2": ["other", "meas_var", "mse_target"],
    "vel": ["other", "init_vel_bound", "init_vel_std", "vel_limit", "tolerance", "peak_cap", "rms_target"],
    "vel2": ["other", "mse_target"], "acc": ["other", "acc_bound", "acc_std"], "acc2": ["other"],
    "frac": ["other", "prob"], "count": ["other"], "other": ["other", "tolerance", "peak_cap", "rms_target"], "other2": ["other", "mse_target"],
}
_SYN = {"velocity": ["velocity", "speed", "rate"], "speed": ["speed", "velocity", "rate"], "rate": ["rate", "velocity", "speed"],
        "angle": ["angle", "position"], "position": ["position", "displacement", "angle"], "displacement": ["displacement", "position"],
        "temperature": ["temperature", "temp"], "pressure": ["pressure"], "current": ["current"], "bias": ["bias", "offset"], "offset": ["offset", "bias"]}


def section_of(text: str, pos: int) -> str:
    marks = [(m.start(), m.group(1).lower()) for m in re.finditer(r"\[(Q[123])\]", text)]
    sec = "q2"
    for st, name in marks:
        if st <= pos:
            sec = name
    return sec


def find_target(q1: str, states: list):
    """Index of the state named in the first answer (alias matching on the attachment's state names); None if unclear."""
    t = q1.lower()
    scores = []
    for i, s in enumerate(states):
        words = s["name"].lower().split()
        last = words[-1]
        syn = _SYN.get(last, [last])
        sc = 0
        if s["name"].lower() in t:
            sc = 3 + len(words)
        elif any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in syn):
            sc = 1 + sum(1 for w in words[:-1] if re.search(r"\b" + re.escape(w) + r"\b", t))
        scores.append(sc)
    if not scores or max(scores) == 0:
        return None
    best = [i for i, sc in enumerate(scores) if sc == max(scores)]
    return best[0] if len(best) == 1 else None


def assemble(attachment: dict, text: str, reading: dict) -> dict:
    """attachment: specification containing the plant block and any structured fields (status 'supplied').
    reading: {"quantities": [{"start","end","role","conf"}...] aligned with find_quantities(text),
              "objective","distribution","horizon","evidence","flags"}.
    Returns {"spec", "issues", "bound": [...]}."""
    sp = copy.deepcopy(attachment)
    plant = sp["plant"]
    issues, bound = [], []
    qs = find_quantities(text)
    roles = {(q["start"], q["end"]): q for q in reading.get("quantities", [])}
    base_family = "lin" if (plant["states"] and plant["states"][0]["unit"] == "m") else "ang"
    so = plant.get("kind") == "second_order"
    vals = {}

    def src(q):
        return f"text[{q.start}:{q.end}] '{text[q.start:q.end].strip()}'"

    for q in qs:
        r = roles.get((q.start, q.end))
        role = r["role"] if r else "other"
        if role not in ALLOWED.get(q.dim, ["other"]):
            role = "other"
        if role == "other":
            continue
        if so and q.dim in ("pos", "vel", "acc", "pos2", "vel2") and q.family and q.family != base_family:
            issues.append({"code": "unit_family", "message": f"{src(q)} is given in {'length' if q.family == 'lin' else 'angle'} units but the model uses "
                                                             f"{'length' if base_family == 'lin' else 'angle'} units", "role": role})
            continue
        ctx = text[max(0, q.start - 40):q.start].lower()
        if role in ("meas_std", "meas_bound") and re.search(r"\bvarian", ctx) and q.dim == "pos":
            issues.append({"code": "variance_unit", "message": f"{src(q)} is called a variance but carries a non-squared unit", "role": role})
            continue
        if q.negative and role != "vel_limit":
            # every bound, level, tolerance, time and probability is a magnitude: a stated negative value is not silently made positive
            issues.append({"code": "sign", "message": f"{src(q)} is negative, but the quantity it was read as ({role}) must be nonnegative; it is not used", "role": role})
            continue
        vals.setdefault(role, []).append(q)
        bound.append({"role": role, "value": q.value, "source": src(q), "conf": (r or {}).get("conf")})

    def single(role, path, transform=lambda v: v, rangeable=True):
        ql = vals.get(role)
        if not ql:
            return None
        vs = sorted({float(f"{transform(q.value):.12g}") for q in ql})
        if len(vs) == 1:
            S.put(sp, path, vs[0], status="interpreted", source=src(ql[0]))
            return vs[0]
        if rangeable:
            sp.setdefault("ranges", {})[path] = [vs[0], vs[-1]]
            S.put(sp, path, None, status="unresolved", source="; ".join(src(q) for q in ql) + " (several values stated: read as a range)")
            issues.append({"code": "several_values", "message": f"several values stated for {path}: read as the range [{vs[0]:.6g}, {vs[-1]:.6g}]", "role": role})
        else:
            issues.append({"code": "conflict", "message": f"conflicting values stated for {path}: {vs}", "role": role})
            S.put(sp, path, None, status="unresolved", source="conflicting statements")
        return None

    # --- timing: period and rate map to the same field
    tq = [(q, q.value) for q in vals.get("Ts", [])] + [(q, 1.0 / q.value) for q in vals.get("fs", []) if q.value > 0]
    if tq:
        vs = sorted({float(f"{v:.12g}") for _, v in tq})
        if len(vs) == 1:
            S.put(sp, "time.Ts", vs[0], status="interpreted", source=src(tq[0][0]))
        else:
            sp.setdefault("ranges", {})["time.Ts"] = [vs[0], vs[-1]]
            S.put(sp, "time.Ts", None, status="unresolved", source="; ".join(src(q) for q, _ in tq) + " (several periods stated)")
            issues.append({"code": "several_values", "message": f"several sample periods stated: {vs}", "role": "Ts"})
    Ts = S.get(sp, "time.Ts") or (plant.get("linear") or {}).get("Ts_native")
    single("meas_bound", "deterministic.V")
    # measurement noise: standard deviation or variance
    rq = [(q, q.value ** 2) for q in vals.get("meas_std", [])] + [(q, q.value) for q in vals.get("meas_var", [])]
    if rq:
        vs = sorted({float(f"{v:.12g}") for _, v in rq})
        if len(vs) == 1:
            S.put(sp, "stochastic.R", [[vs[0]]], status="interpreted", source=src(rq[0][0]))
        else:
            sp.setdefault("ranges", {})["stochastic.R"] = [vs[0], vs[-1]]
            S.put(sp, "stochastic.R", None, status="unresolved", source="; ".join(src(q) for q, _ in rq) + " (several values stated)")
            issues.append({"code": "several_values", "message": f"several measurement-noise variances stated: {vs}", "role": "meas_var"})
    # acceleration bound: total disturbance for an exact nominal model, additive disturbance otherwise
    so_blk = plant.get("second_order") or {}
    exact_model = so and str(so_blk.get("f")) == str(so_blk.get("f0")) and not any(r[0] != r[1] for r in (so_blk.get("param_ranges") or {}).values())
    single("acc_bound", "deterministic.D" if exact_model else "deterministic.d_bound")
    if vals.get("acc_std"):
        sa = vals["acc_std"][0].value
        if Ts:
            G = np.array([[Ts * Ts / 2], [Ts]])
            S.put(sp, "stochastic.Q", (sa * sa * (G @ G.T)).tolist(), status="interpreted", source=src(vals["acc_std"][0]),
                  conditions="acceleration noise constant within each sample: Q = sigma^2 [T^2/2, T]'[T^2/2, T]")
            S.put(sp, "stochastic.Q_kind", "discrete_cov", status="interpreted", source=src(vals["acc_std"][0]))
        else:
            issues.append({"code": "needs_period", "message": "the acceleration noise level was read but the sample period needed to convert it is not known", "role": "acc_std"})
    e1, e2 = vals.get("init_pos_bound"), vals.get("init_vel_bound")
    if e2:
        S.put(sp, "initial.E", [e1[0].value if e1 else 0.0, e2[0].value], status="interpreted",
              source=src(e2[0]) + ("; " + src(e1[0]) if e1 else "; position bound not stated (not needed: the observer starts from the first measurement)"))
    elif e1:
        issues.append({"code": "partial", "message": "an initial position-error bound was read but no velocity-error bound", "role": "init_pos_bound"})
    s1, s2 = vals.get("init_pos_std"), vals.get("init_vel_std")
    if s1 and s2:
        S.put(sp, "initial.P0", [[s1[0].value ** 2, 0.0], [0.0, s2[0].value ** 2]], status="interpreted", source=src(s1[0]) + "; " + src(s2[0]))
    elif s1 or s2:
        issues.append({"code": "partial", "message": "only one of the two initial standard deviations was read", "role": "init_pos_std"})
    if vals.get("vel_limit"):
        w = vals["vel_limit"][0].value
        dom = dict(S.get(sp, "deterministic.domain") or {})
        dom["x2"] = [-w, w]
        S.put(sp, "deterministic.domain", dom, status="interpreted", source=src(vals["vel_limit"][0]))
    ev = reading.get("evidence")
    if ev in ("mechanical_limit", "assumed"):
        S.put(sp, "deterministic.domain_evidence", {"kind": ev, "statement": "stated in the request"}, status="interpreted", source="request text")
    single("tolerance", "objective.tolerance", rangeable=False)
    single("deadline", "objective.deadline", rangeable=False)
    single("peak_cap", "objective.peak_cap", rangeable=False)
    single("prob", "objective.prob", rangeable=False)
    if vals.get("mse_target"):
        single("mse_target", "objective.mse_target", rangeable=False)
    elif vals.get("rms_target"):
        single("rms_target", "objective.mse_target", transform=lambda v: v * v, rangeable=False)
    single("window", "time.t_end", rangeable=False)
    obj = reading.get("objective")
    if obj and obj != "unspecified":
        S.put(sp, "objective.kind", obj, status="interpreted", source="request text (third answer)")
    dist = reading.get("distribution")
    if dist in ("gaussian", "bounded"):
        cur = S.get(sp, "stochastic.distribution")
        if cur is not None and cur != dist:
            issues.append({"code": "conflict", "message": f"the text says the noise is {dist} but the attachment says {cur}", "role": "distribution"})
        else:
            S.put(sp, "stochastic.distribution", dist, status="interpreted", source="request text")
    hor = reading.get("horizon")
    if hor in ("steady_state", "finite"):
        S.put(sp, "objective.horizon", hor, status="interpreted", source="request text")
    fl = list(plant.get("flags") or [])
    for f in reading.get("flags") or []:
        if f == "hidden_dynamics" and f not in fl:
            fl.append(f)
        elif f == "correlated_noise":
            n, ny = len(plant["states"]), max(1, len(plant["outputs"]))
            S.put(sp, "stochastic.cross_cov", [[1.0] * ny for _ in range(n)], status="interpreted",
                  source="request text: process and measurement noise are correlated (placeholder value; the correlation itself is not stated)")
        elif f == "coloured_noise":
            S.put(sp, "stochastic.white", False, status="interpreted", source="request text")
        elif f == "irregular_sampling":
            S.put(sp, "time.irregular", True, status="interpreted", source="request text")
    plant["flags"] = fl
    if S.get(sp, "target.state") is None:
        m = re.search(r"\[Q1\](.*?)(?=\[Q2\]|$)", text, re.S)
        ti = find_target(m.group(1) if m else text, plant["states"])
        if ti is not None:
            S.put(sp, "target.state", ti, status="interpreted", source="first answer (name matched to the model's state list)")
        else:
            issues.append({"code": "target", "message": "the quantity to estimate could not be matched to a state of the attached model", "role": "target"})
    sp["interpretation_issues"] = issues
    return {"spec": sp, "issues": issues, "bound": bound}


def attachment_of(case_given: dict, stats_in_attachment: bool) -> dict:
    """What the user attaches as a structured file for a natural-language request: the plant model, the operating ranges
    of the model file and, for multi-state linear plants, the covariance matrices.  Everything else must come from the text."""
    sp = S.empty_spec()
    sp["plant"] = copy.deepcopy(case_given["plant"])
    keep = ["deterministic.domain"]
    if stats_in_attachment:
        keep += ["stochastic.Q", "stochastic.Q_kind", "stochastic.G", "stochastic.R", "initial.P0", "stochastic.distribution", "stochastic.white",
                 "stochastic.cross_cov", "time.t_end", "time.irregular"]
        if (case_given["plant"].get("linear") or {}).get("clock") == "discrete":
            pass
    for p in keep:
        v = S.get(case_given, p)
        if v is not None:
            if p == "deterministic.domain":
                v = {k: x for k, x in v.items() if k != "x2"}        # the velocity range is stated in the text
            S.put(sp, p, v, status="supplied", source="attached model file")
    for p, r in (case_given.get("ranges") or {}).items():
        if stats_in_attachment and p.startswith(("stochastic", "initial.P0")):
            sp["ranges"][p] = r
            S.put(sp, p, None, status="unresolved", source="attached model file (range)")
    return sp
