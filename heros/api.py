"""Small public facade: build a specification, decide, execute, export and replay.

    from heros import api
    spec = api.new_spec(plant_block, target=1, Ts=1e-3, V=1e-4, D=5.0, E=[1e-3, 2.0], objective={"kind": "hard_bound", ...})
    result = api.solve(spec, y, u)          # decision + estimates + exportable record
    api.replay(result["record"])            # recompute and compare everything
"""
from __future__ import annotations

import copy

import numpy as np

from .core import spec as S
from .coordinator.coordinator import decide, Session
from . import record as R

_SHORT = {"Ts": "time.Ts", "t_end": "time.t_end", "label_resolution": "time.label_resolution", "V": "deterministic.V", "D": "deterministic.D", "d_bound": "deterministic.d_bound",
          "domain": "deterministic.domain", "domain_evidence": "deterministic.domain_evidence", "E": "initial.E", "x0_hat": "initial.x0_hat",
          "P0": "initial.P0", "init_policy": "initial.policy", "Q": "stochastic.Q", "Q_kind": "stochastic.Q_kind", "G": "stochastic.G",
          "R": "stochastic.R", "distribution": "stochastic.distribution", "white": "stochastic.white", "cross_cov": "stochastic.cross_cov",
          "target": "target.state"}


def new_spec(plant: dict, objective: dict | None = None, ranges: dict | None = None, constraints: list | None = None, **fields) -> dict:
    """Structured specification.  Explicit structured fields count as confirmed (status 'supplied')."""
    s = S.empty_spec()
    s["plant"].update(copy.deepcopy(plant))
    for k, v in fields.items():
        path = _SHORT.get(k, k)
        if v is not None:
            S.put(s, path, v.tolist() if isinstance(v, np.ndarray) else v, status="supplied", source="structured input")
    for k, v in (objective or {}).items():
        if v is not None:
            S.put(s, "objective." + k, v, status="supplied", source="structured input")
    for k, r in (ranges or {}).items():
        s["ranges"][_SHORT.get(k, k)] = list(r)
        S.put(s, _SHORT.get(k, k), None, status="unresolved", source="structured input (range)")
    s["constraints"] = [dict(c, paths=[_SHORT.get(p, p) for p in c["paths"]]) for c in (constraints or [])]
    return s


def solve(spec: dict, y=None, u=None, t=None, nlu: dict | None = None, history: list | None = None) -> dict:
    """Decision, estimates and record.  t: recorded sample times of y (optional).  A design certified for one fixed period
    is run on the data only if the recorded times agree with that period; otherwise the release is withdrawn.  A design
    whose model uses the applied input is run only with input values: those supplied as u or, without them, the input
    that the request fixes (record.apply_inputs)."""
    dec = decide(spec)
    dec, timing = R.apply_timestamps(spec, dec, y, t)
    dec, u_run, inputs = R.apply_inputs(spec, dec, y, u)
    out = {"decision": dec, "estimates": None, "timing": timing, "inputs": inputs}
    if dec["disposition"] == "RELEASE" and y is not None:
        out["estimates"] = R.execute(dec, y, u_run)
    out["record"] = R.make_record(spec, dec, y, u, out["estimates"], nlu=nlu, history=history, t=t, timing=timing)
    return out


replay = R.replay
__all__ = ["new_spec", "solve", "replay", "decide", "Session"]
