"""Exportable execution record and full replay.

A record retains everything needed to reconstruct the executed problem and to re-derive the reported result:
the complete normalised specification with evidence and accepted choices, method / build / model identifiers,
the released configuration, initialisation, guarantee summary and certificate numbers, the run-time inputs
(or their SHA-256 when stored externally) and all numerical outputs.

`replay(record)` recomputes the decision from the stored specification, recomputes the estimates from the
stored inputs, and compares EVERY numerical quantity (configuration, initialisation, bounds / covariances,
certificate values, estimates) -- not only the selected label or gain.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import platform

import numpy as np

from . import __version__
from .core import spec as S
from .core.assess import jsonable
from .core import timing as TM
from .coordinator import coordinator as C
from .kalman import specialist as K
from .khalil import specialist as H

RECORD_SCHEMA = "heros-record/1"


def _sha(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(arr, float)).tobytes()).hexdigest()


def _cfg_arrays(cfg):
    return {k: (np.asarray(v, float) if isinstance(v, list) and k in ("A", "B", "C", "Qd", "R", "P0", "x0_hat", "M_ss", "P_ss", "Pf_ss") else v)
            for k, v in cfg.items()}


def _period(decision: dict, spec: dict):
    cfg = decision.get("config") or {}
    if cfg.get("Ts") is not None:
        return float(cfg["Ts"])
    Ts = S.get(spec, "time.Ts")
    return Ts if Ts is not None else (S.get(spec, "plant.linear") or {}).get("Ts_native")


def apply_timestamps(spec: dict, decision: dict, y=None, t=None):
    """The decision for these data.  A design certified for one fixed period is released for recorded data only when the
    recorded timestamps put every sample at its instant t_0 + k Ts (revision 2, finding 1; review cases 01b and 01c);
    otherwise the release is withdrawn and the reason stated.  A question or a "no verified design found" for the request
    is replaced by the same data-level outcome, since no fixed-period design can be released for these data; a request
    that is invalid, awaits confirmation, is out of scope or is proved impossible keeps its outcome, with the timing
    issue added.  Returns (decision, timing_check or None)."""
    if t is None:
        return decision, None
    n = None if y is None else int(np.atleast_1d(np.asarray(y, float)).shape[0])
    res = S.get(spec, "time.label_resolution")
    chk = TM.check_timestamps(t, _period(decision, spec), n, label_resolution=res)
    if chk["ok"]:
        Ts = _period(decision, spec)
        conds = []
        if res and chk.get("max_deviation", 0.0) > TM.label_precision(t, Ts):
            conds.append(f"the samples are taken to be exactly at t_0 + k Ts; the recorded times differ from these instants by up to "
                         f"{chk['max_deviation']:.3g} s, which the request attributes to the rounding of the time labels (resolution {float(res):.6g} s)")
        prec = chk.get("label_precision") or 0.0
        if Ts and prec > TM.PRECISION_NOTE * float(Ts):
            tt = np.asarray(t, float).reshape(-1)
            conds.append(f"the recorded times confirm the sample instants t_0 + k Ts only to within {prec:.3g} s, the precision of time "
                         f"labels near {max(abs(float(tt[0])), abs(float(tt[-1]))):.6g} s; a sampling deviation below that is not detected "
                         "and is not covered")
        if conds and decision["disposition"] == "RELEASE" and decision.get("claim"):
            d = dict(decision)
            d["claim"] = dict(decision["claim"], conditions=list(decision["claim"].get("conditions") or []) + conds)
            return d, chk
        return decision, chk
    if decision["disposition"] not in ("RELEASE", "CLARIFY", "NO_VERIFIED_CANDIDATE"):
        d = dict(decision)
        d["explanations"] = list(decision.get("explanations") or []) + ["recorded timestamps: " + chk["message"]]
        d["timing_issue"] = {"code": chk["code"], "message": chk["message"]}
        return d, chk
    disp = "OUT_OF_SCOPE" if chk["code"] == "irregular_sampling" else "INVALID_INPUT"
    d = {k: decision.get(k) for k in ("coordinator_version", "spec_hash", "policy", "accepted_choices")}
    d.update(disposition=disp, selected=None, qualifying=[], claim=None, config=None, question=None, all_needs=[], alternatives=[],
             tradeoff=None, assessments={}, impossibility=None, harmless=[], explanations=[chk["message"]],
             issues=[{"code": chk["code"], "path": "time.Ts", "message": chk["message"]}],
             display=TIMING_DISPLAY.get(chk["code"], C.DISPLAY[disp]))
    if decision["disposition"] == "RELEASE":
        d["withdrawn_release"] = {"selected": decision.get("selected"), "claim": (decision.get("claim") or {}).get("statement")}
    else:
        d["request_outcome"] = {"disposition": decision["disposition"], "display": decision.get("display")}
    return d, chk


def apply_inputs(spec: dict, decision: dict, y=None, u=None):
    """The applied inputs for running a released design on data (review cases D1 and D2 of revision 2b).

    A released model that uses the input (an observer whose nominal acceleration depends on u, a Kalman model with a
    nonzero input matrix) is run with the supplied input values.  Without them it is run with the input that the request
    fixes (a one-point operating range for u), which then becomes a condition of the claim; otherwise nothing is run for
    these data, since an input taken as zero would make the estimates and the claim false.  Returns (decision, u_run,
    note or None)."""
    if y is None or u is not None or decision["disposition"] != "RELEASE":
        return decision, (None if u is None else np.asarray(u, float)), None
    cfg = decision.get("config") or {}
    if decision.get("selected") == K.METHOD:
        B = np.atleast_2d(np.asarray(cfg.get("B") if cfg.get("B") is not None else [[0.0]], float))
        needs, m, what = bool(np.any(B != 0)), B.shape[1], "the input matrix B of the model is not zero"
    else:
        needs, m, what = H.uses_input(cfg), 1, f"the nominal acceleration f0 = {cfg.get('f0')} depends on u"
    if not needs:
        return decision, None, None
    n = int(np.atleast_1d(np.asarray(y, float)).shape[0]) if np.asarray(y).ndim <= 1 else int(np.asarray(y).shape[0])
    ur = (S.get(spec, "deterministic.domain") or {}).get("u")
    if m == 1 and ur is not None and float(ur[0]) == float(ur[1]):
        c = float(ur[0])
        d = dict(decision)
        d["claim"] = dict(decision["claim"], conditions=list(decision["claim"].get("conditions") or []) + [
            f"no input values came with the data; the run applies the input u = {c:.6g} that the request fixes (operating range [{c:.6g}, {c:.6g}])"])
        return d, np.full((n, 1) if decision.get("selected") == K.METHOD else n, c), {"code": "fixed_input_from_specification", "value": c}
    msg = (f"the released design uses the applied input ({what}), but the data contain no input values; nothing is run: supply the "
           "input column u, or state the input as fixed in the request (a one-point operating range)")
    d0 = {k: decision.get(k) for k in ("coordinator_version", "spec_hash", "policy", "accepted_choices")}
    d0.update(disposition="INVALID_INPUT", selected=None, qualifying=[], claim=None, config=None, question=None, all_needs=[], alternatives=[],
              tradeoff=None, assessments={}, impossibility=None, harmless=[], explanations=[msg],
              issues=[{"code": "missing_input", "path": "u", "message": msg}], display=INPUT_DISPLAY,
              withdrawn_release={"selected": decision.get("selected"), "claim": (decision.get("claim") or {}).get("statement")})
    return d0, None, {"code": "missing_input"}


INPUT_DISPLAY = "The data lack the applied input that the design needs; nothing is run."

TIMING_DISPLAY = {
    "timing_conflict": "The timestamps of the data do not match the stated sample period; nothing is released.",
    "irregular_sampling": "The data are not sampled at a fixed period; no fixed-period certificate applies.",
    "bad_timestamps": "The timestamps of the data are not usable; nothing is released.",
}


def execute(decision: dict, y, u=None) -> dict:
    """Run the released configuration on actual measurements and applied inputs (public run-time path)."""
    if decision["disposition"] != "RELEASE":
        raise ValueError("no released design to execute: disposition is " + decision["disposition"])
    cfg = decision["config"]
    if decision["selected"] == K.METHOD:
        return K.run(_cfg_arrays(cfg), y, u)
    return H.run(cfg, y, u)


def make_record(spec: dict, decision: dict, y=None, u=None, outputs: dict | None = None, nlu: dict | None = None,
                history: list | None = None, store_inputs: bool = True, notes: str = "", t=None, timing: dict | None = None) -> dict:
    rec = {
        "schema": RECORD_SCHEMA,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "build": {"package": "heros-estimation", "version": __version__, "coordinator": C.VERSION, "kalman": K.VERSION,
                  "khalil": H.VERSION, "python": platform.python_version(), "numpy": np.__version__},
        "interpretation_model": nlu or {"used": False},
        "spec": jsonable(spec),
        "spec_hash": S.spec_hash(spec),
        "evidence": jsonable(spec.get("evidence", {})),
        "accepted_choices": decision.get("accepted_choices", []),
        "decision": jsonable(decision),
        "history": jsonable(history or []),
        "notes": notes,
    }
    if y is not None:
        y = np.asarray(y, float)
        rec["inputs"] = {"y_sha256": _sha(y), "n_measurements": int(y.shape[0])}
        if u is not None:
            rec["inputs"]["u_sha256"] = _sha(u)
        if t is not None:
            rec["inputs"]["t_sha256"] = _sha(t)
            rec["inputs"]["timing_check"] = jsonable(timing) if timing is not None else None
        if store_inputs:
            rec["inputs"]["y"] = y.tolist()
            rec["inputs"]["u"] = None if u is None else np.asarray(u, float).tolist()
            if t is not None:
                rec["inputs"]["t"] = np.asarray(t, float).tolist()
    if outputs is not None:
        rec["outputs"] = jsonable({k: v for k, v in outputs.items()})
    return rec


def _flatten(o, prefix=""):
    if isinstance(o, dict):
        for k in sorted(o):
            yield from _flatten(o[k], f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(o, (list, tuple)):
        if o and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in o):
            yield prefix, np.asarray(o, float)
        else:
            for i, v in enumerate(o):
                yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, o


def compare(a, b, rtol=1e-9, atol=1e-12, skip=("created_utc", "seconds", "latency")) -> list:
    """Leaf-by-leaf comparison of two JSON-like structures.  Returns the list of differences."""
    fa, fb = dict(_flatten(a)), dict(_flatten(b))
    diffs = []
    for k in sorted(set(fa) | set(fb)):
        if any(s in k for s in skip):
            continue
        if k not in fa or k not in fb:
            diffs.append({"path": k, "issue": "missing on one side"})
            continue
        x, y = fa[k], fb[k]
        num = lambda z: isinstance(z, (int, float, np.ndarray)) and not isinstance(z, bool)
        if num(x) and num(y):
            x, y = np.asarray(x, float), np.asarray(y, float)
            if x.shape != y.shape:
                diffs.append({"path": k, "issue": f"shape {x.shape} vs {y.shape}"})
            elif not np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True):
                diffs.append({"path": k, "issue": "numeric", "max_abs_diff": float(np.nanmax(np.abs(x - y)))})
        elif x != y:
            diffs.append({"path": k, "issue": "value", "a": str(x)[:80], "b": str(y)[:80]})
    return diffs


def replay(record: dict) -> dict:
    """Recompute and compare the full result.  Returns {"match": bool, "n_compared": int, "differences": [...]}."""
    spec = record["spec"]
    out = {"schema_ok": record.get("schema") == RECORD_SCHEMA, "spec_hash_ok": S.spec_hash(spec) == record["spec_hash"]}
    ins0 = record.get("inputs") or {}
    dec = C.decide(spec)
    if ins0.get("t") is not None:
        if _sha(np.asarray(ins0["t"], float)) != ins0.get("t_sha256"):
            diffs0 = [{"part": "inputs", "path": "t", "issue": "stored timestamps do not match their hash"}]
        else:
            diffs0 = []
        dec = apply_timestamps(spec, dec, ins0.get("y"), ins0["t"])[0]
    else:
        diffs0 = []
    u_run = None
    if ins0.get("y") is not None:
        dec, u_run, _ = apply_inputs(spec, dec, ins0.get("y"), None if ins0.get("u") is None else np.asarray(ins0["u"], float))
    dec = jsonable(dec)
    diffs = diffs0 + [dict(d, part="decision") for d in compare(record["decision"], dec)]
    n = len(dict(_flatten(dec)))
    unverified, blocking = [], []
    has_out = bool(record.get("outputs"))
    ins = record.get("inputs") or {}
    if dec["disposition"] == "RELEASE" and has_out:
        if ins.get("y") is None:
            # outputs that cannot be recomputed are not accepted (review C-03)
            blocking.append("outputs (the record does not contain the measurements needed to recompute them)")
        else:
            y = np.asarray(ins["y"], float)
            u = ins.get("u")
            if _sha(y) != ins.get("y_sha256"):
                diffs.append({"part": "inputs", "path": "y", "issue": "stored measurements do not match their hash"})
            if u is not None and ins.get("u_sha256") is not None and _sha(np.asarray(u, float)) != ins["u_sha256"]:
                diffs.append({"part": "inputs", "path": "u", "issue": "stored inputs do not match their hash"})
            res = jsonable(execute(dec, y, u_run))
            diffs += [dict(d, part="outputs") for d in compare(record.get("outputs", {}), res)]
            n += len(dict(_flatten(res)))
    elif has_out and dec["disposition"] != "RELEASE":
        diffs.append({"part": "outputs", "path": "outputs", "issue": "outputs are stored although the recomputed decision releases nothing"})
    elif dec["disposition"] == "RELEASE":
        # nothing to compare, so nothing about outputs is verified; stated so that a record stripped of its outputs does
        # not look fully verified (review C, recheck N3).  The decision itself was recomputed and compared.
        unverified.append("outputs: none stored, so none verified (" + ("measurements are stored but no outputs: outputs removed?" if ins.get("y") is not None
                          else "a design-only record, or outputs removed") + "); the decision was recomputed")
    unverified = blocking + unverified
    out.update(match=bool(not diffs and not blocking and out["schema_ok"] and out["spec_hash_ok"]), n_compared=n, differences=diffs[:50],
               n_differences=len(diffs), unverified=unverified, outputs_stored=has_out,
               not_recomputed=["interpretation_model (identity of the reader; the reading itself is not repeated)", "history", "notes", "created_utc"])
    return out


def dumps(record: dict) -> str:
    return json.dumps(jsonable(record), indent=1, allow_nan=False)
