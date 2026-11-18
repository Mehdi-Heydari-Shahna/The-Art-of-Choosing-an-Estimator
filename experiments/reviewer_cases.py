"""Replay the six cases supplied with the review of the revision 2 archive through the public interface.

    python -I experiments/reviewer_cases.py --label current
    python -I experiments/reviewer_cases.py --label previous --code <tree of tag revision-2-frozen>

Each case file holds an `arguments` object.  Exactly that object is passed to heros.api.solve; nothing is added, removed or
changed.  The other fields of a case file (audit notes, true velocities, the outcome observed in the review) are read only
to score the result.  The tree under test is the one given with --code (default: this checkout); the previous tree is the
software of tag revision-2-frozen, whose heros/ and independent/ directories equal those of the delivered revision 2
archive.  Results go to results/reviewer_cases_<label>.json; experiments/reviewer_cases_compare.py tabulates both runs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
CASES = HERE / "tests" / "fixtures" / "reviewer_cases_rev2"


def plain(x):
    """JSON-safe copy (numpy arrays and scalars, non-finite floats as strings)."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def brief(dec):
    out = {"disposition": dec["disposition"], "selected": dec.get("selected"), "display": dec.get("display")}
    if dec.get("claim"):
        out["claim"] = dec["claim"].get("statement")
        out["claim_value"] = dec["claim"].get("value")
    if dec.get("impossibility"):
        imp = dec["impossibility"]
        out["impossibility"] = {k: imp.get(k) for k in ("argument", "bound", "level", "sample_time", "kind")}
    out["explanations"] = list(dec.get("explanations") or [])
    return out


def code_digest(root: pathlib.Path) -> str:
    h = hashlib.sha256()
    for p in sorted((root / "heros").rglob("*.py")) + sorted((root / "independent").rglob("*.py")):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--code", default=str(HERE), help="tree whose heros package is under test")
    ap.add_argument("--cases", default=str(CASES), help="directory of case files (default: the six cases of the first review round)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    code = pathlib.Path(a.code).resolve()
    sys.path.insert(0, str(code))
    from heros import api, __version__
    from heros.coordinator import coordinator as C
    assert pathlib.Path(api.__file__).resolve().is_relative_to(code), api.__file__

    rows = []
    for f in sorted(pathlib.Path(a.cases).glob("*.json")):
        raw = f.read_bytes()
        case = json.loads(raw)
        args = copy.deepcopy(case["arguments"])
        out = api.solve(**args)
        dec = out["decision"]
        row = {"file": f.name, "sha256": hashlib.sha256(raw).hexdigest(), "arguments": sorted(case["arguments"]),
               "decision": plain(brief(dec)), "decision_full": plain(dec), "timing": plain(out.get("timing"))}
        est = out.get("estimates")
        tv = case.get("actual_true_velocity_if_given", case.get("actual_true_velocity_if_applicable"))
        if est is not None and "x_hat" not in est:        # Kalman filter: covariance claim, no deterministic error bound to check
            row["estimate_check"] = {"kind": "kalman", "n_estimates": int(np.asarray(est["x_filt"]).shape[0])}
        elif est is not None:
            X = np.asarray(est["x_hat"], float)
            k0 = (case.get("audit_notes") or {}).get("first_scored_sample")
            if k0 is None:                    # score from the requested deadline (first sample at or after it)
                dl, Ts = (args["spec"].get("objective") or {}).get("deadline"), (args["spec"].get("time") or {}).get("Ts")
                k0 = int(np.ceil(dl / Ts - 1e-12)) if dl and Ts else 0
            k0 = int(k0)
            chk = {"first_scored_sample": k0, "n_estimates": int(X.shape[0])}
            if tv is not None:
                n = min(len(tv), X.shape[0])
                err = np.abs(X[k0:n, 1] - np.asarray(tv[k0:n], float))
                chk["max_velocity_error"] = float(err.max())
                chk["at_sample"] = int(k0 + int(err.argmax()))
                if dec.get("claim"):
                    chk["claim_value"] = float(dec["claim"]["value"])
                    chk["claim_covers_error_on_this_record"] = bool(err.max() <= float(dec["claim"]["value"]))
            row["estimate_check"] = chk
        obs = next((v for k, v in case.items() if k.startswith("observed_")), None)
        if obs is not None:
            same = obs.get("disposition") == dec["disposition"]
            if obs.get("claim") and dec.get("claim"):
                same = same and float(obs["claim"]["value"]) == float(dec["claim"]["value"])
            if obs.get("impossibility") and dec.get("impossibility"):
                same = same and float(obs["impossibility"]["bound"]) == float(dec["impossibility"]["bound"])
            row["same_outcome_as_review_record"] = bool(same)
        rows.append(row)
        d = row["decision"]
        print(f"{f.name:58s} {d['disposition']:22s} claim={d.get('claim_value')} "
              f"imp={None if not d.get('impossibility') else d['impossibility']['bound']} "
              f"err={(row.get('estimate_check') or {}).get('max_velocity_error')} review_record_match={row.get('same_outcome_as_review_record')}")
    res = {"label": a.label, "code_tree": "this checkout" if code == HERE.resolve() else code.name, "code_digest": code_digest(code),
           "package_version": __version__, "coordinator_version": C.VERSION, "interface": "heros.api.solve(**arguments)", "cases": rows}
    outp = pathlib.Path(a.out) if a.out else HERE / "results" / f"reviewer_cases_{a.label}.json"
    outp.write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", outp.relative_to(HERE) if outp.is_relative_to(HERE) else outp)


if __name__ == "__main__":
    main()
