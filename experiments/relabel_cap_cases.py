"""Correct outcome of the 20 test requests whose transient cap lies below the later tolerance (revision 2, finding 7).

    python -I experiments/relabel_cap_cases.py            # writes data/revision2_relabel.json

The frozen generator labelled these requests INVALID_INPUT ("cap below tolerance is a contradiction").  The two
requirements are compatible: the cap applies at every sample, the tolerance from the deadline on.  This script decides
each case from the case data alone, without importing the package (the independent references only):

  * At the first sample the only measurement is one position reading, which carries no velocity information.  The
    initial velocities of admissible trajectories form a set of half-width h; whatever velocity estimate is returned
    at t = 0, one admissible velocity lies at least h away.  If the cap is below h, no estimator can meet the request:
    PROVED_IMPOSSIBLE.  Without a stated travel or speed range, every velocity of the initial-error box is admissible.
    With one, only velocities from which the plant can stay inside the ranges for the whole window count (review
    cases 03a and 03b): h is taken from the separately written construction of such trajectories in
    independent.ref_hgo.first_sample_velocity_limit (an inner approximation, so a claim made with it is valid).
  * Otherwise the outcome depends on the certificates and is left to the acceptance test (none of the 20 cases is of
    this kind).
The frozen case files are not edited; the evaluation applies this file when it loads the cases (heros.evaluation.with_revisions).
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from independent import ref_hgo as RH          # noqa: E402  (separately written references, not the package)


def get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main():
    out = {"revision": 2, "finding": 7,
           "rule": ("first-sample limit: cap < half-width of the admissible initial velocities -> PROVED_IMPOSSIBLE; with a stated travel or "
                    "speed range only initial velocities from which the plant can stay inside it for the whole window count (review cases 03)"),
           "cases": {}}
    for line in (ROOT / "data" / "test_structured.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        if c["category"] != "invalid_cap_below_tol":
            continue
        t = c["truth"]
        E, x0, cap, tol = get(t, "initial.E"), get(t, "initial.x0_hat"), get(t, "objective.peak_cap"), get(t, "objective.tolerance")
        dom = get(t, "deterministic.domain") or {}
        assert get(t, "target.state") == 1 and get(t, "objective.kind") == "hard_bound"
        mid = float(x0[1]) if x0 else 0.0
        lo, hi = mid - E[1], mid + E[1]
        if dom.get("x2") is not None:
            lo, hi = max(lo, dom["x2"][0]), min(hi, dom["x2"][1])
        restricted = dom.get("x1") is not None or dom.get("x2") is not None
        if restricted:
            res = RH.first_sample_velocity_limit(t, lo, hi)
            lo, hi = res if res is not None else (0.0, 0.0)
        h = 0.5 * (hi - lo)
        new = "PROVED_IMPOSSIBLE" if cap < h else None
        out["cases"][c["id"]] = {"family": c["family"], "old_final": c["final"]["disposition"], "old_expected": c["expected"]["disposition"],
                                 "new_final": new, "new_expected": new, "cap": cap, "tolerance": tol, "initial_velocity_half_width": h,
                                 "admissible_initial_velocities": [lo, hi],
                                 "travel_or_speed_range_used": restricted,
                                 "reason": (f"cap {cap} < {h:.6g}: at t = 0 one position reading leaves every velocity in [{lo:.6g}, {hi:.6g}] admissible"
                                            + (" (velocities from which the plant can stay inside the stated ranges for the whole window)" if restricted else "")
                                            if new else "not decided by the first-sample limit")}
    n = sum(1 for v in out["cases"].values() if v["new_final"] == "PROVED_IMPOSSIBLE")
    out["summary"] = {"cases": len(out["cases"]), "proved_impossible": n, "undecided": len(out["cases"]) - n}
    (ROOT / "data" / "revision2_relabel.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
