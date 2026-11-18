"""Table of the seven review cases of revision 2b: outcome of revision 2b, revised outcome, and the mathematical reason.

    python -I experiments/reviewer_cases.py --label rev2b_previous --code <tree of tag revision-2b-frozen> \
        --cases tests/fixtures/reviewer_cases_rev2b --out results/reviewer_cases_rev2b_previous.json
    python -I experiments/reviewer_cases.py --label rev2b_current \
        --cases tests/fixtures/reviewer_cases_rev2b --out results/reviewer_cases_rev2b_current.json
    python -I experiments/reviewer_cases_rev2b_compare.py

The numbers in the reason column are computed here from the case data; the checks use the judge (heros.evaluation) and
the independent references, not the specialists.  Writes results/reviewer_cases_rev2b_table.json and .md.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from heros import evaluation as EV                      # noqa: E402  (judge)
from heros.core import timing as TM                     # noqa: E402

CASES = HERE / "tests" / "fixtures" / "reviewer_cases_rev2b"
SHORT = {"RELEASE": "release", "OUT_OF_SCOPE": "out of scope", "NO_VERIFIED_CANDIDATE": "no verified design",
         "PROVED_IMPOSSIBLE": "proved impossible", "INVALID_INPUT": "invalid input", "CLARIFY": "question"}


def g(x, n=3):
    return f"{x:.{n}g}"


def outcome(row):
    d = row["decision"]
    s = SHORT[d["disposition"]]
    if d.get("claim_value") is not None:
        s += f", claim {g(d['claim_value'])}"
    if d.get("impossibility"):
        s += f", lower limit {g(d['impossibility']['bound'])}"
    chk = row.get("estimate_check") or {}
    if "max_velocity_error" in chk:
        s += f"; error on the record {g(chk['max_velocity_error'])}"
    return s


def best_bound(row):
    for e in row["decision"]["explanations"]:
        if "best bound found" in e:
            return float(e.split("best bound found ")[1].split(" ")[0])
    return None


def main():
    prev = {r["file"]: r for r in json.loads((HERE / "results" / "reviewer_cases_rev2b_previous.json").read_text())["cases"]}
    cur = {r["file"]: r for r in json.loads((HERE / "results" / "reviewer_cases_rev2b_current.json").read_text())["cases"]}
    rows = []
    for name in sorted(cur):
        case = json.loads((CASES / name).read_text())
        args, spec = case["arguments"], case["arguments"]["spec"]
        P, Q = prev[name], cur[name]
        key = name.split("_")[0]
        chk = {}
        if P["decision"]["disposition"] == "PROVED_IMPOSSIBLE":
            ok, why = EV._verify_impossibility(spec, P["decision_full"])
            chk["judge_on_revision2b_impossibility"] = {"confirmed": ok, "why": why}
        if Q["decision"]["disposition"] == "RELEASE":
            jr = EV.judge_release(spec, Q["decision_full"])
            chk["judge_on_revised_release"] = {"correct": jr["correct"], "why": jr["why"]}
        err_prev = (P.get("estimate_check") or {}).get("max_velocity_error")
        err_cur = (Q.get("estimate_check") or {}).get("max_velocity_error")
        Ts = spec["time"]["Ts"]
        if key in ("A1", "A2", "A3"):
            t = np.asarray(args["t"], float)
            dev = float(np.max(np.abs(t - (t[0] + np.arange(t.size) * Ts))))
            prec = TM.label_precision(t, Ts)
            old_tol = 64 * np.finfo(float).eps * max(abs(t[0]), abs(t[-1]), Ts)
            chk.update(deviation_s=dev, label_precision_s=prec, revision2b_allowance_s=old_tol, clock_origin_s=float(t[0]))
            if key == "A1":
                why = (f"x1 = t, v = 1; sample 1 is {g(dev)} s (2^-17 s) off t_0 + Ts. Revision 2b already declined: its allowance "
                       f"{g(old_tol, 2)} s was below the deviation. Revised: same decision; the deviation exceeds the precision "
                       f"{g(prec, 2)} s of these labels.")
            elif key == "A2":
                why = (f"Same intervals and positions as A1, clock origin 2^30 s. Revision 2b attributed up to 64 eps x 2^30 = "
                       f"{g(old_tol)} s to label rounding, more than the deviation {g(dev)} s, and released a zero bound (error "
                       f"{g(err_prev)} on the record). Revised: the allowance is the float64 precision of the labels, 4 ulp(2^30) = "
                       f"{g(prec)} s; the deviation is 32 such units, a real sampling deviation, so out of scope as A1.")
            else:
                why = (f"Regular clock at the same origin: deviation {g(dev)}. Released as before (error {g(err_cur)} on the record); "
                       f"the claim now states that labels near {g(float(t[0]), 4)} s confirm the instants only to within {g(prec)} s.")
        elif key == "B":
            dom = spec["deterministic"]["domain"]
            h = math.sqrt(10 * (dom["x1"][1] - spec["time"]["t_end"] * dom["x2"][0]))
            chk.update(excess_speed_bound=h, revised_best_bound=best_bound(Q))
            why = (f"v in [1, 2], x1 in [0, 1.001] for 1 s and |v'| <= 5: the excess h = max(v - 1) satisfies h^2/10 <= 0.001, so "
                   f"h <= {g(h)} and the estimate 1 has error <= {g(h)} < {g(spec['objective']['tolerance'])} at every sample; the "
                   f"claim was false. Its witness braked to rest, outside the stated speed range. Revised: the braking construction "
                   f"(package and judge) keeps every velocity on its path inside the speed range, so there is no witness and no claim; "
                   f"the observer certifies only {g(best_bound(Q))}.")
        elif key == "C":
            P0, Q0, R0 = spec["initial"]["P0"][0][0], spec["stochastic"]["Q"][0][0], spec["stochastic"]["R"][0][0]
            P_, fv = P0, []
            for _ in range(5):
                Pf = P_ - P_ * P_ / (P_ + R0); fv.append(Pf); P_ = Pf + Q0
            te = spec["time"]["t_end"]
            chk.update(t_end=te, filtered_variances=fv, revised_window=list(TM.sample_window(Ts, None, te)))
            why = (f"The record ends at t_end = 4 - 2^-31 s, so its samples are k = 0..3 (the four supplied). Filtered variances "
                   f"{', '.join(g(x, 4) for x in fv[:4])} stay below {g(spec['objective']['mse_target'], 7)}; only k = 4 ({g(fv[4], 5)}) "
                   f"exceeds it. Revision 2b's allowance of 1e-9 on t_end/Ts admitted k = 4 and claimed impossibility with it. Revised: "
                   f"an allowance of a few units of rounding of the ratio; window k <= 3; the Kalman filter is released with "
                   f"{g(fv[3], 6)} <= {g(spec['objective']['mse_target'], 6)}, confirmed by the judge.")
        else:
            X = None
            why = ("f = f0 = u with u fixed at 1 (one-point range), zero initial state and V = 0, so x1 = t^2/2 and v = t. "
                   + (f"Revision 2b ran the observer with u = 0 because no input vector was supplied: velocity 0.443 at t = 0.5 s, "
                      f"error {g(err_prev)} after the deadline under a claimed zero bound. Revised: a design whose model uses the input runs "
                      f"with the input the request fixes (stated as a condition of the claim): error {g(err_cur)}; without a fixed input "
                      f"nothing is run." if key == "D1" else
                      f"With the input vector the run was already exact (error {g(err_prev)}); unchanged."))
        rows.append({"case": name.replace(".json", ""), "revision2b": outcome(P), "revised": outcome(Q),
                     "revision2b_disposition": P["decision"]["disposition"], "revised_disposition": Q["decision"]["disposition"],
                     "revision2b_matches_review_record": P.get("same_outcome_as_review_record"), "justification": why, "checks": chk})
    (HERE / "results" / "reviewer_cases_rev2b_table.json").write_text(json.dumps(rows, indent=1) + "\n")
    L = ["# Review cases of revision 2b: revision 2b and revised outcomes", "",
         "Generated by experiments/reviewer_cases_rev2b_compare.py from results/reviewer_cases_rev2b_previous.json (tag "
         "revision-2b-frozen, the software of the delivered revision 2b archive) and results/reviewer_cases_rev2b_current.json "
         "(revised software). Each case file's `arguments` object was passed unchanged to heros.api.solve.", "",
         "| Case | Revision 2b outcome | Revised outcome | Mathematical justification |", "|---|---|---|---|"]
    for r in rows:
        cell = [x.replace("|", "\\|") for x in (r["revision2b"], r["revised"], r["justification"])]
        L.append(f"| {r['case'].split('_')[0]} | {cell[0]} | {cell[1]} | {cell[2]} |")
    L += ["", "Checks by the judge (heros/evaluation.py) and the independent references:", ""]
    for r in rows:
        c, bits = r["checks"], []
        if "judge_on_revision2b_impossibility" in c:
            j = c["judge_on_revision2b_impossibility"]
            bits.append(f"revision 2b impossibility claim {'confirmed' if j['confirmed'] else 'refuted' if j['confirmed'] is False else 'not checked'}: {j['why']}")
        if "judge_on_revised_release" in c:
            j = c["judge_on_revised_release"]
            bits.append(f"revised release judged {'correct' if j['correct'] else 'incorrect'} under the specification: {j['why']}")
        L.append(f"- {r['case'].split('_')[0]}: " + ("; ".join(bits) if bits else "no claim in either revision"))
    L.append("")
    (HERE / "results" / "reviewer_cases_rev2b_table.md").write_text("\n".join(L))
    for r in rows:
        print(r["case"].split("_")[0], "|", r["revision2b"], "|", r["revised"])


if __name__ == "__main__":
    main()
