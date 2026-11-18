"""Table of the six review cases: outcome of revision 2, revised outcome, and the mathematical reason, with checks.

    python -I experiments/reviewer_cases_compare.py

Reads results/reviewer_cases_previous.json and results/reviewer_cases_current.json (experiments/reviewer_cases.py) and the
case files.  The checks use the judge (heros.evaluation) and the independent references (independent/), not the
specialists that made the decisions:
  * a released claim is checked on the record itself (largest velocity error of the released estimate against the true
    velocity in the case file) and by the judge under the complete facts of the case;
  * a claim of impossibility is checked by the judge's own constructions;
  * the numbers in the reason column are computed here from the case data.
Writes results/reviewer_cases_table.json and results/reviewer_cases_table.md.
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
from heros.core import limits as LM                     # noqa: E402  (package construction, reported for comparison)
from independent import ref_hgo as RH                   # noqa: E402

CASES = HERE / "tests" / "fixtures" / "reviewer_cases_rev2"
SHORT = {"RELEASE": "release", "OUT_OF_SCOPE": "out of scope", "NO_VERIFIED_CANDIDATE": "no verified design",
         "PROVED_IMPOSSIBLE": "proved impossible", "INVALID_INPUT": "invalid input", "CLARIFY": "question"}


def g(x, n=3):
    return f"{x:.{n}g}"


def outcome(row):
    d = row["decision"]
    s = SHORT[d["disposition"]]
    if d.get("claim_value") is not None:
        s += f", claim |error| <= {g(d['claim_value'])}"
    if d.get("impossibility"):
        s += f", lower limit {g(d['impossibility']['bound'])} > {g(d['impossibility']['level'])}"
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
    prev = {r["file"]: r for r in json.loads((HERE / "results" / "reviewer_cases_previous.json").read_text())["cases"]}
    cur = {r["file"]: r for r in json.loads((HERE / "results" / "reviewer_cases_current.json").read_text())["cases"]}
    rows = []
    for name in sorted(cur):
        case = json.loads((CASES / name).read_text())
        spec, args = case["arguments"]["spec"], case["arguments"]
        P, Q = prev[name], cur[name]
        Ts = spec["time"]["Ts"]
        chk = {}
        if P["decision"]["disposition"] == "RELEASE":
            jr = EV.judge_release(spec, P["decision_full"])
            chk["judge_on_revision2_release"] = {"correct": jr["correct"], "why": jr["why"]}
        if P["decision"]["disposition"] == "PROVED_IMPOSSIBLE":
            ok, why = EV._verify_impossibility(spec, P["decision_full"])
            chk["judge_on_revision2_impossibility"] = {"confirmed": ok, "why": why}
        if Q["decision"]["disposition"] == "PROVED_IMPOSSIBLE":
            ok, why = EV._verify_impossibility(spec, Q["decision_full"])
            chk["judge_on_revised_impossibility"] = {"confirmed": ok, "why": why}
        if Q["decision"]["disposition"] == "RELEASE":
            jr = EV.judge_release(spec, Q["decision_full"])
            chk["judge_on_revised_release"] = {"correct": jr["correct"], "why": jr["why"]}
        err_prev = (P.get("estimate_check") or {}).get("max_velocity_error")
        key = name[:3].rstrip("_")
        if key in ("01a", "01b"):
            t = np.asarray(case["audit_notes"]["physical_times"] if key == "01a" else args["t"], float)
            dev = float(np.max(np.abs(t - t[0] - np.arange(t.size) * Ts)))
            chk["deviation_from_grid_s"] = dev
            if key == "01a":
                why = (f"The request declares irregular sampling. The observer's gains and its bound are computed for samples at "
                       f"t_k = k Ts, Ts = {g(Ts)} s; the physical instants {', '.join(g(x) for x in t)} s deviate from that grid by "
                       f"up to {g(dev)} s, so the zero bound does not apply (the released estimate has error {g(err_prev)} on this "
                       f"record). Revised: the observer, like the Kalman filter, is not applicable when irregular sampling is "
                       f"declared, so no method applies.")
            else:
                why = (f"The supplied instants deviate from t_0 + k Ts by up to {g(dev)} s; the record check withdraws the "
                       f"fixed-period release in both revisions. Kept unchanged (the message now names the deviation).")
        elif key == "01c":
            t = np.asarray(args["t"], float)
            dev = float(np.max(np.abs(t - t[0] - np.arange(t.size) * Ts)))
            tol = 64 * np.finfo(float).eps * max(abs(t[0]), abs(t[-1]), Ts)
            v = np.asarray(case["actual_true_velocity_if_given"], float)
            M2 = P["decision_full"]["config"]["M"][1]
            chk.update(deviation_from_grid_s=dev, rounding_allowance_s=tol, gain_times_offset=M2 * float(np.max(np.abs(v))) * dev)
            why = (f"Timestamps are the actual sample instants. Sample 1 is taken {g(dev)} s late; at speed {g(float(np.max(np.abs(v))))} "
                   f"the reading differs by {g(float(np.max(np.abs(v))) * dev)} from the fixed-period model, and the observer gain "
                   f"{g(M2, 4)} turns this into a velocity error {g(M2 * float(np.max(np.abs(v))) * dev)}, matching the error "
                   f"{g(err_prev)} on the record; the zero bound does not cover it. Revision 2 accepted intervals within a relative 1e-6 "
                   f"of Ts. Revised: each instant must equal t_0 + k Ts within the rounding of the labels ({g(tol, 2)} s here) or, "
                   f"if the request states a label resolution r, within r/2, which then becomes a condition of the claim.")
        elif key == "02":
            so = spec["plant"]["second_order"]
            mism = abs(so["param_ranges"]["b"][0] - so["params"]["b"]) * abs(spec["deterministic"]["domain"]["u"][0])
            lower = EV._sampled_disturbance(spec)
            bb = best_bound(Q)
            M = np.asarray(P["decision_full"]["config"]["M"], float)
            wc = max(RH.worst_case_exact(Ts, M, np.eye(2)[spec["target"]["state"]], spec["initial"]["E"], mism,
                                         spec["deterministic"]["V"], n, form="current", sub=8, init="box") for n in (5, 10, 40, 80))
            chk.update(mismatch=mism, judge_sampled_mismatch=lower, revised_best_bound=bb, revision2_gains_exact_worst_case=wc)
            why = (f"The range b in [2, 2] states the true gain, so x2' - f0 = (2 - 1) u = {g(mism)} with u = 1 (the judge's own "
                   f"search finds {g(lower)}). Revision 2 put the nominal b = 1 into the true model, used D = 0 and claimed 0; the "
                   f"error on the record for t >= 0.5 s is {g(err_prev)}. Revised: D = {g(mism)} is derived from the stated true "
                   f"value. With D = {g(mism)} the exact worst case of the Revision 2 gains is {g(wc)} (independent computation; the "
                   f"constant mismatch of this record attains it), and the best bound of the observer family is {g(bb)} > "
                   f"{g(spec['objective']['tolerance'])}, so no design is released. Not impossible: with the exact "
                   f"initial state and V = 0, the estimate 2 t_k of the true model has zero error.")
        elif key == "03a":
            dom = spec["deterministic"]["domain"]
            vmax = dom["x1"][1] / spec["time"]["t_end"]
            pk = LM.viable_initial_velocities(spec, -1.0, 1.0)
            ind = RH.first_sample_velocity_limit(spec, -1.0, 1.0)
            bb = best_bound(Q)
            chk.update(exact_speed_limit=vmax, package_inner_velocities=list(pk[:2]) if pk else None,
                       independent_inner_velocities=list(ind) if ind else None, revised_best_bound=bb)
            why = (f"x2' = 0 and x1(0) = 0, so x1(t) = v t; the travel |x1| <= {g(dom['x1'][1])} holds up to t = "
                   f"{g(spec['time']['t_end'])} s, hence |v| <= {g(vmax)}. The estimate 0 has error <= {g(vmax)} < "
                   f"{g(spec['objective']['tolerance'])} at every sample, so the claim of impossibility (which used [-1, 1]) is false. "
                   f"Revised: the first-sample limit uses only initial velocities whose trajectories stay inside the stated ranges for the "
                   f"whole window, here [{g(pk[0])}, {g(pk[1])}] (independent construction [{g(ind[0])}, {g(ind[1])}]); half-width "
                   f"{g(0.5 * (pk[1] - pk[0]))} < {g(spec['objective']['tolerance'])}, so no claim. The observer's certificate "
                   f"does not use the travel range (best bound {g(bb)} > {g(spec['objective']['tolerance'])}), so it is not released.")
        elif key == "03b":
            dom, D, V = spec["deterministic"]["domain"], spec["deterministic"]["D"], spec["deterministic"]["V"]
            w = dom["x1"][1] - dom["x1"][0]
            vb = math.sqrt(2 * D * w)
            pk = LM.viable_initial_velocities(spec, -1.0, 1.0)
            ind = RH.first_sample_velocity_limit(spec, -1.0, 1.0)
            ts = math.ceil(RH.lower_limit_two_signals(D, V)["t_star"] / Ts - 1e-9) * Ts      # first sample of the construction
            room = RH.two_signal_travel_room(V, D, ts, spec["time"]["t_end"])
            bb = best_bound(Q)
            chk.update(speed_bound_from_travel=vb, review_interval_bound=0.001 / 0.1 + D * 0.1 / 2,
                       package_inner_velocities=list(pk[:2]) if pk else None,
                       independent_inner_velocities=list(ind) if ind else None, two_signal_travel_room=room, revised_best_bound=bb)
            why = (f"|x2'| <= D = {g(D)} and the travel width is {g(w)} m for the whole window, so every admissible velocity has "
                   f"|v| <= sqrt(2 D w) = {g(vb)} (the review's interval argument gives 0.06); the estimate 0 meets the cap "
                   f"{g(spec['objective']['peak_cap'])} and the tolerance at every sample, so the claim of impossibility (which used "
                   f"[-1, 1]) is false. Revised: inner set of admissible initial velocities [{g(pk[0])}, {g(pk[1])}] (independent "
                   f"[{g(ind[0])}, {g(ind[1])}]), half-width below the cap; the two-signal construction needs travel "
                   f"+-{g(room)} m for V = {g(V)} and the travel is +-{g(dom['x1'][1])} m, so no claim. The observer's best bound "
                   f"{g(bb)} > {g(spec['objective']['tolerance'])}, so it is not released.")
        rows.append({"case": name.replace(".json", ""), "revision2": outcome(P), "revised": outcome(Q),
                     "revision2_disposition": P["decision"]["disposition"], "revised_disposition": Q["decision"]["disposition"],
                     "revision2_matches_review_record": P.get("same_outcome_as_review_record"),
                     "justification": why, "checks": chk})
    (HERE / "results" / "reviewer_cases_table.json").write_text(json.dumps(rows, indent=1) + "\n")
    lines = ["# Review cases of 18 November 2026: revision 2 and revised outcomes", "",
             "Generated by experiments/reviewer_cases_compare.py from results/reviewer_cases_previous.json (tag revision-2-frozen, "
             "the software of the delivered revision 2 archive) and results/reviewer_cases_current.json (revised software). "
             "Each case file's `arguments` object was passed unchanged to heros.api.solve.", "",
             "| Case | Revision 2 outcome | Revised outcome | Mathematical justification |", "|---|---|---|---|"]
    for r in rows:
        cell = [x.replace("|", "\\|") for x in (r["revision2"], r["revised"], r["justification"])]
        lines.append(f"| {r['case'][:3].rstrip('_')} | {cell[0]} | {cell[1]} | {cell[2]} |")
    lines += ["", "Checks by the judge (heros/evaluation.py) and the independent references (independent/ref_hgo.py):", ""]
    for r in rows:
        c = r["checks"]
        bits = []
        if "judge_on_revision2_release" in c:
            j = c["judge_on_revision2_release"]
            bits.append(f"Revision 2 release judged {'correct' if j['correct'] else 'incorrect'} under the specification: {j['why']}")
            if j["correct"] and "rounding_allowance_s" in c:
                bits.append("the specification does not contain the sample instants, so this check covers samples exactly at k Ts; "
                            "the record itself violates the zero bound (error above), which is the refutation")
        if "judge_on_revision2_impossibility" in c:
            j = c["judge_on_revision2_impossibility"]
            bits.append(f"Revision 2 impossibility claim {'confirmed' if j['confirmed'] else 'refuted' if j['confirmed'] is False else 'not checked'}: {j['why']}")
        if not bits:
            bits.append("no claim in either revision")
        lines.append(f"- {r['case'][:3].rstrip('_')}: " + "; ".join(bits))
    lines.append("")
    (HERE / "results" / "reviewer_cases_table.md").write_text("\n".join(lines))
    for r in rows:
        print(r["case"], "|", r["revision2"], "|", r["revised"])
        print("   ", json.dumps(r["checks"])[:700])


if __name__ == "__main__":
    main()
