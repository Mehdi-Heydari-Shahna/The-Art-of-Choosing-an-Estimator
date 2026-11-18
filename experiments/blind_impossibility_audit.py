"""Post-hoc audit of the false impossibility claims made without review (the reported run, trained reader, coordinator,
blind user; run 6: results/run6/blind_impossibility_audit.json).

    python experiments/blind_impossibility_audit.py

Each such row is replayed with the same reader, request and scripted user; the replayed disposition must equal the stored
one.  For each claim the script records the argument used and the material fields whose reading differs from the hidden
complete specification.  Writes results/blind_impossibility_audit.json.  Nothing here changes how the test is scored.
"""
import json, collections
from common import ROOT, load_jsonl
import evaluate as E
import nl_common as NC
from heros import evaluation as EV
from heros.core import spec as S
from heros.core.assess import jsonable

PATHS = ["objective.kind", "objective.tolerance", "objective.deadline", "objective.peak_cap", "objective.prob", "objective.mse_target",
         "objective.horizon", "initial.E", "deterministic.V", "deterministic.D", "time.Ts", "time.t_end"]


def _same(a, b):
    return json.dumps(jsonable(a), sort_keys=True) == json.dumps(jsonable(b), sort_keys=True)


def main():
    rows = list(load_jsonl(ROOT / "results" / "test_nl_rows.jsonl"))
    targets = [r for r in rows if r["outcome"] == "false_impossibility" and r["policy"] == "coordinator"
               and r["user"] == "blind" and r["reader"].startswith("final_")]
    items = {(c["id"], a): (c, t) for c, t, a, g in NC.authored("test")}
    out, by_kind, by_field = [], collections.Counter(), collections.Counter()
    replay_ok = True
    for r in targets:
        case, text = items[(r["id"], r["author"])]
        o = EV.run_request(E._reader(r["reader"]), case, text, "coordinator", "blind")
        d, sp = o["decision"], o["final_spec"]
        imp = d.get("impossibility") or {}
        same = d["disposition"] == r["disposition"]
        replay_ok &= same
        diff = {p: [S.get(sp, p), S.get(case["truth"], p)] for p in PATHS if not _same(S.get(sp, p), S.get(case["truth"], p))}
        cap_misread = "objective.peak_cap" in diff and diff["objective.peak_cap"][1] is None
        tol_true = S.get(case["truth"], "objective.tolerance")
        cap_is_tolerance = bool(cap_misread and tol_true is not None
                                and abs(diff["objective.peak_cap"][0] - tol_true) <= 1e-9 * abs(tol_true))
        kind = imp.get("kind") or "optimum_or_other"
        by_kind[kind] += 1
        for p in diff:
            by_field[p] += 1
        out.append({"id": r["id"], "reader": r["reader"], "author": r["author"], "category": r["category"],
                    "stored_disposition": r["disposition"], "replayed_disposition": d["disposition"], "replay_matches": same,
                    "argument_kind": kind, "argument": imp.get("argument"), "cap_read_where_none_was_stated": cap_misread,
                    "cap_equals_stated_tolerance": cap_is_tolerance,
                    "misread_fields": diff})
    n = len(out)
    first_cap = sum(1 for x in out if x["argument_kind"] == "initial_sample" and x["cap_read_where_none_was_stated"])
    first_cap_tol = sum(1 for x in out if x["argument_kind"] == "initial_sample" and x["cap_equals_stated_tolerance"])
    res = {"note": "post-hoc audit; rows of results/test_nl_rows.jsonl (run 7, tag revision-2b-frozen), replayed with the frozen software",
           "n_false_impossibility": n, "replay_matches_all": bool(replay_ok),
           "by_argument_kind": dict(by_kind), "first_sample_with_misread_cap": first_cap,
           "first_sample_cap_equals_stated_tolerance": first_cap_tol,
           "per_reader": dict(collections.Counter(x["reader"] for x in out)),
           "misread_field_counts": dict(by_field), "rows": out}
    (ROOT / "results" / "blind_impossibility_audit.json").write_text(json.dumps(jsonable(res), indent=1))
    print("false impossibility claims:", n, "| replay matches:", replay_ok, "| first-sample on a misread cap:", first_cap,
          "| of which the stated tolerance:", first_cap_tol, "|", dict(by_kind))


if __name__ == "__main__":
    main()
