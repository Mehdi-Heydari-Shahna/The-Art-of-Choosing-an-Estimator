"""Re-score stored acceptance-test rows with the revision-2 labels and judge (the stored rows are not modified).

    python experiments/rescore_revision2.py results/run5

Two things changed in the scoring: the corrected outcome of the 20 requests with a transient cap below the later
tolerance (data/revision2_relabel.json), and the independent check of impossibility claims (a claim is refuted by a
verified design for the complete facts, and a worst-case limit never confirms a probability level).  Both use only the
hidden complete specification and the stored disposition, so stored rows can be re-scored without re-running the
dialogues.  Writes <dir>/rescored_revision2/ with the rows and a report of every changed outcome and of the re-scored
outcome counts per file, policy, reader and user.
"""
import json, sys, pathlib, collections
from common import ROOT, load_jsonl
from heros import evaluation as EV
from heros.core.assess import jsonable

NONREL = EV.NONRELEASE
src = pathlib.Path(sys.argv[1]).resolve()
cases = {c["id"]: EV.with_revisions(c) for c in load_jsonl(ROOT / "data" / "test_structured.jsonl")}
out_dir = src / "rescored_revision2"
out_dir.mkdir(exist_ok=True)
report = {"source": str(src.relative_to(ROOT)), "changed": [], "summary": {}, "totals": []}
totals = collections.Counter()
_cache = {}
for name in ("test_structured_rows.jsonl", "test_nl_rows.jsonl"):
    p = src / name
    if not p.exists():
        continue
    rows = []
    for r in load_jsonl(p):
        c = cases[r["id"]]
        new = dict(r, expected_final=c["final"]["disposition"])
        disp = r.get("disposition")
        if not r.get("released") and r["outcome"] not in ("error",):
            fin = c["final"]["disposition"]
            if fin == "RELEASE":
                new["outcome"] = "unnecessary_refusal" if disp in NONREL else "unresolved"
            else:
                new["outcome"] = "correct_nonrelease" if disp == fin else ("nonrelease_other_reason" if disp in NONREL else "unresolved")
            if disp == "PROVED_IMPOSSIBLE":
                if r["id"] not in _cache:
                    _cache[r["id"]] = EV._verify_impossibility(c["truth"], {})
                ok, why = _cache[r["id"]]
                new["impossibility_verified"] = None if ok is None else bool(ok)
                if ok is not None and not ok:
                    new["outcome"] = "false_impossibility"; new["why"] = why
        if new["outcome"] != r["outcome"]:
            report["changed"].append({"file": name, "id": r["id"], "category": r.get("category"), "policy": r.get("policy"), "reader": r.get("reader"),
                                      "user": r.get("user"), "author": r.get("author"), "disposition": disp, "before": r["outcome"], "after": new["outcome"]})
        rows.append(new)
        totals[(name, new.get("policy"), new.get("reader") or "-", new.get("user") or "-", new["outcome"])] += 1
    with open(out_dir / name, "w") as f:
        for r in rows:
            f.write(json.dumps(jsonable(r)) + "\n")
by = collections.Counter((c["file"], c.get("reader") or "-", c.get("user") or "-", c["policy"], c["category"], c["before"], c["after"]) for c in report["changed"])
report["summary"] = [{"file": k[0], "reader": k[1], "user": k[2], "policy": k[3], "category": k[4], "before": k[5], "after": k[6], "n": v}
                     for k, v in sorted(by.items(), key=str)]
report["totals"] = [{"file": k[0], "policy": k[1], "reader": k[2], "user": k[3], "outcome": k[4], "n": v} for k, v in sorted(totals.items(), key=str)]
(out_dir / "rescore_report.json").write_text(json.dumps(report, indent=1))
for s in report["summary"]:
    print(s["n"], s["file"][:13], s["reader"], s["user"], s["policy"], s["category"], s["before"], "->", s["after"])
print("rows changed:", len(report["changed"]), "->", out_dir.relative_to(ROOT))
