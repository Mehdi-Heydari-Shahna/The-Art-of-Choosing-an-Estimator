"""Re-score the PROVED_IMPOSSIBLE outcomes of stored acceptance-test rows with the current independent check.

    python experiments/rescore_impossibility.py results/run3

Added after recheck B (finding N1): the judge compared the result of its impossibility check with `is False`, which
misses numpy.False_, so most false impossibility claims were counted as refusals; and an impossibility claim for a
request that asks for the smallest error variance had no check.  The check uses only the hidden complete
specification, so stored rows can be re-scored without re-running the dialogues.  Writes <dir>/rescored/ with the
re-scored rows and a list of the rows whose outcome changed; the stored rows are not modified.
"""
import json, sys, pathlib, collections
from common import ROOT, load_jsonl
from heros import evaluation as EV
from heros.core.assess import jsonable

src = pathlib.Path(sys.argv[1]).resolve()
cases = {c["id"]: c for c in load_jsonl(ROOT / "data" / "test_structured.jsonl")}
out_dir = src / "rescored"
out_dir.mkdir(exist_ok=True)
report = {"source": str(src.relative_to(ROOT)), "changed": [], "counts": {}}
for name in ("test_structured_rows.jsonl", "test_nl_rows.jsonl"):
    p = src / name
    if not p.exists():
        continue
    rows, cnt = [], collections.Counter()
    for r in load_jsonl(p):
        if r.get("disposition") == "PROVED_IMPOSSIBLE":
            ok, why = EV._verify_impossibility(cases[r["id"]]["truth"], {})
            new = dict(r, impossibility_verified=None if ok is None else bool(ok))
            if ok is not None and not ok:
                new.update(outcome="false_impossibility", why=why)
            if new["outcome"] != r["outcome"]:
                report["changed"].append({"file": name, "id": r["id"], "policy": r.get("policy"), "reader": r.get("reader"), "user": r.get("user"),
                                          "author": r.get("author"), "before": r["outcome"], "after": new["outcome"], "why": new.get("why")})
            cnt[(r.get("reader") or "-", r.get("user") or "-", r.get("policy"), new["outcome"])] += 1
            r = new
        rows.append(r)
    with open(out_dir / name, "w") as f:
        for r in rows:
            f.write(json.dumps(jsonable(r)) + "\n")
    report["counts"][name] = {"|".join(k): v for k, v in sorted(cnt.items())}
(out_dir / "rescore_report.json").write_text(json.dumps(report, indent=1))
by = collections.Counter((c["file"], c.get("reader"), c.get("user"), c["policy"], c["before"], c["after"]) for c in report["changed"])
for k, v in sorted(by.items(), key=str):
    print(v, k)
print("rows changed:", len(report["changed"]), "->", out_dir.relative_to(ROOT))
