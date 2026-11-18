"""Outcome changes between two stored acceptance-test runs (row by row).

    python experiments/compare_runs.py results/before_change results/run2_structured_reconstructed out.json

Rows are matched on (id, policy, reader, user, author); a row counts as changed if its outcome, disposition or
selected method differs.  Added after recheck B (finding N2) so that the history of the runs, and the effect of each
round of corrections, can be stated from the rows.  The stored rows are not modified.
"""
import json, sys, pathlib, collections
from common import ROOT

a_dir, b_dir, out = (pathlib.Path(x).resolve() for x in sys.argv[1:4])
key = lambda r: (r["id"], r.get("policy"), r.get("reader"), r.get("user"), r.get("author"))
report = {"a": str(a_dir.relative_to(ROOT)), "b": str(b_dir.relative_to(ROOT)), "files": {}}
for name in ("test_structured_rows.jsonl", "test_nl_rows.jsonl"):
    pa, pb = a_dir / name, b_dir / name
    if not (pa.exists() and pb.exists()):
        continue
    A = {key(r): r for r in map(json.loads, open(pa))}
    B = {key(r): r for r in map(json.loads, open(pb))}
    common = sorted(set(A) & set(B), key=str)
    changed = []
    for k in common:
        ra, rb = A[k], B[k]
        if any(ra.get(f) != rb.get(f) for f in ("outcome", "disposition", "selected")):
            changed.append({"id": k[0], "policy": k[1], "reader": k[2], "user": k[3], "author": k[4],
                            "before": [ra.get("outcome"), ra.get("disposition"), ra.get("selected")],
                            "after": [rb.get("outcome"), rb.get("disposition"), rb.get("selected")], "why_after": (rb.get("why") or "")[:200]})
    summary = collections.Counter((c["policy"], c["reader"], c["user"], c["before"][0], c["after"][0]) for c in changed)
    report["files"][name] = {"rows_a": len(A), "rows_b": len(B), "matched": len(common), "only_a": len(set(A) - set(B)), "only_b": len(set(B) - set(A)),
                             "changed": len(changed), "summary": [{"policy": k[0], "reader": k[1], "user": k[2], "before": k[3], "after": k[4], "n": v}
                                                                   for k, v in sorted(summary.items(), key=str)],
                             "rows": changed}
    print(f"{name}: {len(common)} matched rows ({len(set(A) - set(B))} only in a, {len(set(B) - set(A))} only in b), {len(changed)} changed")
    for k, v in sorted(summary.items(), key=str):
        print("  ", v, k)
out.write_text(json.dumps(report, indent=1))
print("wrote", out.relative_to(ROOT))
