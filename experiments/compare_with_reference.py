"""Compare a fresh acceptance-test run (results/) with the reported run kept in results/reference/ (the earlier runs, made
before the review corrections, are kept separately; see docs/PROTOCOL.md)."""
import json, sys
from common import RES

REF = RES / "reference"
ok = True
QUICK = "--quick" in sys.argv
for name in ("test_structured_rows.jsonl", "test_nl_rows.jsonl"):
    a, b = RES / (name.replace(".jsonl", "_quick.jsonl") if QUICK else name), REF / name
    if not (a.exists() and b.exists()):
        print(f"{name}: FAILED - missing {'re-run rows' if not a.exists() else 'reference rows'} ({a if not a.exists() else b})")
        ok = False
        continue
    key = lambda r: (r["id"], r.get("policy"), r.get("reader"), r.get("user"), r.get("author"))
    ra = {key(r): r for r in map(json.loads, open(a))}
    rb = {key(r): r for r in map(json.loads, open(b))}
    common = [k for k in rb if k in ra]
    diff = [k for k in common if any(ra[k].get(f) != rb[k].get(f) for f in ("outcome", "disposition", "selected", "n_questions", "edits"))]
    print(f"{name}: {len(rb)} reference rows, {len(common)} re-run, {len(diff)} with a different outcome, disposition, method, "
          f"question count or correction count")
    for k in diff[:10]:
        print("   ", k, "->", {f: (rb[k].get(f), ra.get(k, {}).get(f)) for f in ("outcome", "disposition", "selected")})
    ok &= not diff
sys.exit(0 if ok else 1)
