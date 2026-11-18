"""Post-hoc, descriptive (not part of the frozen protocol): where the fixed-priority selector and the coordinator
release different methods on the structured test set, how do the two designs compare on the requested criterion?

    python experiments/criterion_effect.py      -> results/criterion_effect.json
Both designs are released correctly in these cases (judged by the independent references); the comparison uses the
certified criterion values that the coordinator computes for every qualifying method on the complete specification."""
import json
from common import ROOT, RES, save, load_jsonl
from heros.coordinator.coordinator import decide
from heros.core import spec as S

rows = load_jsonl(RES / "test_structured_rows.jsonl")
cases = {c["id"]: c for c in load_jsonl(ROOT / "data" / "test_structured.jsonl")}
co = {r["id"]: r for r in rows if r["policy"] == "coordinator"}
se = {r["id"]: r for r in rows if r["policy"] == "selector"}
out = []
for i in sorted(co):
    if co[i].get("selected") and se[i].get("selected") and co[i]["selected"] != se[i]["selected"]:
        d = decide(cases[i]["truth"])
        cv = (d.get("tradeoff") or {}).get("values") or {}
        out.append({"id": i, "category": co[i]["category"], "objective": S.get(cases[i]["truth"], "objective.kind"),
                    "coordinator": co[i]["selected"], "selector": se[i]["selected"], "criterion_values": cv,
                    "ratio_selector_over_coordinator": (cv[se[i]["selected"]] / cv[co[i]["selected"]]) if cv.get(co[i]["selected"]) else None})
save("criterion_effect.json", {"n_cases": len(co), "n_different_choice": len(out), "cases": out,
                                "note": "post-hoc descriptive analysis (not in the frozen protocol); for a probability objective the value is one minus the guaranteed probability level (smaller is better)"})
for o in out:
    print(o["id"], o["category"], o["coordinator"], o["selector"], o["criterion_values"], o["ratio_selector_over_coordinator"])
