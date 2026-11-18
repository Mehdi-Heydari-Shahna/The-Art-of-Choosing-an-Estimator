"""Tabulate the outcomes of the review findings on the previous and the current revision.

    python experiments/review_findings_compare.py        # prints a Markdown table; writes results/review_findings_table.md
Reads results/review_findings_previous.json and results/review_findings_current.json (experiments/review_findings.py).
"""
import json
from common import RES


def outcome(res):
    if "first_line" in res:                     # command-line case (F1)
        first = res["first_line"].replace("Disposition: ", "")
        est = res.get("estimates_line") or "no estimates"
        return f"{first}; {est.split(';')[0]}"
    d = res.get("disposition", "")
    parts = [d]
    for k, fmt in (("claim_value", "claim {:.6g}"), ("exported_P_filt_first", "P(0|0) exported {:.6g}"), ("exported_envelope_k0", "envelope at t=0 {:.6g}")):
        if res.get(k) is not None:
            parts.append(fmt.format(float(res[k])))
    return ", ".join(parts)


def main():
    a = json.loads((RES / "review_findings_previous.json").read_text())
    b = json.loads((RES / "review_findings_current.json").read_text())
    rows = ["| Finding | Case | Previous revision (" + a["code_commit"] + ") | Current revision (" + b["code_commit"] + ") |", "|---|---|---|---|"]
    for fa, fb in zip(a["findings"], b["findings"]):
        if fa["id"] == "F7":
            rows.append(f"| F7 | the 20 test requests with a cap below the tolerance | {fa['counts']} | {fb['counts']} |")
            continue
        for ca, cb in zip(fa["cases"], fb["cases"]):
            rows.append(f"| {fa['id']} | {ca['case']} | {outcome(ca['result'])} | {outcome(cb['result'])} |")
    text = "\n".join(rows)
    (RES / "review_findings_table.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
