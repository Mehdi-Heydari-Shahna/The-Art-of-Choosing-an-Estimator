"""Run the validation requests V01 to V12 (validation/cases.json; expected answers in validation/expected.json, both first
committed in commit cb8d3e3, tagged validation-frozen and, for the software of revision 2, revision-2-frozen) through the
public interface of the software in this tree and compare.

    python experiments/run_validation.py        # results/validation.json and paper/sections/validation_table.tex

The software under test is identified by the digest of heros/ and independent/ (and by git when available); it is named
after RELEASE.json only when that digest matches. The first run, on revision 2, is kept unchanged in
results/validation_first_run_revision2.json, and its replay on the tag revision-2-frozen in
results/validation_replay_revision2_tag.json; the table states both counts.

Each request goes through heros.api.solve exactly as a user would send it (structured fields count as confirmed; a
record with timestamps is passed with them).  For a release, the returned claim is also re-evaluated with the formulas
of validation/expected_answers.py (no package code): the scalar Kalman recursion, or the term-by-term worst-case sum
of the released observer gain.
"""
import hashlib, json, math, subprocess, sys, pathlib
import numpy as np
from common import ROOT, save
from heros import api

sys.path.insert(0, str(ROOT / "validation"))
import expected_answers as XA          # noqa: E402  (independent formulas; imports numpy only)

ANSWER = {"RELEASE": "release", "PROVED_IMPOSSIBLE": "decline: impossible", "NO_VERIFIED_CANDIDATE": "decline: no verified design"}


def returned_answer(dec):
    d = dec["disposition"]
    if d == "CLARIFY":
        return "ask: " + dec["question"]["path"]
    if d in ("INVALID_INPUT", "OUT_OF_SCOPE") and any(i.get("code") in ("timing_conflict", "irregular_sampling") for i in dec.get("issues") or []):
        return "decline: data contradict the request"
    return ANSWER.get(d, "decline: " + d.lower())


def code_digest(root):
    h = hashlib.sha256()
    for p in sorted((root / "heros").rglob("*.py")) + sorted((root / "independent").rglob("*.py")):
        h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def software_identity():
    """Digest of the code under test, git commit and tags when available, and the release name if RELEASE.json matches."""
    ident = {"code_digest": code_digest(ROOT)}
    try:
        g = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=20)
        r = g("rev-parse", "HEAD")
        if r.returncode == 0:
            ident["git_commit"] = r.stdout.strip()
            ident["git_tags_at_commit"] = g("tag", "--points-at", "HEAD").stdout.split()
    except (OSError, subprocess.SubprocessError):
        pass
    rel = ROOT / "RELEASE.json"
    if rel.exists():
        info = json.loads(rel.read_text())
        if info.get("code_digest") == ident["code_digest"]:
            ident.update({k: info[k] for k in ("release", "release_tag", "software_tag", "software_commit") if k in info})
    return ident


def main():
    cases = json.loads((ROOT / "validation" / "cases.json").read_text())["cases"]
    exp = json.loads((ROOT / "validation" / "expected.json").read_text())["expected"]
    rows = []
    for c in cases:
        sp = dict(c["spec"])
        plant, obj = sp.pop("plant"), sp.pop("objective")
        spec = api.new_spec(plant, objective=obj, **sp)
        data = c.get("data")
        res = api.solve(spec, None if not data else np.array(data["y"]), None if not data else np.array(data["u"]),
                        t=None if not data else np.array(data["t"]))
        dec = res["decision"]
        got = returned_answer(dec)
        e = exp[c["id"]]
        row = {"id": c["id"], "request": c["request"], "expected": e["answer"], "expected_value": e["value"], "returned": got,
               "display": dec.get("display"), "agree": got == e["answer"], "reason": e["reason"]}
        if dec["disposition"] == "RELEASE":
            cl = dec["claim"]; row["claim_value"] = cl.get("value"); row["selected"] = dec["selected"]
            if dec["selected"] == "high_gain_observer":
                cfg = dec["config"]
                indep = XA.observer_ultimate(cfg["Ts"], cfg["eps"], cfg["D"], cfg["V"])
                row["independent_value"] = indep
                row["value_check"] = bool(abs(indep - cl["value"]) <= 1e-9 * indep and cl["value"] <= e["value"])
            else:
                row["independent_value"] = e["value"]
                row["value_check"] = bool(abs(cl["value"] - e["value"]) <= 1e-6 * e["value"])
            row["agree"] = row["agree"] and row["value_check"]
        elif dec.get("impossibility"):
            row["claimed_limit"] = dec["impossibility"].get("bound")
            row["limit_check"] = bool(abs(row["claimed_limit"] - e["value"]) <= 1e-9 * max(1.0, abs(e["value"])))
            row["agree"] = row["agree"] and row["limit_check"]
        rows.append(row)
        print(f"{c['id']}  expected {e['answer']:38s} returned {got:38s} {'agree' if row['agree'] else 'DIFFERENT'}")
    ident = software_identity()
    first = json.loads((ROOT / "results" / "validation_first_run_revision2.json").read_text())
    out = {"n": len(rows), "agree": sum(r["agree"] for r in rows), "rows": rows, "software": ident,
           "expected_answers": {"file": "validation/expected.json",
                                "sha256": hashlib.sha256((ROOT / "validation" / "expected.json").read_bytes()).hexdigest(),
                                "first_committed": "cb8d3e3 (tag validation-frozen), before the first run"},
           "first_run": {"file": "results/validation_first_run_revision2.json", "software_tag": "revision-2-frozen",
                         "created_utc": first.get("_meta", {}).get("created_utc"), "n": first["n"], "agree": first["agree"],
                         "same_answers_as_this_run": all(x["returned"] == y["returned"] for x, y in zip(first["rows"], rows))}}
    save("validation.json", out)
    write_table(rows, out)
    print(f"{out['agree']} of {out['n']} agree")


def tex(s):
    return (s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")
            .replace("<=", "$\\le$").replace(">=", "$\\ge$"))


SHORT = {
    "V01": ("Random-walk level, finite record; smallest mean-square error", "release", "Kalman recursion from $P_0$; largest $P_{k|k}$ at $t=0$"),
    "V02": ("AR(1) level; steady-state MSE $\\le 0.5$", "release", "scalar Riccati equation gives $0.375$"),
    "V03": ("Axis, $D=4$, $V=10^{-4}$; speed error eventually $\\le 0.06$", "release", "exact worst case of one observer gain below $0.06$"),
    "V04": ("Axis, total $D=6$ and additive $d=1.5$; $\\le 0.045$", "release", "exact model: the additive bound applies"),
    "V05": ("Axis, guaranteed limit; encoder bound missing", "ask $V$", "every gain passes $v_k$ with nonzero weight"),
    "V06": ("Level gauge, smallest MSE; noise variance missing", "ask $R$", "gain $P/(P+R)$ depends on $R$"),
    "V07": ("Axis, limit after $20$ ms; initial error missing", "ask $E$", "initial error still weighted at the deadline"),
    "V08": ("Axis, speed error eventually $\\le 0.035$", "impossible", "two-signal limit $2\\sqrt{DV}=0.04$"),
    "V09": ("Axis, never above $0.5$; initial speed $\\pm 2$", "impossible", "one reading carries no speed information"),
    "V10": ("AR(1) level; steady-state MSE $\\le 0.3$", "impossible", "optimum $0.375$ over all estimators"),
    "V11": ("Axis, $|e|\\le 0.02$ with probability $0.9$", "no verified design", "Chebyshev proves $0.84$; no impossibility argument"),
    "V12": ("Request at $1$ kHz, record every $2$ ms", "data conflict", "certificate is for one period"),
}
RET = {"release": "release", "ask: deterministic.V": "ask $V$", "ask: stochastic.R": "ask $R$", "ask: initial.E": "ask $E$",
       "decline: impossible": "impossible", "decline: no verified design": "no verified design", "decline: data contradict the request": "data conflict"}


def write_table(rows, out):
    sw, fr = out["software"], out["first_run"]
    now = sw.get("release", "the software of this tree")
    now_tag = f" (tag {sw['software_tag']})" if sw.get("software_tag") else ""
    L = ["% generated by experiments/run_validation.py from results/validation.json -- do not edit",
         "\\begin{table*}[tp]",
         f"\\caption{{Validation requests V01 to V12, first run on revision 2 and rerun on {now}{now_tag} through the public interface.}}",
         "\\label{tab:validation}", "\\scriptsize\\centering", "\\setlength{\\tabcolsep}{3.5pt}",
         "\\begin{tabularx}{\\textwidth}{@{}l L l l L@{}}", "\\toprule",
         " & Request & Expected & Returned & Mathematical reason \\\\", "\\midrule"]
    for r in rows:
        req, expd, why = SHORT[r["id"]]
        ret = RET.get(r["returned"], tex(r["returned"]))
        L.append(f"{r['id']} & {req} & {expd} & {ret} & {why} \\\\")
    L += ["\\bottomrule", "\\end{tabularx}",
          f"\\tablenote{{The expected answers were derived without the software and committed together with the revision 2 software "
          f"(commit cb8d3e3, tags validation-frozen and revision-2-frozen) before the first run, in which {fr['agree']} of {fr['n']} "
          f"agreed; on {now}{now_tag}, {out['agree']} of {out['n']} agree"
          + (", with the same answers" if fr["same_answers_as_this_run"] else "") +
          ". For each release the returned bound was recomputed with the same independent formulas.}", "\\end{table*}"]
    (ROOT / "paper" / "sections" / "validation_table.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
