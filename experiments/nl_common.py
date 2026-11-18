"""Shared helpers for the natural-language evaluations (development selection and acceptance test)."""
import json, time
import numpy as np
from common import ROOT, load_jsonl
from heros import evaluation as EV
from heros.core import spec as S
from heros.nlu import assemble as AS, templates as T
from heros.nlu.quantities import find_quantities


def authored(split):
    """[(case, text, author_id, gold)] for independently authored requests of 'dev' (author 0) or 'test' (authors 1-3)."""
    out = []
    if split == "dev":
        cases = {c["id"]: c for c in load_jsonl(ROOT / "data" / "dev_extra_cases.jsonl")}
        files = [("author0", "requests_dev_author0.json", "gold_dev_author0.json")]
    else:
        cases = {c["id"]: EV.with_revisions(c) for c in load_jsonl(ROOT / "data" / "test_structured.jsonl")}
        files = [(f"author{a}", f"requests_test_author{a}.json", f"gold_test_author{a}.json") for a in (1, 2, 3)]
    for author, rf, gf in files:
        p = ROOT / "data" / "authors" / rf
        if not p.exists():
            continue
        gold = {g["id"]: g["gold"] for g in json.loads((ROOT / "data" / "authors" / gf).read_text())}
        for r in json.loads(p.read_text()):
            text = "[Q1] " + r["q1"].strip() + " [Q2] " + r["q2"].strip() + " [Q3] " + r["q3"].strip()
            if EV.nl_evaluable(cases[r["id"]]):
                out.append((cases[r["id"]], text, author, gold[r["id"]]))
    return out


def templated(split="dev"):
    return [(c, T.full_text(c["request"]), "template_" + c["request"]["style"], None) for c in load_jsonl(ROOT / "data" / f"{split}.jsonl") if EV.nl_evaluable(c)]


def gold_roles(text, gold, case=None):
    """Gold role of every parsed quantity of an authored request, by matching SI values with the fact sheet.
    Returns (roles aligned with find_quantities(text), number of sheet facts that no parsed quantity matches)."""
    qs = find_quantities(text)
    facts = [dict(f) for f in gold["facts"]]
    roles = ["other"] * len(qs)
    used = set()
    missed = 0
    for f in facts:
        hit = None
        for i, q in enumerate(qs):
            if i in used:
                continue
            same = abs(q.value - f["value"]) <= 1e-6 * abs(f["value"])
            inv = f["role"] in ("Ts", "fs") and q.value > 0 and abs(1.0 / q.value - f["value"]) <= 1e-6 * abs(f["value"]) and q.dim in ("time", "freq")
            if (same and (q.dim == f["dim"] or f["dim"].startswith("other") or (f["role"] in ("Ts", "fs") and q.dim in ("time", "freq")))) or inv:
                hit = i
                role = f["role"]
                if f["role"] in ("Ts", "fs"):
                    role = "Ts" if q.dim == "time" else "fs"
                if f["role"] in ("rms_target", "mse_target") and same:
                    role = f["role"]
                roles[i] = role
                break
        if hit is None:
            # squared / rooted statements of the same fact (variance vs standard deviation; mse vs rms)
            for i, q in enumerate(qs):
                if i in used:
                    continue
                alt = {"meas_std": ("meas_var", f["value"] ** 2), "meas_var": ("meas_std", f["value"] ** 0.5),
                       "rms_target": ("mse_target", f["value"] ** 2), "mse_target": ("rms_target", f["value"] ** 0.5)}.get(f["role"])
                if alt and abs(q.value - alt[1]) <= 1e-6 * abs(alt[1]):
                    hit = i; roles[i] = alt[0]
                    break
        if hit is None:
            missed += 1
        else:
            used.add(hit)
    return roles, missed


def reading_metrics(reader, items, with_dialogue=True):
    """Role accuracy, label accuracies, exact reading (attentive review needs no edit) and latency over a list of items."""
    n_q = n_q_ok = n_lab = 0
    lab_ok = {"objective": 0, "distribution": 0, "horizon": 0, "evidence": 0, "flags": 0}
    exact = 0; edits = []; lat = []; missed = 0; per_role = {}
    for case, text, author, gold in items:
        t0 = time.perf_counter()
        rd = reader.read(text, case["given"]["plant"], EV.stats_in_attachment(case["family"]))
        lat.append(time.perf_counter() - t0)
        qs = find_quantities(text)
        if gold is None:
            ex = EV.example_from_request(case, case["request"])
            groles = ex["roles"]
            glab = case["request"]["labels"]
            glab = {"objective": glab["objective"], "distribution": glab["distribution"] or "unstated", "horizon": glab["horizon"], "evidence": glab["evidence"], "flags": glab["flags"]}
        else:
            groles, m = gold_roles(text, gold)
            missed += m
            glab = None
        pred = {(q["start"], q["end"]): q["role"] for q in rd["quantities"]}
        for q, g in zip(qs, groles):
            p = pred.get((q.start, q.end), "other")
            n_q += 1; n_q_ok += int(p == g)
            d = per_role.setdefault(g, [0, 0]); d[0] += 1; d[1] += int(p == g)
        if glab is not None:
            n_lab += 1
            for k in ("objective", "distribution", "horizon", "evidence"):
                lab_ok[k] += int(rd.get(k) == glab[k])
            lab_ok["flags"] += int(sorted(rd.get("flags") or []) == sorted(glab["flags"]))
        if with_dialogue:
            att = AS.attachment_of(case["given"], EV.stats_in_attachment(case["family"]))
            asm = AS.assemble(att, text, rd)
            _, e = EV.ScriptedUser(case["truth"], "attentive").review(asm["spec"], EV.nl_stated(case))
            edits.append(e); exact += int(e == 0)
    out = {"n_requests": len(items), "n_quantities": n_q, "role_accuracy": n_q_ok / max(1, n_q), "exact_reading": exact / max(1, len(items)),
           "mean_edits": float(np.mean(edits)) if edits else None, "ms_per_request": 1000 * float(np.mean(lat)), "sheet_facts_not_parsed": missed,
           "per_role": {k: {"n": v[0], "acc": v[1] / v[0]} for k, v in sorted(per_role.items())}}
    if n_lab:
        out["label_accuracy"] = {k: v / n_lab for k, v in lab_ok.items()}
    return out
