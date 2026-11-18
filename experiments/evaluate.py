"""Acceptance test (run once, after docs/PROTOCOL.md was frozen).

    python experiments/evaluate.py structured      # 600 structured specifications x policies
    python experiments/evaluate.py nl              # independently authored requests x readers x user modes
    python experiments/evaluate.py reading         # reading layer alone on the authored test requests (per reader)
    python experiments/evaluate.py summarise       # tables from the raw rows
Raw outcomes: results/test_structured_rows.jsonl, results/test_nl_rows.jsonl (one row per request and configuration).
"""
import json, math, sys, time, collections
from multiprocessing import Pool
import numpy as np
from common import ROOT, RES, save, load_jsonl
from heros import evaluation as EV
from heros.core.assess import jsonable

STRUCT_POLICIES = ["coordinator", "kalman_only", "khalil_only", "selector", "ask_all", "default_fill"]
QUICK = "--quick" in sys.argv          # reproduction check: coordinator only (structured) and the deployed reader (natural language)
SUFFIX = "_quick" if QUICK else ""
HELD_OUT = {"gantry_axis", "duffing_stage", "hydraulic_boom", "cart_pole"}


def _row(case, out, extra):
    j = EV.judge(case, out)
    j.update(id=case["id"], family=case["family"], category=case["category"], held_out=case["family"] in HELD_OUT,
             expected_first=case["expected"]["disposition"], seconds=out["seconds"], questions=out["questions"], **extra)
    j.pop("checks", None)
    return j


def _struct_job(args):
    policy, cases = args
    rows = []
    for c in cases:
        try:
            if policy == "coordinator_limited_user":
                # sensitivity variant (added after review B4): the user cannot give an exact value for any field of which the
                # request stated only a range
                out = EV.run_dialogue(c["given"], c["truth"], "coordinator", unknown_paths=set((c["given"].get("ranges") or {}).keys()))
            else:
                out = EV.run_dialogue(c["given"], c["truth"], policy)
            rows.append(_row(c, out, {"policy": policy, "n_answers": out.get("n_answers"), "n_unknown": out.get("n_unknown")}))
        except Exception as ex:  # noqa  (kept as a failure row; never dropped)
            rows.append({"id": c["id"], "family": c["family"], "category": c["category"], "policy": policy, "outcome": "error", "why": f"{type(ex).__name__}: {ex}",
                         "held_out": c["family"] in HELD_OUT, "expected_final": c["final"]["disposition"], "released": False, "n_questions": 0, "edits": 0, "seconds": 0.0})
    return rows


def run_structured():
    cases = [EV.with_revisions(c) for c in load_jsonl(ROOT / "data" / "test_structured.jsonl")]
    if QUICK:
        half = len(cases) // 2
        jobs = [("coordinator", cases[:half]), ("coordinator", cases[half:])]
    else:
        jobs = [(p, cases) for p in STRUCT_POLICIES + ["coordinator_limited_user"]]
    t0 = time.time()
    with Pool(2) as pool:
        res = pool.map(_struct_job, jobs, chunksize=1)
    with open(RES / f"test_structured_rows{SUFFIX}.jsonl", "w") as f:
        for rows in res:
            for r in rows:
                f.write(json.dumps(jsonable(r)) + "\n")
    print("structured done in", round(time.time() - t0), "s")


_READERS = {}


def _reader(name):
    if name not in _READERS:
        from heros.nlu import readers as RD
        import eval_dev as ED
        if name == "null":
            _READERS[name] = RD.NullReader()
        elif name == "rules":
            _READERS[name] = RD.RuleReader()
        elif name.startswith("final_") and (ROOT / "models" / name / "reader_numpy.npz").exists():
            # NumPy inference of the same checkpoint (identical readings: results/numpy_reader_check.json); no PyTorch needed
            from heros.nlu.numpy_reader import NumpyGRUReader
            _READERS[name] = NumpyGRUReader(ROOT / "models" / name)
        else:
            _READERS[name] = ED.load(name)[0]
    return _READERS[name]


def _nl_job(args):
    name, mode, policy, items = args
    reader = _reader(name)
    rows = []
    for case, text, author in items:
        try:
            out = EV.run_request(reader, case, text, policy, mode)
            out["seconds"] = out["seconds"] + out["seconds_read"]          # time per request = reading + dialogue (judge excluded)
            rows.append(_row(case, out, {"reader": name, "user": mode, "policy": policy, "author": author, "seconds_read": out["seconds_read"],
                                         "proposal": out["reading"].get("proposal"), "n_issues": len(out["issues"]),
                                         "n_answers": out.get("n_answers"), "n_unknown": out.get("n_unknown")}))
        except Exception as ex:  # noqa
            rows.append({"id": case["id"], "family": case["family"], "category": case["category"], "reader": name, "user": mode, "policy": policy, "author": author,
                         "outcome": "error", "why": f"{type(ex).__name__}: {ex}", "held_out": case["family"] in HELD_OUT, "expected_final": case["final"]["disposition"],
                         "released": False, "n_questions": 0, "edits": 0, "seconds": 0.0})
    return rows


def run_nl():
    import nl_common as NC
    sel = json.loads((ROOT / "models" / "selection.json").read_text())
    arch = sel["selected_architecture"]
    items = [(c, t, a) for c, t, a, g in NC.authored("test")]
    print("evaluable authored test requests:", len(items))
    finals = [f"final_{arch}_s{s}" for s in (0, 1, 2)]
    jobs = []
    if QUICK:
        half = len(items) // 2
        jobs = [(finals[0], "attentive", "coordinator", items[:half]), (finals[0], "attentive", "coordinator", items[half:])]
    for name in ([] if QUICK else ["null", "rules", "select_neural_frozen_s0"] + finals):
        for mode in ("attentive", "blind"):
            jobs.append((name, mode, "coordinator", items))
    if not QUICK:
        for mode in ("attentive", "blind"):
            jobs.append((finals[0], mode, "single_model", items))
        jobs.append((finals[0], "attentive", "ask_all", items))
    t0 = time.time()
    with Pool(2) as pool:
        res = pool.map(_nl_job, jobs, chunksize=1)
    with open(RES / f"test_nl_rows{SUFFIX}.jsonl", "w") as f:
        for rows in res:
            for r in rows:
                f.write(json.dumps(jsonable(r)) + "\n")
    print("nl done in", round(time.time() - t0), "s")


def run_reading():
    """Layer evaluation: the reading alone (roles of quantities, exact reading, proposal of the outcome) on the authored test requests."""
    import nl_common as NC
    sel = json.loads((ROOT / "models" / "selection.json").read_text())
    arch = sel["selected_architecture"]
    items = NC.authored("test")
    names = ["null", "rules", "select_neural_frozen_s0"] + [f"final_{arch}_s{s}" for s in (0, 1, 2)]
    out = {}
    for name in names:
        rd = _reader(name)
        m = NC.reading_metrics(rd, items)
        by_author = {}
        for a in sorted({i[2] for i in items}):
            ma = NC.reading_metrics(rd, [i for i in items if i[2] == a])
            by_author[a] = {k: ma[k] for k in ("n_requests", "role_accuracy", "exact_reading", "mean_edits")}
        held = NC.reading_metrics(rd, [i for i in items if i[0]["family"] in HELD_OUT])
        m["by_author"] = by_author
        m["held_out_families"] = {k: held[k] for k in ("n_requests", "role_accuracy", "exact_reading", "mean_edits")}
        # proposal head: does the reader's proposed first outcome equal the coordinator's first outcome on the correct structured specification?
        ok = n = 0
        for case, text, author, gold in items:
            r = rd.read(text, case["given"]["plant"], EV.stats_in_attachment(case["family"]))
            if r.get("proposal") is None:
                continue
            e = case["expected"]
            want = ("RELEASE:" + (e["selected"] or "")) if e["disposition"] == "RELEASE" else (("CLARIFY:" + e["ask"]) if e["disposition"] == "CLARIFY" else e["disposition"])
            n += 1
            ok += int(r["proposal"] == want or (e["disposition"] == "RELEASE" and e["selected"] is None and r["proposal"].startswith("RELEASE")))
        m["proposal_accuracy"] = (ok / n) if n else None
        m["proposal_n"] = n
        out[name] = m
        print(f"{name:28s} roles {m['role_accuracy']:.3f} exact {m['exact_reading']:.3f} edits {m['mean_edits']:.2f} proposal {m['proposal_accuracy']}")
    save("test_reading.json", out)


# ---------------------------------------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def cluster_boot(rows, fn, key, B=2000, seed=0):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(names), len(names))
        sample = [r for i in pick for r in groups[names[i]]]
        v = fn(sample)
        if v is not None:
            vals.append(v)
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] if vals else [None, None]


def metrics(rows, cluster_key):
    n = len(rows)
    cnt = collections.Counter(r["outcome"] for r in rows)
    rel = sum(1 for r in rows if r.get("released"))
    inc = cnt["incorrect_release"]
    releasable = sum(1 for r in rows if r["expected_final"] == "RELEASE")
    cov = sum(1 for r in rows if r["outcome"] == "correct_release" and r["expected_final"] == "RELEASE")
    nonrel = n - releasable
    m = {"n": n, "outcomes": dict(cnt), "released": rel, "incorrect_releases": inc,
         "incorrect_per_request": inc / n, "incorrect_per_request_ci": wilson(inc, n), "incorrect_per_released": (inc / rel) if rel else None,
         "incorrect_per_released_ci": wilson(inc, rel) if rel else [None, None],
         "n_releasable": releasable, "useful_coverage": cov / releasable if releasable else None, "useful_coverage_ci": wilson(cov, releasable),
         "unnecessary_refusals": cnt["unnecessary_refusal"], "unresolved": cnt["unresolved"], "errors": cnt["error"],
         "n_not_releasable": nonrel, "correct_nonrelease": cnt["correct_nonrelease"], "nonrelease_other_reason": cnt["nonrelease_other_reason"],
         "questions_per_request": float(np.mean([r["n_questions"] for r in rows])), "edits_per_request": float(np.mean([r.get("edits", 0) for r in rows])),
         "answers_per_request": float(np.mean([r.get("n_answers") or 0 for r in rows])), "unknown_answers_per_request": float(np.mean([r.get("n_unknown") or 0 for r in rows])),
         "false_impossibility": cnt["false_impossibility"],
         "impossibility_unverified": sum(1 for r in rows if r.get("disposition") == "PROVED_IMPOSSIBLE" and r.get("impossibility_verified") is None),
         "seconds_per_request_median": float(np.median([r["seconds"] for r in rows])), "seconds_per_request_p95": float(np.quantile([r["seconds"] for r in rows], 0.95))}
    m["incorrect_per_request_cluster_ci"] = cluster_boot(rows, lambda s: sum(1 for r in s if r["outcome"] == "incorrect_release") / len(s), cluster_key)
    m["useful_coverage_cluster_ci"] = cluster_boot(rows, lambda s: (sum(1 for r in s if r["outcome"] == "correct_release" and r["expected_final"] == "RELEASE") /
                                                                    max(1, sum(1 for r in s if r["expected_final"] == "RELEASE"))) if any(r["expected_final"] == "RELEASE" for r in s) else None, cluster_key)
    return m


def summarise():
    out = {}
    p = RES / "test_structured_rows.jsonl"
    if p.exists():
        rows = load_jsonl(p)
        S_ = {}
        for pol in STRUCT_POLICIES + ["coordinator_limited_user"]:
            rs = [r for r in rows if r["policy"] == pol]
            if not rs:
                continue
            S_[pol] = metrics(rs, lambda r: r["family"])
            S_[pol]["held_out_families"] = metrics([r for r in rs if r["held_out"]], lambda r: r["family"])
            S_[pol]["training_families"] = metrics([r for r in rs if not r["held_out"]], lambda r: r["family"])
            S_[pol]["by_family"] = {f: {k: v for k, v in metrics([r for r in rs if r["family"] == f], lambda r: r["id"]).items() if k in ("n", "outcomes", "incorrect_releases", "useful_coverage", "questions_per_request")}
                                    for f in sorted({r["family"] for r in rs})}
            S_[pol]["by_category"] = {c: dict(collections.Counter(r["outcome"] for r in rs if r["category"] == c)) for c in sorted({r["category"] for r in rs})}
            sec = [r.get("seconds", 0.0) for r in rs]
            S_[pol]["seconds_per_request_max"] = float(max(sec)); S_[pol]["requests_over_1s"] = int(sum(x > 1.0 for x in sec))
            S_[pol]["answers_unknown_total"] = int(sum(r.get("n_unknown") or 0 for r in rs))
            S_[pol]["failures"] = [{k: r.get(k) for k in ("id", "family", "category", "outcome", "disposition", "selected", "why")} for r in rs
                                   if r["outcome"] in ("incorrect_release", "error", "unnecessary_refusal", "unresolved", "nonrelease_other_reason")][:60]
        out["structured"] = S_
        co = [r for r in rows if r["policy"] == "coordinator"]; aa = [r for r in rows if r["policy"] == "ask_all"]
        qa, qc = np.mean([r["n_questions"] for r in aa]), np.mean([r["n_questions"] for r in co])
        out["structured_question_saving"] = {"coordinator": float(qc), "ask_all": float(qa), "relative_reduction": float(1 - qc / qa) if qa else None}
    p = RES / "test_nl_rows.jsonl"
    if p.exists():
        rows = load_jsonl(p)
        N_ = {}
        keys = sorted({(r["reader"], r["user"], r["policy"]) for r in rows})
        for rd, us, pol in keys:
            rs = [r for r in rows if (r["reader"], r["user"], r["policy"]) == (rd, us, pol)]
            m = metrics(rs, lambda r: (r["family"], r["author"]))
            m["read_ms_median"] = float(np.median([1000 * r.get("seconds_read", 0) for r in rs]))
            m["by_author"] = {a: {k: v for k, v in metrics([r for r in rs if r["author"] == a], lambda r: r["family"]).items()
                                  if k in ("n", "outcomes", "incorrect_releases", "incorrect_per_request", "useful_coverage", "edits_per_request", "questions_per_request")}
                              for a in sorted({r["author"] for r in rs})}
            m["held_out_families"] = {k: v for k, v in metrics([r for r in rs if r["held_out"]], lambda r: (r["family"], r["author"])).items()
                                      if k in ("n", "outcomes", "incorrect_releases", "incorrect_per_request", "useful_coverage", "edits_per_request")}
            m["zero_edit_requests"] = sum(1 for r in rs if r.get("edits", 0) == 0) / len(rs)
            # revision 2: the first reading was entirely correct (no field corrected) / the user had to correct at least one field
            m["initial_reading_correct_share"] = m["zero_edit_requests"]
            m["corrected_requests_share"] = 1.0 - m["zero_edit_requests"]
            ce = [r.get("edits", 0) for r in rs if r.get("edits", 0) > 0]
            m["edits_per_corrected_request"] = float(np.mean(ce)) if ce else 0.0
            m["by_category"] = {c: dict(collections.Counter(r["outcome"] for r in rs if r["category"] == c)) for c in sorted({r["category"] for r in rs})}
            m["failures"] = [{k: r.get(k) for k in ("id", "family", "category", "author", "outcome", "disposition", "selected", "why")} for r in rs
                             if r["outcome"] in ("incorrect_release", "error")][:60]
            N_[f"{rd}|{us}|{pol}"] = m
        out["nl"] = N_
        # held-out versus training families for every configuration (review B7)
        keep = ("n", "outcomes", "incorrect_releases", "incorrect_per_request", "incorrect_per_request_ci", "useful_coverage", "edits_per_request",
                "questions_per_request")
        out["nl_family_split"] = {}
        for rd, us, pol in keys:
            rs = [r for r in rows if (r["reader"], r["user"], r["policy"]) == (rd, us, pol)]
            out["nl_family_split"][f"{rd}|{us}|{pol}"] = {
                "held_out": {k: v for k, v in metrics([r for r in rs if r["held_out"]], lambda r: (r["family"], r["author"])).items() if k in keep},
                "training": {k: v for k, v in metrics([r for r in rs if not r["held_out"]], lambda r: (r["family"], r["author"])).items() if k in keep},
                "by_family": {f: {k: v for k, v in metrics([r for r in rs if r["family"] == f], lambda r: r["author"]).items() if k in ("n", "incorrect_releases", "useful_coverage", "edits_per_request")}
                              for f in sorted({r["family"] for r in rs})}}
        # paired comparisons of the trained reader with the baselines on the same requests (cluster bootstrap over (family, author))
        def paired(a_rows, b_rows, B=2000, seed=1):
            A = {r["id"]: r for r in a_rows}; Bm = {r["id"]: r for r in b_rows}
            ids = sorted(set(A) & set(Bm))
            inc = lambda r: float(r["outcome"] == "incorrect_release")
            cov = lambda r: float(r["outcome"] == "correct_release" and r["expected_final"] == "RELEASE")
            rel = lambda r: r["expected_final"] == "RELEASE"
            groups = collections.defaultdict(list)
            for i in ids:
                groups[(A[i]["family"], A[i]["author"])].append(i)
            names = sorted(groups)
            def stat(sel):
                di = np.mean([inc(A[i]) - inc(Bm[i]) for i in sel])
                rr = [i for i in sel if rel(A[i])]
                dc = np.mean([cov(A[i]) - cov(Bm[i]) for i in rr]) if rr else np.nan
                de = np.mean([A[i].get("edits", 0) - Bm[i].get("edits", 0) for i in sel])
                return di, dc, de
            point = stat(ids)
            rng = np.random.default_rng(seed)
            boots = []
            for _ in range(B):
                pick = rng.integers(0, len(names), len(names))
                sel = [i for k in pick for i in groups[names[k]]]
                boots.append(stat(sel))
            boots = np.array(boots)
            ci = lambda j: [float(np.nanquantile(boots[:, j], 0.025)), float(np.nanquantile(boots[:, j], 0.975))]
            return {"n": len(ids), "d_incorrect_per_request": float(point[0]), "ci_incorrect": ci(0), "d_useful_coverage": float(point[1]), "ci_coverage": ci(1),
                    "d_edits_per_request": float(point[2]), "ci_edits": ci(2)}
        out["nl_paired"] = {}
        for us in ("attentive", "blind"):
            for t in sorted({k.split("|")[0] for k in N_ if k.startswith("final_")}):
                for base in ("null", "rules", "select_neural_frozen_s0"):
                    a_rows = [r for r in rows if (r["reader"], r["user"], r["policy"]) == (t, us, "coordinator")]
                    b_rows = [r for r in rows if (r["reader"], r["user"], r["policy"]) == (base, us, "coordinator")]
                    if a_rows and b_rows:
                        out["nl_paired"][f"{t}-{base}|{us}"] = paired(a_rows, b_rows)
        # oracle: the coordinator on the correct structured specifications of the same requests (no language layer)
        ps = RES / "test_structured_rows.jsonl"
        if ps.exists():
            ids = {r["id"] for r in rows}
            orc = [r for r in load_jsonl(ps) if r["policy"] == "coordinator" and r["id"] in ids]
            if orc:
                out["nl_oracle_structured"] = metrics(orc, lambda r: r["family"])
        # seed variability of the trained reader
        tr = sorted({r["reader"] for r in rows if r["reader"].startswith("final_")})
        # overlap between seeds (review B10) and the categories of the failures without review (review B9)
        def ids_where(rd, us, cond):
            return {r["id"] for r in rows if (r["reader"], r["user"], r["policy"]) == (rd, us, "coordinator") and cond(r)}
        exact = [ids_where(t, "attentive", lambda r: r.get("edits", 0) == 0) for t in tr]
        fail = [ids_where(t, "blind", lambda r: r["outcome"] == "incorrect_release") for t in tr]
        cats = {t: sorted({r["category"] for r in rows if (r["reader"], r["user"], r["policy"]) == (t, "blind", "coordinator")
                           and r["outcome"] == "incorrect_release"}) for t in tr}
        if tr:
            out["nl_seed_overlap"] = {
                "exact_reading_all_seeds": len(set.intersection(*exact)), "exact_reading_any_seed": len(set.union(*exact)),
                "blind_incorrect_union": len(set.union(*fail)), "blind_incorrect_all_seeds": len(set.intersection(*fail)),
                "blind_incorrect_one_seed_only": sum(1 for i in set.union(*fail) if sum(i in f for f in fail) == 1),
                "blind_incorrect_categories": cats, "blind_incorrect_categories_union": sorted(set().union(*cats.values())),
                "n_categories_total": len({r["category"] for r in rows})}
        for us in ("attentive", "blind"):
            vals = {k: [N_[f"{t}|{us}|coordinator"][k] for t in tr if f"{t}|{us}|coordinator" in N_]
                    for k in ("incorrect_per_request", "useful_coverage", "edits_per_request", "zero_edit_requests", "corrected_requests_share",
                              "edits_per_corrected_request")}
            out.setdefault("nl_seed_variability", {})[us] = {k: {"values": v, "mean": float(np.mean(v)), "min": float(min(v)), "max": float(max(v))} for k, v in vals.items() if v}
    save("test_summary.json", out)
    for sec in ("structured", "nl"):
        for k, m in out.get(sec, {}).items():
            print(f"{sec:10s} {k:52s} n={m['n']} inc={m['incorrect_releases']:3d} cov={m['useful_coverage'] if m['useful_coverage'] is None else round(m['useful_coverage'],3)} "
                  f"q={m['questions_per_request']:.2f} edits={m['edits_per_request']:.2f} unnec={m['unnecessary_refusals']} unres={m['unresolved']} err={m['errors']} "
                  f"false_imposs={m['false_impossibility']}")


if __name__ == "__main__":
    {"structured": run_structured, "nl": run_nl, "reading": run_reading, "summarise": summarise}[sys.argv[1]]()
