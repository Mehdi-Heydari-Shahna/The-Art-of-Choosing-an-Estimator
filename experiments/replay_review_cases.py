"""Replay the seven review cases of revision 2b on a release tree and check each against its expected outcome.

Each case file's `arguments` object is passed unchanged to the public interface `heros.api.solve` of the tree given by
--code (default: this repository). For every file the complete output is written to <out>/<case>.output.json, and one
row with the expected result, the decision, the numerical result and pass/fail goes to <out>/summary.json and
<out>/summary.md. Reported separately from the acceptance test.

    python -I experiments/replay_review_cases.py --code <release tree> --out results/review_cases_rev2b_release

The expected results are those of the review (README of the cases): A1 and A2 carry the same timing deviation at
different clock origins and must not receive the exact-period guarantee; A3 is on its grid and is released; B may end
without a verified design but must not be declared impossible; C has four observations in its window and its largest
in-window covariance is about 0.61589404 against the level 0.61680778; D1 (input fixed by the specification) and D2
(input as a vector) give the same trajectory, with velocity 0.5 at t = 0.5 s.
"""
import argparse, copy, hashlib, json, math, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parents[1]
CASES = HERE / "tests" / "fixtures" / "reviewer_cases_rev2b"
SHA256 = {   # as stored with the cases (tests/fixtures/reviewer_cases_rev2b/README.md)
    "A1_relative_times_with_jitter.json": "bcd3146029200c1f4ad079440b56b7133080a889fb6a8e2bf4549e50d5574205",
    "A2_absolute_times_same_jitter.json": "1ea8898195e9d2d54b2d0af7546a6ec638c8af73725037f6502ef278161ea7e5",
    "A3_absolute_times_on_grid.json": "0f16e039d59098833dcc09422e09067542b4486d7101e848147f208cf6d82337",
    "B_forward_only_carriage.json": "792a60c9bdb1206513eee56cfc65e83b81a42aea5a94e30506f5c70efcb44177",
    "C_exact_window_ends_before_next_sample.json": "bdd1c1d4936bc2150f611c0f76fa71937ef5e3003c0fa902e6e8857a1fa1d2dc",
    "D1_fixed_input_only_in_specification.json": "80fa1f2801e6d699fc9820f72bb0c0caee6add0f85f018e9eb9ae4cf2d173a2d",
    "D2_explicit_known_input_control.json": "2169eb93373461aca41cc0c124062866e5c056f8f3898ec7451580be46dfd5a2",
}
C_LEVEL, C_EXPECTED = 0.61680778, 0.61589404


def code_digest(root):
    h = hashlib.sha256()
    for p in sorted((root / "heros").rglob("*.py")) + sorted((root / "independent").rglob("*.py")):
        h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def plain(o):
    """JSON-ready copy: arrays to lists, non-finite floats to strings."""
    import numpy as np
    if isinstance(o, dict):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return plain(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else str(f)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def git_info(root):
    def g(*a):
        r = subprocess.run(["git", *a], cwd=root, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    commit = g("rev-parse", "HEAD")
    return {"commit": commit, "tags_at_commit": (g("tag", "--points-at", "HEAD") or "").split() if commit else [],
            "describe": g("describe", "--tags", "--always")} if commit else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default=str(HERE))
    ap.add_argument("--cases", default=str(CASES))
    ap.add_argument("--out", default=str(HERE / "results" / "review_cases_rev2b_release"))
    ap.add_argument("--release", default="", help="free-text name of the release under test (archive name, tag)")
    a = ap.parse_args()
    code, cases, out = pathlib.Path(a.code).resolve(), pathlib.Path(a.cases), pathlib.Path(a.out)
    sys.path.insert(0, str(code))
    import numpy as np
    from heros import api
    out.mkdir(parents=True, exist_ok=True)
    raw, rows = {}, []
    for name, sha in SHA256.items():
        data = (cases / name).read_bytes()
        got_sha = hashlib.sha256(data).hexdigest()
        case = json.loads(data)
        res = api.solve(**copy.deepcopy(case["arguments"]))
        raw[name] = res
        (out / name.replace(".json", ".output.json")).write_text(json.dumps(plain(res), indent=1))
        dec = res["decision"]; disp = dec["disposition"]; claim = (dec.get("claim") or {}).get("value")
        imp = (dec.get("impossibility") or {}).get("lower_bound") if isinstance(dec.get("impossibility"), dict) else None
        reason = ""
        if disp != "RELEASE":
            reason = "; ".join((dec.get("explanations") or [])[:1]) or str((dec.get("issues") or [{}])[0].get("message", ""))
        row = {"file": name, "sha256_ok": got_sha == sha, "disposition": disp, "selected": dec.get("selected"),
               "claim_value": claim, "lower_limit": imp, "reason": reason[:300]}
        case_id = name.split("_")[0]
        if case_id in ("A1", "A2"):
            row["expected"] = "exact-period guarantee declined (same timing deviation, clock origin %s)" % ("0" if case_id == "A1" else "2^30 s")
            row["numerical"] = "max deviation %.3g s vs precision %.3g s" % (res["timing"]["max_deviation"], res["timing"]["label_precision"]) \
                if res.get("timing") and res["timing"].get("max_deviation") is not None else "timing check: %s" % (res.get("timing") or {}).get("code")
            row["pass"] = disp != "RELEASE"
        elif case_id == "A3":
            row["expected"] = "released (samples on the grid)"
            row["numerical"] = "claim %s; label precision %.3g s" % (claim, res["timing"]["label_precision"])
            row["pass"] = disp == "RELEASE"
        elif case_id == "B":
            row["expected"] = "no impossibility claim; 'no verified design' acceptable"
            row["numerical"] = "disposition %s, lower limit %s" % (disp, imp)
            row["pass"] = disp != "PROVED_IMPOSSIBLE" and (disp == "NO_VERIFIED_CANDIDATE" or (disp == "RELEASE" and claim is not None))
        elif case_id == "C":
            P = np.asarray(res["estimates"]["P_filt"], float).reshape(len(res["estimates"]["t"]), -1)[:, 0] if res.get("estimates") else None
            n_in = int(dec["config"]["N"]) + 1 if disp == "RELEASE" else None
            row["expected"] = "four observations in the window; largest in-window covariance about %.8f <= %.8f" % (C_EXPECTED, C_LEVEL)
            row["numerical"] = "observations in window %s; covariances %s; largest %.10f" % (
                n_in, None if P is None else [round(float(x), 8) for x in P], float(np.max(P))) if P is not None else "no estimates"
            row["pass"] = (disp == "RELEASE" and n_in == 4 and P is not None and abs(float(np.max(P)) - C_EXPECTED) < 5e-9
                           and float(np.max(P)) <= C_LEVEL and claim is not None and abs(claim - float(np.max(P))) < 1e-12)
        elif case_id in ("D1", "D2"):
            t = np.asarray(res["estimates"]["t"], float); x = np.asarray(res["estimates"]["x_hat"], float)
            k = int(np.argmin(np.abs(t - 0.5)))
            row["expected"] = "velocity 0.5 at t = 0.5 s; same trajectory for D1 and D2"
            row["numerical"] = "velocity at t = %.3f s: %.17g (error %.2g)" % (t[k], x[k, 1], abs(x[k, 1] - 0.5))
            row["pass"] = disp == "RELEASE" and abs(x[k, 1] - 0.5) < 1e-12
        rows.append(row)
    # D1 and D2 must give the same trajectory
    x1 = np.asarray(raw["D1_fixed_input_only_in_specification.json"]["estimates"]["x_hat"], float)
    x2 = np.asarray(raw["D2_explicit_known_input_control.json"]["estimates"]["x_hat"], float)
    same = x1.shape == x2.shape and float(np.max(np.abs(x1 - x2))) <= 1e-12
    for r in rows:
        if r["file"].startswith(("D1", "D2")):
            r["numerical"] += "; D1 and D2 trajectories differ by at most %.2g" % float(np.max(np.abs(x1 - x2)))
            r["pass"] = r["pass"] and same
    meta = {"script": "experiments/replay_review_cases.py", "release": a.release, "code_tree": code.name,
            "code_digest": code_digest(code), "git": git_info(code), "cases": str(cases.name),
            "n": len(rows), "n_pass": sum(r["pass"] for r in rows), "all_sha256_ok": all(r["sha256_ok"] for r in rows),
            "python": sys.version.split()[0], "numpy": np.__version__}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "rows": plain(rows)}, indent=1))
    L = [f"# Review cases of revision 2b on {a.release or code.name}", "",
         f"Code digest (heros/ and independent/): `{meta['code_digest']}`" + (f"; git {meta['git']['describe']} ({meta['git']['commit'][:7]})" if meta["git"] else ""), "",
         f"{meta['n_pass']} of {meta['n']} pass; case files match their SHA-256: {meta['all_sha256_ok']}. Raw outputs: `<case>.output.json` in this folder.", "",
         "| File | Expected | Decision | Numerical result | Pass |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        L.append(f"| `{r['file']}` | {r['expected']} | {r['disposition']}{(' (' + r['selected'] + ')') if r['selected'] else ''} | {r['numerical']} | {'pass' if r['pass'] else 'FAIL'} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
