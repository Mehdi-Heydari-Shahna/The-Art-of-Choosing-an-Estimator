"""Build the frozen datasets (deterministic for the seeds below) and their manifest.

  data/train.jsonl            cases of the TRAIN families, template styles A, B, C            (training)
  data/dev.jsonl              cases of the TRAIN families, template style D                   (model selection, thresholds)
  data/test_structured.jsonl  600 cases over ALL families incl. the held-out ones             (acceptance test, structured)
  data/authors/sheets_*.json  fact sheets handed to the independent request authors (one dev author, three test authors)
Usage:  python experiments/make_data.py --out /tmp/regen     (regenerate into another directory; see verify_data.py)
Grouping: every case has its own randomly drawn base specification; train / dev / test use disjoint seeds, the
held-out families occur only in the test set, and wording styles are disjoint across the three splits.
"""
import hashlib, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from heros import datagen as DG
from heros.core.assess import jsonable
from heros.nlu import templates as T
from heros.sim import plants as PL

FROZEN = pathlib.Path(__file__).resolve().parents[1] / "data"
ROOT = FROZEN
if "--out" in sys.argv:                 # regenerate elsewhere, e.g. to verify the frozen files against the manifest
    ROOT = pathlib.Path(sys.argv[sys.argv.index("--out") + 1]).resolve()
SEEDS = {"train": 20261117, "dev": 20261118, "test": 20261119, "sheets": 20261120}
N = {"train": 3600, "dev": 600, "test": 600}
ALL_FAMILIES = PL.TRAIN_FAMILIES + PL.HELD_OUT_FAMILIES
# family schedule (relative frequencies): the two unsupported families have a single category each and are kept rare;
# the double-integrator families carry the largest number of categories and get the largest share
TRAIN_MIX = {"servo_axis": 9, "mass_spring_damper": 6, "rotary_joint": 6, "dc_motor": 4, "thermal": 5, "two_mass": 4, "bearing_only": 1}
TEST_MIX = {"servo_axis": 6, "gantry_axis": 9, "mass_spring_damper": 5, "rotary_joint": 5, "duffing_stage": 7, "dc_motor": 4, "thermal": 4,
            "two_mass": 4, "hydraulic_boom": 6, "bearing_only": 1, "cart_pole": 1}


def schedule(mix, seed):
    fams = [f for f, k in mix.items() for _ in range(k)]
    return [fams[i] for i in np.random.default_rng(seed).permutation(len(fams))]

WHAT = {"Ts": "sample period", "fs": "sampling rate", "meas_bound": "hard worst-case bound on the measurement error of the sensor",
        "meas_std": "standard deviation (RMS) of the measurement noise", "meas_var": "variance of the measurement noise (per sample)",
        "acc_bound": "hard bound on the unmodelled / disturbance acceleration",
        "acc_std": "standard deviation of the random disturbance acceleration (white, constant within one sample)",
        "init_pos_bound": "hard bound on the initial error of the first state (position/angle)", "init_vel_bound": "hard bound on the initial error of the second state (velocity)",
        "init_pos_std": "standard deviation of the initial uncertainty of the first state", "init_vel_std": "standard deviation of the initial uncertainty of the second state",
        "vel_limit": "largest magnitude the second state (velocity) can reach during operation",
        "tolerance": "required accuracy (largest allowed estimation error) of the estimated quantity", "deadline": "time after start from which the accuracy requirement applies",
        "peak_cap": "largest allowed estimation error during the initial transient", "prob": "probability with which the accuracy requirement must hold",
        "mse_target": "largest allowed mean-square estimation error", "rms_target": "largest allowed RMS estimation error", "window": "duration of the record"}
OBJ = {"hard_bound": "a guaranteed (worst-case, for every admissible noise and disturbance) error limit", "mse_optimal": "the estimate with the smallest mean-square error (statistically optimal)",
       "mse_bound": "a mean-square (average) accuracy level", "probability": "an accuracy level that holds with a stated probability",
       "unspecified": "NOT STATED - do not say what success means beyond a vague wish for good accuracy (or say nothing)"}
ALSO = {"dist:gaussian": "all noises are Gaussian", "dist:bounded": "the noises are not Gaussian (bounded/uniform); only means and variances are known",
        "hor:steady_state": "only steady-state behaviour matters / a constant-gain estimator is fine", "hor:finite": "the start-up transient matters / the time-varying gain is wanted",
        "ev:mechanical_limit": "the stated velocity range is enforced by hardware (limiter / mechanical limit)", "ev:assumed": "the stated velocity range is an assumption the user accepts",
        "flag:hidden_dynamics": "there are actuator dynamics (a lag) that the model does not include", "flag:correlated_noise": "process noise and measurement noise are correlated with each other",
        "flag:coloured_noise": "the measurement noise is correlated in time (not white)", "flag:irregular_sampling": "samples arrive irregularly and some are missing"}


def dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(jsonable(r), sort_keys=True) + "\n")
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def sheet(case, rng):
    from heros.nlu.quantities import fmt_quantity
    F = T.case_facts(case, rng)
    plant = case["given"]["plant"]
    items = []
    for i, f in enumerate(F["facts"]):
        disp = (T._fmt_other(f["value"], T._sq_unit(f["family"]) if f["dim"] == "other2" else f["family"]) if f["dim"] in ("other", "other2")
                else fmt_quantity(f["value"], f["dim"], f["family"], rng)[0])
        items.append({"key": f"{f['role']}#{i}", "what": WHAT[f["role"]], "value": disp})
    also = []
    if F["distribution"] in ("gaussian", "bounded"):
        also.append(ALSO["dist:" + F["distribution"]])
    if F["horizon"] in ("steady_state", "finite"):
        also.append(ALSO["hor:" + F["horizon"]])
    if F["evidence"] in ("mechanical_limit", "assumed") and any(f["role"] == "vel_limit" for f in F["facts"]):
        also.append(ALSO["ev:" + F["evidence"]])
    also += [ALSO["flag:" + fl] for fl in F["flags"]]
    sysdesc = (f"{case['family'].replace('_', ' ')}; states: " + ", ".join(f"{s['name']} [{s['unit']}]" for s in plant["states"]) +
               "; measured: " + ", ".join(o["name"] for o in plant["outputs"]) + ("; input: " + ", ".join(i["name"] for i in plant["inputs"]) if plant["inputs"] else ""))
    return {"id": case["id"], "system": sysdesc, "estimate": F["target"], "facts_to_state": items, "success_means": OBJ[F["objective"]],
            "also_convey": also, "model_attached": True,
            "gold": {"facts": F["facts"], "objective": F["objective"], "distribution": F["distribution"], "horizon": F["horizon"], "flags": F["flags"],
                     "evidence": F["evidence"], "target": F["target"]}}


def main():
    if ROOT == FROZEN and (FROZEN / "manifest.json").exists() and "--force" not in sys.argv:
        raise SystemExit("data/ already holds the frozen data sets (the authored requests were written from its fact sheets). "
                         "Regenerate into another directory with --out DIR and compare with experiments/verify_data.py; "
                         "--force overwrites the frozen files.")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "authors").mkdir(exist_ok=True)
    manifest = {"seeds": SEEDS, "counts": {}, "sha256": {}, "families": {"train": PL.TRAIN_FAMILIES, "held_out": PL.HELD_OUT_FAMILIES},
                "family_mix": {"train_dev": TRAIN_MIX, "test": TEST_MIX}}
    train = DG.make_set(N["train"], schedule(TRAIN_MIX, 1), SEEDS["train"], "train")
    dev = DG.make_set(N["dev"], schedule(TRAIN_MIX, 2), SEEDS["dev"], "dev")
    test = DG.make_set(N["test"], schedule(TEST_MIX, 3), SEEDS["test"], "test")
    rng = np.random.default_rng(SEEDS["train"] + 7)
    for i, c in enumerate(train):
        c["request"] = T.realise(c, T.TRAIN_STYLES[i % 3], rng)
    rng = np.random.default_rng(SEEDS["dev"] + 7)
    for c in dev:
        c["request"] = T.realise(c, "D", rng)
    for name, rows in (("train", train), ("dev", dev), ("test_structured", test)):
        manifest["sha256"][name + ".jsonl"] = dump(ROOT / f"{name}.jsonl", rows)
        manifest["counts"][name] = len(rows)
    # fact sheets: 240 test cases (three authors x 80), stratified over families; 60 extra dev cases (one author)
    rng = np.random.default_rng(SEEDS["sheets"])
    nl_ok = [c for c in test if c["given"]["plant"]["states"]]
    order = rng.permutation(len(nl_ok))
    picked = [nl_ok[i] for i in order[:240]]          # simple random sample of the structured test set (same family mix)
    for a in range(3):
        sub = picked[a::3]
        sheets = [sheet(c, rng) for c in sub]
        (ROOT / "authors" / f"sheets_test_author{a + 1}.json").write_text(json.dumps(jsonable([{k: v for k, v in s.items() if k != "gold"} for s in sheets]), indent=1))
        (ROOT / "authors" / f"gold_test_author{a + 1}.json").write_text(json.dumps(jsonable(sheets), indent=1))
    devx = DG.make_set(60, schedule(TRAIN_MIX, 4), SEEDS["dev"] + 99, "devx")
    manifest["sha256"]["dev_extra_cases.jsonl"] = dump(ROOT / "dev_extra_cases.jsonl", devx)
    sheets = [sheet(c, rng) for c in devx]
    (ROOT / "authors" / "sheets_dev_author0.json").write_text(json.dumps(jsonable([{k: v for k, v in s.items() if k != "gold"} for s in sheets]), indent=1))
    (ROOT / "authors" / "gold_dev_author0.json").write_text(json.dumps(jsonable(sheets), indent=1))
    manifest["counts"].update(test_nl=240, dev_extra=60)
    manifest["test_nl_ids"] = [c["id"] for c in picked]
    import collections
    manifest["test_family_counts"] = dict(collections.Counter(c["family"] for c in test))
    manifest["test_category_counts"] = dict(collections.Counter(c["category"] for c in test))
    manifest["test_nl_family_counts"] = dict(collections.Counter(c["family"] for c in picked))
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("counts", "test_family_counts", "test_nl_family_counts")}, indent=1))


def rerealise():
    """Re-write the templated requests of the stored train / dev cases (same cases, same seeds) after a change of the
    display code.  Cases, the structured test set and the authors' fact sheets are NOT regenerated."""
    manifest = json.loads((ROOT / "manifest.json").read_text())
    changed = {}
    for name, seed, style in (("train", SEEDS["train"] + 7, None), ("dev", SEEDS["dev"] + 7, "D")):
        rows = [json.loads(l) for l in open(ROOT / f"{name}.jsonl")]
        rng = np.random.default_rng(seed)
        n = 0
        for i, c in enumerate(rows):
            new = T.realise(c, style or T.TRAIN_STYLES[i % 3], rng)
            n += int(new["sections"] != c["request"]["sections"])
            c["request"] = new
        manifest["sha256"][name + ".jsonl"] = dump(ROOT / f"{name}.jsonl", rows)
        changed[name] = n
    manifest.setdefault("notes", []).append(
        "train/dev requests re-realised after fixing the display of squared compound units ('(m/s)^2' was printed as 'm/s^2'); "
        f"requests whose text changed: {changed}. Cases, test set and fact sheets unchanged; one test fact sheet (test-00182, author 1) "
        "was written with the defective unit string and its request is kept as authored.")
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for f in ("train_examples.json", "dev_examples.json"):
        if (ROOT / f).exists():
            (ROOT / f).unlink()
    print("re-realised; changed requests:", changed)


if __name__ == "__main__":
    rerealise() if "--rerealise" in sys.argv else main()
