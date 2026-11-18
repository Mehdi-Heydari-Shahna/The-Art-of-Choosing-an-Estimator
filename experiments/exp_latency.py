"""Measured computation and workflow cost (no human timing is claimed).

    python experiments/exp_latency.py
Writes results/latency.json: reading time per request for each reader, coordinator decision time per specification,
run-time cost per estimator step, peak memory, model sizes, training times, and the scripted interaction cost
(questions and readback corrections) taken from the acceptance-test rows when they exist.
"""
import json, resource, time, tracemalloc
import numpy as np
from common import ROOT, RES, save, load_jsonl
from heros import api
from heros.nlu import readers as RD, templates as T
from heros.kalman import specialist as K
from heros.khalil import specialist as H
from heros import record as R
from heros.interface import service as SV
import eval_dev as ED

MODELS = ROOT / "models"


def timeit(fn, n):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return {"median_ms": 1e3 * float(np.median(ts)), "p95_ms": 1e3 * float(np.quantile(ts, 0.95)), "n": n}


def main():
    out = {"hardware": "2 CPU cores, no GPU (build machine); single process", "note": "wall-clock times of this machine; no human effort is measured"}
    dev = [c for c in load_jsonl(ROOT / "data" / "dev.jsonl")][:150]
    texts = [(T.full_text(c["request"]), c["given"]["plant"]) for c in dev]
    readers = {"rules": RD.RuleReader()}
    for d in sorted(MODELS.glob("*/config.json")):
        name = d.parent.name
        if name.startswith("final_") and name.endswith("_s0") or name.startswith("select_"):
            try:
                readers[name] = ED.load(name)[0]
            except Exception as ex:  # noqa
                out.setdefault("not_loaded", {})[name] = str(ex)
    out["reading"] = {}
    for name, rd in readers.items():
        it = iter(texts * 2)
        out["reading"][name] = timeit(lambda: (lambda tp: rd.read(tp[0], tp[1]))(next(it)), 200)
        cfg = MODELS / name / "config.json"
        if cfg.exists():
            c = json.loads(cfg.read_text())
            f = MODELS / name / c["file"]
            out["reading"][name].update(n_trainable_parameters=c.get("n_trainable_parameters"), train_seconds=c.get("train_seconds"), file_bytes=f.stat().st_size)
    # coordinator decisions on development specifications (complete after the scripted answers = "truth")
    it = iter(dev)
    out["decision_incomplete_spec"] = timeit(lambda: api.decide(next(it)["given"]), 100)
    it = iter(dev)
    out["decision_complete_spec"] = timeit(lambda: api.decide(next(it)["truth"]), 100)
    # run-time cost per step of the two released estimators (examples of the package)
    EX = ROOT / "examples"
    spec = json.loads((EX / "hydraulic_boom_spec.json").read_text())
    spec = api.new_spec(spec.pop("plant"), objective=spec.pop("objective"), **spec)
    y, u = SV.read_data((EX / "hydraulic_boom_data.csv").read_text())
    dec = api.decide(spec)
    dts = []
    for _ in range(7):                       # median of repeated runs (the first call also pays one-time costs)
        t = time.perf_counter(); est = R.execute(dec, y, u); dts.append(time.perf_counter() - t)
    dt = float(np.median(dts))
    out["kalman_runtime"] = {"steps": int(len(y)), "microseconds_per_step": 1e6 * dt / len(y), "states": 4, "outputs": 2,
                             "note": "median of 7 runs of the whole record, time-varying gains computed in the run"}
    model = json.loads((EX / "rotary_joint_model.json").read_text())
    script = json.loads((EX / "rotary_joint_answers.json").read_text())
    yj, uj = SV.read_data((EX / "rotary_joint_data.csv").read_text())
    t0 = time.perf_counter()
    conv = SV.Conversation(model, script["answers"], yj, uj)
    conv.confirm(conv.readback()["spec_hash"])
    res = conv.step()
    n_q = 0
    while res["decision"]["disposition"] == "CLARIFY" and n_q < 4:
        p = res["decision"]["question"]["path"]; conv.answer(p, script["clarifications"][p]); n_q += 1
        res = conv.step()
    out["joint_request_end_to_end"] = {"seconds_compute": time.perf_counter() - t0, "questions": n_q, "disposition": res["decision"]["disposition"],
                                       "operations": "3 answers, 1 attached model, 1 data file, 1 confirmation, %d clarification answer(s)" % n_q,
                                       "includes": "reading, readback, decisions, interval certificate, estimation over %d samples, record" % len(yj)}
    dts = []
    for _ in range(7):
        t = time.perf_counter(); H.run(res["decision"]["config"], yj, uj); dts.append(time.perf_counter() - t)
    out["observer_runtime"] = {"steps": int(len(yj)), "microseconds_per_step": 1e6 * float(np.median(dts)) / len(yj), "note": "median of 7 runs"}
    t = time.perf_counter(); rep = api.replay(json.loads(R.dumps(res["record"]))); out["replay_seconds"] = time.perf_counter() - t
    out["replay_match"] = bool(rep["match"]); out["replay_quantities_compared"] = rep["n_compared"]
    tracemalloc.start(); api.decide(dev[0]["truth"]); cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    out["decision_peak_python_allocation_mb"] = peak / 1e6
    out["process_peak_rss_mb"] = [int(l.split()[1]) / 1024.0 for l in open("/proc/self/status") if l.startswith("VmHWM")][0]
    vec_loaded = any(n in readers for n in ("select_neural_pretrained_s0", "select_neural_frozen_s0"))
    out["process_peak_rss_note"] = ("this measurement process also loaded the 300-dimensional pretrained vectors of two candidate readers" if vec_loaded
                                    else "the candidate readers with pretrained vectors were not loaded in this process (see not_loaded)")
    if out.get("not_loaded"):
        print("WARNING: some readers could not be loaded; their reading times are missing:", ", ".join(out["not_loaded"]), flush=True)
    # the deployed configuration in a fresh process: NumPy reader + the joint request end to end (no PyTorch imported)
    import subprocess, sys
    code = ("import sys, json, resource, time; sys.modules['torch'] = None; sys.path.insert(0, %r)\n"
            "from heros.interface import service as SV\n"
            "ex = %r\n"
            "m = json.load(open(ex + '/rotary_joint_model.json')); s = json.load(open(ex + '/rotary_joint_answers.json'))\n"
            "y, u = SV.read_data(open(ex + '/rotary_joint_data.csv').read())\n"
            "t = time.perf_counter(); c = SV.Conversation(m, s['answers'], y, u); c.confirm(c.readback()['spec_hash']); r = c.step()\n"
            "while r['decision']['disposition'] == 'CLARIFY':\n"
            "    p = r['decision']['question']['path']; c.answer(p, s['clarifications'][p]); r = c.step()\n"
            "print(json.dumps({'reader': type(SV.load_reader()[0]).__name__, 'seconds': time.perf_counter() - t, 'disposition': r['decision']['disposition'],"
            " 'peak_rss_mb': [int(l.split()[1]) / 1024.0 for l in open('/proc/self/status') if l.startswith('VmHWM')][0]}))") % (str(ROOT), str(ROOT / "examples"))
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    out["deployed_fresh_process"] = json.loads(res.stdout.strip().splitlines()[-1]) if res.returncode == 0 else {"error": res.stderr[-500:]}
    from heros.nlu.numpy_reader import NumpyGRUReader
    sel = json.loads((MODELS / "selected.json").read_text())
    nr = NumpyGRUReader(MODELS / sel["id"])
    it = iter(texts * 2)
    out["reading"]["deployed_numpy"] = timeit(lambda: (lambda tp: nr.read(tp[0], tp[1]))(next(it)), 200)
    p = RES / "test_summary.json"
    if p.exists():
        S_ = json.loads(p.read_text())
        st = S_.get("structured", {})
        out["scripted_interaction_cost"] = {k: {"questions_per_request": v["questions_per_request"], "seconds_per_request_median": v["seconds_per_request_median"],
                                                "seconds_per_request_p95": v["seconds_per_request_p95"]} for k, v in st.items()}
    save("latency.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "reading"}, indent=1)[:1500])
    for k, v in out["reading"].items():
        print(k, v)


if __name__ == "__main__":
    main()
