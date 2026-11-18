"""Export the final checkpoints to NumPy and check that the NumPy readings equal the PyTorch readings.

    python experiments/check_numpy_reader.py
Compares, for every development and test request (templated and authored), the role of every number, the request
labels, the flags and the proposal; reports the largest difference of the confidence values.  Writes
results/numpy_reader_check.json.  (A test request is only *read* here, by both implementations of the same frozen
checkpoint; nothing is tuned.)"""
import json
import numpy as np
from common import ROOT, save
import nl_common as NC
import eval_dev as ED
from heros.nlu.numpy_reader import NumpyGRUReader, export_numpy
from heros import evaluation as EV

items = NC.templated("dev") + NC.authored("dev") + NC.authored("test")
out = {"n_requests": len(items)}
for s in (0, 1, 2):
    name = f"final_neural_scratch_s{s}"
    tr, _ = ED.load(name)
    export_numpy(tr, ROOT / "models" / name)
    nr = NumpyGRUReader(ROOT / "models" / name)
    diff_req, dconf, nq = [], 0.0, 0
    for case, text, author, gold in items:
        plant = case["given"]["plant"]; sa = EV.stats_in_attachment(case["family"])
        a, b = tr.read(text, plant, sa), nr.read(text, plant, sa)
        same = [x["role"] for x in a["quantities"]] == [x["role"] for x in b["quantities"]] and all(
            a[k] == b[k] for k in ("objective", "distribution", "horizon", "evidence", "flags", "proposal"))
        nq += len(a["quantities"])
        if not same:
            diff_req.append(case["id"])
        for x, y in zip(a["quantities"], b["quantities"]):
            dconf = max(dconf, abs(x["conf"] - y["conf"]))
        dconf = max(dconf, abs(a["proposal_conf"] - b["proposal_conf"]), abs(a["objective_conf"] - b["objective_conf"]))
    out[name] = {"requests_with_any_difference": diff_req, "n_quantities": nq, "max_confidence_difference": dconf}
    print(name, "requests that differ:", len(diff_req), "| max confidence difference:", f"{dconf:.2e}", flush=True)
save("numpy_reader_check.json", out)
