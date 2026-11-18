"""Train the candidate readers and write checkpoints, configurations and logs.

    python experiments/train_readers.py --stage select     # all candidate architectures, seed 0 (model selection on dev)
    python experiments/train_readers.py --stage final      # the selected architecture, three predeclared seeds (0, 1, 2)
Checkpoints: models/<name>/reader.(pkl|pt), config.json, train_log.json.  Hardware: CPU only.
"""
import argparse, hashlib, json, pathlib, platform, time
import numpy as np
from common import ROOT, load_jsonl
from heros import evaluation as EV
from heros.coordinator import coordinator as C
from heros.nlu import readers as RD

MODELS = ROOT / "models"
SEEDS = [0, 1, 2]


def proposal_of(case):
    e = case["expected"]
    if e["disposition"] == "RELEASE":
        if e["selected"] is None:
            return EV.proposal_label(C.decide(case["given"]))
        return "RELEASE:" + e["selected"]
    if e["disposition"] == "CLARIFY":
        return "CLARIFY:" + e["ask"]
    return e["disposition"]


def examples(split):
    cache = ROOT / "data" / f"{split}_examples.json"
    if cache.exists():
        return json.loads(cache.read_text())
    ex = [EV.example_from_request(c, c["request"], proposal_of(c)) for c in load_jsonl(ROOT / "data" / f"{split}.jsonl")]
    cache.write_text(json.dumps(ex))
    return ex


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def train_one(name, make, train, extra=None):
    d = MODELS / name
    d.mkdir(parents=True, exist_ok=True)
    log = []
    t0 = time.time()
    r = make().fit(train, log=log)
    secs = time.time() - t0
    path = d / ("reader.pkl" if isinstance(r, RD.LinearReader) else "reader.pt")
    r.save(path)
    cfg = {"name": name, "class": type(r).__name__, "n_trainable_parameters": getattr(r, "n_params", None), "train_seconds": round(secs, 1),
           "n_train_requests": len(train), "n_train_quantities": int(sum(len(e["roles"]) for e in train)),
           "data_sha256": json.loads((ROOT / "data" / "manifest.json").read_text())["sha256"], "hardware": "2 CPU cores, no GPU",
           "python": platform.python_version(), "sha256": sha(path), "file": path.name}
    cfg.update(extra or {})
    for k in ("C", "seed", "bootstrap", "variant", "hidden", "epochs", "lr"):
        if hasattr(r, k):
            cfg[k] = getattr(r, k)
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    (d / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"trained {name}: {secs:.1f} s, params {cfg['n_trainable_parameters']}")
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["select", "final"], required=True)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--only", default=None, help="comma-separated candidate names of the select stage (to split the work over processes)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seeds", default=None, help="comma-separated subset of the predeclared seeds 0,1,2 (final stage; to split the work over processes)")
    a = ap.parse_args()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    only = set(a.only.split(",")) if a.only else None
    want = lambda n: only is None or n in only
    train = examples("train")
    vec_path = MODELS / "vectors_en_core_web_md.npz"
    pre = {"pretrained_base": "en_core_web_md==3.8.0 static vectors (spaCy model release; sha256 5e6329fe3fecedb1d1a02c3ea2172ee0fede6cea6e4aefb6a02d832dba78a310 for the wheel)",
           "base_license": "MIT (spaCy model package)", "base_retrieval": "pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl"}
    if a.stage == "select":
        vectors = RD.load_vectors(str(vec_path))
        if want("linear"):
            train_one("select_linear_s0", lambda: RD.LinearReader(C=3.0, seed=0), train)
        if want("neural_pretrained"):
            train_one("select_neural_pretrained_s0", lambda: RD.NeuralReader("pretrained", 0, vectors=vectors), train, pre)
        if want("neural_scratch"):
            train_one("select_neural_scratch_s0", lambda: RD.NeuralReader("scratch", 0), train)
        if want("neural_frozen"):
            train_one("select_neural_frozen_s0", lambda: RD.NeuralReader("frozen", 0, vectors=vectors, epochs=10), train, pre)
    else:
        arch = a.arch or json.loads((MODELS / "selection.json").read_text())["selected_architecture"]
        for s in ([int(z) for z in a.seeds.split(",")] if a.seeds else SEEDS):
            assert s in SEEDS
            if arch == "linear":
                train_one(f"final_linear_s{s}", lambda: RD.LinearReader(C=3.0, seed=s, bootstrap=True), train,
                          {"seed_meaning": "bootstrap resample of the training requests (the optimiser itself is deterministic)"})
            else:
                vectors = RD.load_vectors(str(vec_path)) if arch != "neural_scratch" else None
                train_one(f"final_{arch}_s{s}", lambda: RD.NeuralReader(arch.replace("neural_", ""), s, vectors=vectors), train, pre if vectors else None)
