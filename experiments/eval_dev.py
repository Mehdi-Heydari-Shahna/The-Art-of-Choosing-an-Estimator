"""Model selection on DEVELOPMENT data only (template family D and the independently authored development requests).

Pre-declared rule (docs/PROTOCOL.md, written before this script was first run on the candidates):
  score = mean of the exact-reading rates on the two development parts (templated, independently authored);
  among the trained candidates within 0.02 (absolute) of the best score, choose the one with the fewest trainable
  parameters; report all candidates, including the supplied models and the deterministic baseline.
"""
import json, sys
from common import ROOT, save
from heros.nlu import readers as RD
import nl_common as NC

MODELS = ROOT / "models"


_VEC = None


def vectors():
    """The pretrained static vectors, loaded once per process (about 1 GB in memory)."""
    global _VEC
    if _VEC is None:
        _VEC = RD.load_vectors(str(MODELS / "vectors_en_core_web_md.npz"))
    return _VEC


def load(name):
    d = MODELS / name
    cfg = json.loads((d / "config.json").read_text())
    if cfg["class"] == "LinearReader":
        return RD.LinearReader.load(d / cfg["file"]), cfg
    return RD.NeuralReader.load(d / cfg["file"], vectors() if cfg.get("variant") != "scratch" else None), cfg


if __name__ == "__main__":
    tmpl = NC.templated("dev")
    auth = NC.authored("dev")
    print("dev items:", len(tmpl), "templated,", len(auth), "authored")
    # candidates are built one at a time (lazily) and released after scoring, to keep memory within a small machine
    cands = {"null": lambda: (RD.NullReader(), {}), "rules": lambda: (RD.RuleReader(), {"n_trainable_parameters": 0})}
    for name in ("select_linear_s0", "select_neural_pretrained_s0", "select_neural_scratch_s0", "select_neural_frozen_s0"):
        if (MODELS / name / "config.json").exists():
            cands[name] = (lambda n=name: load(n))
    if "--legacy" in sys.argv:
        import legacy_readers as LR
        for cls in (LR.LegacyKhalilReader, LR.LegacyKalmanReader):
            cands[cls.name] = (lambda c=cls: (c(), {"note": "supplied model, applied without re-training; labels mapped where possible"}))
    res = {}
    import gc
    for name, make in cands.items():
        try:
            reader, cfg = make()
        except Exception as ex:  # noqa
            res[name] = {"error": f"{type(ex).__name__}: {ex}"}; print(name, res[name]); continue
        r = {"templated_D": NC.reading_metrics(reader, tmpl), "authored_dev": NC.reading_metrics(reader, auth) if auth else None,
             "n_trainable_parameters": cfg.get("n_trainable_parameters"), "train_seconds": cfg.get("train_seconds"), "note": cfg.get("note")}
        parts = [r["templated_D"]["exact_reading"]] + ([r["authored_dev"]["exact_reading"]] if auth else [])
        r["selection_score"] = sum(parts) / len(parts)
        res[name] = r
        print(f"{name:34s} score {r['selection_score']:.3f} | tmpl exact {r['templated_D']['exact_reading']:.3f} role {r['templated_D']['role_accuracy']:.3f} "
              + (f"| auth exact {r['authored_dev']['exact_reading']:.3f} role {r['authored_dev']['role_accuracy']:.3f} " if auth else "")
              + f"| {r['templated_D']['ms_per_request']:.1f} ms/req | params {r['n_trainable_parameters']}", flush=True)
        del reader
        gc.collect()
    trained = {k: v for k, v in res.items() if k.startswith("select_") and "selection_score" in v}
    if trained:
        best = max(v["selection_score"] for v in trained.values())
        ok = {k: v for k, v in trained.items() if v["selection_score"] >= best - 0.02}
        chosen = min(ok, key=lambda k: ok[k]["n_trainable_parameters"])
        arch = chosen[len("select_"):-len("_s0")]
        sel = {"selected_architecture": arch, "rule": "fewest trainable parameters among candidates within 0.02 of the best development score",
               "scores": {k: v["selection_score"] for k, v in trained.items()}, "best_score": best}
        if "--write" in sys.argv:
            (MODELS / "selection.json").write_text(json.dumps(sel, indent=1))
        res["selection"] = sel
        print("selected:", arch)
    save("dev_selection.json", res)
