"""Wrappers that expose the SUPPLIED (legacy) trained models as readers of the new interface, for benchmarking only.

Both models were trained on the wording and label sets of their own packages; here they are applied without any
re-training ("zero-shot" on the new task) and their labels are mapped to the new roles where a mapping exists.
They cannot produce the request-level labels of the new interface (objective kind, distribution, ...), which are
left at their defaults; the comparison is therefore reported on quantity roles and on exact reading.
"""
import os, pathlib, sys
# directory that contains the extracted supplied packages heros_kalman/ and heros_khalil/ (not part of this project)
SRC = pathlib.Path(os.environ.get("HEROS_LEGACY_SRC", pathlib.Path(__file__).resolve().parents[2] / "src"))
from heros.nlu.quantities import find_quantities

KH_MAP = {"V": "meas_bound", "V_rms": "meas_std", "D": "acc_bound", "E1": "init_pos_bound", "E2": "init_vel_bound", "T": "deadline",
          "rho": "tolerance", "P": "peak_cap", "sample_period": "Ts", "state_limit": "vel_limit"}
KA_MAP = {"DT": "Ts", "RATE": "fs", "V_STD": "meas_std", "V_VAR": "meas_var", "W_STD": "acc_std", "S0_STD": {"pos": "init_pos_std", "vel": "init_vel_std"}}
MAPPABLE_KHALIL = set(KH_MAP.values())
MAPPABLE_KALMAN = {"Ts", "fs", "meas_std", "meas_var", "acc_std", "init_pos_std", "init_vel_std"}


class LegacyKhalilReader:
    name = "supplied_khalil_spacy_finetuned"

    def __init__(self, src=SRC):
        sys.path.insert(0, str(src / "heros_khalil" / "repo" / "src"))
        from hgo_agent.binding import SpacyBinder
        self.b = SpacyBinder(str(src / "heros_khalil" / "repo" / "models" / "finetuned" / "model-best"), cat_threshold=0.8)

    def read(self, text, plant=None, stats_in_attachment=False):
        plain = text.replace("[Q1]", "    ").replace("[Q2]", "    ").replace("[Q3]", "    ")
        try:
            prop = self.b.propose(plain)
        except Exception:  # noqa
            prop = {"entities": [], "cats": {}}
        qs = find_quantities(text)
        out = []
        for q in qs:
            lab = [l for s, e, l in prop["entities"] if s < q.end and e > q.start]
            role = KH_MAP.get(lab[0], "other") if lab else "other"
            out.append({"start": q.start, "end": q.end, "role": role, "conf": None})
        flags = ["hidden_dynamics"] if prop.get("cats", {}).get("extra_dynamics") else []
        return {"quantities": out, "objective": "hard_bound" if any(o["role"] == "tolerance" for o in out) else "unspecified", "distribution": "unstated",
                "horizon": "unstated", "evidence": "unstated", "flags": flags, "proposal": None}


class LegacyKalmanReader:
    name = "supplied_kalman_roberta_lora"

    def __init__(self, src=SRC):
        sys.path.insert(0, str(src / "heros_kalman"))
        from kalman_agent.nlu.model import ParserModel
        from kalman_agent.nlu import text as T
        self.T = T
        self.m = ParserModel.load(str(src / "heros_kalman" / "checkpoints" / "lora_top6"))

    def read(self, text, plant=None, stats_in_attachment=False):
        plain = text.replace("[Q1]", "    ").replace("[Q2]", "    ").replace("[Q3]", "    ")
        try:
            parsed = self.m.parse(plain)
            toks = self.T.tokenize(plain)
        except Exception:  # noqa
            parsed, toks = None, []
        qs = find_quantities(text)
        out = []
        for q in qs:
            role = "other"
            if parsed is not None:
                for tk, r in zip(toks, parsed.roles):
                    s = getattr(tk, "start", None)
                    if r != "O" and s is not None and q.start <= s < q.end:
                        m = KA_MAP.get(r)
                        role = (m.get(q.dim, "other") if isinstance(m, dict) else m) or "other"
                        break
            out.append({"start": q.start, "end": q.end, "role": role, "conf": None})
        obj = "unspecified"
        if parsed is not None and parsed.task == "kf":
            obj = "mse_optimal"
        return {"quantities": out, "objective": obj, "distribution": "unstated", "horizon": "unstated", "evidence": "unstated", "flags": [], "proposal": None}
