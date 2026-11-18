"""Interface logic shared by the command line and the local web page (no web framework, standard library only).

Flow:  three answers + attached model  ->  reading (trained reader)  ->  readback  ->  confirmation
       ->  coordinator decision (question / release / explanation)  ->  estimates, plot, exportable record.
"""
from __future__ import annotations

import base64
import copy
import csv
import io
import json
import pathlib

import numpy as np

from .. import record as R
from ..core import spec as S
from ..core.assess import jsonable
from ..coordinator.coordinator import Session
from ..nlu import assemble as AS
from ..nlu.readers import RuleReader, LinearReader

ROOT = pathlib.Path(__file__).resolve().parents[2]
QUESTIONS = ["What do you want to estimate?",
             "What do you measure, and what model and uncertainty information do you have?",
             "What does successful estimation mean for you?"]
_READER = None


def load_reader():
    """The selected trained reader (models/selected.json); the rule reader if no trained artifact is installed."""
    global _READER
    if _READER is None:
        sel = ROOT / "models" / "selected.json"
        try:
            info = json.loads(sel.read_text())
            if info["kind"] == "linear":
                _READER = (LinearReader.load(ROOT / "models" / info["path"]), info)
            elif info.get("numpy_export"):
                # NumPy inference of the same checkpoint (identical readings to the PyTorch path, results/numpy_reader_check.json)
                from ..nlu.numpy_reader import NumpyGRUReader
                d = ROOT / "models" / pathlib.Path(info["numpy_export"]).parent
                for f, h in info["numpy_sha256"].items():
                    import hashlib
                    if hashlib.sha256((d / f).read_bytes()).hexdigest() != h:
                        raise ValueError(f"{f} does not match models/selected.json")
                _READER = (NumpyGRUReader(d), info)
            else:
                from ..nlu.readers import NeuralReader, load_vectors
                vec = load_vectors(str(ROOT / "models" / info["vectors"])) if info.get("vectors") else None   # the selected reader needs none
                _READER = (NeuralReader.load(ROOT / "models" / info["path"], vec), info)
        except Exception as ex:  # noqa
            _READER = (RuleReader(), {"kind": "rules", "note": f"trained reader not available ({type(ex).__name__}); keyword rules used"})
    return _READER


def read_data(text: str):
    """CSV with header containing y (or y1,y2,..) and optionally u (or u1,..) and t.  Returns (y, u); see read_data_t."""
    y, u, _ = read_data_t(text)
    return y, u


def read_data_t(text: str):
    """As read_data, and also the recorded sample times (column t or time, in seconds; None if the file has none).
    The times are compared with the period of the design before it is run (heros.record.apply_timestamps)."""
    rows = list(csv.DictReader(io.StringIO(text.strip())))
    if not rows:
        raise ValueError("empty data file")
    cols = [c.strip() for c in rows[0].keys()]
    tcol = [c for c in rows[0].keys() if c.strip().lower() in ("t", "time", "t_s", "time_s")]
    t = np.array([float(r[tcol[0]]) for r in rows]) if tcol else None
    y, u = _read_yu(rows, cols)
    return y, u, t


def _read_yu(rows, cols):
    ycols = [c for c in cols if c.lower().startswith("y")]
    ucols = [c for c in cols if c.lower().startswith("u")]
    if not ycols:
        raise ValueError("the data file needs a column named y (or y1, y2, ...)")
    y = np.array([[float(r[c]) for c in ycols] for r in rows])
    u = np.array([[float(r[c]) for c in ucols] for r in rows]) if ucols else None
    return (y[:, 0] if y.shape[1] == 1 else y), (None if u is None else (u[:, 0] if u.shape[1] == 1 else u))


class Conversation:
    """One user interaction.  The same object backs the CLI and the web page."""

    def __init__(self, model: dict, answers: list, y=None, u=None, t=None):
        reader, info = load_reader()
        self.reader_info = {"kind": info.get("kind"), "id": info.get("id"), "sha256": info.get("sha256"), "note": info.get("note"),
                            "implementation": type(reader).__name__, "numpy_sha256": info.get("numpy_sha256")}
        att = S.empty_spec()
        if "plant" in model:
            for k, v in model.items():
                if k in att and isinstance(v, dict):
                    att[k].update(copy.deepcopy(v))
            for p in S.MATERIAL_PATHS:
                if S.get(att, p) is not None:
                    S.put(att, p, S.get(att, p), status="supplied", source="attached model file")
        else:
            att["plant"].update(copy.deepcopy(model))
        self.text = "[Q1] " + answers[0].strip() + " [Q2] " + answers[1].strip() + " [Q3] " + answers[2].strip()
        reading = reader.read(self.text, att["plant"], S.get(att, "stochastic.R") is not None)
        asm = AS.assemble(att, self.text, reading)
        self.issues = asm["issues"]
        self.proposal = reading.get("proposal")
        self.session = Session(asm["spec"])
        self.y, self.u, self.t = y, u, t
        self.last = None

    def readback(self) -> dict:
        return {"spec_hash": S.spec_hash(self.session.spec), "rows": jsonable(self.session.readback()), "issues": self.issues,
                "proposal_of_the_reader": self.proposal, "reader": self.reader_info}

    def correct(self, changes: dict):
        self.session.update(changes, status="supplied", source="readback correction")

    def confirm(self, spec_hash: str):
        self.session.confirm(spec_hash)

    def answer(self, path: str, value):
        self.session.update({path: value}, status="supplied", source="answer to a clarification question")

    def unknown(self, path: str):
        """The engineer cannot supply the asked value: the coordinator then ends without a release for that question."""
        self.session.spec.setdefault("unavailable", []).append(path)

    def accept_choice(self, path: str, value):
        self.session.update({path: value}, status="accepted_choice", source="design choice accepted by the user")

    def step(self) -> dict:
        dec = self.session.decide()
        dec, timing = R.apply_timestamps(self.session.spec, dec, self.y, self.t)
        dec, u_run, inputs = R.apply_inputs(self.session.spec, dec, self.y, self.u)
        out = {"decision": dec, "estimates": None, "timing": timing, "inputs": inputs}
        if dec["disposition"] == "RELEASE" and self.y is not None:
            out["estimates"] = R.execute(dec, self.y, u_run)
        out["record"] = R.make_record(self.session.spec, dec, self.y, self.u, out["estimates"], nlu=dict(self.reader_info, used=True, request_text=self.text),
                                      history=self.session.history, t=self.t, timing=timing)
        self.last = out
        return out


def summary(dec: dict) -> str:
    L = [dec.get("display") or dec["disposition"], f"Disposition: {dec['disposition']}"]
    if dec["disposition"] == "RELEASE":
        cfg = dec["config"]
        L.append(f"Selected method: {dec['selected']}  (qualifying: {', '.join(dec['qualifying'])})")
        L.append("Reason: " + dec["explanations"][0])
        L.append("Claim: " + dec["claim"]["statement"])
        L.append("Uncertainty information: " + dec["claim"]["uncertainty"])
        L.append("Conditions:")
        L += ["  - " + c for c in dec["claim"]["conditions"]]
        if "eps" in cfg:
            L.append(f"Configuration: eps = {cfg['eps']:.6g} s, gains M = [{cfg['M'][0]:.6g}, {cfg['M'][1]:.6g}], period {cfg['Ts']:.6g} s; {cfg['init']['premise']}")
        else:
            L.append(f"Configuration: {cfg['mode']} Kalman filter, period {cfg['Ts']:.6g} s, initial estimate {cfg['x0_provenance']}")
        for h in dec.get("harmless", []):
            L.append(f"Omission shown to be irrelevant: {h['path']} - {h['why']}")
        for lim in dec.get("limits", []):
            L.append("Scope: " + lim)
        if dec.get("tradeoff"):
            L.append("Trade-off: " + json.dumps(dec["tradeoff"]["values"]) + " - " + dec["tradeoff"]["note"])
    elif dec["disposition"] == "CLARIFY":
        q = dec["question"]
        L.append("Question: " + q["question"])
        L.append("It affects: " + q["why"])
        if q.get("witness"):
            L.append("Admissible completions with different outcomes: " + json.dumps(jsonable({k: q["witness"][k] for k in ("completion_a", "completion_b") if k in q["witness"]})))
    elif dec["disposition"] == "CONFIRM":
        L.append("Please confirm the interpreted facts: " + ", ".join(dec["pending_confirmation"]))
    else:
        L += ["  - " + e for e in dec["explanations"]]
    return "\n".join(L)


def plot_png(dec: dict, est: dict, title: str = "") -> bytes:
    """Estimate of the target with the appropriate uncertainty quantity (probabilistic band or deterministic envelope)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = dec["config"]
    tgt = cfg.get("target") or 0
    t = np.asarray(est["t"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    if "x_hat" in est:
        x = np.asarray(est["x_hat"])[:, tgt]
        ax[0].plot(t, x, lw=1.2, color="#1f4e79", label="estimate")
        if est.get("envelope") is not None:
            env = np.asarray(est["envelope"])
            ax[0].fill_between(t, x - env, x + env, color="#1f4e79", alpha=0.18, lw=0, label="guaranteed error envelope")
            ax[1].semilogy(t, env, color="#1f4e79"); ax[1].set_ylabel("deterministic error bound")
    else:
        x = np.asarray(est["x_filt"])[:, tgt]
        sd = np.sqrt(np.asarray(est["P_filt"])[:, tgt, tgt])
        ax[0].plot(t, x, lw=1.2, color="#7a3b00", label="estimate")
        ax[0].fill_between(t, x - 2 * sd, x + 2 * sd, color="#7a3b00", alpha=0.18, lw=0, label="+/- 2 standard deviations")
        ax[1].semilogy(t, sd, color="#7a3b00"); ax[1].set_ylabel("error standard deviation")
    ax[0].set_xlabel("time [s]"); ax[1].set_xlabel("time [s]"); ax[0].set_ylabel("estimated quantity"); ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title or dec["selected"], fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    return buf.getvalue()


def plot_b64(dec, est, title=""):
    return base64.b64encode(plot_png(dec, est, title)).decode()
