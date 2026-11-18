"""NumPy inference for the deployed recurrent reader (no PyTorch needed at run time).

Training uses PyTorch (readers.NeuralReader).  For deployment the weights of a 'scratch' checkpoint are exported once
to `reader_numpy.npz` (+ vocabulary in `reader_numpy.json`) by `export_numpy(...)`, and `NumpyGRUReader` evaluates the
same network: token embedding -> one-layer bidirectional GRU (PyTorch gate equations, eval mode, no dropout) ->
role layer at the number positions, attention pooling -> request-level heads.  Equality of the readings with the
PyTorch implementation on all development and test requests is checked by experiments/check_numpy_reader.py.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from .templates import ROLES, OBJECTIVES, DISTRIBUTIONS, HORIZONS, EVIDENCE, FLAGS
from .readers import tokens_of, plant_meta_tokens, _apply_mask

HEADS = ("objective", "distribution", "horizon", "evidence", "flags", "proposal")


def export_numpy(torch_reader, out_dir) -> dict:
    """Write reader_numpy.npz / reader_numpy.json from a loaded readers.NeuralReader of variant 'scratch'."""
    if torch_reader.variant != "scratch":
        raise ValueError("only the 'scratch' variant (no pretrained vectors) is exported")
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in torch_reader.net.state_dict().items() if k != "pv.weight"}
    out = pathlib.Path(out_dir)
    np.savez(out / "reader_numpy.npz", **sd)
    meta = {"variant": "scratch", "hidden": torch_reader.hidden, "tr_vocab": torch_reader.tr_vocab, "proposals": torch_reader.proposals}
    (out / "reader_numpy.json").write_text(json.dumps(meta))
    return meta


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyGRUReader:
    name = "neural_numpy"

    def __init__(self, directory):
        d = pathlib.Path(directory)
        meta = json.loads((d / "reader_numpy.json").read_text())
        z = np.load(d / "reader_numpy.npz")
        self.W = {k: z[k].astype(np.float64) for k in z.files}
        self.tr_vocab, self.proposals, self.H = meta["tr_vocab"], meta["proposals"], meta["hidden"]

    def _gru(self, x, sfx):
        W, H = self.W, self.H
        Wi, Wh = W["enc.weight_ih_l0" + sfx], W["enc.weight_hh_l0" + sfx]
        bi, bh = W["enc.bias_ih_l0" + sfx], W["enc.bias_hh_l0" + sfx]
        gi = x @ Wi.T + bi                                   # (T, 3H), gate order r, z, n
        h = np.zeros(H)
        out = np.zeros((x.shape[0], H))
        for t in range(x.shape[0]):
            gh = Wh @ h + bh
            r = _sig(gi[t, :H] + gh[:H])
            zz = _sig(gi[t, H:2 * H] + gh[H:2 * H])
            n = np.tanh(gi[t, 2 * H:] + r * gh[2 * H:])
            h = (1.0 - zz) * n + zz * h
            out[t] = h
        return out

    def _forward(self, ids, qpos):
        W = self.W
        x = W["tr.weight"][ids]
        hf = self._gru(x, "")
        hb = self._gru(x[::-1], "_reverse")[::-1]
        h = np.concatenate([hf, hb], axis=1)
        Hq = h[qpos] if len(qpos) else h[:0]
        rl = Hq @ W["role.weight"].T + W["role.bias"]
        a = (h @ W["att.weight"].T + W["att.bias"])[:, 0]
        w = np.exp(a - a.max()); w /= w.sum()
        pooled = (w[:, None] * h).sum(0)
        heads = {k: W[f"heads.{k}.weight"] @ pooled + W[f"heads.{k}.bias"] for k in HEADS}
        return rl, heads

    def read(self, text, plant=None, stats_in_attachment=False):
        toks, qpos, qs = tokens_of(text)
        meta = plant_meta_tokens(plant, stats_in_attachment) if plant else []
        ids = [self.tr_vocab.get(t, 0) for t in meta + toks]
        rl, heads = self._forward(np.array(ids, dtype=int), [p + len(meta) for p in qpos])
        quantities = []
        for i, q in enumerate(qs):
            sc = _apply_mask(rl[i].astype(float), q.dim)
            k = int(np.argmax(sc))
            fin = sc[np.isfinite(sc)]
            quantities.append({"start": q.start, "end": q.end, "role": ROLES[k], "conf": float(np.exp(sc[k] - np.logaddexp.reduce(fin)))})
        g = lambda k, names: names[int(np.argmax(heads[k]))]
        sm = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()
        pc = sm(heads["proposal"])
        return {"quantities": quantities, "objective": g("objective", OBJECTIVES), "distribution": g("distribution", DISTRIBUTIONS),
                "horizon": g("horizon", HORIZONS), "evidence": g("evidence", EVIDENCE),
                "flags": [f for f, zv in zip(FLAGS, heads["flags"].tolist()) if zv > 0.0],
                "proposal": self.proposals[int(np.argmax(pc))], "proposal_conf": float(pc.max()),
                "objective_conf": float(sm(heads["objective"]).max())}
