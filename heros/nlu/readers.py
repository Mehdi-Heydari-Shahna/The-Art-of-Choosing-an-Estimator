"""Readers: propose a role for every numeric mention and request-level labels (and a coordinator proposal).

    RuleReader      deterministic keyword cues (baseline; cues written from the training wording families only)
    LinearReader    logistic regression on hashed context-window features  (the simple trained approach)
    NeuralReader    bidirectional GRU over static word vectors with role, label and proposal heads (compact neural approach)
    NullReader      reads nothing (floor)

What is learned: roles of numbers, request-level labels, and a proposal of the coordinator's decision.
What is NOT learned: number and unit reading (quantities.py), the role/dimension compatibility mask, the
mapping to specification fields (assemble.py), every numerical method and every check.
"""
from __future__ import annotations

import json
import math
import pickle
import re

import numpy as np

from .assemble import ALLOWED, section_of
from .quantities import find_quantities
from .templates import ROLES, OBJECTIVES, DISTRIBUTIONS, HORIZONS, EVIDENCE, FLAGS

_TOK = re.compile(r"<[a-z0-9_:]+>|\[q[123]\]|[a-zA-Z]+(?:'[a-z]+)?|<=|>=|\+/-|[^\sA-Za-z0-9]")


def mask_text(text: str, qs=None):
    """Replace every numeric mention by a placeholder <q:dim>; return (masked text, placeholder order)."""
    qs = find_quantities(text) if qs is None else qs
    out, last = [], 0
    for q in qs:
        out.append(text[last:q.start]); out.append(f" <q:{q.dim}> "); last = q.end
    out.append(text[last:])
    return "".join(out), qs


def tokens_of(text: str, qs=None):
    """Lower-case tokens of the masked text and the token index of each quantity placeholder."""
    masked, qs = mask_text(text, qs)
    toks = [t.lower() for t in _TOK.findall(masked)]
    qpos = [i for i, t in enumerate(toks) if t.startswith("<q:")]
    return toks, qpos, qs


def plant_meta_tokens(plant: dict, stats_in_attachment: bool) -> list:
    lin = plant.get("linear") or {}
    return [f"<kind:{plant.get('kind')}>", f"<n:{len(plant.get('states') or [])}>", f"<ny:{len(plant.get('outputs') or [])}>",
            f"<lin:{int(bool(plant.get('linear')))}>", f"<so:{int(bool(plant.get('second_order')))}>", f"<clock:{lin.get('clock')}>",
            f"<stats:{int(bool(stats_in_attachment))}>"]


def _apply_mask(scores: np.ndarray, dim: str) -> np.ndarray:
    allowed = ALLOWED.get(dim, ["other"])
    m = np.full(len(ROLES), -np.inf)
    for r in allowed:
        m[ROLES.index(r)] = 0.0
    return scores + m


class NullReader:
    name = "null"

    def read(self, text, plant=None, stats_in_attachment=False):
        return {"quantities": [], "objective": "unspecified", "distribution": "unstated", "horizon": "unstated", "evidence": "unstated", "flags": [],
                "proposal": None}


# -------------------------------------------------------------------------------------------------------------
# rules
# -------------------------------------------------------------------------------------------------------------
_CUES = {
    "Ts": ["period", "interval", "every", "ts", "sample time", "cycle", "per"], "fs": [],
    "deadline": ["after", "settling", "from", "onward", "passed", "wait", "first"], "window": ["record", "run", "lasts", "long"],
    "meas_bound": ["never", "bounded", "worst", "max", "<=", "hard", "exceeds", "promise", "off by"],
    "meas_std": ["standard deviation", "sigma", "rms", "std", "roughly", "typical"],
    "init_pos_bound": ["initial", "start", "starting", "(0)"], "init_pos_std": ["initial", "start", "starting"],
    "init_vel_bound": ["initial", "start", "starting", "(0)"], "init_vel_std": ["initial", "start", "starting"],
    "vel_limit": ["range", "faster", "restricted", "magnitude does not", "stays below", "|"],
    "tolerance": ["within", "accuracy", "error <=", "good to", "must not exceed", "worse than", "need"],
    "peak_cap": ["transient", "settling", "even at the start", "capped", "cap"],
    "rms_target": ["rms error"], "mse_target": ["mean-square", "mse"], "meas_var": ["variance", "r ="],
    "acc_bound": ["bounded", "<=", "max", "exceed", "at most", "below", "more than"], "acc_std": ["standard deviation", "sigma", "std", "white", "one-sigma"],
    "prob": ["probability", "confidence", "of the time"],
}
_OBJ_CUES = {"hard_bound": ["guarantee", "worst case", "no matter", "hard", "not on average", "not just on average"],
             "mse_optimal": ["smallest mean-square", "minimum-variance", "minimum mse", "min-variance", "best estimate", "statistically optimal"],
             "mse_bound": ["mean-square accuracy", "mse spec", "on average is fine", "mse <=", "rms error", "mean-square error"],
             "probability": ["probab", "confidence", "of the time", "doesn't have to hold always"]}


class RuleReader:
    name = "rules"

    def read(self, text, plant=None, stats_in_attachment=False):
        qs = find_quantities(text)
        low = text.lower()
        out = []
        for q in qs:
            sec = section_of(text, q.start)
            left = low[max(0, q.start - 60):q.start]
            right = low[q.end:q.end + 30]
            for cut in (";", ".", ","):                       # keep the clause of the quantity
                if cut in left:
                    left = left[left.rindex(cut) + 1:]
            ctx = left + " <q> " + right.split(";")[0].split(".")[0]
            allowed = [r for r in ALLOWED.get(q.dim, ["other"]) if r != "other"]
            best, bs = "other", 0
            for r in allowed:
                sc = sum(1 for c in _CUES.get(r, []) if c in ctx)
                if r in ("init_pos_bound", "init_vel_bound") and any(c in ctx for c in ("std", "sigma", "standard deviation", "uncertainty has")):
                    sc -= 2
                if r in ("init_pos_std", "init_vel_std") and any(c in ctx for c in ("std", "sigma", "standard deviation")):
                    sc += 2
                if r in ("meas_bound", "meas_std") and any(c in ctx for c in ("initial", "start")):
                    sc -= 2
                if r == "tolerance" and sec == "q3":
                    sc += 1
                if r == "deadline" and sec == "q3":
                    sc += 1
                if r == "fs":
                    sc = 2
                if r == "Ts" and sec == "q2":
                    sc += 1
                if r == "prob" and sec == "q3":
                    sc += 1
                if sc > bs:
                    best, bs = r, sc
            out.append({"start": q.start, "end": q.end, "role": best, "conf": None})
        q3 = low[low.index("[q3]"):] if "[q3]" in low else low
        obj = "unspecified"
        for k in ("hard_bound", "mse_optimal", "probability", "mse_bound"):
            if any(c in q3 for c in _OBJ_CUES[k]):
                obj = k
                break
        dist = "gaussian" if ("gaussian" in low and "not gaussian" not in low and "non-gaussian" not in low) else (
            "bounded" if any(c in low for c in ("uniform", "not gaussian", "non-gaussian", "only the means", "not the bell")) else "unstated")
        hor = "steady_state" if any(c in low for c in ("steady", "constant gain", "constant-gain", "fixed gain", "settled")) else (
            "finite" if any(c in low for c in ("transient from", "time-varying", "start-up phase", "include transient")) else "unstated")
        ev = "mechanical_limit" if any(c in low for c in ("limiter", "mechanical stop", "drive limit", "physically cannot", "drive electronics")) else (
            "assumed" if "assum" in low else "unstated")
        flags = [f for f, cues in (("hidden_dynamics", ["not included in the model", "unmodelled lag", "not modelled", "not in the model", "missing from the model", "unmodelled actuator"]),
                                   ("correlated_noise", ["correlated with each other", "noise correlated", "sensor noise are correlated"]),
                                   ("coloured_noise", ["not white", "coloured", "correlated in time"]),
                                   ("irregular_sampling", ["irregular", "dropouts", "missing"])) if any(c in low for c in cues)]
        return {"quantities": out, "objective": obj, "distribution": dist, "horizon": hor, "evidence": ev, "flags": flags, "proposal": None}


# -------------------------------------------------------------------------------------------------------------
# shared feature extraction for the trained readers
# -------------------------------------------------------------------------------------------------------------
def quantity_features(toks, qpos, i):
    p = qpos[i]
    f = {"dim=" + toks[p]: 1.0}
    sec = "q2"
    for t in toks[:p]:
        if t in ("[q1]", "[q2]", "[q3]"):
            sec = t
    f["sec=" + sec] = 1.0
    # clause-limited windows
    stop = {".", ";", "[q1]", "[q2]", "[q3]"}
    L, R = [], []
    j = p - 1
    while j >= 0 and len(L) < 10 and toks[j] not in stop:
        L.append(toks[j]); j -= 1
    j = p + 1
    while j < len(toks) and len(R) < 7 and toks[j] not in stop:
        R.append(toks[j]); j += 1
    for k, t in enumerate(L[:4]):
        f[f"L{k + 1}={t}"] = 1.0
    for k, t in enumerate(R[:3]):
        f[f"R{k + 1}={t}"] = 1.0
    for t in L:
        f["Lw=" + t] = 1.0
    for t in R:
        f["Rw=" + t] = 1.0
    for a, b_ in zip(L[1:], L[:-1]):
        f[f"Lb={a}_{b_}"] = 1.0
    f["nq_before=" + str(min(3, sum(1 for q in qpos if q < p and all(t not in stop for t in toks[q:p]))))] = 1.0
    return f


def request_text(toks, meta=None):
    return " ".join((meta or []) + toks)


class LinearReader:
    """Multinomial logistic regression per task on sparse hashed features (scikit-learn)."""
    name = "linear"

    def __init__(self, C=3.0, seed=0, bootstrap=False):
        self.C, self.seed, self.bootstrap = C, seed, bootstrap
        self.models = {}

    def fit(self, examples, log=None):
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, len(examples), len(examples)) if self.bootstrap else np.arange(len(examples))
        ex = [examples[i] for i in idx]
        X, y = [], []
        for e in ex:
            for i, role in enumerate(e["roles"]):
                X.append(quantity_features(e["toks"], e["qpos"], i)); y.append(role)
        self.qvec = DictVectorizer()
        self.qclf = LogisticRegression(C=self.C, max_iter=2000)
        self.qclf.fit(self.qvec.fit_transform(X), y)
        texts = [request_text(e["toks"]) for e in ex]
        self.tvec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"\S+", min_df=1)
        Xt = self.tvec.fit_transform(texts)
        for head in ("objective", "distribution", "horizon", "evidence"):
            self.models[head] = LogisticRegression(C=self.C * 3, max_iter=2000).fit(Xt, [e[head] for e in ex])
        for fl in FLAGS:
            yb = [int(fl in e["flags"]) for e in ex]
            self.models["flag:" + fl] = LogisticRegression(C=self.C * 3, max_iter=2000).fit(Xt, yb) if 0 < sum(yb) < len(yb) else None
        self.pvec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"\S+", min_df=1)
        Xp = self.pvec.fit_transform([request_text(e["toks"], e["meta"]) for e in ex])
        self.models["proposal"] = LogisticRegression(C=self.C * 3, max_iter=3000).fit(Xp, [e["proposal"] for e in ex])
        self.n_params = int(self.qclf.coef_.size + sum(m.coef_.size for m in self.models.values() if m is not None))
        return self

    def read(self, text, plant=None, stats_in_attachment=False):
        toks, qpos, qs = tokens_of(text)
        quantities = []
        if qs:
            Xq = self.qvec.transform([quantity_features(toks, qpos, i) for i in range(len(qs))])
            lp = self.qclf.predict_log_proba(Xq)
            classes = list(self.qclf.classes_)
            for i, q in enumerate(qs):
                sc = np.full(len(ROLES), -np.inf)
                for c, v in zip(classes, lp[i]):
                    sc[ROLES.index(c)] = v
                sc = _apply_mask(sc, q.dim)
                k = int(np.argmax(sc))
                pr = np.exp(sc - np.logaddexp.reduce(sc[np.isfinite(sc)]))
                quantities.append({"start": q.start, "end": q.end, "role": ROLES[k], "conf": float(pr[k])})
        Xt = self.tvec.transform([request_text(toks)])
        out = {"quantities": quantities}
        for head in ("objective", "distribution", "horizon", "evidence"):
            out[head] = str(self.models[head].predict(Xt)[0])
            out[head + "_conf"] = float(self.models[head].predict_proba(Xt).max())
        out["flags"] = [fl for fl in FLAGS if self.models.get("flag:" + fl) is not None and self.models["flag:" + fl].predict(Xt)[0] == 1]
        meta = plant_meta_tokens(plant, stats_in_attachment) if plant else []
        Xp = self.pvec.transform([request_text(toks, meta)])
        out["proposal"] = str(self.models["proposal"].predict(Xp)[0])
        out["proposal_conf"] = float(self.models["proposal"].predict_proba(Xp).max())
        return out

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


# -------------------------------------------------------------------------------------------------------------
# neural reader (PyTorch, CPU)
# -------------------------------------------------------------------------------------------------------------
def _torch():
    """torch, restricted to one thread: the networks are tiny and requests are processed one at a time, so additional
    threads only add scheduling overhead (measured on the 2-core build machine)."""
    import torch
    if not getattr(_torch, "done", False):
        try:
            torch.set_num_threads(1)
        except Exception:  # noqa
            pass
        _torch.done = True
    return torch


class NeuralReader:
    """BiGRU tagger.  variant: 'pretrained' (frozen static word vectors + trained projection, encoder and heads),
    'scratch' (no pretrained vectors), 'frozen' (pretrained vectors, NO trained encoder: linear heads on window averages)."""
    name = "neural"

    def __init__(self, variant="pretrained", seed=0, hidden=96, epochs=14, lr=2e-3, vectors=None):
        self.variant, self.seed, self.hidden, self.epochs, self.lr = variant, seed, hidden, epochs, lr
        self.vectors = vectors            # (words list, matrix) of the static vectors, or None
        self.net = None

    # -- vocabulary ------------------------------------------------------------------------------------------
    def _build_vocab(self, examples):
        words = {}
        for e in examples:
            for t in e["meta"] + e["toks"]:
                words[t] = words.get(t, 0) + 1
        self.tr_vocab = {w: i + 1 for i, w in enumerate(sorted(words))}          # 0 = unknown / padding
        self.proposals = sorted({e["proposal"] for e in examples})
        if self.vectors is not None and self.variant != "scratch":
            vw, vm = self.vectors
            self.pv_index = {w: i + 1 for i, w in enumerate(vw)}
            self.pv_matrix = np.vstack([np.zeros((1, vm.shape[1]), np.float32), vm.astype(np.float32)])
        else:
            self.pv_index, self.pv_matrix = {}, np.zeros((1, 8), np.float32)

    def _ids(self, toks):
        return [self.tr_vocab.get(t, 0) for t in toks], [self.pv_index.get(t, 0) for t in toks]

    def _make_net(self):
        torch = _torch(); import torch.nn as nn
        torch.manual_seed(self.seed)
        P = self

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.pv = nn.Embedding.from_pretrained(torch.tensor(P.pv_matrix), freeze=True, padding_idx=0)
                s.tr = nn.Embedding(len(P.tr_vocab) + 1, 48, padding_idx=0)
                din = (64 if P.pv_index else 0) + 48
                s.proj = nn.Linear(P.pv_matrix.shape[1], 64) if P.pv_index else None
                s.frozen = P.variant == "frozen"
                if s.frozen:
                    s.enc = None
                    dout = P.pv_matrix.shape[1] * 3
                else:
                    s.enc = nn.GRU(din, P.hidden, batch_first=True, bidirectional=True)
                    dout = 2 * P.hidden
                s.drop = nn.Dropout(0.3)
                s.role = nn.Linear(dout, len(ROLES))
                s.att = nn.Linear(dout, 1) if not s.frozen else None
                dreq = dout if not s.frozen else P.pv_matrix.shape[1]
                s.heads = nn.ModuleDict({"objective": nn.Linear(dreq, len(OBJECTIVES)), "distribution": nn.Linear(dreq, len(DISTRIBUTIONS)),
                                         "horizon": nn.Linear(dreq, len(HORIZONS)), "evidence": nn.Linear(dreq, len(EVIDENCE)),
                                         "flags": nn.Linear(dreq, len(FLAGS)), "proposal": nn.Linear(dreq, len(P.proposals))})

            def forward(s, tr_ids, pv_ids, qpos):
                torch = _torch()
                if s.frozen:
                    v = s.pv(pv_ids)[0]                                    # (T, d)
                    T = v.shape[0]
                    feats = []
                    for p in qpos:
                        a, b = max(0, p - 6), min(T, p + 5)
                        left = v[a:p].mean(0) if p > a else torch.zeros(v.shape[1])
                        right = v[p + 1:b].mean(0) if b > p + 1 else torch.zeros(v.shape[1])
                        feats.append(torch.cat([left, right, v[a:b].mean(0)]))
                    H = torch.stack(feats) if feats else torch.zeros((0, v.shape[1] * 3))
                    pooled = v.mean(0)
                else:
                    x = s.tr(tr_ids)
                    if s.proj is not None:
                        x = torch.cat([torch.tanh(s.proj(s.pv(pv_ids))), x], dim=-1)
                    h, _ = s.enc(s.drop(x))
                    h = s.drop(h[0])
                    H = h[qpos] if len(qpos) else h[:0]
                    w = torch.softmax(s.att(h).squeeze(-1), dim=0)
                    pooled = (w[:, None] * h).sum(0)
                return s.role(H), {k: m(pooled) for k, m in s.heads.items()}

        return Net()

    def fit(self, examples, log=None):
        torch = _torch(); import torch.nn.functional as Fn
        self._build_vocab(examples)
        self.net = self._make_net()
        rng = np.random.default_rng(self.seed)
        params = [p for p in self.net.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=self.lr)
        self.n_params = int(sum(p.numel() for p in params))
        data = []
        for e in examples:
            toks = e["meta"] + e["toks"]
            off = len(e["meta"])
            tr, pv = self._ids(toks)
            data.append((torch.tensor([tr]), torch.tensor([pv]), [p + off for p in e["qpos"]], torch.tensor([ROLES.index(r) for r in e["roles"]], dtype=torch.long),
                         {"objective": OBJECTIVES.index(e["objective"]), "distribution": DISTRIBUTIONS.index(e["distribution"]), "horizon": HORIZONS.index(e["horizon"]),
                          "evidence": EVIDENCE.index(e["evidence"]), "proposal": self.proposals.index(e["proposal"])},
                         torch.tensor([float(f in e["flags"]) for f in FLAGS])))
        for ep in range(self.epochs):
            self.net.train()
            order = rng.permutation(len(data))
            tot = 0.0
            for bi in range(0, len(order), 16):
                opt.zero_grad()
                loss = 0.0
                for i in order[bi:bi + 16]:
                    tr, pv, qpos, roles, lab, fl = data[i]
                    rl, heads = self.net(tr, pv, qpos)
                    if len(qpos):
                        loss = loss + Fn.cross_entropy(rl, roles)
                    for k, v in lab.items():
                        loss = loss + 0.5 * Fn.cross_entropy(heads[k][None, :], torch.tensor([v]))
                    loss = loss + 0.5 * Fn.binary_cross_entropy_with_logits(heads["flags"], fl)
                loss = loss / 16
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                tot += float(loss)
            if log is not None:
                log.append({"epoch": ep + 1, "train_loss": tot / max(1, len(order) // 16)})
        self.net.eval()
        return self

    def read(self, text, plant=None, stats_in_attachment=False):
        torch = _torch()
        toks, qpos, qs = tokens_of(text)
        meta = plant_meta_tokens(plant, stats_in_attachment) if plant else ["<kind:None>"] * 0
        tr, pv = self._ids(meta + toks)
        with torch.no_grad():
            rl, heads = self.net(torch.tensor([tr]), torch.tensor([pv]), [p + len(meta) for p in qpos])
        quantities = []
        rl = rl.numpy()
        for i, q in enumerate(qs):
            sc = _apply_mask(rl[i].astype(float), q.dim)
            k = int(np.argmax(sc))
            fin = sc[np.isfinite(sc)]
            quantities.append({"start": q.start, "end": q.end, "role": ROLES[k], "conf": float(np.exp(sc[k] - np.logaddexp.reduce(fin)))})
        g = lambda k, names: names[int(heads[k].argmax())]
        pc = torch.softmax(heads["proposal"], 0)
        return {"quantities": quantities, "objective": g("objective", OBJECTIVES), "distribution": g("distribution", DISTRIBUTIONS),
                "horizon": g("horizon", HORIZONS), "evidence": g("evidence", EVIDENCE),
                "flags": [f for f, z in zip(FLAGS, heads["flags"].tolist()) if z > 0.0],
                "proposal": self.proposals[int(pc.argmax())], "proposal_conf": float(pc.max()),
                "objective_conf": float(torch.softmax(heads["objective"], 0).max())}

    def save(self, path):
        torch = _torch()
        state = {"variant": self.variant, "seed": self.seed, "hidden": self.hidden, "tr_vocab": self.tr_vocab, "proposals": self.proposals,
                 "pv_words": [w for w, _ in sorted(self.pv_index.items(), key=lambda t: t[1])], "state_dict": {k: v for k, v in self.net.state_dict().items() if k != "pv.weight"},
                 "pv_dim": int(self.pv_matrix.shape[1])}
        torch.save(state, path)

    @staticmethod
    def load(path, vectors=None):
        torch = _torch()
        st = torch.load(path, weights_only=False)
        r = NeuralReader(st["variant"], st["seed"], st["hidden"], vectors=vectors)
        r.tr_vocab, r.proposals = st["tr_vocab"], st["proposals"]
        if st["pv_words"]:
            vw, vm = vectors
            idx = {w: i for i, w in enumerate(vw)}
            r.pv_index = {w: i + 1 for i, w in enumerate(st["pv_words"])}
            r.pv_matrix = np.vstack([np.zeros((1, vm.shape[1]), np.float32), vm[[idx[w] for w in st["pv_words"]]].astype(np.float32)])
        else:
            r.pv_index, r.pv_matrix = {}, np.zeros((1, st["pv_dim"]), np.float32)
        r.net = r._make_net()
        r.net.load_state_dict(st["state_dict"], strict=False)
        r.net.eval()
        return r


def _check_vectors(path, words, M):
    """Verify (words, matrix) against the recorded digests next to the cache (review C-18), if they are there."""
    import os, hashlib, json as _json
    ref = os.path.join(os.path.dirname(path), "vectors_en_core_web_md.sha256.json")
    if os.path.exists(ref):
        r = _json.loads(open(ref).read())
        if (hashlib.sha256("\n".join(words).encode()).hexdigest() != r["words_sha256"]
                or hashlib.sha256(np.ascontiguousarray(M, dtype=np.float32).tobytes()).hexdigest() != r["matrix_sha256"]):
            raise ValueError("word vectors do not match models/vectors_en_core_web_md.sha256.json")


def load_vectors(path=None):
    """Static word vectors of the pinned spaCy pipeline en_core_web_md 3.8.0 as (words, matrix).  If `path` (npz cache) exists it is used.
    Both a cache that is read and one that is first created from spaCy are checked against the recorded digests."""
    import os
    if path and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        words, M = list(z["words"]), z["matrix"]
        _check_vectors(path, words, M)
        return words, M
    import spacy
    nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"])
    vec = nlp.vocab.vectors
    words, rows = [], []
    for key, row in vec.key2row.items():
        w = nlp.vocab.strings[key]
        if w == w.lower() and len(w) < 25:
            words.append(w); rows.append(row)
    M = np.asarray(vec.data)[rows].astype(np.float32)
    if path:
        _check_vectors(path, words, M)              # recheck C, C-18: also when the cache is first created
        np.savez_compressed(path, words=np.array(words), matrix=M)
    return words, M
