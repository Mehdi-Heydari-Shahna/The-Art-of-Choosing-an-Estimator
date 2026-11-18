"""Dialogue runner, scripted user, baseline policies and the independent judge used by the evaluation.

The scripted user is NOT a person: it answers a question with the hidden true value (or "unknown" when the truth
has no such fact), and it reviews a readback either attentively (every interpreted field that differs from what
the request states is corrected, every stated fact that was not read is added; each edit is counted) or blindly
(the readback is confirmed as shown).  The blind mode exposes what misreadings would release.

The judge re-verifies every released claim against the hidden truth with the independently written references
(independent/ref_kalman.py, independent/ref_hgo.py); it does not consult the coordinator's own numbers.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np

from .core import spec as S
from .core.assess import jsonable
from .coordinator import coordinator as C
from .kalman import specialist as K
from .khalil import specialist as H
from .nlu import assemble as AS
from .nlu import templates as T
from .nlu.readers import tokens_of, plant_meta_tokens
from .sim import plants as PL

NONRELEASE = ("NO_VERIFIED_CANDIDATE", "PROVED_IMPOSSIBLE", "OUT_OF_SCOPE", "INVALID_INPUT")
ASKABLE = ["objective.kind", "objective.tolerance", "objective.deadline", "objective.mse_target", "objective.prob", "time.Ts", "deterministic.V",
           "deterministic.D", "deterministic.d_bound", "deterministic.domain_evidence", "initial.E", "stochastic.R", "stochastic.Q", "stochastic.Q_kind",
           "initial.P0", "stochastic.distribution", "objective.horizon"]


def stats_in_attachment(family: str) -> bool:
    return not (family in PL.BOTH_METHODS or family in PL.KHALIL_ONLY)


def nl_evaluable(case: dict) -> bool:
    """Categories whose defining feature cannot be expressed in the three answers are not part of the natural-language sets:
    a 'reversed range' is two stated values, which every reader (and every person) reads as an ordinary range."""
    return case["category"] != "invalid_reversed_range"


def nl_stated(case: dict) -> dict:
    """The specification that a natural-language request for this case actually STATES (text plus attachment).
    Fields of the structured `given` specification that no answer carries are removed, so that the attentive review does
    not count them as misreadings: the record length (unless the case is about it) and the whiteness flag of
    text-stated noise levels."""
    st = copy.deepcopy(case["given"])
    if not stats_in_attachment(case["family"]):
        if case["category"] != "invalid_deadline_after_window":
            st["time"]["t_end"] = None
        if S.get(st, "stochastic.white") is True:
            st["stochastic"]["white"] = None
    return st


# ---------------------------------------------------------------------------------------------------------
# training examples
# ---------------------------------------------------------------------------------------------------------
def proposal_label(dec: dict) -> str:
    d = dec["disposition"]
    if d == "RELEASE":
        return "RELEASE:" + dec["selected"]
    if d == "CLARIFY":
        return "CLARIFY:" + dec["question"]["path"]
    return d


def example_from_request(case: dict, req: dict, proposal: str | None = None) -> dict:
    """Supervised example for the readers from a templated request (gold spans known)."""
    text = T.full_text(req)
    off = T.SECTION_OFFSETS(req)
    toks, qpos, qs = tokens_of(text)
    gold = [(off[g["section"]] + g["start"], off[g["section"]] + g["start"] + len(g["text"]), g["role"]) for g in req["gold_quantities"]]
    roles = []
    for q in qs:
        r = [role for a, b, role in gold if a < q.end and b > q.start]
        roles.append(r[0] if r else "other")
    lab = req["labels"]
    return {"id": case["id"], "text": text, "toks": toks, "qpos": qpos, "roles": roles,
            "meta": plant_meta_tokens(case["given"]["plant"], stats_in_attachment(case["family"])),
            "objective": lab["objective"], "distribution": lab["distribution"] or "unstated", "horizon": lab["horizon"], "evidence": lab["evidence"],
            "flags": lab["flags"], "proposal": proposal or "NONE"}


# ---------------------------------------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------------------------------------
def _decide(spec, policy):
    if policy in ("coordinator", "ask_all", "default_fill"):
        return C.decide(spec)
    if policy == "kalman_only":
        return C.decide(spec, methods=[K.METHOD])
    if policy == "khalil_only":
        return C.decide(spec, methods=[H.METHOD])
    if policy == "selector":
        return C.decide(spec, ranking="priority")
    if policy.startswith("single_model:"):
        m = policy.split(":", 1)[1]
        if m in (K.METHOD, H.METHOD):
            sp = copy.deepcopy(spec)
            # no completion reasoning: an unresolved range is replaced by its midpoint
            for p, r in list((sp.get("ranges") or {}).items()):
                if S.get(sp, p) is None and r and r[0] <= r[1]:
                    v = 0.5 * (r[0] + r[1])
                    S.put(sp, p, [[v]] if p == "stochastic.R" else v, status="supplied", source="midpoint of the stated range (no completion reasoning)")
            sp["ranges"] = {}
            return C.decide(sp, methods=[m])
        d = C._base(spec)
        d["disposition"] = m if m in NONRELEASE else "NO_VERIFIED_CANDIDATE"
        d["explanations"] = ["decision taken by the single model without specialist checks"]
        return d
    raise ValueError(policy)


DEFAULTS = {"deterministic.V": 0.0, "deterministic.D": 0.0, "deterministic.d_bound": 0.0, "initial.E": [0.0, 0.0], "objective.kind": "mse_optimal",
            "objective.horizon": "steady_state", "stochastic.distribution": "gaussian",
            "deterministic.domain_evidence": {"kind": "assumed", "statement": "default"}}


def _missing_fields(spec):
    """Every unresolved material field that is relevant to the plant class (used by the ask-everything baseline)."""
    kind = S.get(spec, "plant.kind")
    out = []
    obj = S.get(spec, "objective.kind")
    cand = ["objective.kind", "time.Ts"]
    if kind == "second_order":
        cand += ["deterministic.V", "initial.E"]
        so = S.get(spec, "plant.second_order") or {}
        exact = str(so.get("f")) == str(so.get("f0")) and not any(r[0] != r[1] for r in (so.get("param_ranges") or {}).values())
        cand += ["deterministic.D"] if exact else ["deterministic.d_bound", "deterministic.domain_evidence"]
    if S.get(spec, "plant.linear"):
        cand += ["stochastic.R", "stochastic.Q", "initial.P0", "stochastic.distribution"]
    if obj in ("hard_bound", "probability"):
        cand += ["objective.tolerance"]
    if obj == "hard_bound":
        cand += ["objective.deadline"]
    if obj == "mse_bound":
        cand += ["objective.mse_target"]
    if obj in ("mse_optimal", "mse_bound", "probability"):
        cand += ["objective.horizon"]
    lin = S.get(spec, "plant.linear") or {}
    for p in cand:
        if p == "time.Ts" and lin.get("clock") == "discrete" and lin.get("Ts_native") is not None:
            continue
        if S.get(spec, p) is None:
            out.append(p)
    return out


class ScriptedUser:
    """A program, not a person.  mode: 'attentive' (reviews the readback against what the request states), 'blind'
    (confirms the readback unread); unknown_paths: fields the user cannot answer exactly (e.g. those for which only a
    range was stated: the 'limited' variant of the structured test)."""

    def __init__(self, truth: dict, mode: str = "attentive", unknown_paths=None):
        self.truth, self.mode = truth, mode
        self.unknown_paths = set(unknown_paths or [])
        self.n_answers = 0
        self.n_unknown = 0

    def answer(self, path: str):
        """True value of a field, or None ('unknown')."""
        if path in self.unknown_paths:
            self.n_unknown += 1
            return None
        v = S.get(self.truth, path)
        if v is None:
            self.n_unknown += 1
        else:
            self.n_answers += 1
        return copy.deepcopy(v)

    def review(self, spec: dict, stated: dict) -> tuple:
        """Readback review.  `stated` is the specification that the request text actually states (the `given` spec).
        Returns (spec after review, number of edits)."""
        if self.mode == "blind":
            return S.confirm(spec), 0
        sp = copy.deepcopy(spec)
        edits = 0
        paths = set(S.MATERIAL_PATHS) | {"time.t_end", "time.irregular"}
        for p in sorted(paths):
            cur, ref = S.get(sp, p), S.get(stated, p)
            st = (sp.get("evidence") or {}).get(p, {}).get("status")
            if st not in ("interpreted", None, "unresolved") and cur is not None:
                continue                                    # supplied by the attachment
            rng_cur, rng_ref = (sp.get("ranges") or {}).get(p), (stated.get("ranges") or {}).get(p)
            same = _same(cur, ref) and _same(rng_cur, rng_ref)
            if p == "initial.E" and cur is not None and ref is not None and _same(cur[1], ref[1]):
                same = True                                 # the position bound is not used with first-measurement initialisation
            if p == "deterministic.domain_evidence" and cur is not None and ref is not None and cur.get("kind") == ref.get("kind"):
                same = True
            if p == "stochastic.cross_cov" and cur is not None and ref is not None:
                same = True
            if p == "deterministic.domain" and cur is not None and ref is not None and _same(cur.get("x2"), ref.get("x2")):
                same = True
            if not same:
                edits += 1
                if ref is None:
                    S.put(sp, p, None, status="unresolved", source="readback correction")
                else:
                    S.put(sp, p, copy.deepcopy(ref), status="supplied", source="readback correction")
                if rng_ref is not None:
                    sp.setdefault("ranges", {})[p] = list(rng_ref)
                else:
                    (sp.get("ranges") or {}).pop(p, None)
        fl_ref = list(S.get(stated, "plant.flags") or [])
        if sorted(S.get(sp, "plant.flags") or []) != sorted(fl_ref):
            edits += 1
            sp["plant"]["flags"] = fl_ref
        return S.confirm(sp), edits


def _same(a, b, rtol=1e-6):
    if a is None or b is None:
        return a is None and b is None
    try:
        x, y = np.asarray(a, float), np.asarray(b, float)
        return x.shape == y.shape and np.allclose(x, y, rtol=rtol, atol=1e-300)
    except (TypeError, ValueError):
        return a == b


def run_dialogue(spec0: dict, truth: dict, policy: str = "coordinator", user_mode: str = "attentive", stated: dict | None = None,
                 max_rounds: int = 8, proposal: str | None = None, unknown_paths=None) -> dict:
    """Run one request to its final outcome.  spec0: structured specification or assembled reading."""
    user = ScriptedUser(truth, user_mode, unknown_paths)
    sess = C.Session(spec0)
    out = {"questions": [], "edits": 0, "confirmations": 0, "rounds": 0, "policy": policy}
    t0 = time.perf_counter()
    if S.unconfirmed(sess.spec) or stated is not None and policy != "structured":
        if S.unconfirmed(sess.spec) or spec0.get("interpretation_issues") is not None:
            sess.spec, e = user.review(sess.spec, stated if stated is not None else sess.spec)
            out["edits"] += e
            out["confirmations"] += 1
    pol = policy
    if policy == "single_model":
        m = (proposal or "").split(":")
        pol = "single_model:" + (m[1] if m[0] == "RELEASE" and len(m) > 1 else (m[0] if m[0] in NONRELEASE else "__clarify__"))
        if pol.endswith("__clarify__"):
            # the single model wants to ask: it asks its proposed field, then decides with the method it would then pick (HGO for
            # chain-form plants, Kalman otherwise) -- still without cross-method comparison
            path = m[1] if len(m) > 1 else None
            if path:
                v = user.answer(path)
                out["questions"].append(path)
                if v is not None:
                    sess.update({path: v})
                    if path == "objective.kind":
                        for p in ("objective.tolerance", "objective.deadline", "objective.mse_target", "objective.prob", "objective.horizon"):
                            if S.get(sess.spec, p) is None and S.get(truth, p) is not None:
                                sess.update({p: S.get(truth, p)})
                                user.n_answers += 1
            obj = S.get(sess.spec, "objective.kind")
            pol = "single_model:" + (H.METHOD if (S.get(sess.spec, "plant.kind") == "second_order" and obj in ("hard_bound", None)) else K.METHOD)
    if policy == "ask_all":
        for p in _missing_fields(sess.spec):
            out["questions"].append(p)
            v = user.answer(p)
            if v is not None:
                sess.update({p: v})
        for p in _missing_fields(sess.spec):          # the objective answer may open further fields
            if p not in out["questions"]:
                out["questions"].append(p)
                v = user.answer(p)
                if v is not None:
                    sess.update({p: v})
    if policy == "default_fill":
        for p in _missing_fields(sess.spec):
            if p in DEFAULTS:
                sess.update({p: copy.deepcopy(DEFAULTS[p])}, status="accepted_choice", source="default (never asked)")
        for p, r in list((sess.spec.get("ranges") or {}).items()):
            if S.get(sess.spec, p) is None and r and r[0] <= r[1]:
                v = 0.5 * (r[0] + r[1])
                sess.update({p: [[v]] if p == "stochastic.R" else v}, status="accepted_choice", source="default (midpoint)")
    dec = None
    for rnd in range(max_rounds):
        out["rounds"] = rnd + 1
        dec = _decide(sess.spec, pol)
        if dec["disposition"] != "CLARIFY":
            break
        if policy == "default_fill":
            break
        path = dec["question"]["path"]
        if path in out["questions"] and policy != "ask_all":
            sess.spec.setdefault("unavailable", []).append(path)
            continue
        out["questions"].append(path) if path not in out["questions"] else None
        v = user.answer(path)
        if v is None:
            sess.spec.setdefault("unavailable", []).append(path)
            continue
        sess.update({path: v})
        if path == "objective.kind":
            for p in ("objective.tolerance", "objective.deadline", "objective.mse_target", "objective.prob", "objective.horizon"):
                if S.get(sess.spec, p) is None and S.get(truth, p) is not None:
                    sess.update({p: S.get(truth, p)})
                    user.n_answers += 1
    out.update(decision=dec, final_spec=sess.spec, seconds=time.perf_counter() - t0, n_questions=len(out["questions"]),
               n_answers=user.n_answers, n_unknown=user.n_unknown)
    return out


def run_request(reader, case: dict, text: str, policy: str = "coordinator", user_mode: str = "attentive") -> dict:
    """End-to-end run of one natural-language request with the given reader."""
    att = AS.attachment_of(case["given"], stats_in_attachment(case["family"]))
    t0 = time.perf_counter()
    reading = reader.read(text, att["plant"], stats_in_attachment(case["family"]))
    t_read = time.perf_counter() - t0
    asm = AS.assemble(att, text, reading)
    out = run_dialogue(asm["spec"], case["truth"], policy, user_mode, stated=nl_stated(case), proposal=reading.get("proposal"))
    out.update(reading=reading, issues=asm["issues"], seconds_read=t_read, assembled=asm["spec"])
    return out


# ---------------------------------------------------------------------------------------------------------
# independent judge
# ---------------------------------------------------------------------------------------------------------
_WC_CACHE = {}


def _true_model(truth):
    from independent import ref_kalman as RK
    lin = S.get(truth, "plant.linear")
    A = np.atleast_2d(np.asarray(lin["A"], float)); C_ = np.atleast_2d(np.asarray(lin["C"], float))
    B = np.atleast_2d(np.asarray(lin["B"], float)) if lin.get("B") is not None else np.zeros((A.shape[0], 1))
    if lin["clock"] == "continuous":
        Ts = S.get(truth, "time.Ts")
        Ad, Bd = RK.zoh_quadrature(A, B, Ts)
        Q = np.atleast_2d(np.asarray(S.get(truth, "stochastic.Q"), float))
        if S.get(truth, "stochastic.Q_kind") == "continuous_psd":
            G = S.get(truth, "stochastic.G")
            G = np.eye(A.shape[0]) if G is None else np.atleast_2d(np.asarray(G, float))
            Qd = RK.process_cov_quadrature(A, G, Q, Ts)
        else:
            Qd = Q
    else:
        Ts = lin.get("Ts_native") or S.get(truth, "time.Ts")
        Ad, Bd, Qd = A, B, np.atleast_2d(np.asarray(S.get(truth, "stochastic.Q"), float))
    return Ad, Bd, C_, Qd, np.atleast_2d(np.asarray(S.get(truth, "stochastic.R"), float)), Ts


def _fixed_gain_cov_seq(A, C_, Qd, R, P0, Ms):
    """Independent (re-written) covariance recursion of a given gain sequence; returns filtered covariances."""
    P = np.array(P0, float); out = []
    n = A.shape[0]
    for M in Ms:
        J = np.eye(n) - M @ C_
        Pf = J @ P @ J.T + M @ R @ M.T
        out.append(Pf)
        P = A @ Pf @ A.T + Qd
    return out



def _released_gain_seq(cfg):
    """Gain sequence of a released Kalman configuration, recomputed here from the released matrices (independent of the
    package's gain code): constant for a steady-state design, the Riccati recursion from the released P0 otherwise."""
    if cfg.get("mode") == "steady_state":
        M = np.asarray(cfg["M_ss"], float)
        while True:
            yield M
    A, C_ = np.asarray(cfg["A"], float), np.asarray(cfg["C"], float)
    Qd, R, P = np.asarray(cfg["Qd"], float), np.atleast_2d(np.asarray(cfg["R"], float)), np.atleast_2d(np.asarray(cfg["P0"], float))
    n = A.shape[0]
    while True:
        G = C_ @ P @ C_.T + R
        M = np.linalg.solve(G, C_ @ P).T
        yield M
        J = np.eye(n) - M @ C_
        P = A @ (J @ P @ J.T + M @ R @ M.T) @ A.T + Qd


def _optimal_gain_seq(A, C_, Qd, R, P0):
    """Gains of the optimal (time-varying) filter under the TRUE model and initial covariance."""
    return _released_gain_seq({"mode": "time_varying", "A": A, "C": C_, "Qd": Qd, "R": R, "P0": P0})


def _window_max(A, C_, Qd, R, P0, gains, tgt, kd, n_end, kmax=400000, rtol=1e-14):
    """Independent check of a finite-time claim: largest error variance of the target over the samples kd..n_end, or kd..inf
    when n_end is None (iterated until gain and covariance have settled; the limit, from a Lyapunov solution, is included).
    Returns (value, settled)."""
    from independent import ref_kalman as RK
    P = np.atleast_2d(np.asarray(P0, float)); n = A.shape[0]
    best, k, quiet, prev, Mprev = -np.inf, 0, 0, None, None
    last = n_end if n_end is not None else kmax
    for M in gains:
        J = np.eye(n) - M @ C_
        Pf = J @ P @ J.T + M @ R @ M.T
        v = float(Pf[tgt, tgt])
        if k >= kd:
            best = max(best, v)
        if n_end is None and k > kd:
            dv = abs(v - prev) <= rtol * max(abs(v), 1e-300)
            dm = Mprev is not None and np.max(np.abs(M - Mprev)) <= rtol * max(np.max(np.abs(M)), 1e-300)
            quiet = quiet + 1 if (dv and dm) else 0
            if quiet >= 200:
                lim = float(RK.lyapunov_mse_of_gain(A, C_, Qd, R, M)[tgt, tgt])
                return max(best, lim), True
        if k >= last:
            return best, n_end is not None
        prev, Mprev = v, M
        P = A @ Pf @ A.T + Qd
        k += 1
    return best, False


def judge_release(truth: dict, dec: dict) -> dict:
    """Is the released claim true under the hidden complete specification?  Returns {"correct": bool, "why": str, ...}."""
    from independent import ref_kalman as RK, ref_hgo as RH
    cfg, claim = dec["config"], dec["claim"]
    tgt = S.get(truth, "target.state")
    obj = S.get(truth, "objective.kind")
    res = {"correct": False, "why": "", "checks": {}}
    if S.get(truth, "plant.flags") or S.get(truth, "plant.kind") in (None, "other"):
        res["why"] = "a design was released for a request outside the supported scope"
        return res
    if S.get(truth, "time.irregular"):
        res["why"] = "a design was released although the samples are irregular (outside the method register)"
        return res
    te_, dl_ = S.get(truth, "time.t_end"), S.get(truth, "objective.deadline")
    if te_ is not None and dl_ is not None and te_ < dl_:
        res["why"] = f"a design was released although the record ends ({te_} s) before the deadline ({dl_} s)"
        return res
    cap_ = S.get(truth, "objective.peak_cap")
    if cap_ is not None and (claim or {}).get("uncertainty") != "deterministic error envelope":
        res["why"] = "the request caps the transient error, which no statistical claim certifies"
        return res
    if obj is None or claim is None:
        res["why"] = "no objective in the truth / no claim"
        return res
    ctype = claim["type"]
    want = {"hard_bound": ("hard_bound",), "mse_optimal": ("mmse", "lmmse"), "mse_bound": ("mse_bound", "mse_bound_robust", "mmse", "lmmse"),
            "probability": ("probability",)}[obj]
    if ctype not in want:
        res["why"] = f"claim type {ctype} does not answer the requested objective {obj}"
        return res
    if cfg.get("target") is not None and tgt is not None and cfg.get("target") != tgt:
        res["why"] = f"estimate refers to state {cfg.get('target')} but state {tgt} was requested"
        return res
    Ts_true = S.get(truth, "time.Ts") or (S.get(truth, "plant.linear") or {}).get("Ts_native")
    if Ts_true is not None and abs(cfg["Ts"] - Ts_true) > 1e-9 * Ts_true:
        res["why"] = f"design computed for period {cfg['Ts']} s, true period {Ts_true} s"
        return res
    if dec["selected"] == K.METHOD and claim.get("uncertainty") != "deterministic error envelope":
        pr_true = {k: r for k, r in ((S.get(truth, "plant.second_order") or {}).get("param_ranges") or {}).items() if r and r[1] > r[0]}
        if pr_true:
            res["why"] = ("a statistical claim was released for the nominal model although the true parameters are known only as ranges "
                          f"({sorted(pr_true)}); the error covariance of the nominal filter is not that of the true plant")
            return res
        if any(S.get(truth, p) is None for p in ("stochastic.Q", "stochastic.R")):
            res["why"] = "a statistical claim was released although the truth has no statistical model"
            return res
        if S.get(truth, "stochastic.white") is False or (S.get(truth, "stochastic.cross_cov") is not None and np.any(np.asarray(S.get(truth, "stochastic.cross_cov")) != 0)):
            res["why"] = "white, uncorrelated noise assumed against the truth"
            return res
        Ad, Bd, C_, Qd, R, Ts = _true_model(truth)
        if not (np.allclose(np.asarray(cfg["A"]), Ad, rtol=1e-8, atol=1e-11 * max(1.0, abs(Ad).max())) and np.allclose(np.asarray(cfg["C"]), C_)
                and np.allclose(np.asarray(cfg["B"]), Bd, rtol=1e-8, atol=1e-11 * max(1.0, abs(Bd).max() if Bd.size else 1.0))):
            res["why"] = "released model differs from the true model"
            return res
        steady = cfg.get("mode") == "steady_state"
        hor_true = S.get(truth, "objective.horizon") or "finite"
        dl = S.get(truth, "objective.deadline")
        te = S.get(truth, "time.t_end")
        kd, n_end = _window(Ts, dl, te)
        asymptotic = hor_true == "steady_state" and dl is None
        if asymptotic:
            # the request concerns the limit k -> infinity: the released gain's limiting error variance
            if steady:
                M_lim = np.asarray(cfg["M_ss"], float)
            else:
                g = _released_gain_seq(cfg); M_lim = None
                for _ in range(400000):
                    Mn = next(g)
                    if M_lim is not None and np.max(np.abs(Mn - M_lim)) <= 1e-14 * max(np.max(np.abs(Mn)), 1e-300):
                        break
                    M_lim = Mn
            true_mse = float(RK.lyapunov_mse_of_gain(Ad, C_, Qd, R, M_lim)[tgt, tgt])
            opt = float(RK.riccati_fixed_point(Ad, C_, Qd, R)[1][tgt, tgt])
        else:
            # a statement at finite times (deadline or finite horizon): computed from the TRUE initial covariance
            P0t = S.get(truth, "initial.P0")
            if P0t is None:
                res["why"] = "a finite-time error statement was released although the truth has no initial covariance"
                return res
            P0t = np.atleast_2d(np.asarray(P0t, float))
            true_mse, ok1 = _window_max(Ad, C_, Qd, R, P0t, _released_gain_seq(cfg), tgt, kd, n_end)
            opt, ok2 = _window_max(Ad, C_, Qd, R, P0t, _optimal_gain_seq(Ad, C_, Qd, R, P0t), tgt, kd, n_end)
            if not (ok1 and ok2):
                res["why"] = "the error variance did not settle within the judge's iteration limit; the claim could not be verified"
                return res
        res["checks"].update(true_mse=true_mse, optimum=opt, claimed=claim.get("value"), asymptotic=asymptotic)
        if obj == "mse_optimal":
            if hor_true == "steady_state" and not steady:
                pass                                         # a time-varying optimal filter also answers a steady-state request
            if hor_true == "finite" and steady:
                res["why"] = "a steady-state filter was released for a request that includes the transient"
                return res
            if steady and not asymptotic:
                # constant gain released for a deadline in steady-state mode: optimal only in the limit, and the stated
                # value must bound the variance over the requested samples
                lim = float(RK.lyapunov_mse_of_gain(Ad, C_, Qd, R, np.asarray(cfg["M_ss"], float))[tgt, tgt])
                lim_opt = float(RK.riccati_fixed_point(Ad, C_, Qd, R)[1][tgt, tgt])
                ok = lim <= lim_opt * (1 + 1e-6) and claim["value"] >= true_mse * (1 - 1e-6) and claim["value"] <= true_mse * (1 + 1e-5)
            else:
                ok = true_mse <= opt * (1 + 1e-6) and abs(claim["value"] - true_mse) <= 1e-5 * true_mse
            dist_t = S.get(truth, "stochastic.distribution")
            if ctype == "mmse" and dist_t != "gaussian":
                ok = False; res["why"] = "conditional-mean claim although the truth is not Gaussian"
            res["correct"] = bool(ok)
            res["why"] = res["why"] or ("optimal under the truth" if ok else f"released gain is not optimal under the truth (mse {true_mse:.4g} vs optimum {opt:.4g}) or the claimed value is wrong")
        elif obj == "mse_bound":
            target = S.get(truth, "objective.mse_target")
            ok = true_mse <= target * (1 + 1e-9) and claim["value"] >= true_mse * (1 - 1e-6)
            res["correct"] = bool(ok); res["why"] = "true mean-square error within the target" if ok else f"true mse {true_mse:.4g} vs target {target:.4g}, claimed {claim['value']:.4g}"
        else:
            tol, pr = S.get(truth, "objective.tolerance"), S.get(truth, "objective.prob")
            gauss = S.get(truth, "stochastic.distribution") == "gaussian"
            level = math.erf(tol / math.sqrt(2 * true_mse)) if gauss else max(0.0, 1 - true_mse / tol ** 2)
            ok = level >= pr - 1e-12 and claim["value"] <= level + 1e-9
            res["checks"]["true_level"] = level
            res["correct"] = bool(ok); res["why"] = "provable level meets the request" if ok else f"true provable level {level:.4f} vs requested {pr}, claimed {claim['value']:.4f}"
        return res
    # ---- deterministic claims (HGO, or a constant Kalman gain on the double integrator)
    if obj not in ("hard_bound", "probability", "mse_bound"):
        res["why"] = "deterministic claim for a statistical objective"
        return res
    if obj == "mse_bound":
        # an HGO released for a mean-square level: verify with the independent Lyapunov solution
        Ad, Bd, C_, Qd, R, Ts = _true_model(truth)
        M = np.asarray(cfg["M"], float).reshape(2, 1)
        dl = S.get(truth, "objective.deadline")
        if (S.get(truth, "objective.horizon") or "finite") == "steady_state" and dl is None:
            true_mse = float(RK.lyapunov_mse_of_gain(Ad, C_, Qd, R, M)[tgt, tgt])
        else:
            P0t = S.get(truth, "initial.P0")
            if P0t is None:
                res["why"] = "a finite-time mean-square statement was released although the truth has no initial covariance"
                return res
            te = S.get(truth, "time.t_end")
            kd, ne = _window(Ts, dl, te)

            def _const():
                while True:
                    yield M
            true_mse, settled = _window_max(Ad, C_, Qd, R, np.atleast_2d(np.asarray(P0t, float)), _const(), tgt, kd, ne)
            if not settled:
                res["why"] = "the error variance did not settle within the judge's iteration limit; the claim could not be verified"
                return res
        target = S.get(truth, "objective.mse_target")
        res["checks"].update(true_mse=true_mse, claimed=claim.get("value"))
        res["correct"] = bool(true_mse <= target * (1 + 1e-9) and claim.get("value") is not None and claim["value"] >= true_mse * (1 - 1e-6))
        res["why"] = f"true mse {true_mse:.4g} vs target {target:.4g}, claimed {claim.get('value')}"
        return res
    V = S.get(truth, "deterministic.V")
    if V is None:
        res["why"] = "a hard-bound claim was released although the truth has no hard measurement bound"
        return res
    # true total disturbance: supplied D, or sampled lower bound of the derived one (necessary condition for the released D)
    Dt = S.get(truth, "deterministic.D")
    M = np.asarray(cfg["M"] if "M" in cfg else cfg["M_ss"], float).reshape(2)
    D_rel = cfg.get("D") if "D" in cfg else (cfg.get("hard_bound") or {}).get("D", S.get(truth, "deterministic.D"))
    so_t = S.get(truth, "plant.second_order") or {}
    if Dt is not None and D_rel is not None and D_rel < Dt * (1 - 1e-12):
        # the release used a bound tighter than the supplied total bound: it must come from the stated model (the additive bound
        # plus the mismatch); a sampled value of the true mismatch is a NECESSARY check of it, as for a derived bound
        if S.get(truth, "deterministic.d_bound") is None or not so_t.get("f"):
            res["why"] = f"released disturbance bound {D_rel} is below the supplied total bound {Dt} and no model statement supports it"
            return res
        try:
            Ds = _sampled_disturbance(truth)
        except Exception as ex:  # noqa
            res["why"] = f"the tighter released disturbance bound could not be checked ({type(ex).__name__})"
            return res
        res["checks"]["sampled_true_mismatch"] = Ds
        if D_rel < Ds * (1 - 1e-9):
            res["why"] = f"released disturbance bound {D_rel} is below a sampled value {Ds:.6g} of the true mismatch"
            return res
        Dt = D_rel
    if Dt is None:
        so = S.get(truth, "plant.second_order") or {}
        if S.get(truth, "deterministic.d_bound") is None or (S.get(truth, "deterministic.domain_evidence") or {}).get("kind") in (None, "none"):
            res["why"] = "a hard-bound claim was released without the disturbance bound / domain evidence that the truth requires"
            return res
        Dt = _sampled_disturbance(truth)
        res["checks"]["sampled_true_mismatch"] = Dt
        if D_rel is None or D_rel < Dt * (1 - 1e-9):
            res["why"] = f"released disturbance bound {D_rel} is below a sampled value {Dt:.6g} of the true mismatch"
            return res
        Dt = D_rel                      # the released (larger) bound is used: the sampled value is a NECESSARY check only
    Et = S.get(truth, "initial.E")
    dl, tol, cap = S.get(truth, "objective.deadline"), S.get(truth, "objective.tolerance"), S.get(truth, "objective.peak_cap")
    Ts = cfg["Ts"]
    c = np.eye(2)[tgt]
    kd, ne = _window(Ts, dl, S.get(truth, "time.t_end"))
    if dl is None:
        ne = None
    pol = (cfg.get("init") or {}).get("policy") or ("first_measurement" if cfg.get("init_policy") == "first_measurement" or "M_ss" in cfg else "given")
    init = "first_measurement" if pol == "first_measurement" else "box"
    if dl is not None or cap is not None:
        if Et is None:
            res["why"] = "a finite-time claim was released although the truth has no initial-error bound"
            return res
        E = [V if pol == "first_measurement" else Et[0], Et[1]]
        pts = sorted({k for k in (kd, kd + 1, kd + 7, kd + 40) if ne is None or k <= ne})
    else:
        E = [0.0, 0.0]
        rho = max(abs(np.linalg.eigvals(_phi(Ts, M))))
        if rho >= 1:
            res["why"] = "released gain is not stable"
            return res
        pts = [int(min(600, max(60, 30 / max(1e-9, -math.log(rho)))))]
    worst = 0.0
    for n in pts:
        key = (Ts, M[0], M[1], tgt, E[0], E[1], Dt, V, n, init)
        if key not in _WC_CACHE:
            _WC_CACHE[key] = float(RH.worst_case_exact(Ts, M, c, E, Dt, V, n, form="current", sub=4, init=init))
        worst = max(worst, _WC_CACHE[key])
    if cap is not None:
        peak_seen = 0.0
        for n in range(0, max(pts) + 1):
            key = (Ts, M[0], M[1], tgt, E[0], E[1], Dt, V, n, init)
            if key not in _WC_CACHE:
                _WC_CACHE[key] = float(RH.worst_case_exact(Ts, M, c, E, Dt, V, n, form="current", sub=4, init=init))
            peak_seen = max(peak_seen, _WC_CACHE[key])
        res["checks"]["independent_peak"] = peak_seen
        if peak_seen > cap * (1 + 1e-9):
            res["why"] = f"independent worst case {peak_seen:.6g} exceeds the requested transient cap {cap}"
            return res
    res["checks"].update(independent_worst_case=worst, claimed=claim["value"], tolerance=tol, points=pts)
    ok = worst <= claim["value"] * (1 + 1e-7) and claim["value"] <= tol * (1 + 1e-12)
    if obj == "probability":
        ok = worst <= tol * (1 + 1e-9)
    res["correct"] = bool(ok)
    res["why"] = "independent worst case within the claimed bound and the tolerance" if ok else f"independent worst case {worst:.6g}, claimed {claim['value']:.6g}, tolerance {tol}"
    return res


def _ratio_slack(r):
    """Judge's allowance when a ratio of two stated times is compared with an integer: 8 units of float64 rounding of
    the ratio (decimal input), nothing more, so an endpoint stated 2^-31 s before a sample excludes that sample."""
    return 8 * 2.220446049250313e-16 * max(1.0, abs(r))


def _window(Ts, dl, te):
    """Judge's own reading of a requirement window: samples k with k Ts >= deadline and k Ts <= t_end (allowance:
    _ratio_slack; revision 2b used 1e-9, which admitted a sample after an endpoint 0.47 ns before it, review case C)."""
    kd = int(math.ceil(dl / Ts - _ratio_slack(dl / Ts))) if dl is not None else 0
    ne = int(math.floor(te / Ts + _ratio_slack(te / Ts))) if te is not None else None
    return max(kd, 0), ne


def _phi(Ts, M):
    return np.array([[1.0, Ts], [0.0, 1.0]]) @ (np.eye(2) - np.outer(M, [1.0, 0.0]))


def _sampled_disturbance(truth, n=40000, seed=0, starts=16, sweeps=3, grid=401):
    """LOWER estimate of sup |x2'(t) - f0(y_k, u_k)| over the admissible set (independent of the interval code):
    |f(x1, x2, u; params) - f0(x1 + delta, u)| + d_bound, with x1, x2, u in the operating domain, parameters in their
    ranges and the measurement offset |delta| <= V + |x2|max Ts (y_k kept within V of the domain).  A necessary check
    only.  Search: random samples in which each coordinate is independently at a bound or inside its range (faces and
    edges of the box are covered, e.g. an interior angle with input and parameters at their bounds), plus pure corners;
    from the best samples, coordinate sweeps on a dense grid (robust to the kink of |.| and to box corners, where a
    gradient method from a corner stopped at once: review B, recheck N3), then L-BFGS-B."""
    import sympy as sp_
    from scipy.optimize import minimize
    so = S.get(truth, "plant.second_order")
    dom = S.get(truth, "deterministic.domain") or {}
    rng = np.random.default_rng(seed)
    pnames = sorted(so["params"])
    pr = so.get("param_ranges") or {}
    names = ["x1", "x2", "u"] + pnames
    syms = [sp_.Symbol(nm) for nm in names]
    f = sp_.lambdify(syms, sp_.sympify(so["f"]), "numpy")
    f0e = sp_.sympify(str(so["f0"])).subs({sp_.Symbol(k): v for k, v in so["params"].items()})
    f0 = sp_.lambdify([sp_.Symbol("y"), sp_.Symbol("u")], f0e, "numpy")
    V, Ts = S.get(truth, "deterministic.V"), S.get(truth, "time.Ts")
    w = max(abs(dom["x2"][0]), abs(dom["x2"][1])) if dom.get("x2") else 0.0
    off = V + w * Ts
    lo = [dom.get("x1", (0.0, 0.0))[0], dom.get("x2", (0.0, 0.0))[0], dom.get("u", (0.0, 0.0))[0]] + [pr.get(k, (so["params"][k],) * 2)[0] for k in pnames] + [-off]
    hi = [dom.get("x1", (0.0, 0.0))[1], dom.get("x2", (0.0, 0.0))[1], dom.get("u", (0.0, 0.0))[1]] + [pr.get(k, (so["params"][k],) * 2)[1] for k in pnames] + [off]
    lo, hi = np.array(lo, float), np.array(hi, float)
    d = lo.size
    X = rng.uniform(lo, hi, (n, d))
    side = np.where(rng.random((n, d)) < 0.5, lo, hi)
    X = np.where(rng.random((n, d)) < 0.5, side, X)                   # each coordinate at a bound or inside
    X[: n // 8] = side[: n // 8]                                       # pure corners
    ylo, yhi = lo[0] - V, hi[0] + V

    def g(z):
        y = np.clip(z[..., 0] + z[..., -1], ylo, yhi)
        args = [z[..., i] for i in range(d - 1)]
        return np.abs(f(*args) - f0(y, z[..., 2]) * np.ones_like(y))
    vals = g(X)
    best = float(np.max(vals))
    for idx in np.argsort(vals)[-starts:]:
        z = X[idx].copy()
        for _ in range(sweeps):
            for i in range(d):
                if hi[i] > lo[i]:
                    Z = np.repeat(z[None], grid, 0)
                    Z[:, i] = np.linspace(lo[i], hi[i], grid)
                    gv = g(Z)
                    j = int(np.argmax(gv))
                    if gv[j] > g(z[None])[0]:
                        z = Z[j].copy()
        best = max(best, float(g(z[None])[0]))
        r = minimize(lambda q: -float(g(q[None])[0]), z, bounds=list(zip(lo, hi)), method="L-BFGS-B")
        best = max(best, -float(r.fun))
    return best + S.get(truth, "deterministic.d_bound")


def _refuted_by_verified_design(truth):
    """A design that the coordinator releases for the complete truth AND that the judge verifies independently refutes any
    claim that no estimator can meet the request.  Returns (True, why) when such a design exists, else (False, why)."""
    try:
        d = C.decide(truth)
    except Exception as ex:  # noqa
        return False, f"no design for the truth ({type(ex).__name__})"
    if d["disposition"] != "RELEASE":
        return False, "no design is released for the truth"
    jr = judge_release(truth, d)
    return bool(jr["correct"]), ("a design for the complete facts meets the request and is verified independently (" + jr["why"] + ")"
                                 if jr["correct"] else "the design for the truth is not verified: " + jr["why"])


def _judge_initial_sample(truth):
    """Judge's own check of the first-sample limit: True if a requirement that applies at t = 0 is below the half-width of the
    velocities (position: min(V, admitted positions)) that one position reading cannot distinguish."""
    if S.get(truth, "objective.kind") != "hard_bound" or S.get(truth, "plant.kind") != "second_order":
        return None
    if (S.get(truth, "plant.second_order") or {}).get("output", "x1") != "x1":
        return None
    tol, cap, dl, Ts = (S.get(truth, "objective.tolerance"), S.get(truth, "objective.peak_cap"), S.get(truth, "objective.deadline"),
                        S.get(truth, "time.Ts"))
    levels = [cap] if cap is not None else []
    if tol is not None and dl is not None and Ts and _window(Ts, dl, None)[0] == 0:
        levels.append(tol)
    E, x0, dom = S.get(truth, "initial.E"), S.get(truth, "initial.x0_hat"), S.get(truth, "deterministic.domain") or {}
    if not levels or E is None:
        return None
    tgt = S.get(truth, "target.state")
    restricted = dom.get("x1") is not None or dom.get("x2") is not None
    if tgt == 1:
        mid = float(x0[1]) if x0 is not None else 0.0
        lo, hi = mid - E[1], mid + E[1]
        if dom.get("x2") is not None:
            lo, hi = max(lo, dom["x2"][0]), min(hi, dom["x2"][1])
        if restricted and min(levels) < 0.5 * (hi - lo):
            # a stated travel or speed range restricts the admissible initial velocities (review cases 03): the judge's own
            # construction of trajectories that stay inside the ranges for the whole window (independent.ref_hgo)
            from independent import ref_hgo as RH
            res = RH.first_sample_velocity_limit(truth, lo, hi)
            if res is None:
                return False, "independent construction: no two admissible initial velocities inside the stated ranges"
            lo, hi = res
        h = 0.5 * (hi - lo)
    elif tgt == 0 and restricted:
        return None
    elif tgt == 0 and S.get(truth, "deterministic.V") is not None:
        lo, hi = -math.inf, math.inf
        if x0 is not None:
            lo, hi = x0[0] - E[0], x0[0] + E[0]
        if dom.get("x1") is not None:
            lo, hi = max(lo, dom["x1"][0]), min(hi, dom["x1"][1])
        h = min(S.get(truth, "deterministic.V"), 0.5 * (hi - lo))
    else:
        return None
    return bool(min(levels) < h), f"requirement {min(levels):.6g} at t = 0 versus half-width {h:.6g} of the states one reading cannot distinguish"


def _judge_two_signal(truth):
    """Judge's own check of the two-signal limit (construction of independent.ref_hgo.lower_limit_two_signals: the two signals
    rest at -V and +V and reach velocities -+2 sqrt(DV) after 2 sqrt(V/D)).  Returns (ok, why) or (None, why)."""
    from independent import ref_hgo as RH
    import sympy as sp_
    if S.get(truth, "target.state") != 1:
        return False, "impossibility claimed for a target other than the velocity"
    V, Ts = S.get(truth, "deterministic.V"), S.get(truth, "time.Ts")
    so = S.get(truth, "plant.second_order") or {}
    if so.get("f") and {str(x) for x in sp_.sympify(so["f"]).free_symbols} & {"x1", "x2"}:
        return None, "the true acceleration depends on the state: the two-signal construction is not checked here"
    cands = [S.get(truth, "deterministic.D")] + ([S.get(truth, "deterministic.d_bound")] if so.get("f") else [])
    cands = [c for c in cands if c is not None]
    if not cands or V is None:
        return None, "no bound on a free acceleration error"
    D = min(cands)
    if D <= 0 or V <= 0:
        return False, "no free acceleration error or no measurement error: the two-signal construction does not apply"
    ref = RH.lower_limit_two_signals(D, V)
    half = 0.5 * ref["claimed_gap"]
    dom = S.get(truth, "deterministic.domain") or {}
    E, x0 = S.get(truth, "initial.E"), S.get(truth, "initial.x0_hat")
    if dom.get("x2") is not None and not (dom["x2"][0] <= -half and half <= dom["x2"][1]):
        return False, "the stated speed range excludes the velocities of the construction"
    if dom.get("x1") is not None and dom["x1"][1] - dom["x1"][0] < 2 * V:
        return False, "the stated travel range is narrower than the construction needs"
    x1r = dom.get("x1")
    if E is not None and x0 is not None and (abs(x0[1]) > E[1] or E[0] < V):
        return False, "the initial-error bounds exclude the resting states of the construction"
    if E is not None and x0 is None and E[1] < 0:
        return False, "invalid initial bound"
    if dom.get("u") is not None and not (dom["u"][0] <= 0.0 <= dom["u"][1]):
        return None, "the construction rests at u = 0, which the stated input range excludes"
    f0 = sp_.sympify(str(so.get("f0", "0"))).subs({sp_.Symbol(k): v for k, v in (so.get("params") or {}).items()})
    rest = {sp_.Symbol("u"): 0, sp_.Symbol("y"): 0, sp_.Symbol("x1"): 0, sp_.Symbol("x2"): 0}
    if abs(float(f0.subs(rest))) > 0 or (so.get("f") and abs(float(sp_.sympify(so["f"]).subs({sp_.Symbol(k): v for k, v in (so.get("params") or {}).items()}).subs(rest))) > 0):
        return None, "the models do not rest at u = 0"
    tol, cap, dl = S.get(truth, "objective.tolerance"), S.get(truth, "objective.peak_cap"), S.get(truth, "objective.deadline")
    t_end = S.get(truth, "time.t_end")
    kd, ne = _window(Ts, dl, t_end)
    f0_y = {str(x) for x in f0.free_symbols} & {"y", "x1"}
    ok = False
    room_why = None
    for level, start, end in ((tol, (kd * Ts if dl is not None else None), ne if dl is not None else None), (cap, 0.0, ne)):
        if level is None or not level < half:
            continue
        r_ = max(ref["t_star"], 0.0 if start is None else start) / Ts
        k = int(math.ceil(r_ - _ratio_slack(r_)))
        if start is not None and end is not None and k > end:
            continue
        # after the sample both signals must stay admissible for the rest of the window: they brake with D and need travel
        # +-room about the centre (review cases 03); a nominal model that depends on the reading would reduce the free
        # deceleration near the travel ends, which this check does not model
        continues = t_end is None or float(t_end) > k * Ts
        if continues and f0_y:
            room_why = "the nominal model depends on the reading; braking after the sample is not checked here"
            continue
        if x1r is not None:
            room = RH.two_signal_travel_room(V, D, k * Ts, t_end)
            if x1r[1] - x1r[0] < 2 * room:
                room_why = f"the stated travel range is narrower than the {2 * room:.6g} that the signals need, braking after the sample"
                continue
        ok = True
    if not ref.get("ok", False):
        return None, "the independent construction did not verify"
    if not ok and room_why and room_why.startswith("the nominal model"):
        return None, "independent two-signal construction: " + room_why
    return ok, (f"independent two-signal construction: limit {half:.6g}, needs a requirement below it at a sample at or after "
                f"{ref['t_star']:.6g} s of the window and travel for the braking that follows" + ("" if ok else f" (not met: {room_why or 'no such sample'})"))


_VI_CACHE = {}


def _verify_impossibility(truth, dec):
    """Independent check of a PROVED_IMPOSSIBLE outcome.  Returns (True / False / None, why); None = no independent check
    for this case is available here.  False means refuted: by a verified design for the complete facts, by the request
    kind (an optimal filter always exists; a worst-case limit does not refute a probability level), or by a construction
    whose conditions fail.  The check depends on the complete facts only (not on the argument offered), so its result is
    kept for each distinct truth: the interval constructions for a stated travel range take seconds."""
    key = S.canonical(truth)
    if key not in _VI_CACHE:
        _VI_CACHE[key] = _verify_impossibility_for(truth)
    return _VI_CACHE[key]


def _verify_impossibility_for(truth):
    from independent import ref_kalman as RK
    obj = S.get(truth, "objective.kind")
    try:
        if obj == "mse_optimal":
            return False, "the request asks for the smallest error variance; an optimal filter always exists, so no impossibility claim is true"
        refuted, why = _refuted_by_verified_design(truth)
        if refuted:
            return False, why
        if obj == "probability":
            return None, "no independent check: a worst-case limit does not show that a probability level is unattainable"
        if obj == "hard_bound":
            first = _judge_initial_sample(truth)
            if first is not None and first[0]:
                return True, "independent first-sample check: " + first[1]
            ok, why2 = _judge_two_signal(truth)
            if ok:
                return True, why2
            if first is not None or ok is False:
                return False, "independent checks do not confirm the claim: " + (first[1] + "; " if first else "") + why2
            return None, why2
        if obj == "mse_bound" and not (S.get(truth, "objective.horizon") == "steady_state" and S.get(truth, "objective.deadline") is None):
            # a finite window (deadline or finite horizon): the optimal filter from the true initial covariance over the
            # requested samples, by the judge's own recursion (second recheck A)
            P0t = S.get(truth, "initial.P0")
            if P0t is None:
                return None, "no independent check: the truth has no initial covariance"
            Ad, _, C_, Qd, R, Ts = _true_model(truth)
            dl, te = S.get(truth, "objective.deadline"), S.get(truth, "time.t_end")
            kd, n_end = _window(Ts, dl, te)
            tgt = S.get(truth, "target.state")
            P0t = np.atleast_2d(np.asarray(P0t, float))
            wmax, settled = _window_max(Ad, C_, Qd, R, P0t, _optimal_gain_seq(Ad, C_, Qd, R, P0t), tgt, kd, n_end)
            if not settled:
                return None, "no independent check: the optimal error variance did not settle within the judge's iteration limit"
            ok = S.get(truth, "objective.mse_target") < wmax * (1 - 1e-9)
            return bool(ok), f"independent optimal-filter window maximum {wmax:.6g} vs target {S.get(truth, 'objective.mse_target'):.6g}"
        if obj == "mse_bound" and (S.get(truth, "objective.horizon") == "steady_state") and S.get(truth, "objective.deadline") is None:
            A, _, C, Qd, R, _ = _true_model(truth)           # independent discretisation (quadrature), not the package's
            P = RK.riccati_fixed_point(A, C, Qd, R)
            P = P[0] if isinstance(P, tuple) else P
            G = C @ P @ C.T + R
            Pf = P - P @ C.T @ np.linalg.solve(G, C @ P)
            tgt = S.get(truth, "target.state")
            ok = S.get(truth, "objective.mse_target") < Pf[tgt, tgt] * (1 - 1e-9)
            return bool(ok), f"independent Riccati optimum {Pf[tgt, tgt]:.6g} vs target {S.get(truth, 'objective.mse_target'):.6g}"
    except Exception as ex:  # noqa
        return None, f"no independent check: {type(ex).__name__}"
    return None, "no independent check for this case"


_REVISIONS = None


def with_revisions(case: dict) -> dict:
    """The case with documented corrections of its designed outcome applied (data/revision2_relabel.json, revision 2,
    finding 7).  The frozen case files themselves are not edited."""
    global _REVISIONS
    if _REVISIONS is None:
        import json as _json
        import pathlib as _pl
        p = _pl.Path(__file__).resolve().parents[1] / "data" / "revision2_relabel.json"
        _REVISIONS = _json.loads(p.read_text())["cases"] if p.exists() else {}
    r = _REVISIONS.get(case.get("id"))
    if not r or not r.get("new_final"):
        return case
    c = dict(case)
    c["final"] = dict(case["final"], disposition=r["new_final"], selected=None)
    c["expected"] = dict(case["expected"], disposition=r["new_expected"], selected=None, ask=None)
    c["revision"] = {"finding": 7, "old_final": r["old_final"], "reason": r["reason"]}
    return c


def judge(case: dict, outcome: dict) -> dict:
    """Classify the final outcome of one request against the designed expectation and the independent verification."""
    dec = outcome["decision"]
    disp = dec["disposition"]
    fin = case["final"]
    r = {"disposition": disp, "selected": dec.get("selected"), "expected_final": fin["disposition"], "n_questions": outcome["n_questions"],
         "edits": outcome.get("edits", 0)}
    if disp == "RELEASE":
        jr = judge_release(case["truth"], dec)
        if fin["disposition"] == "INVALID_INPUT":
            jr = {"correct": False, "why": "a design was released for an invalid request (contradictory or impossible as stated)", "checks": {}}
        r.update(released=True, correct_release=jr["correct"], why=jr["why"], checks=jr["checks"])
        r["outcome"] = "correct_release" if jr["correct"] else "incorrect_release"
        if jr["correct"] and fin["disposition"] != "RELEASE":
            r["outcome"] = "correct_release"          # a verified release where the design expected none is still a true claim
            r["unexpected_release"] = True
    else:
        r["released"] = False
        if fin["disposition"] == "RELEASE":
            r["outcome"] = "unnecessary_refusal" if disp in NONRELEASE else "unresolved"
        else:
            r["outcome"] = "correct_nonrelease" if disp == fin["disposition"] else ("nonrelease_other_reason" if disp in NONRELEASE else "unresolved")
        r["why"] = "; ".join(dec.get("explanations", [])[:2])
        if disp == "PROVED_IMPOSSIBLE":
            ok, why = _verify_impossibility(case["truth"], dec)
            r["impossibility_verified"] = None if ok is None else bool(ok)
            if ok is not None and not ok:             # numpy booleans included (an identity test missed np.False_)
                r["outcome"] = "false_impossibility"
                r["why"] = why
    return jsonable(r)
