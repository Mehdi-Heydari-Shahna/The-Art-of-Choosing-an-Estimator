"""Source-linked case generator for training, development and acceptance testing.

Every case is derived from a base specification of one model family (heros.sim.plants) by one named transformation
(`category`), and records:
    truth      the complete specification (hidden ground truth; answers of the scripted user come from it)
    given      what the request supplies (omissions, ranges, flags, contradictions)
    expected   the justified first decision BY CONSTRUCTION of the transformation (not by running the coordinator)
    final      the justified outcome once every question has been answered truthfully
    facts      the facts a natural-language request must convey (role, SI value, dimension)
    labels     request-level labels (objective kind, distribution, horizon, flags, evidence kind)
    history    family, base id, category and the parameters of the transformation
Tolerance levels are positioned relative to quantities computed with the numerical methods (best certified bound,
optimal mean-square error, the two-signal lower limit); the acceptance judge re-verifies every released claim
with the independently written references.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from . import api
from .core import peak, spec as S
from .kalman import specialist as K
from .khalil import specialist as H
from .sim import plants as PL

CATEGORIES = [
    "complete_hard", "complete_mse_opt", "complete_mse_bound", "complete_prob", "both_hard",
    "harmless_no_init_ultimate", "harmless_no_P0_steady", "harmless_no_distribution",
    "missing_V", "missing_E2_deadline", "missing_R", "missing_Ts", "missing_objective", "missing_domain_evidence", "missing_D",
    "rms_not_bound", "range_V_ok", "range_V_split", "range_R_mse_opt", "range_R_mse_bound", "range_Ts",
    "impossible_hard", "impossible_mse", "no_candidate_hard", "no_candidate_objective",
    "flag_hidden_dynamics", "flag_correlated_noise", "flag_coloured_noise", "flag_irregular", "unsupported_family", "feedthrough_output",
    "invalid_timing", "invalid_reversed_range", "invalid_cap_below_tol", "invalid_deadline_after_window",
]
RELEASE, CLARIFY = "RELEASE", "CLARIFY"


def _r(x, k=3):
    return 0.0 if x == 0 else float(f"{x:.{k}g}")


def _best_hard(Ts, tgt, E, D, V, k_dead, ultimate=False):
    c = np.eye(2)[tgt]
    eps, val, pk = H.search_eps(Ts, c, E, D, V, k_dead, None, None, ultimate_only=ultimate)
    return eps, val, pk


def base_case(family: str, rng) -> dict:
    """Complete truth specification pieces for one family (no objective yet)."""
    fam = PL.make(family, rng)
    sc, plant = fam["scales"], fam["plant"]
    b = {"family": family, "plant": plant, "scales": sc, "kind": plant["kind"]}
    if family in PL.BOTH_METHODS:
        Ts, V, D, E = sc["Ts"], sc["V"], sc["D"], sc["E"]
        G = np.array([[Ts * Ts / 2], [Ts]])
        # second-order information consistent with the hard bounds (uniform within the bounds), stated as round standard deviations
        sa, sv, s1, s2 = _r(D / math.sqrt(3)), _r(V / math.sqrt(3)), _r(E[0] / math.sqrt(3)), _r(E[1] / math.sqrt(3))
        b["fields"] = dict(Ts=Ts, V=V, D=D, E=E, Q=(sa * sa * (G @ G.T)).tolist(), Q_kind="discrete_cov", R=[[sv * sv]],
                           P0=[[s1 * s1, 0.0], [0.0, s2 * s2]], distribution="bounded", white=True)
        b["sigma_a"], b["sigma_v"] = sa, sv
    elif plant["kind"] == "second_order":
        b["fields"] = dict(Ts=sc["Ts"], V=sc["V"], d_bound=sc["d_bound"], domain=sc["domain"], E=sc["E"],
                           domain_evidence={"kind": "mechanical_limit", "statement": "hardware speed limiter and saturated actuator"})
        if family == "mass_spring_damper":
            # a second-order noise model is also available for this linear plant
            b["stoch"] = dict(Q=[[_r(sc["d_bound"] ** 2 / 3, 6)]], Q_kind="continuous_psd", G=[[0.0], [1.0]], R=[[_r(sc["V"] ** 2 / 3, 6)]],
                              P0=[[_r(sc["E"][0] ** 2 / 3, 6), 0.0], [0.0, _r(sc["E"][1] ** 2 / 3, 6)]], distribution="bounded", white=True)
    elif plant["kind"] == "linear":
        n = len(plant["states"])
        b["fields"] = dict(Ts=sc["Ts"], Q=sc["Qc"], Q_kind="continuous_psd", G=sc["G"], R=np.diag(np.square(sc["r_std"])).tolist(),
                           P0=np.diag(np.square(sc["P0_std"])).tolist(), distribution="gaussian", white=True, t_end=_r(60 * sc["Ts"]))
    else:
        b["fields"] = dict(Ts=sc.get("Ts"))
    return b


def applicable_categories(family: str) -> list:
    if family in PL.UNSUPPORTED:
        return ["unsupported_family"]
    if family in PL.BOTH_METHODS:
        return ["complete_hard", "complete_mse_opt", "complete_mse_bound", "complete_prob", "both_hard", "harmless_no_init_ultimate",
                "harmless_no_P0_steady", "harmless_no_distribution", "missing_V", "missing_E2_deadline", "missing_R", "missing_Ts",
                "missing_objective", "missing_D", "rms_not_bound", "range_V_ok", "range_V_split", "range_R_mse_opt", "range_R_mse_bound",
                "range_Ts", "impossible_hard", "impossible_mse", "no_candidate_hard", "flag_hidden_dynamics", "flag_correlated_noise",
                "flag_coloured_noise", "flag_irregular", "feedthrough_output", "invalid_reversed_range", "invalid_cap_below_tol",
                "invalid_deadline_after_window"]
    if family in PL.KHALIL_ONLY or family == "mass_spring_damper":
        cats = ["complete_hard", "harmless_no_init_ultimate", "missing_V", "missing_E2_deadline", "missing_Ts", "missing_objective",
                "missing_domain_evidence", "range_V_ok", "range_V_split", "range_Ts", "no_candidate_objective", "flag_hidden_dynamics",
                "feedthrough_output", "invalid_reversed_range", "invalid_cap_below_tol", "rms_not_bound"]
        if family == "mass_spring_damper":
            cats += ["complete_mse_opt", "missing_R"]
            cats.remove("no_candidate_objective")
        return cats
    return ["complete_mse_opt", "complete_mse_bound", "complete_prob", "harmless_no_P0_steady", "harmless_no_distribution", "missing_R",
            "missing_objective", "range_R_mse_opt", "range_R_mse_bound", "impossible_mse", "no_candidate_objective", "flag_correlated_noise",
            "flag_coloured_noise", "flag_irregular", "invalid_timing"]


def _spec(b, objective, extra=None, drop=(), ranges=None, flags=None, stoch=False):
    f = dict(b["fields"])
    if stoch and "stoch" in b:
        f.update(b["stoch"])
    f.update(extra or {})
    for k in drop:
        f.pop(k, None)
    plant = copy.deepcopy(b["plant"])
    if flags:
        plant["flags"] = list(flags)
    return api.new_spec(plant, objective=objective, ranges=ranges, **f)


def make_case(family: str, category: str, rng, cid: str) -> dict | None:
    """One case; returns None if the category cannot be instantiated for the drawn base (caller redraws)."""
    b = base_case(family, rng)
    f = b["fields"]
    plant = b["plant"]
    hist = {"family": family, "category": category, "base": cid}
    both = family in PL.BOTH_METHODS
    so = plant["kind"] == "second_order"
    tgt = 1 if so else int(rng.integers(0, len(plant["states"])))
    labels = {"flags": [], "evidence_kind": None, "horizon": None, "distribution": None, "objective": None, "q_kind_stated": None}
    exp = {"disposition": RELEASE, "selected": None, "ask": None}
    fin = {"disposition": RELEASE, "selected": None}
    extra, drop, omit, ranges, flags = {"target": tgt}, [], [], None, None
    obj = {}
    stoch = False

    # ---- numeric anchors -------------------------------------------------------------------------------
    def hard_anchor(deadline=True):
        Ts = f["Ts"]
        if both:
            D = f["D"]
        else:
            D, _ = H.total_disturbance(_spec(b, {"kind": "hard_bound", "tolerance": 1.0}, {"target": 1}))
        E = [f["V"], f["E"][1]]               # first-measurement initialisation: E1_eff = V
        if deadline:
            kd = int(rng.integers(30, 400))
            eps, val, _ = _best_hard(Ts, 1, E, D, f["V"], kd)
            return kd * Ts, val, D
        eps, val, _ = _best_hard(Ts, 1, E, D, f["V"], 0, ultimate=True)
        return None, val, D

    def kf_anchor(horizon="steady_state"):
        sp = _spec(b, {"kind": "mse_optimal", "horizon": horizon}, {"target": tgt}, stoch=True)
        a = K.assess(sp)
        if a.status != "qualifies":
            return None
        return a.metrics["mse_post_deadline_max"], a

    cat = category
    if cat in ("complete_hard", "both_hard", "missing_V", "missing_E2_deadline", "missing_Ts", "missing_D", "missing_domain_evidence",
               "range_V_ok", "range_V_split", "range_Ts", "rms_not_bound", "invalid_cap_below_tol", "invalid_reversed_range",
               "invalid_deadline_after_window", "feedthrough_output", "flag_hidden_dynamics"):
        if not so:
            return None
        extra["target"] = tgt = 1
        dl, best, D = hard_anchor(True)
        tol = _r(best * float(rng.uniform(1.25, 2.5)))
        obj = {"kind": "hard_bound", "tolerance": tol, "deadline": _r(dl, 6)}
        labels["objective"] = "hard_bound"
        exp["selected"] = fin["selected"] = H.METHOD
        if not both:
            labels["evidence_kind"] = "mechanical_limit"
        if cat == "both_hard":
            if not both:
                return None
            stoch = True
            exp["selected"] = fin["selected"] = None          # whichever has the better certified bound
            labels["distribution"] = "bounded"; labels["horizon"] = None
            hist["note"] = "both specialists receive the same hard bounds and second-order noise model"
        else:
            # hard-bound requests do not carry the statistical model
            drop += ["Q", "Q_kind", "R", "P0", "distribution", "white"]
        if cat == "missing_V":
            omit.append("V"); exp.update(disposition=CLARIFY, ask="deterministic.V")
        elif cat == "missing_E2_deadline":
            omit.append("E"); exp.update(disposition=CLARIFY, ask="initial.E")
        elif cat == "missing_Ts":
            omit.append("Ts"); exp.update(disposition=CLARIFY, ask="time.Ts")
        elif cat == "missing_D":
            if not both:
                return None
            omit.append("D"); exp.update(disposition=CLARIFY, ask="deterministic.D")
        elif cat == "missing_domain_evidence":
            if both:
                return None
            omit.append("domain_evidence"); labels["evidence_kind"] = "none"
            exp.update(disposition=CLARIFY, ask="deterministic.domain_evidence")
        elif cat == "rms_not_bound":
            # the request states an RMS noise level; a hard bound is not supplied
            omit.append("V"); extra["R"] = [[_r(f["V"] / math.sqrt(3)) ** 2]]
            if "R" in drop:
                drop.remove("R")
            exp.update(disposition=CLARIFY, ask="deterministic.V"); hist["rms_stated"] = _r(f["V"] / math.sqrt(3))
        elif cat in ("range_V_ok", "range_V_split"):
            omit.append("V")
            if cat == "range_V_ok":
                ranges = {"V": [_r(f["V"] * float(rng.uniform(0.3, 0.8))), f["V"]]}
            else:
                # upper end large enough that no design is certified there
                hi = f["V"] * float(rng.uniform(15, 40))
                ranges = {"V": [_r(f["V"] * 0.5), _r(hi)]}
                exp.update(disposition=CLARIFY, ask="deterministic.V")
        elif cat == "range_Ts":
            omit.append("Ts"); ranges = {"Ts": [f["Ts"], _r(2 * f["Ts"])]}
            exp.update(disposition=CLARIFY, ask="time.Ts")
        elif cat == "invalid_cap_below_tol":
            # Frozen label of revision 1 (kept so that the frozen data files regenerate).  The label is wrong: a cap below the
            # later tolerance is not a contradiction.  The corrected outcome of every such test case is in
            # data/revision2_relabel.json (experiments/relabel_cap_cases.py) and is applied by heros.evaluation.with_revisions.
            obj["peak_cap"] = _r(tol * 0.5); exp["disposition"] = fin["disposition"] = "INVALID_INPUT"
        elif cat == "invalid_reversed_range":
            omit.append("V"); ranges = {"V": [_r(f["V"] * 2), _r(f["V"] * 0.5)]}
            exp["disposition"] = fin["disposition"] = "INVALID_INPUT"
        elif cat == "invalid_deadline_after_window":
            extra["t_end"] = _r(dl * 0.5, 6); exp["disposition"] = fin["disposition"] = "INVALID_INPUT"
        elif cat == "feedthrough_output":
            plant = copy.deepcopy(plant); plant["second_order"]["output"] = "x1 + 0.1*u"
            plant["flags"] = []; plant["linear"] = None
            b = dict(b, plant=plant)
            labels["flags"] = ["parameter_dependent_output"]
            exp["disposition"] = fin["disposition"] = "NO_VERIFIED_CANDIDATE"; exp["selected"] = fin["selected"] = None
        elif cat == "flag_hidden_dynamics":
            flags = ["hidden_dynamics"]; labels["flags"] = ["hidden_dynamics"]
            exp["disposition"] = fin["disposition"] = "OUT_OF_SCOPE"; exp["selected"] = fin["selected"] = None
    elif cat == "harmless_no_init_ultimate":
        if not so:
            return None
        extra["target"] = tgt = 1
        _, best, D = hard_anchor(False)
        obj = {"kind": "hard_bound", "tolerance": _r(best * float(rng.uniform(1.25, 2.5)))}
        labels["objective"] = "hard_bound"
        drop += ["E", "Q", "Q_kind", "R", "P0", "distribution", "white"]
        exp["selected"] = fin["selected"] = H.METHOD
        if not both:
            labels["evidence_kind"] = "mechanical_limit"
    elif cat in ("impossible_hard", "no_candidate_hard"):
        if not both:
            return None
        extra["target"] = tgt = 1
        _, best, D = hard_anchor(False)
        low = peak.lower_limit_velocity(D, f["V"])
        if cat == "impossible_hard":
            tol = _r(low * float(rng.uniform(0.4, 0.9)))
            exp["disposition"] = fin["disposition"] = "PROVED_IMPOSSIBLE"
        else:
            if best <= low * 1.02:
                return None
            tol = _r(low + (best - low) * float(rng.uniform(0.3, 0.7)), 4)
            if not (low < tol < best):
                return None
            exp["disposition"] = fin["disposition"] = "NO_VERIFIED_CANDIDATE"
        obj = {"kind": "hard_bound", "tolerance": tol}
        labels["objective"] = "hard_bound"
        drop += ["Q", "Q_kind", "R", "P0", "distribution", "white"]
        exp["selected"] = fin["selected"] = None
    elif cat in ("complete_mse_opt", "complete_mse_bound", "complete_prob", "harmless_no_P0_steady", "harmless_no_distribution", "missing_R",
                 "range_R_mse_opt", "range_R_mse_bound", "impossible_mse", "flag_correlated_noise", "flag_coloured_noise", "flag_irregular",
                 "invalid_timing", "missing_objective"):
        if so and not both and family != "mass_spring_damper":
            if cat != "missing_objective":
                return None
        stoch = True
        horizon = "steady_state" if (cat in ("harmless_no_P0_steady", "complete_mse_bound", "impossible_mse", "range_R_mse_bound", "range_R_mse_opt",
                                              "complete_prob") or rng.random() < 0.5) else "finite"
        if so and not both and family != "mass_spring_damper":
            # nonlinear chain-form plant with a missing objective: hard-bound truth
            dl, best, D = hard_anchor(True)
            obj_true = {"kind": "hard_bound", "tolerance": _r(best * 1.8), "deadline": _r(dl, 6)}
            labels["evidence_kind"] = "mechanical_limit"
            exp.update(disposition=CLARIFY, ask="objective.kind"); fin["selected"] = H.METHOD
            truth = _spec(b, obj_true, {"target": 1})
            given = _spec(b, {"tolerance": obj_true["tolerance"], "deadline": obj_true["deadline"]}, {"target": 1})
            return _finish(cid, b, truth, given, exp, fin, labels, hist, 1)
        anc = kf_anchor(horizon)
        if anc is None:
            return None
        mse_opt, a = anc
        labels["horizon"] = horizon
        labels["distribution"] = f.get("distribution") or (b.get("stoch") or {}).get("distribution")
        labels["q_kind_stated"] = (dict(f, **b.get("stoch", {}))).get("Q_kind")
        exp["selected"] = fin["selected"] = K.METHOD
        if horizon == "finite" and plant["kind"] != "linear":
            extra["t_end"] = _r(200 * f["Ts"])
        if not both:
            drop += [k for k in ("V", "D", "d_bound", "domain", "domain_evidence", "E") if k in f]
        obj = {"kind": "mse_optimal", "horizon": horizon}
        labels["objective"] = "mse_optimal"
        if cat in ("complete_mse_bound", "range_R_mse_bound", "impossible_mse"):
            labels["objective"] = "mse_bound"
            if cat == "impossible_mse":
                obj = {"kind": "mse_bound", "horizon": horizon, "mse_target": _r(math.sqrt(mse_opt * float(rng.uniform(0.2, 0.8)))) ** 2}
                exp["disposition"] = fin["disposition"] = "PROVED_IMPOSSIBLE"; exp["selected"] = fin["selected"] = None
            else:
                obj = {"kind": "mse_bound", "horizon": horizon, "mse_target": _r(math.sqrt(mse_opt * float(rng.uniform(3.0, 8.0)))) ** 2}
        if cat == "complete_prob":
            labels["objective"] = "probability"
            sd = math.sqrt(mse_opt)
            gauss = labels["distribution"] == "gaussian"
            r = _r(sd * (3.0 if gauss else 5.0))
            obj = {"kind": "probability", "horizon": horizon, "tolerance": r, "prob": 0.95 if gauss else 0.9}
            if both:
                exp["selected"] = fin["selected"] = None      # the HGO may also give a probability-one statement
        if cat == "harmless_no_P0_steady":
            drop.append("P0")
        elif cat == "harmless_no_distribution":
            drop.append("distribution"); labels["distribution"] = None
        elif cat == "missing_R":
            omit.append("R"); exp.update(disposition=CLARIFY, ask="stochastic.R")
        elif cat in ("range_R_mse_opt", "range_R_mse_bound"):
            Rm = np.asarray(dict(f, **b.get("stoch", {}))["R"], float)
            if Rm.shape != (1, 1):
                return None
            r0 = float(Rm[0, 0])
            omit.append("R")
            if cat == "range_R_mse_opt":
                ranges = {"R": [_r(r0 * 0.5), _r(r0 * 2.0)]}
                exp.update(disposition=CLARIFY, ask="stochastic.R")
            else:
                ranges = {"R": [_r(r0 * 0.6), r0]}          # true value at the upper end: target met for the whole range
        elif cat == "flag_correlated_noise":
            n = len(plant["states"]); ny = len(plant["outputs"])
            extra["cross_cov"] = (1e-6 * np.ones((n, ny))).tolist(); labels["flags"] = ["correlated_noise"]
            exp["disposition"] = fin["disposition"] = "NO_VERIFIED_CANDIDATE"; exp["selected"] = fin["selected"] = None
        elif cat == "flag_coloured_noise":
            extra["white"] = False; labels["flags"] = ["coloured_noise"]
            exp["disposition"] = fin["disposition"] = "NO_VERIFIED_CANDIDATE"; exp["selected"] = fin["selected"] = None
        elif cat == "flag_irregular":
            extra["time.irregular"] = True; labels["flags"] = ["irregular_sampling"]
            exp["disposition"] = fin["disposition"] = "NO_VERIFIED_CANDIDATE"; exp["selected"] = fin["selected"] = None
        elif cat == "invalid_timing":
            # discrete model at its native period, request states half of it
            sp0 = _spec(b, obj, extra, stoch=True)
            cfg = K.design(sp0)
            plant = copy.deepcopy(plant)
            plant["linear"] = {"clock": "discrete", "A": cfg["A"].tolist(), "B": cfg["B"].tolist(), "C": cfg["C"].tolist(), "Ts_native": cfg["Ts"]}
            b = dict(b, plant=plant, fields=dict(f, Q=cfg["Qd"].tolist(), Q_kind="discrete_cov", Ts=_r(cfg["Ts"] / 2)))
            b["fields"].pop("G", None)
            exp["disposition"] = fin["disposition"] = "INVALID_INPUT"; exp["selected"] = fin["selected"] = None
        elif cat == "missing_objective":
            truth = _spec(b, obj, extra, drop=drop, stoch=True)
            given = _spec(b, {"horizon": horizon}, extra, drop=drop, stoch=True)
            exp.update(disposition=CLARIFY, ask="objective.kind")
            return _finish(cid, b, truth, given, exp, fin, labels, hist, tgt)
    elif cat == "no_candidate_objective":
        if both or family == "mass_spring_damper":
            return None
        if so:
            # nonlinear plant, minimum-MSE request: no stochastic result for the HGO, no linear model for the Kalman filter
            obj = {"kind": "mse_optimal", "horizon": "steady_state"}; labels["objective"] = "mse_optimal"; labels["horizon"] = "steady_state"
            extra["target"] = tgt = 1
            extra.update(R=[[_r(f["V"] / math.sqrt(3)) ** 2]], distribution="bounded")
            labels["distribution"] = "bounded"; labels["evidence_kind"] = "mechanical_limit"
        else:
            # linear multi-state plant, guaranteed error limit requested
            obj = {"kind": "hard_bound", "tolerance": 0.1, "deadline": _r(20 * f["Ts"])}; labels["objective"] = "hard_bound"
            extra.update(V=_r(float(np.sqrt(np.asarray(f["R"])[0][0]) * 3)))
            labels["distribution"] = f.get("distribution")
            stoch = True
        exp["disposition"] = fin["disposition"] = "NO_VERIFIED_CANDIDATE"; exp["selected"] = fin["selected"] = None
    elif cat == "unsupported_family":
        obj = {"kind": "mse_optimal", "horizon": "finite"}; labels["objective"] = "mse_optimal"
        labels["flags"] = list(plant.get("flags") or [])
        extra = {"target": 0, "R": [[1e-4]]}
        exp["disposition"] = fin["disposition"] = "OUT_OF_SCOPE"; exp["selected"] = fin["selected"] = None
    else:
        return None

    truth = _spec(b, obj, extra, drop=drop, stoch=stoch, flags=flags)
    given = _spec(b, obj, extra, drop=drop + omit, ranges=ranges, flags=flags, stoch=stoch)
    if exp["disposition"] in ("INVALID_INPUT",):
        truth = copy.deepcopy(given)                       # the request itself is invalid; there is nothing to complete
    return _finish(cid, b, truth, given, exp, fin, labels, hist, tgt)


def _finish(cid, b, truth, given, exp, fin, labels, hist, tgt):
    hist["truth_hash"] = S.spec_hash(truth); hist["given_hash"] = S.spec_hash(given)
    return {"id": cid, "family": hist["family"], "category": hist["category"], "truth": truth, "given": given, "expected": exp,
            "final": fin, "labels": labels, "history": hist, "target": tgt}


def make_set(n: int, families: list, seed: int, prefix: str, categories=None, weights=None) -> list:
    """n cases cycling over families and their applicable categories (deterministic for a seed)."""
    rng = np.random.default_rng(seed)
    out, k, tries = [], 0, 0
    fam_cycle = list(families)
    cat_pos = {f: int(rng.integers(0, 50)) for f in sorted(set(fam_cycle))}
    while len(out) < n and tries < 50 * n:
        tries += 1
        fam = fam_cycle[k % len(fam_cycle)]
        cats = [c for c in applicable_categories(fam) if categories is None or c in categories]
        if not cats:
            k += 1
            continue
        cat = cats[cat_pos[fam] % len(cats)]
        cat_pos[fam] += 1
        case = make_case(fam, cat, rng, f"{prefix}-{len(out):05d}")
        k += 1
        if case is not None:
            out.append(case)
    return out
