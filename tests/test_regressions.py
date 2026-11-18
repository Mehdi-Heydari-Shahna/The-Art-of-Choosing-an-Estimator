"""Regression evidence for every finding of SOURCE_INVENTORY.md, exercised through the new public path.
The legacy behaviour itself is reproduced by regressions/reproduce_legacy_defects.py (output: regressions/legacy_defects.json)."""
import copy, json, math
import numpy as np
import pytest
from heros import api, record as R
from heros.core import spec as S, peak, bounds
from heros.coordinator.coordinator import Session, decide
from heros.kalman import specialist as K
from heros.khalil import specialist as H
from heros.sim import plants as PL, loop


def di_plant(b=1.0):
    return {"family": "servo_axis", "kind": "second_order", "states": [{"name": "angle", "unit": "rad"}, {"name": "speed", "unit": "rad/s"}],
            "inputs": [{"name": "torque", "unit": "N m"}], "outputs": [{"name": "encoder", "unit": "rad"}],
            "linear": {"clock": "continuous", "A": [[0.0, 1.0], [0.0, 0.0]], "B": [[0.0], [b]], "C": [[1.0, 0.0]], "Ts_native": None},
            "second_order": {"f": "b*u", "f0": "b*u", "params": {"b": b}, "param_ranges": {}, "output": "x1"}, "flags": []}


def hard_spec(**kw):
    base = dict(target=1, Ts=0.001, V=1e-4, D=5.0, E=[1e-3, 2.0], objective={"kind": "hard_bound", "tolerance": 0.2, "deadline": 0.1})
    base.update(kw)
    return api.new_spec(di_plant(), **base)


# ---------------------------------------------------------------- Kalman findings
def test_K1_timing_conflict_is_an_explicit_result():
    T = 0.1
    plant = {"family": "generic", "kind": "linear", "states": [{"name": "x1", "unit": "-"}, {"name": "x2", "unit": "-"}], "inputs": [{"name": "u", "unit": "-"}],
             "outputs": [{"name": "y", "unit": "-"}], "linear": {"clock": "discrete", "A": [[1.0, T], [0.0, 1.0]], "B": [[0.0], [T]], "C": [[1.0, 0.0]], "Ts_native": T},
             "second_order": None, "flags": []}
    kw = dict(target=1, Q=(1e-4 * np.eye(2)).tolist(), Q_kind="discrete_cov", R=[[1e-2]], P0=np.eye(2).tolist(), objective={"kind": "mse_optimal", "horizon": "finite"})
    d = api.decide(api.new_spec(plant, Ts=0.05, **kw))
    assert d["disposition"] == "INVALID_INPUT" and any("timing conflict" in e or "native period" in e for e in d["explanations"])
    d = api.decide(api.new_spec(plant, Ts=0.1, **kw))
    assert d["disposition"] == "RELEASE" and d["config"]["Ts"] == 0.1 and np.allclose(d["config"]["A"], plant["linear"]["A"])
    d = api.decide(api.new_spec(plant, **kw))                       # no period stated: the native one is used and recorded
    assert d["disposition"] == "RELEASE" and d["config"]["Ts"] == 0.1


def test_K2_witnesses_and_completions_are_admissible():
    sp = hard_spec(objective={"kind": "mse_optimal", "horizon": "steady_state"}, Q=(25.0 * np.outer([5e-7, 1e-3], [5e-7, 1e-3])).tolist(),
                   Q_kind="discrete_cov", ranges={"R": [5e-9, 6e-9]})
    d = api.decide(sp)
    assert d["disposition"] == "CLARIFY" and d["question"]["path"] == "stochastic.R"
    w = d["question"]["witness"]
    for comp in (w["completion_a"], w["completion_b"]):
        assert 5e-9 <= comp["stochastic.R"] <= 6e-9
    assert w["admissible"] and w["outcome_a"]["gain"] != w["outcome_b"]["gain"]
    with pytest.raises(ValueError):
        S.complete(sp, {"stochastic.R": 1.25e-9})                    # an inadmissible completion is rejected, never returned
    # equality constraint (tutorial example): prior + noise variance = 2, both unknown
    plant = {"family": "scalar", "kind": "linear", "states": [{"name": "x", "unit": "-"}], "inputs": [], "outputs": [{"name": "y", "unit": "-"}],
             "linear": {"clock": "discrete", "A": [[1.0]], "B": [[0.0]], "C": [[1.0]], "Ts_native": 1.0}, "second_order": None, "flags": []}
    sp = api.new_spec(plant, target=0, Q=[[0.0]], Q_kind="discrete_cov", objective={"kind": "mse_optimal", "horizon": "finite"}, t_end=1.0,
                      ranges={"P0": [0.5, 1.0], "R": [1.0, 1.5]}, constraints=[{"paths": ["P0", "R"], "sum": 2.0}])
    d = api.decide(sp)
    assert d["disposition"] == "CLARIFY"
    w = d["question"]["witness"]
    for comp in (w["completion_a"], w["completion_b"]):
        assert comp["initial.P0"] + comp["stochastic.R"] == pytest.approx(2.0)
    assert sorted([w["outcome_a"]["first_gain"][0][0], w["outcome_b"]["first_gain"][0][0]]) == pytest.approx([0.25, 0.5], rel=1e-12)


def test_K3_process_noise_semantics_are_explicit():
    fam = PL.thermal(np.random.default_rng(0)); sc = fam["scales"]
    kw = dict(target=1, Ts=sc["Ts"], G=sc["G"], R=[[sc["r_std"][0] ** 2]], P0=np.diag(np.square(sc["P0_std"])).tolist(), objective={"kind": "mse_optimal", "horizon": "finite"})
    d = api.decide(api.new_spec(fam["plant"], Q=sc["Qc"], **kw))                 # kind not stated -> question
    assert d["disposition"] == "CLARIFY" and d["question"]["path"] == "stochastic.Q_kind"
    d = api.decide(api.new_spec(fam["plant"], Q=sc["Qc"], Q_kind="continuous_psd", **kw))
    assert d["disposition"] == "RELEASE" and any("Van Loan" in x["how"] for x in d["config"]["derivations"])
    Qd_psd = np.array(d["config"]["Qd"])
    Qfull = (np.array(sc["G"]) @ np.array(sc["Qc"]) @ np.array(sc["G"]).T).tolist()
    d2 = api.decide(api.new_spec(fam["plant"], Q=Qfull, Q_kind="discrete_cov", **{k: v for k, v in kw.items() if k != "G"}))
    assert d2["disposition"] == "RELEASE" and not np.allclose(np.array(d2["config"]["Qd"]), Qd_psd)   # the two readings differ (factor ~Ts)
    # a spectral density with a discrete model is refused with the reason
    cfg = K.design(api.new_spec(fam["plant"], Q=sc["Qc"], Q_kind="continuous_psd", **kw))
    plant = copy.deepcopy(fam["plant"]); plant["linear"] = {"clock": "discrete", "A": cfg["A"].tolist(), "B": cfg["B"].tolist(), "C": cfg["C"].tolist(), "Ts_native": cfg["Ts"]}
    d3 = api.decide(api.new_spec(plant, Q=sc["Qc"], Q_kind="continuous_psd", **{k: v for k, v in kw.items() if k != "Ts"}))
    assert d3["disposition"] != "RELEASE" and any("spectral density" in e for e in d3["explanations"])


def test_K4_K10_record_replays_everything_and_detects_tampering():
    rng = np.random.default_rng(2)
    sp = hard_spec()
    y = 0.01 * rng.standard_normal(301)
    res = api.solve(sp, y, np.zeros(300))
    rec = json.loads(R.dumps(res["record"]))
    for key in ("spec", "evidence", "decision", "inputs", "outputs", "build", "spec_hash"):
        assert key in rec
    assert api.replay(rec)["match"]
    for path, val in ((("decision", "claim", "value"), 0.01), (("decision", "config", "init", "E1_effective"), 0.0),
                      (("decision", "config", "M"), [0.1, 1.0]), (("outputs", "envelope"), None)):
        bad = json.loads(json.dumps(rec)); cur = bad
        for k in path[:-1]:
            cur = cur[k]
        cur[path[-1]] = (np.asarray(cur[path[-1]]) * 1.001).tolist() if val is None else val
        rp = api.replay(bad)
        assert not rp["match"] and rp["n_differences"] >= 1


# ---------------------------------------------------------------- Khalil findings
def test_H1_finite_time_and_asymptotic_claims_are_distinct():
    # legacy counterexample: D = 0.5, V = E1 = 0, E2 = 2, gain 1: the admissible error 1 + (1 + 1.5 t) e^-t exceeds 1 at every finite time
    t = np.array([0.0, 1.0, 10.0, 30.0])
    W = peak.ct_envelope(1.0, [0.0, 2.0], 0.5, 0.0, t)
    assert np.allclose(W, 1 + (1 + 1.5 * t) * np.exp(-t)) and np.all(W > 1.0) and peak.ct_ultimate(1.0, 0.5, 0.0) == 1.0
    # public path: without a deadline the claim is an ultimate bound and says so; with a deadline the bound covers every later sample
    d0 = api.decide(hard_spec(objective={"kind": "hard_bound", "tolerance": 0.2}))
    assert d0["disposition"] == "RELEASE" and "lim sup" in d0["claim"]["statement"] and "no finite-time claim" in d0["claim"]["statement"]
    d1 = api.decide(hard_spec())
    c = d1["assessments"]["high_gain_observer"]["metrics"]["worst_case"]
    assert "t_k >=" in d1["claim"]["statement"] and c["post_deadline_upper"] >= c["W_at_deadline_upper"] >= c["ultimate_lower"]
    M = np.array(d1["config"]["M"]); kd = c["k_deadline"]
    W = peak.worst_case_lti(0.001, M, np.array([0.0, 1.0]), d1["config"]["E_eff"], 5.0, 1e-4, kd + 5000)["W"]
    assert W[kd:].max() <= d1["claim"]["value"] * (1 + 1e-12)


def test_H2_new_deadline_triggers_full_reevaluation():
    s = Session(hard_spec(E=None, objective={"kind": "hard_bound", "tolerance": 0.2}))
    d = s.decide()
    assert d["disposition"] == "RELEASE" and any(h["path"] == "initial.E" for h in d["harmless"])
    s.update({"objective.deadline": 1.0}, source="scenario")
    d = s.decide()
    assert d["disposition"] == "CLARIFY" and d["question"]["path"] == "initial.E"
    s.update({"initial.E": [1e-3, 100.0]})
    d = s.decide()
    assert d["disposition"] == "RELEASE"
    M = np.array(d["config"]["M"])
    W = peak.worst_case_lti(0.001, M, np.array([0.0, 1.0]), [1e-4, 100.0], 5.0, 1e-4, 6000)["W"]
    assert W[1000:].max() <= 0.2                                      # the admissible initial error 100 is covered at the deadline


def test_H3_initialisation_matches_the_certificate_premise():
    d = api.decide(hard_spec(V=0.01, E=[0.001, 1.0], objective={"kind": "hard_bound", "tolerance": 5.0, "deadline": 1.0}))
    cfg = d["config"]
    assert cfg["init"]["policy"] == "first_measurement" and cfg["init"]["E1_effective"] == 0.01 and cfg["E_eff"][0] == 0.01
    assert "not used" in cfg["init"]["premise"]
    d = api.decide(hard_spec(V=0.01, E=[0.001, 1.0], x0_hat=[0.3, 0.0], objective={"kind": "hard_bound", "tolerance": 5.0, "deadline": 1.0}))
    assert d["config"]["init"]["policy"] == "given" and d["config"]["E_eff"][0] == 0.001
    est = api.solve(hard_spec(V=0.01, E=[0.001, 1.0], x0_hat=[0.3, 0.0], objective={"kind": "hard_bound", "tolerance": 5.0, "deadline": 1.0}), np.full(5, 0.3), np.zeros(5))["estimates"]
    assert abs(est["x_hat"][0, 0] - 0.3) < 1e-12


def test_H4_sample_period_range_gets_a_precise_disposition():
    sp = hard_spec(Ts=None, ranges={"Ts": [0.001, 0.002]})
    for dec in (api.decide(sp), Session(sp).decide(), api.solve(sp)["decision"]):
        assert dec["disposition"] == "CLARIFY" and dec["question"]["path"] == "time.Ts"
    assert api.decide(hard_spec(Ts=None, ranges={"Ts": [0.001, 0.001]}))["disposition"] == "RELEASE"


def test_H5_output_transformations_are_not_accepted():
    plant = di_plant(); plant["second_order"]["output"] = "x1 + 0.1*u"; plant["linear"] = None
    d = api.decide(api.new_spec(plant, target=1, Ts=0.001, V=1e-4, D=5.0, E=[1e-3, 2.0], objective={"kind": "hard_bound", "tolerance": 0.2, "deadline": 0.1}))
    assert d["disposition"] == "NO_VERIFIED_CANDIDATE" and any("input derivative" in e for e in d["explanations"])
    plant = di_plant(); plant["second_order"]["output"] = "k*x1"; plant["linear"] = None
    d = api.decide(api.new_spec(plant, target=1, Ts=0.001, V=1e-4, D=5.0, E=[1e-3, 2.0], objective={"kind": "hard_bound", "tolerance": 0.2, "deadline": 0.1}))
    assert d["disposition"] == "NO_VERIFIED_CANDIDATE"


def _joint_spec(**kw):
    fam = PL.rotary_joint(nominal=True); sc = fam["scales"]
    base = dict(target=1, Ts=sc["Ts"], V=sc["V"], d_bound=sc["d_bound"], domain=sc["domain"], E=sc["E"],
                domain_evidence={"kind": "mechanical_limit", "statement": "speed limiter"}, objective={"kind": "hard_bound", "tolerance": 0.6, "deadline": 0.1})
    base.update(kw)
    return api.new_spec(fam["plant"], **base), fam


def test_H6_domain_evidence_is_required_and_monitored():
    sp, fam = _joint_spec(domain_evidence=None)
    d = api.decide(sp)
    assert d["disposition"] == "CLARIFY" and d["question"]["path"] == "deterministic.domain_evidence"
    sp, fam = _joint_spec()
    d = api.decide(sp)
    assert d["disposition"] == "RELEASE" and "operating domain" in " ".join(d["claim"]["conditions"])
    # a trajectory that leaves the velocity domain is flagged by the run-time monitor
    sc = fam["scales"]; N = 1500; rng = np.random.default_rng(0)
    f = loop.chain_rhs(fam["plant"]["second_order"]["f"], fam["truth"])
    sim = loop.simulate_chain(f, [0.3, 1.0], sc["Ts"], N, rng.uniform(-sc["V"], sc["V"], N + 1), lambda t: 0.0, u_seq=np.full(N + 1, 4.0))
    est = api.solve(sp, sim["y"], sim["u"])["estimates"]
    assert abs(sim["x"][:, 1]).max() > 4.0 and est["assumption_monitor"]["certain_exit_index"] is not None


def test_H7_derived_disturbance_bound_is_conservative():
    sp, fam = _joint_spec()
    D, det = H.total_disturbance(sp)
    so = fam["plant"]["second_order"]; rng = np.random.default_rng(1)
    f0 = loop.chain_rhs(so["f0"].replace("y", "x1"), so["params"])
    worst = 0.0
    for _ in range(4000):
        prm = {k: rng.uniform(*so["param_ranges"][k]) for k in so["param_ranges"]}
        x1, x2, u = rng.uniform(-3.2, 3.2), rng.uniform(-4, 4), rng.uniform(-6, 6)
        if rng.random() < 0.5:          # corners are where the supremum sits
            prm = {k: so["param_ranges"][k][rng.integers(0, 2)] for k in so["param_ranges"]}; x2 = rng.choice([-4.0, 4.0]); u = rng.choice([-6.0, 6.0])
        worst = max(worst, abs(loop.chain_rhs(so["f"], prm)(x1, x2, u) - f0(x1, x2, u)))
    assert det["model_mismatch"] >= worst and det["model_mismatch"] <= 1.05 * worst + 1e-9
    assert D >= det["model_mismatch"] + det["additive"] + det["sample_hold_and_noise"] - 1e-12
    assert bounds.up(__import__("mpmath").iv.mpf([0.1, 0.1000001])) >= 0.1000001


def test_H8_ranges_thresholds_and_literal_questions():
    assert api.decide(hard_spec(V=None, ranges={"V": [2e-4, 1e-4]}))["disposition"] == "INVALID_INPUT"
    d = api.decide(hard_spec(V=None, ranges={"V": [5e-5, 2e-3]}))
    assert d["disposition"] == "CLARIFY" and 5e-5 <= d["question"]["threshold"] <= 2e-3
    thr = d["question"]["threshold"]
    assert api.decide(hard_spec(V=thr))["disposition"] == "RELEASE"                 # "if yes, a certified design exists" is true
    assert api.decide(hard_spec(V=thr * 1.05))["disposition"] != "RELEASE"          # and beyond it none is verified
    for comp in (d["question"]["witness"]["completion_a"], d["question"]["witness"]["completion_b"]):
        assert 5e-5 <= comp["deterministic.V"] <= 2e-3
    d = api.decide(hard_spec(V=None, ranges={"V": [5e-5, 1e-4]}))
    assert d["disposition"] == "RELEASE" and "universal_over_completions" in d
    assert d["config"]["V"] == 1e-4                                                  # worst corner used


def test_H9_release_requires_confirmation_of_the_final_specification():
    sp = hard_spec()
    S.put(sp, "deterministic.V", 1e-4, status="interpreted", source="text[10:18] '0.1 mrad'")
    s = Session(sp)
    d = s.decide()
    assert d["disposition"] == "CONFIRM" and d["pending_confirmation"] == ["deterministic.V"] and d["config"] is None
    h = d["spec_hash"]
    s.update({"objective.tolerance": 0.3})                 # a material fact changes after the readback
    with pytest.raises(ValueError):
        s.confirm(h)                                       # stale confirmation is rejected
    s.confirm(s.decide()["spec_hash"])
    d = s.decide()
    assert d["disposition"] == "RELEASE" and d["spec_hash"] == S.spec_hash(s.spec)
    assert api.solve(sp)["decision"]["disposition"] == "CONFIRM"          # the one-call API cannot release an unconfirmed reading either


def test_H8b_initial_error_question_is_literally_true():
    # generous deadline: every tested bound admits a design -> the question must not claim a critical value
    d = api.decide(hard_spec(E=None))
    q = d["question"]
    assert d["disposition"] == "CLARIFY" and q["path"] == "initial.E" and q.get("threshold") is None and "every bound tested" in q["question"]
    assert api.decide(hard_spec(E=[1e-3, 9.9e5]))["disposition"] == "RELEASE"
    # tight deadline: a finite critical value exists; below it a design is certified, 5 % above it none is
    tight = dict(objective={"kind": "hard_bound", "tolerance": 0.2, "deadline": 0.004})
    d = api.decide(hard_spec(E=None, **tight))
    thr = d["question"]["threshold"]
    assert thr is not None and 0 < thr < 1e6 and "at most" in d["question"]["question"]
    assert api.decide(hard_spec(E=[1e-3, thr], **tight))["disposition"] == "RELEASE"
    assert api.decide(hard_spec(E=[1e-3, 1.05 * thr], **tight))["disposition"] != "RELEASE"


def test_K5_language_failures_are_visible_not_silent():
    """Legacy Kalman finding 5: signs, variance units, number words, dropped fields, finite versus infinite horizon."""
    import json, pathlib
    from heros.nlu.quantities import find_quantities
    from heros.nlu.readers import RuleReader
    from heros.nlu import assemble as AS
    from heros.core import spec as S
    from heros.coordinator.coordinator import Session
    model = json.loads((pathlib.Path(__file__).resolve().parents[1] / "examples" / "servo_axis_model.json").read_text())
    att = S.empty_spec(); att["plant"].update(model["plant"] if "plant" in model else model)

    def run(q2, q3):
        text = "[Q1] the shaft speed [Q2] " + q2 + " [Q3] " + q3
        return AS.assemble(att, text, RuleReader().read(text, att["plant"]))

    # number words and squared units are read by the deterministic reader
    assert [round(q.value, 9) for q in find_quantities("two hundred hertz")] == [200.0]
    assert find_quantities("variance 0.04 (mrad)^2")[0].dim == "pos2" and abs(find_quantities("variance 0.04 (mrad)^2")[0].value - 4e-8) < 1e-20
    # a negative magnitude is reported and not used
    a = run("encoder sampled every 1 ms, the encoder error never exceeds -0.1 mrad, disturbance acceleration below 6 rad/s^2", "guaranteed error below 0.09 rad/s in the long run")
    assert any(i["code"] == "sign" for i in a["issues"]) and S.get(a["spec"], "deterministic.V") is None
    # a "variance" with a non-squared unit is reported and not used
    text = "[Q1] the shaft speed [Q2] encoder sampled every 1 ms, measurement noise variance 0.1 mrad [Q3] smallest mean-square error"
    a = AS.assemble(att, text, RuleReader().read(text, att["plant"]))
    assert S.get(a["spec"], "stochastic.R") is None                       # never converted silently
    q = find_quantities(text)[1]
    forced = {"quantities": [{"start": q.start, "end": q.end, "role": "meas_std", "conf": 1.0}], "objective": "mse_optimal"}
    a = AS.assemble(att, text, forced)                                     # even if a reader calls it a standard deviation
    assert any(i["code"] == "variance_unit" for i in a["issues"]) and S.get(a["spec"], "stochastic.R") is None
    # a dropped field is asked for after confirmation; it is never filled with a default
    a = run("encoder sampled every 1 ms, disturbance acceleration below 6 rad/s^2", "guaranteed error below 0.09 rad/s in the long run")
    ses = Session(a["spec"]); ses.confirm(S.spec_hash(ses.spec))
    d = ses.decide()
    assert d["disposition"] == "CLARIFY" and d["question"]["path"] == "deterministic.V"
    # a deadline (finite time) and an ultimate limit are different requests
    a1 = run("encoder sampled every 1 ms, the encoder error never exceeds 0.1 mrad, disturbance acceleration below 6 rad/s^2", "guaranteed error below 0.09 rad/s in the long run")
    a2 = run("encoder sampled every 1 ms, the encoder error never exceeds 0.1 mrad, disturbance acceleration below 6 rad/s^2", "guaranteed error below 0.09 rad/s from 50 ms after start")
    assert S.get(a1["spec"], "objective.deadline") is None and abs(S.get(a2["spec"], "objective.deadline") - 0.05) < 1e-12
