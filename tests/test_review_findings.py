"""Regression tests for the findings of the four internal reviews: each test is a counterexample from a review."""
import copy, json, math, pathlib
import numpy as np
import pytest
from heros import api, record as R
from heros.core import spec as S
from heros import evaluation as EV

Ts, b = 0.001, 4.0
SERVO = {"family": "servo_axis", "kind": "second_order", "states": [{"name": "angle", "unit": "rad"}, {"name": "speed", "unit": "rad/s"}],
         "inputs": [{"name": "torque", "unit": "N m"}], "outputs": [{"name": "encoder", "unit": "rad"}],
         "linear": {"clock": "continuous", "A": [[0.0, 1.0], [0.0, 0.0]], "B": [[0.0], [b]], "C": [[1.0, 0.0]], "Ts_native": None},
         "second_order": {"f": "b*u", "f0": "b*u", "params": {"b": b}, "param_ranges": {}, "output": "x1"}, "flags": []}
G = np.array([[Ts * Ts / 2], [Ts]])


def stat(D, V):
    return dict(Q=((D * D / 3) * (G @ G.T)).tolist(), Q_kind="discrete_cov", R=[[V * V / 3]])


def test_A1_no_impossibility_claim_for_a_state_dependent_plant():
    plant = copy.deepcopy(SERVO)
    plant.update(family="custom", linear=None)
    plant["second_order"] = {"f": "-1000*x2 + u", "f0": "u", "params": {}, "param_ranges": {}, "output": "x1"}
    spec = api.new_spec(plant, target=1, Ts=Ts, V=1e-4, d_bound=0.5, domain={"x1": [-1, 1], "x2": [-1.0, 1.0], "u": [-0.1, 0.1]},
                        domain_evidence={"kind": "mechanical_limit", "statement": "stops"}, E=[1e-3, 1.0],
                        objective={"kind": "hard_bound", "tolerance": 0.01, "deadline": 0.1})
    assert api.decide(spec)["disposition"] != "PROVED_IMPOSSIBLE"


def test_A2_observer_mse_is_judged_on_the_same_window():
    spec = api.new_spec(copy.deepcopy(SERVO), target=1, Ts=Ts, P0=np.diag([1e-6, 1.0]).tolist(), distribution="gaussian", t_end=1.0,
                        objective={"kind": "mse_bound", "horizon": "finite", "mse_target": 1.385e-4, "deadline": 0.005}, **stat(6.0, 1e-4))
    d = api.decide(spec)
    assert not (d["disposition"] == "RELEASE" and d["selected"] == "high_gain_observer")


def test_A3_steady_state_gain_is_not_released_for_a_deadline_it_misses():
    base = dict(target=1, Ts=Ts, distribution="gaussian", t_end=2.0, **stat(0.05, 1e-4))
    v_ss = api.decide(api.new_spec(copy.deepcopy(SERVO), objective={"kind": "mse_optimal", "horizon": "steady_state"}, **base))["claim"]["value"]
    obj = {"kind": "mse_bound", "horizon": "steady_state", "mse_target": 2 * v_ss, "deadline": 0.2}
    d = api.decide(api.new_spec(copy.deepcopy(SERVO), objective=obj, P0=np.diag([1e-6, 1.0]).tolist(), **base))
    assert d["disposition"] != "RELEASE"
    d = api.decide(api.new_spec(copy.deepcopy(SERVO), objective=obj, **base))          # P0 now matters: it is asked for
    assert d["disposition"] == "CLARIFY" and d["question"]["path"] == "initial.P0"


def test_A4_disturbance_bound_covers_measurements_outside_the_range():
    plant = {"family": "custom", "kind": "second_order", "states": [{"name": "position", "unit": "m"}, {"name": "velocity", "unit": "m/s"}],
             "inputs": [{"name": "force", "unit": "N"}], "outputs": [{"name": "probe", "unit": "m"}], "linear": None,
             "second_order": {"f": "u - kk*x1**3", "f0": "u - kk*y**3", "params": {"kk": 1e7}, "param_ranges": {}, "output": "x1"}, "flags": []}
    spec = api.new_spec(plant, target=1, Ts=Ts, V=0.01, d_bound=0.1, domain={"x1": [-0.01, 0.01], "x2": [-0.1, 0.1], "u": [-20.0, 20.0]},
                        domain_evidence={"kind": "mechanical_limit", "statement": "end stops"}, E=[0.01, 0.05],
                        objective={"kind": "hard_bound", "tolerance": 1.4, "deadline": 0.2})
    from heros.khalil import specialist as H
    D, det = H.total_disturbance(spec)
    # admissible: x1 = 0.01, y = 0.02: |f(x1) - f0(y)| = kk (0.02^3 - 0.01^3) = 70
    assert D >= 70.0 + 0.1


def test_A5_constrained_ranges_use_every_vertex():
    spec = api.new_spec(copy.deepcopy(SERVO), target=1, Ts=0.01, E=[0.1, 1.0], objective={"kind": "hard_bound", "tolerance": 0.938864},
                        ranges={"V": [0.9, 1.0], "D": [0.05, 0.25]}, constraints=[{"paths": ["V", "D"], "sum": 1.1}])
    d = api.decide(spec)
    assert d["disposition"] != "RELEASE"


def test_A6_kalman_hard_bound_uses_the_observer_initialisation():
    spec = api.new_spec(copy.deepcopy(SERVO), target=1, Ts=Ts, V=1e-4, D=6.0, E=[2e-3, 3.0], distribution="bounded", **stat(6.0, 1e-4),
                        P0=np.diag([4e-6 / 3, 3.0]).tolist(), objective={"kind": "hard_bound", "tolerance": 0.0906, "deadline": 0.05})
    d = api.decide(spec, methods=["kalman_filter"])
    assert d["disposition"] == "RELEASE" and d["config"]["init_policy"] == "first_measurement"


def test_A7_no_hidden_window():
    spec = api.new_spec(copy.deepcopy(SERVO), target=1, Ts=Ts, P0=(1e-12 * np.eye(2)).tolist(), distribution="gaussian",
                        objective={"kind": "mse_bound", "horizon": "finite", "mse_target": 2.5e-12}, **stat(1e-4, 1e-3))
    assert api.decide(spec)["disposition"] != "RELEASE"


def test_A8_sidebar_example_with_its_stated_range():
    sc = {"family": "scalar", "kind": "linear", "states": [{"name": "x", "unit": "-"}], "inputs": [], "outputs": [{"name": "y", "unit": "-"}],
          "linear": {"clock": "discrete", "A": [[1.0]], "B": [[0.0]], "C": [[1.0]], "Ts_native": 1.0}, "second_order": None, "flags": []}
    kw = dict(target=0, Q=[[0.0]], Q_kind="discrete_cov", distribution="gaussian", t_end=1.0,
              ranges={"P0": [0.5, 1.0], "R": [1.0, 1.5]}, constraints=[{"paths": ["P0", "R"], "sum": 2.0}])
    d = api.decide(api.new_spec(sc, objective={"kind": "mse_bound", "horizon": "finite", "mse_target": 0.65}, **kw))
    assert d["disposition"] == "RELEASE" and d["claim"]["value"] <= 0.65
    d = api.decide(api.new_spec(sc, objective={"kind": "mse_optimal", "horizon": "finite"}, **kw))
    assert d["disposition"] == "CLARIFY"


def test_A14_a_range_on_an_unused_field_is_not_asked_about():
    spec = api.new_spec(copy.deepcopy(SERVO), target=1, Ts=Ts, distribution="bounded", **stat(6.0, 1e-4),
                        objective={"kind": "mse_optimal", "horizon": "steady_state"}, ranges={"V": [1e-4, 2e-4]})
    assert api.decide(spec)["disposition"] == "RELEASE"


def test_B1_judge_rejects_releases_that_ignore_facts_of_the_truth():
    spec = api.new_spec(copy.deepcopy(SERVO), target=1, Ts=Ts, V=1e-4, D=6.0, E=[2e-3, 3.0],
                        objective={"kind": "hard_bound", "tolerance": 0.0906, "deadline": 0.05})
    dec = api.decide(spec)
    assert dec["disposition"] == "RELEASE"
    for change in ({"time.irregular": True}, {"time.t_end": 0.01}, {"objective.peak_cap": 1e-3}):
        truth = copy.deepcopy(spec)
        for k, v in change.items():
            S.put(truth, k, v, status="supplied")
        assert not EV.judge_release(truth, dec)["correct"], change


def test_C03_replay_refuses_outputs_it_cannot_recompute():
    spec = api.new_spec(copy.deepcopy(SERVO), target=1, Ts=Ts, V=1e-4, D=6.0, E=[2e-3, 3.0],
                        objective={"kind": "hard_bound", "tolerance": 0.0906, "deadline": 0.05})
    y = np.sin(np.arange(200) * Ts)
    rec = json.loads(R.dumps(api.solve(spec, y, np.zeros(200))["record"]))
    assert api.replay(rec)["match"]
    rec["inputs"] = {}
    assert not api.replay(rec)["match"]


def test_recheck_C_N3_a_record_without_outputs_does_not_look_fully_verified():
    spec = api.new_spec(copy.deepcopy(SERVO), target=1, Ts=Ts, V=1e-4, D=6.0, E=[2e-3, 3.0],
                        objective={"kind": "hard_bound", "tolerance": 0.0906, "deadline": 0.05})
    y = np.sin(np.arange(200) * Ts)
    full = json.loads(R.dumps(api.solve(spec, y, np.zeros(200))["record"]))
    rep = api.replay(full)
    assert rep["match"] and rep["outputs_stored"] and not rep["unverified"]
    for strip_inputs in (False, True):
        rec = copy.deepcopy(full)
        rec.pop("outputs")
        if strip_inputs:
            rec["inputs"] = {}
        rep = api.replay(rec)
        assert not rep["outputs_stored"] and any(u.startswith("outputs: none stored") for u in rep["unverified"])


def test_B20_judge_checks_finite_time_statistical_claims_from_the_true_initial_covariance():
    base = dict(target=1, Ts=Ts, distribution="gaussian", t_end=2.0, P0=np.diag([1e-6, 1.0]).tolist(), **stat(0.05, 1e-4))
    # steady-state optimal gain asked for "from 0.2 s on": the stated value is the variance over those samples from P0
    spec = api.new_spec(copy.deepcopy(SERVO), objective={"kind": "mse_optimal", "horizon": "steady_state", "deadline": 0.2}, **base)
    d = api.decide(spec)
    assert d["disposition"] == "RELEASE" and d["config"]["mode"] == "steady_state"
    assert EV.judge_release(spec, d)["correct"]
    bad = copy.deepcopy(d)
    bad["claim"]["value"] = float(d["config"]["Pf_ss"][1][1])        # the asymptotic value (the earlier defect)
    assert bad["claim"]["value"] < 0.5 * d["claim"]["value"]
    assert not EV.judge_release(spec, bad)["correct"]
    # a time-varying optimal filter without an end time: the claim covers every sample, the limit included
    spec2 = api.new_spec(copy.deepcopy(SERVO), objective={"kind": "mse_optimal", "horizon": "finite"},
                         **{k: v for k, v in base.items() if k != "t_end"})
    d2 = api.decide(spec2)
    assert d2["disposition"] == "RELEASE" and d2["config"]["N"] is None
    assert EV.judge_release(spec2, d2)["correct"]
    # an observer released for a mean-square level: the stated value is checked too
    spec3 = api.new_spec(copy.deepcopy(SERVO), objective={"kind": "mse_bound", "horizon": "steady_state", "mse_target": 1.0, "deadline": 0.2}, **base)
    d3 = api.decide(spec3)
    assert d3["disposition"] == "RELEASE" and EV.judge_release(spec3, d3)["correct"]
    bad3 = copy.deepcopy(d3); bad3["claim"]["value"] *= 0.5
    assert not EV.judge_release(spec3, bad3)["correct"]


def test_C13_unknown_exact_period_ends_without_a_release():
    spec = api.new_spec(copy.deepcopy(SERVO), target=1, V=1e-4, D=6.0, E=[2e-3, 3.0], objective={"kind": "hard_bound", "tolerance": 0.09})
    spec.setdefault("ranges", {})["time.Ts"] = [0.0009, 0.0011]
    d = api.decide(spec)
    assert d["disposition"] == "CLARIFY" and d["question"]["path"] == "time.Ts"
    spec["unavailable"] = ["time.Ts"]
    assert api.decide(spec)["disposition"] == "NO_VERIFIED_CANDIDATE"


def test_A17_constraint_corners_tolerate_rounding():
    spec = api.new_spec(copy.deepcopy(SERVO), target=1, Ts=0.01, E=[0.1, 1.0], objective={"kind": "hard_bound", "tolerance": 5.0},
                        ranges={"V": [0.9, 1.0], "D": [0.1, 0.2]}, constraints=[{"paths": ["V", "D"], "sum": 1.1}])
    assert api.decide(spec)["disposition"] != "INVALID_INPUT"


def test_recheck_A_N1_window_rule_judges_each_state_on_its_own_scale():
    from heros.kalman import specialist as K
    from independent import ref_kalman as RK
    A = np.diag([1.0, 0.5]); C = np.eye(2); R = np.eye(2)
    Qd = np.diag([1e-8, 7.5e5]); P0 = np.diag([1e-12, 1e6])
    vmax, _, steps, ok = K.window_variance(A, C, Qd, R, P0, 0, 0, None)
    q, r = 1e-8, 1.0                                                  # the slow state is a scalar random walk
    pp = 0.5 * (q + np.sqrt(q * q + 4 * q * r))
    lim = pp * r / (pp + r)                                           # its filtered variance rises to this limit
    assert ok and abs(vmax[0] - lim) <= 1e-6 * lim and lim > 1e-5
    # the reference iteration judges each entry on its own scale too (it stopped 6 % high with a rule relative to max|P|)
    assert abs(float(RK.riccati_fixed_point(A, C, Qd, R)[1][0, 0]) - lim) <= 1e-9 * lim
    vmax_c, _, _, ok_c = K.window_variance(A, C, Qd, R, P0, 0, 0, None, gains=np.diag([0.5, 0.5])[None])
    Pl = K.steady_cov_of_constant_gain(A, C, Qd, R, np.diag([0.5, 0.5]))
    assert ok_c and vmax_c[0] >= float(Pl[0, 0]) * (1 - 1e-9)


def test_recheck_A_N2_constrained_hard_bound_states_the_largest_value():
    spec = api.new_spec(copy.deepcopy(SERVO), target=1, Ts=0.01, E=[0.1, 1.0], objective={"kind": "hard_bound", "tolerance": 1.2},
                        ranges={"V": [0.9, 1.0], "D": [0.05, 0.25]}, constraints=[{"paths": ["D", "V"], "sum": 1.1}])
    d = api.decide(spec)
    assert d["disposition"] == "RELEASE"
    vals = d["universal_over_completions"]["values_at_vertices"]
    assert d["claim"]["value"] >= max(vals) * (1 - 1e-12)
    assert not any(c.startswith("|measurement error| <= ") for c in d["claim"]["conditions"])


def test_recheck_A_m3_constraints_that_share_a_field_have_vertices():
    cons = [{"paths": ["D", "V"], "sum": 1.1}, {"paths": ["D", "deterministic.d_bound"], "sum": 0.3}]
    rng = {"V": [0.9, 1.0], "D": [0.05, 0.25], "deterministic.d_bound": [0.0, 0.2]}
    spec = api.new_spec(copy.deepcopy(SERVO), target=1, Ts=0.01, E=[0.1, 1.0], objective={"kind": "hard_bound", "tolerance": 1.2},
                        ranges=rng, constraints=cons)
    d = api.decide(spec)
    assert d["disposition"] == "RELEASE"
    verts = d["universal_over_completions"]["vertices"]
    V, D, db = "deterministic.V", "deterministic.D", "deterministic.d_bound"
    assert len(verts) == 2 and all(abs(v[V] + v[D] - 1.1) < 1e-12 and abs(v[D] + v[db] - 0.3) < 1e-12 for v in verts)
    assert d["claim"]["value"] >= max(d["universal_over_completions"]["values_at_vertices"]) * (1 - 1e-12)


def test_recheck_A_m4_no_lower_limit_when_a_state_dependent_model_accompanies_a_supplied_D():
    out = {}
    for f in ("b*u", "b*u - 2*x1"):
        pl = copy.deepcopy(SERVO); pl["second_order"]["f"] = f
        spec = api.new_spec(pl, target=1, Ts=0.001, V=1e-4, D=6.0, objective={"kind": "hard_bound", "tolerance": 0.03})
        out[f] = api.decide(spec)["disposition"]
    assert out == {"b*u": "PROVED_IMPOSSIBLE", "b*u - 2*x1": "NO_VERIFIED_CANDIDATE"}


def test_recheck_B_N1_false_impossibility_is_scored_for_numpy_booleans():
    # the two-signal branch of the judge's check used to return numpy.False_, which an identity test missed
    case = {"truth": api.new_spec(copy.deepcopy(SERVO), target=1, Ts=0.001, V=1e-4, D=6.0, objective={"kind": "hard_bound", "tolerance": 0.06}),
            "final": {"disposition": "RELEASE"}}
    dec = {"disposition": "PROVED_IMPOSSIBLE", "selected": None, "explanations": ["claimed"]}
    r = EV.judge(case, {"decision": dec, "n_questions": 0})
    assert r["outcome"] == "false_impossibility" and r["impossibility_verified"] is False
    case["truth"]["objective"] = {"kind": "mse_optimal"}
    assert EV.judge(case, {"decision": dec, "n_questions": 0})["outcome"] == "false_impossibility"


def test_recheck_B_N3_judge_finds_the_interior_maximum_of_the_model_mismatch():
    # the fields of test case test-00560 (rotary joint) that the search uses, inline so that the test also runs from the
    # light archive, which leaves out data/test_structured.jsonl; the mismatch peaks at an interior angle
    t = {"plant": {"second_order": {"f": "(u - mgl*sin(x1) - bv*x2 - Fc*tanh(x2/vs))/J", "f0": "(u - mgl*sin(y))/J", "output": "x1",
                                    "param_ranges": {"Fc": [0.04745, 0.14235], "J": [0.04218, 0.04662], "bv": [0.0206, 0.0618],
                                                     "mgl": [3.6005, 3.9795], "vs": [0.05, 0.05]},
                                    "params": {"Fc": 0.0949, "J": 0.0444, "bv": 0.0412, "mgl": 3.79, "vs": 0.05}}},
         "deterministic": {"V": 4.34e-05, "d_bound": 0.124, "domain": {"u": [-11.4, 11.4], "x1": [-3.2, 3.2], "x2": [-4.0, 4.0]}},
         "time": {"Ts": 0.001}}
    assert EV._sampled_disturbance(t) >= 31.86      # reviewer B's dense supremum 31.8656 (the corner-started search gave 23.74)


def test_second_recheck_B_reference_riccati_refuses_a_non_stabilizing_fixed_point():
    from independent import ref_kalman as RK
    with pytest.raises(RuntimeError):
        RK.riccati_fixed_point(np.eye(2), np.array([[1.0, 0.0]]), np.diag([1e-4, 0.0]), np.eye(1))
    P, Pf, M = RK.riccati_fixed_point(np.eye(1), np.eye(1), 1e-4 * np.eye(1), np.eye(1))   # random walk: stabilizing
    assert abs(P[0, 0] - 0.5 * (1e-4 + np.sqrt(1e-8 + 4e-4))) <= 1e-10 * P[0, 0]


# ---- second recheck A ----------------------------------------------------------------------------------------------
_SERVO_PRECISE = {"family": "servo_axis", "kind": "second_order", "states": [{"name": "shaft angle", "unit": "rad"}, {"name": "shaft speed", "unit": "rad/s"}],
                  "inputs": [{"name": "motor torque", "unit": "N m"}], "outputs": [{"name": "encoder angle", "unit": "rad"}],
                  "linear": {"clock": "continuous", "A": [[0.0, 1.0], [0.0, 0.0]], "B": [[0.0], [1.0]], "C": [[1.0, 0.0]], "Ts_native": None},
                  "second_order": {"f": "b*u", "f0": "b*u", "params": {"b": 1.0}, "param_ranges": {}, "output": "x1"}, "flags": []}


@pytest.mark.parametrize("qa, r, target, expected", [(1e6, 1e-16, 1.9990e-5, "PROVED_IMPOSSIBLE"), (1e8, 1e-14, 2.0001e-3, "RELEASE")])
def test_second_recheck_A_new1_badly_conditioned_riccati_limit_is_refined(qa, r, target, expected):
    # precise encoder with rank-1 acceleration noise: the library DARE solution was off by up to 8e-4 relative in the
    # filtered velocity variance (true limits 1.99996e-5 and 1.99996e-3), giving a false release and a false impossibility
    Ts = 1e-3
    G = np.array([[Ts * Ts / 2], [Ts]])
    for horizon, P0 in (("finite", [[1e-20, 0.0], [0.0, 1e-12]]), ("steady_state", None)):
        spec = api.new_spec(copy.deepcopy(_SERVO_PRECISE), target=1, Ts=Ts, Q=(qa * (G @ G.T)).tolist(), Q_kind="discrete_cov", R=[[r]], P0=P0,
                            distribution="gaussian", white=True, objective={"kind": "mse_bound", "horizon": horizon, "mse_target": target})
        d = api.decide(spec)
        assert d["disposition"] == expected
        if expected == "RELEASE":
            assert EV.judge_release(spec, d)["correct"]
        else:
            assert EV._verify_impossibility(spec, d)[0] is True


def test_second_recheck_A_new2_constrained_claims_state_every_certified_value():
    # peak cap and deadline under an equality constraint: both stated numbers are the largest certified values at the vertices
    spec = api.new_spec(copy.deepcopy(SERVO), target=1, Ts=0.01, E=[1.0, 2.0],
                        objective={"kind": "hard_bound", "tolerance": 3.0, "deadline": 0.5, "peak_cap": 50.0},
                        ranges={"V": [0.9, 1.0], "D": [0.05, 0.25]}, constraints=[{"paths": ["D", "V"], "sum": 1.1}])
    d = api.decide(spec)
    assert d["disposition"] == "RELEASE"
    vals = d["universal_over_completions"]["values_at_vertices"]
    assert d["claim"]["value"] >= max(vals) * (1 - 1e-12) and d["claim"]["peak_value"] >= max(vals) * (1 - 1e-12)
    nums = [float(x) for x in __import__("re").findall(r"<= ([0-9.]+)", d["claim"]["statement"])]
    assert len(nums) == 2 and min(nums) >= max(vals) * (1 - 1e-12)


def test_second_recheck_A_supplied_D_with_an_input_only_model_gives_no_lower_limit():
    # Revision 2: the two-signal construction holds the plant at rest at an admissible input.  For an input-only true model
    # that vanishes at u = 0 the acceleration error at rest is the free disturbance, so the limit holds (it was withheld
    # before, conservatively); when the stated input range excludes every rest input, no limit is claimed.
    out = {}
    for f, dom in (("b*u", None), ("2*b*u", None), ("2*b*u", {"u": [0.5, 1.0]})):
        pl = copy.deepcopy(SERVO); pl["second_order"]["f"] = f
        spec = api.new_spec(pl, target=1, Ts=0.001, V=1e-4, D=6.0, domain=dom, objective={"kind": "hard_bound", "tolerance": 0.03})
        out[(f, dom is None)] = api.decide(spec)["disposition"]
    assert out == {("b*u", True): "PROVED_IMPOSSIBLE", ("2*b*u", True): "PROVED_IMPOSSIBLE", ("2*b*u", False): "NO_VERIFIED_CANDIDATE"}


def test_second_recheck_A_slow_mode_next_to_a_decreasing_state():
    from heros.kalman import specialist as K
    q = 1e-20
    A = np.diag([0.5, 1.0]); C = np.eye(2); Qd = np.diag([1.0, q]); R = np.eye(2)
    L = (q + np.sqrt(q * q + 4 * q)) / 2
    vm, _, _, ok = K.window_variance(A, C, Qd, R, np.diag([100.0, L * (1 - 4e-5)]), 1, 0, None)
    assert ok and abs(vm[0] - L / (1 + L)) <= 1e-12 * L


def test_second_recheck_A_reference_riccati_by_doubling():
    from independent import ref_kalman as RK
    for q, r in ((1e-4, 1.0), (1e-8, 1.0), (1e-15, 1e18), (1e-20, 1.0)):
        P = RK.riccati_fixed_point(np.eye(1), np.eye(1), q * np.eye(1), r * np.eye(1))[0][0, 0]
        assert abs(P / (0.5 * (q + np.sqrt(q * q + 4 * q * r))) - 1) <= 1e-12


def test_second_recheck_A_constrained_claim_bounds_interior_completions_with_derived_D():
    # derived D whose Lipschitz constant depends on V (concave): the vertex argument needs the widened range; the stated
    # value must bound the certified worst case at interior admissible completions too
    from heros.coordinator import coordinator as CO
    plant = {"family": "rotary_joint", "kind": "second_order", "states": [{"name": "joint angle", "unit": "rad"}, {"name": "joint velocity", "unit": "rad/s"}],
             "inputs": [{"name": "joint torque", "unit": "N m"}], "outputs": [{"name": "joint encoder", "unit": "rad"}], "linear": None,
             "second_order": {"f": "(u - mgl*sin(x1))/J", "f0": "(u - mgl*sin(y))/J", "params": {"J": 1.0, "mgl": 0.48}, "param_ranges": {}, "output": "x1"},
             "flags": []}
    spec = api.new_spec(plant, target=1, Ts=0.01, objective={"kind": "hard_bound", "tolerance": 5.0},
                        domain={"x1": [1.0, 2.0], "x2": [-0.5, 0.5], "u": [-1.0, 1.0]},
                        domain_evidence={"kind": "mechanical_limit", "statement": "joint end stops and speed limiter"},
                        ranges={"V": [0.75, 1.0], "d_bound": [0.0, 0.25]}, constraints=[{"paths": ["d_bound", "V"], "sum": 1.0}])
    d = api.decide(spec)
    assert d["disposition"] == "RELEASE"
    vals = [CO._verify_release(d, spec, {"deterministic.V": float(V), "deterministic.d_bound": float(1.0 - V)})[1] for V in np.linspace(0.75, 1.0, 11)]
    assert d["claim"]["value"] >= max(vals) * (1 - 1e-12)
    assert any("operating domain" in c for c in d["claim"]["conditions"])      # the domain clause of the derived bound is kept
