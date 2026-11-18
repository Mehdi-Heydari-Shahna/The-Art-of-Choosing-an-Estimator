"""Regression tests for the seven review cases supplied for the revision 2b archive (tests/fixtures/reviewer_cases_rev2b,
the files of the review attachment, unchanged): the timestamp check under a change of clock origin (A1 to A3), the
first-sample limit for a plant that must keep moving forward (B), a record that ends just before a sample (C), and a
design whose model uses an input that the data do not contain (D1, D2).  Expected answers come from closed forms, from
the review's own mathematical notes, or from the judge and the independent references."""
import copy, hashlib, json, math, pathlib
import numpy as np
import pytest
from heros import api, record as R
from heros import evaluation as EV
from heros.core import limits as LM, timing as TM
from independent import ref_hgo as RH

ROOT = pathlib.Path(__file__).resolve().parents[1]
FX = ROOT / "tests" / "fixtures" / "reviewer_cases_rev2b"
SHA = {"A1_relative_times_with_jitter.json": "bcd3146029200c1f4ad0",
       "A2_absolute_times_same_jitter.json": "1ea8898195e9d2d54b2d",
       "A3_absolute_times_on_grid.json": "0f16e039d59098833dcc",
       "B_forward_only_carriage.json": "792a60c9bdb1206513ee",
       "C_exact_window_ends_before_next_sample.json": "bdd1c1d4936bc2150f61",
       "D1_fixed_input_only_in_specification.json": "80fa1f2801e6d699fc98",
       "D2_explicit_known_input_control.json": "2169eb93373461aca41c"}
REVISED = {"A1": "OUT_OF_SCOPE", "A2": "OUT_OF_SCOPE", "A3": "RELEASE", "B": "NO_VERIFIED_CANDIDATE", "C": "RELEASE",
           "D1": "RELEASE", "D2": "RELEASE"}


def _case(prefix):
    (f,) = [p for p in SHA if p.startswith(prefix + "_")]
    return json.loads((FX / f).read_text())


def _solve(prefix):
    return api.solve(**copy.deepcopy(_case(prefix)["arguments"]))       # the arguments object as supplied, nothing else


def test_fixtures_are_the_files_as_received():
    for f, h in SHA.items():
        assert hashlib.sha256((FX / f).read_bytes()).hexdigest().startswith(h), f


@pytest.mark.parametrize("prefix", sorted(REVISED))
def test_revised_outcome_through_the_public_interface(prefix):
    out = _solve(prefix)
    d = out["decision"]
    assert d["disposition"] == REVISED[prefix], d["explanations"][:2]
    assert d.get("impossibility") is None
    rp = api.replay(json.loads(R.dumps(out["record"])))
    assert rp["match"], rp["differences"]


# ---------------------------------------------------------------- A: change of clock origin
def test_A_same_sampling_deviation_for_either_clock_origin():
    t1, t2, t3 = (np.asarray(_case(p)["arguments"]["t"], float) for p in ("A1", "A2", "A3"))
    Ts = _case("A1")["arguments"]["spec"]["time"]["Ts"]
    c1, c2, c3 = (TM.check_timestamps(t, Ts) for t in (t1, t2, t3))
    assert c1["max_deviation"] == c2["max_deviation"] == 2.0 ** -17          # sample 1 is 2^-17 s early on both clocks
    assert c1["code"] == c2["code"] == "irregular_sampling" and c3["ok"]
    # the labels near 2^30 s resolve 2^-22 s; four such units stay far below the deviation
    assert c2["label_precision"] == 4 * 2.0 ** -22 < 2.0 ** -17
    # the release on the regular clock states how precisely its labels confirm the instants
    d3 = _solve("A3")["decision"]
    assert any("only to within 9.54e-07 s" in c for c in d3["claim"]["conditions"])
    assert not any("only to within" in c for c in api.solve(**copy.deepcopy(dict(_case("A3")["arguments"], t=list(t3 - t3[0]))))["decision"]["claim"]["conditions"])


def test_A_labels_too_coarse_for_the_period_are_not_used():
    Ts = 2.0 ** -10
    t = 2.0 ** 40 + np.arange(5) * Ts                    # float64 resolves only 2^-12 s near 2^40 s
    chk = TM.check_timestamps(t, Ts)
    assert chk["code"] == "bad_timestamps" and "relative to the first sample" in chk["message"]
    assert TM.check_timestamps(t - t[0], Ts)["ok"]


# ---------------------------------------------------------------- B: the carriage keeps moving forward
def test_B_no_witness_outside_the_stated_speed_range():
    spec = copy.deepcopy(_case("B")["arguments"]["spec"])
    # the review's bound: v >= 1, |v'| <= 5 and x1(1) <= 1.001 give v <= 1.1 throughout, so the estimate 1 meets 0.2
    h = math.sqrt(10 * 0.001)
    assert h <= 0.1 + 1e-12 < spec["objective"]["tolerance"]
    assert LM.viable_initial_velocities(spec, 1.0, 2.0) is None
    assert RH.first_sample_velocity_limit(spec, 1.0, 2.0) is None
    assert EV._verify_impossibility(spec, None)[0] is False
    # narrow change: with a speed range that admits rest, braking from 2 m/s at 5 m/s^2 needs 0.4 m of the 1.001 m
    spec["deterministic"]["domain"]["x2"] = [-0.5, 2.0]
    d = api.decide(spec)
    assert d["disposition"] == "PROVED_IMPOSSIBLE" and abs(d["impossibility"]["bound"] - 0.5) < 1e-9
    assert EV._verify_impossibility(spec, d)[0] is True


# ---------------------------------------------------------------- C: the record ends just before a sample
def test_C_window_ends_before_the_next_sample():
    a = _case("C")["arguments"]
    Ts, te = a["spec"]["time"]["Ts"], a["spec"]["time"]["t_end"]
    assert te == 4 - 2.0 ** -31
    assert TM.sample_window(Ts, None, te) == (0, 3) and EV._window(Ts, None, te) == (0, 3)
    # decimal input still counts as the intended sample
    assert TM.sample_window(0.1, 0.7, 0.3) == (7, 3) and EV._window(0.1, 0.7, 0.3) == (7, 3)
    assert TM.sample_window(0.001, 0.06, None)[0] == 60
    # filtered variances of the scalar random walk (review note): only the fifth sample exceeds the target
    P, fv = 0.1, []
    for k in range(5):
        Pf = P - P * P / (P + 1.0); fv.append(Pf); P = Pf + 1.0
    target = a["spec"]["objective"]["mse_target"]
    assert max(fv[:4]) < target < fv[4]
    d = _solve("C")["decision"]
    assert d["selected"] == "kalman_filter" and abs(d["claim"]["value"] - fv[3]) < 1e-12
    assert EV.judge_release(a["spec"], d)["correct"]
    assert EV._verify_impossibility(a["spec"], None)[0] is False


# ---------------------------------------------------------------- D: the input is fixed in the specification only
def test_D1_runs_with_the_fixed_input_and_matches_the_analytic_trajectory():
    case = _case("D1")
    out = _solve("D1")
    X = np.asarray(out["estimates"]["x_hat"], float)
    v = np.asarray(case["actual_true_velocity_if_applicable"], float)
    assert out["inputs"] == {"code": "fixed_input_from_specification", "value": 1.0}
    assert abs(X[5, 1] - 0.5) < 1e-12 and np.max(np.abs(X[5:, 1] - v[5:])) < 1e-12
    assert any("the run applies the input u = 1" in c for c in out["decision"]["claim"]["conditions"])
    out2 = _solve("D2")
    assert np.max(np.abs(np.asarray(out2["estimates"]["x_hat"])[:, 1] - X[:, 1])) < 1e-12


def test_D_nothing_is_run_without_input_values_when_the_model_needs_them():
    a = copy.deepcopy(_case("D1")["arguments"])
    a["spec"]["deterministic"]["domain"] = {"u": [0.5, 1.5]}
    out = api.solve(**a)
    assert out["decision"]["disposition"] == "INVALID_INPUT" and out["estimates"] is None
    assert out["decision"]["issues"][0]["code"] == "missing_input" and "withdrawn_release" in out["decision"]
    assert api.replay(json.loads(R.dumps(out["record"])))["match"]
    # a model that does not use the input runs without input values
    b = copy.deepcopy(_case("A3")["arguments"]); b.pop("u")
    assert api.solve(**b)["decision"]["disposition"] == "RELEASE"
