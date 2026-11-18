"""Regression tests for the six review cases of 18 November 2026 (tests/fixtures/reviewer_cases_rev2, the files of the
review attachment, unchanged) and for the three rules they exposed: irregular sampling (declared, or shown by the recorded
instants), a true parameter value stated as a one-point range, and lower limits under a stated travel range.  The expected
answers are derived in the tests (closed forms) or by the judge and the independent references, not by the specialists."""
import copy, hashlib, json, math, pathlib
import numpy as np
import pytest
from heros import api
from heros import evaluation as EV
from heros.core import limits as LM, timing as TM
from heros.khalil import specialist as H
from independent import ref_hgo as RH

ROOT = pathlib.Path(__file__).resolve().parents[1]
FX = ROOT / "tests" / "fixtures" / "reviewer_cases_rev2"
SHA = {"01a_declared_irregular_no_timestamps.json": "4f4da48fca472782ec0b",
       "01b_same_record_supplied_timestamps.json": "c6a5c3133b4cac5b9af2",
       "01c_exact_timestamps_within_acceptance_tolerance.json": "41f8ceb1238d18a03ff5",
       "02_true_gain_two_nominal_gain_one.json": "878e3d492de0d34a2709",
       "03a_first_sample_known_travel.json": "c7631c20a1a4aece3121",
       "03b_first_sample_known_travel_acceleration.json": "534fc325bc227d7fbadb"}
REVISED = {"01a": "NO_VERIFIED_CANDIDATE", "01b": "OUT_OF_SCOPE", "01c": "OUT_OF_SCOPE", "02": "NO_VERIFIED_CANDIDATE",
           "03a": "NO_VERIFIED_CANDIDATE", "03b": "NO_VERIFIED_CANDIDATE"}
Ts, b = 0.001, 4.0
SERVO = {"family": "servo_axis", "kind": "second_order", "states": [{"name": "angle", "unit": "rad"}, {"name": "speed", "unit": "rad/s"}],
         "inputs": [{"name": "torque", "unit": "N m"}], "outputs": [{"name": "encoder", "unit": "rad"}],
         "linear": {"clock": "continuous", "A": [[0.0, 1.0], [0.0, 0.0]], "B": [[0.0], [b]], "C": [[1.0, 0.0]], "Ts_native": None},
         "second_order": {"f": "b*u", "f0": "b*u", "params": {"b": b}, "param_ranges": {}, "output": "x1"}, "flags": []}


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
    assert d.get("claim") is None and d.get("impossibility") is None and out["estimates"] is None


# ---------------------------------------------------------------- 01 irregular sampling
def test_01b_keeps_the_revision2_outcome():
    case = _case("01b")
    out = _solve("01b")
    assert case["observed_rev2_result"]["disposition"] == out["decision"]["disposition"] == "OUT_OF_SCOPE"
    assert out["timing"]["code"] == "irregular_sampling" and abs(out["timing"]["max_deviation"] - 1e-3) < 1e-15


def test_01a_declared_irregular_sampling_excludes_both_methods():
    d = _solve("01a")["decision"]
    assert d["explanations"][0].startswith("the request states irregular sampling")
    assert any(e.startswith("high_gain_observer not applicable: irregular sampling") for e in d["explanations"])
    # the declaration is what decides: the same request without it is released for fixed-period data
    spec = copy.deepcopy(_case("01a")["arguments"]["spec"])
    spec["time"]["irregular"] = False
    assert api.decide(spec)["disposition"] == "RELEASE"


def test_01c_sampling_deviation_versus_label_rounding():
    case = _case("01c")
    a = case["arguments"]
    t = np.asarray(a["t"], float)
    chk = TM.check_timestamps(t, 1e-3)
    assert chk["code"] == "irregular_sampling" and abs(chk["max_deviation"] - 5e-10) < 1e-15
    assert TM.label_tolerance(t, 1e-3) == 4 * np.spacing(0.002)                # float64 precision of the largest time
    # the error that the deviation causes with the observer released by revision 2 (gains of results/reviewer_cases_previous.json):
    # the reading is 5e-10 above the fixed-period model, times the velocity gain
    assert abs(929.924647106835 * 1.0 * 5e-10 - 4.6496e-7) < 1e-10
    # the same times, stated to be labels rounded to 2 ns: accepted; the claim carries the statement as a condition
    spec = copy.deepcopy(a["spec"])
    spec["time"]["label_resolution"] = 2e-9
    out = api.solve(spec, a["y"], a["u"], t=t)
    assert out["decision"]["disposition"] == "RELEASE"
    assert any("rounding of the time labels" in c for c in out["decision"]["claim"]["conditions"])
    # a resolution whose half is below the deviation does not cover it
    spec["time"]["label_resolution"] = 5e-10
    assert api.solve(spec, a["y"], a["u"], t=t)["decision"]["disposition"] == "OUT_OF_SCOPE"
    # the structured interface accepts the field and validates it
    s2 = api.new_spec(copy.deepcopy(SERVO), objective={"kind": "hard_bound", "tolerance": 0.09}, target=1, Ts=Ts, label_resolution=1e-6)
    assert s2["time"]["label_resolution"] == 1e-6
    s2["time"]["label_resolution"] = -1.0
    assert api.decide(s2)["disposition"] == "INVALID_INPUT"


# ---------------------------------------------------------------- 02 true parameter stated as a one-point range
def test_02_mismatch_derived_from_the_stated_true_value():
    spec = copy.deepcopy(_case("02")["arguments"]["spec"])
    D, info = H._model_disturbance(spec)
    assert abs(D - 1.0) < 1e-12 and abs(info["model_mismatch"] - 1.0) < 1e-12          # (2 - 1) u with u = 1
    assert abs(EV._sampled_disturbance(spec) - 1.0) < 1e-9                              # the judge's own search
    d = api.decide(spec)
    m = d["assessments"]["high_gain_observer"]["metrics"]
    assert d["disposition"] == "NO_VERIFIED_CANDIDATE" and m["D"] == 1.0 and m["best_float_bound"] > 1e-3
    # independent: exact worst case of the gains released by revision 2 under D = 1 (t >= 0.5 s), equal to the error on the record
    M = np.array([0.9987273661986602, 9.299246471068349])
    w = max(RH.worst_case_exact(0.1, M, np.array([0.0, 1.0]), [0.0, 0.0], 1.0, 0.0, n, sub=8) for n in (5, 10, 40, 80))
    assert abs(w - 0.0573987413) < 1e-9 and abs(w - m["best_float_bound"]) < 1e-9
    # a one-point range equal to the nominal value changes nothing: exact model, zero mismatch
    spec["plant"]["second_order"]["param_ranges"] = {"b": [1.0, 1.0]}
    assert H._model_disturbance(spec)[0] == 0.0 and api.decide(spec)["disposition"] == "RELEASE"


def test_02_kalman_refuses_a_statistical_claim_when_the_true_value_differs_from_the_model():
    obj = {"kind": "mse_optimal", "horizon": "steady_state"}
    G = np.array([[Ts * Ts / 2], [Ts]])
    st = dict(Q=((12.0) * (G @ G.T)).tolist(), Q_kind="discrete_cov", R=[[1e-8 / 3]], P0=[[1.3e-6, 0.0], [0.0, 3.0]],
              distribution="bounded", white=True)
    pl = copy.deepcopy(SERVO); pl["second_order"]["param_ranges"] = {"b": [5.0, 5.0]}
    d = api.decide(api.new_spec(pl, objective=obj, target=1, Ts=Ts, **st))
    assert d["disposition"] != "RELEASE"
    assert any("differ from the nominal model" in e for e in d["explanations"])
    pl["second_order"]["param_ranges"] = {"b": [4.0, 4.0]}
    assert api.decide(api.new_spec(pl, objective=obj, target=1, Ts=Ts, **st))["disposition"] == "RELEASE"


# ---------------------------------------------------------------- 03 lower limits under a stated travel range
def test_03a_first_sample_limit_uses_the_travel_range():
    spec = copy.deepcopy(_case("03a")["arguments"]["spec"])
    # closed form: x2' = 0, x1(0) = 0 and |x1(t)| <= 0.1 up to t = 1 s give |v| <= 0.1
    vmax = spec["deterministic"]["domain"]["x1"][1] / spec["time"]["t_end"]
    lo, hi, _ = LM.viable_initial_velocities(spec, -1.0, 1.0)
    ilo, ihi = RH.first_sample_velocity_limit(spec, -1.0, 1.0)
    assert abs(hi - vmax) < 1e-9 and abs(lo + vmax) < 1e-9 and abs(ihi - vmax) < 1e-9 and abs(ilo + vmax) < 1e-9
    assert vmax < spec["objective"]["tolerance"]                    # the estimate 0 meets 0.2: no impossibility
    assert EV._verify_impossibility(spec, None)[0] is False
    # without the travel range the claim of revision 2 is correct and is still made
    spec["deterministic"]["domain"] = {"x2": [-1, 1]}
    d = api.decide(spec)
    assert d["disposition"] == "PROVED_IMPOSSIBLE" and d["impossibility"]["kind"] == "initial_sample"
    assert abs(d["impossibility"]["bound"] - 1.0) < 1e-9 and EV._verify_impossibility(spec, d)[0] is True


def test_03b_no_claim_when_the_travel_bounds_every_admissible_speed():
    spec = copy.deepcopy(_case("03b")["arguments"]["spec"])
    D, w = spec["deterministic"]["D"], 0.001
    # closed form: braking with |x2'| <= D inside a travel of width w bounds every admissible speed by sqrt(2 D w)
    assert math.sqrt(2 * D * w) < spec["objective"]["peak_cap"]
    lo, hi, _ = LM.viable_initial_velocities(spec, -1.0, 1.0)
    assert abs(hi - math.sqrt(D * w)) < 1e-9 and abs(lo + math.sqrt(D * w)) < 1e-9      # inner set: braking from the centre
    ilo, ihi = RH.first_sample_velocity_limit(spec, -1.0, 1.0)
    assert abs(ihi - hi) < 1e-9 and abs(ilo - lo) < 1e-9
    assert EV._verify_impossibility(spec, None)[0] is False


def test_two_signal_limit_needs_travel_for_the_braking_after_the_sample():
    V, D = 1e-4, 6.0
    room = V + (2 * math.sqrt(D * V)) ** 2 / (2 * D)               # = 3 V: reach +-V, then brake with D
    assert abs(RH.two_signal_travel_room(V, D, 0.01, None) - room) < 1e-15
    base = dict(target=1, Ts=Ts, V=V, D=D, E=[2e-3, 3.0])
    obj = {"kind": "hard_bound", "tolerance": 0.03}
    wide = api.new_spec(copy.deepcopy(SERVO), objective=obj, domain={"x1": [-1e-3, 1e-3]}, **base)
    d = api.decide(wide)
    assert d["disposition"] == "PROVED_IMPOSSIBLE" and EV._verify_impossibility(wide, d)[0] is True
    narrow = api.new_spec(copy.deepcopy(SERVO), objective=obj, domain={"x1": [-0.9 * room, 0.9 * room]}, **base)
    d = api.decide(narrow)
    assert d["disposition"] != "PROVED_IMPOSSIBLE"
    assert EV._verify_impossibility(narrow, d)[0] is not True


# ---------------------------------------------------------------- constructions behind the first-sample limit
def test_braking_construction_rests_with_the_input_it_holds():
    """An input is held until the next sample instant, so a trajectory that stops between samples rests only if its own
    braking input admits zero acceleration there.  Servo axis x2' = b u + d, |d| <= 0.01, |u| <= 1, travel +-1 mm: the
    strong input u = -1 cannot hold the axis at rest (|b u| > 0.01), so only u = 0 brakes a stopping trajectory, with
    deceleration 0.01, which admits speeds up to sqrt(2 * 0.01 * 1e-3)."""
    db, room = 0.01, 1e-3
    spec = api.new_spec(copy.deepcopy(SERVO), objective={"kind": "hard_bound", "tolerance": 0.002, "deadline": 0.0}, target=1, Ts=Ts,
                        V=1e-4, d_bound=db, E=[0.0, 1.0], x0_hat=[0.0, 0.0], domain={"x1": [-room, room], "u": [-1.0, 1.0]},
                        domain_evidence={"kind": "mechanical_limit", "statement": "end stops"})
    exact = math.sqrt(2 * db * room)
    lo, hi, info = LM.viable_initial_velocities(spec, -1.0, 1.0)
    ilo, ihi = RH.first_sample_velocity_limit(spec, -1.0, 1.0)
    for v in (hi, -lo, ihi, -ilo):
        assert abs(v - exact) < 1e-6 * exact
    assert info["u_up"] == 0.0 and abs(b * info["u_up"]) <= db             # the braking input can hold the axis at rest
    assert hi < math.sqrt(2 * (b * 1.0 + db) * room) / 10                   # far below what braking with u = -1 would give
    d = api.decide(spec)
    assert d["disposition"] == "PROVED_IMPOSSIBLE" and abs(d["impossibility"]["bound"] - exact) < 1e-6 * exact
    assert EV._verify_impossibility(spec, d)[0] is True


def test_energy_construction_for_a_spring_stage():
    """Duffing stage x2' = (u - k1 x1 - k3 x1^3 - c x2)/m + d (test-00343): with u = 0 and d = 0 the energy
    x2^2/2 + (k1 x1^2/2 + k3 x1^4/4)/m does not increase (c >= 0), and at the travel ends +-0.01 it exceeds the energy of
    every initial velocity of the prior |x2(0)| <= 0.166, so the whole prior is admissible."""
    row = json.loads(next(l for l in (ROOT / "tests" / "fixtures" / "revision2_cases.jsonl").read_text().splitlines() if '"test-00343"' in l))
    t = row["truth"]
    so = t["plant"]["second_order"]["params"]
    phi_end = (so["k1"] * 0.01 ** 2 / 2 + so["k3"] * 0.01 ** 4 / 4) / so["m"]
    E1 = t["initial"]["E"][1]
    assert E1 ** 2 / 2 < phi_end and E1 <= t["deterministic"]["domain"]["x2"][1]
    lo, hi, info = LM.viable_initial_velocities(t, -E1, E1)
    assert info["route"] == "energy" and abs(hi - E1) < 1e-12 and abs(lo + E1) < 1e-12
    assert RH.first_sample_velocity_limit(t, -E1, E1) == (-E1, E1)
    d = api.decide(t)
    assert d["disposition"] == "PROVED_IMPOSSIBLE" and abs(d["impossibility"]["bound"] - E1) < 1e-12
    assert "energy" in d["impossibility"]["argument"] and EV._verify_impossibility(t, d)[0] is True
