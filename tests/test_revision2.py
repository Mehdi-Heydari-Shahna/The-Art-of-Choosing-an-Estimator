"""Regression tests for revision 2: one or more counterexamples for each of the seven findings (F1-F7) of the request,
with the expected answer derived independently in the test (closed forms, an explicit simulation or the reference
implementations in independent/)."""
import copy, json, math, pathlib, subprocess, sys
import numpy as np
from heros import api, record as R
from heros import evaluation as EV
from heros.core import peak, spec as S, timing as TM
from heros.khalil import specialist as H
from heros.interface import service as SV
from independent import ref_hgo as RH

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _cases(ids):
    """Rows of data/test_structured.jsonl that these tests need, from tests/fixtures/revision2_cases.jsonl, so that the
    tests also run from the light archive (which leaves out the case files until experiments/verify_data.py --install
    re-creates them).  When the case file is present, the fixture must equal its rows."""
    fx = {json.loads(l)["id"]: l for l in (ROOT / "tests" / "fixtures" / "revision2_cases.jsonl").read_text().splitlines() if l}
    full = ROOT / "data" / "test_structured.jsonl"
    if full.exists():
        src = {json.loads(l)["id"]: l for l in full.read_text().splitlines() if l}
        assert all(fx[i] == src[i] for i in ids), "tests/fixtures/revision2_cases.jsonl differs from data/test_structured.jsonl"
    return {i: json.loads(fx[i]) for i in ids}
Ts, b = 0.001, 4.0
SERVO = {"family": "servo_axis", "kind": "second_order", "states": [{"name": "angle", "unit": "rad"}, {"name": "speed", "unit": "rad/s"}],
         "inputs": [{"name": "torque", "unit": "N m"}], "outputs": [{"name": "encoder", "unit": "rad"}],
         "linear": {"clock": "continuous", "A": [[0.0, 1.0], [0.0, 0.0]], "B": [[0.0], [b]], "C": [[1.0, 0.0]], "Ts_native": None},
         "second_order": {"f": "b*u", "f0": "b*u", "params": {"b": b}, "param_ranges": {}, "output": "x1"}, "flags": []}
WALK = {"family": "random_walk", "kind": "linear", "states": [{"name": "level", "unit": "m"}], "inputs": [{"name": "none", "unit": "-"}],
        "outputs": [{"name": "gauge", "unit": "m"}], "linear": {"clock": "discrete", "A": [[1.0]], "B": [[0.0]], "C": [[1.0]], "Ts_native": 1.0},
        "second_order": None, "flags": []}
G = np.array([[Ts * Ts / 2], [Ts]])
HARD = dict(target=1, Ts=Ts, V=1e-4, D=6.0, E=[2e-3, 3.0])


def stat(D, V):
    return dict(Q=((D * D / 3) * (G @ G.T)).tolist(), Q_kind="discrete_cov", R=[[V * V / 3]], P0=[[1.33333e-6, 0.0], [0.0, 3.0]],
                distribution="bounded", white=True)


def servo_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = np.array([0.1, 0.5]); ys, us = [], []
    for k in range(n):
        u = 0.5 * math.sin(2 * math.pi * 2 * k * Ts); ys.append(x[0] + rng.uniform(-1e-4, 1e-4)); us.append(u)
        a = b * u + rng.uniform(-6, 6); x = np.array([x[0] + Ts * x[1] + 0.5 * Ts * Ts * a, x[1] + Ts * a])
    return np.array(ys), np.array(us)


# ---------------------------------------------------------------- F1 period and timestamps
def test_F1_timestamp_check_classifies_the_record():
    t = np.arange(11) * 1e-3
    assert TM.check_timestamps(t, 1e-3)["ok"]
    assert TM.check_timestamps(np.round(t, 4), 1e-3)["ok"]                       # printed to 0.1 ms
    assert TM.check_timestamps(2 * t, 1e-3)["code"] == "timing_conflict"
    assert TM.check_timestamps(np.delete(t, 5), 1e-3)["code"] == "irregular_sampling"
    assert TM.check_timestamps(t[::-1], 1e-3)["code"] == "bad_timestamps"
    assert TM.check_timestamps(t, 1e-3, n_samples=10)["code"] == "bad_timestamps"


def test_F1_release_is_withdrawn_when_timestamps_disagree_and_replay_agrees():
    spec = api.new_spec(copy.deepcopy(SERVO), objective={"kind": "hard_bound", "tolerance": 0.09, "deadline": 0.05}, **HARD)
    y, u = servo_data()
    ok = api.solve(spec, y, u, t=np.arange(len(y)) * Ts)
    assert ok["decision"]["disposition"] == "RELEASE" and ok["estimates"] is not None
    bad = api.solve(spec, y, u, t=np.arange(len(y)) * 2 * Ts)
    assert bad["decision"]["disposition"] == "INVALID_INPUT" and bad["estimates"] is None
    assert bad["decision"]["issues"][0]["code"] == "timing_conflict" and "withdrawn_release" in bad["decision"]
    gap = api.solve(spec, y[:-1], u[:-1], t=np.delete(np.arange(len(y)) * Ts, 100))
    assert gap["decision"]["disposition"] == "OUT_OF_SCOPE" and gap["estimates"] is None
    for res in (ok, bad, gap):
        rp = api.replay(json.loads(R.dumps(res["record"])))
        assert rp["match"], rp["differences"]


def test_F1_command_line_reads_the_time_column(tmp_path):
    spec = json.loads((ROOT / "examples" / "servo_axis_spec.json").read_text())
    for k in ("Q", "Q_kind", "R", "P0", "distribution", "white"):
        spec.pop(k)
    spec["objective"] = {"kind": "hard_bound", "tolerance": spec.pop("_hard_tolerance"), "deadline": spec.pop("_deadline")}
    (tmp_path / "s.json").write_text(json.dumps(spec))
    rows = (ROOT / "examples" / "servo_axis_data.csv").read_text().splitlines()
    (tmp_path / "d.csv").write_text("\n".join([rows[0]] + rows[1::2]))         # every second sample: 2 ms spacing
    p = subprocess.run([sys.executable, "-m", "heros", "solve", "--spec", str(tmp_path / "s.json"), "--data", str(tmp_path / "d.csv")],
                       cwd=ROOT, capture_output=True, text=True)
    assert "Disposition: INVALID_INPUT" in p.stdout and "Estimates:" not in p.stdout
    y, u, t = SV.read_data_t((ROOT / "examples" / "servo_axis_data.csv").read_text())
    assert t is not None and len(t) == len(y) and SV.read_data((ROOT / "examples" / "servo_axis_data.csv").read_text())[0].shape == y.shape


# ---------------------------------------------------------------- F2 samples inside the window
def test_F2_window_indices():
    assert TM.sample_window(1.0, None, 3.5) == (0, 3)
    assert TM.sample_window(0.001, 0.05, 0.06) == (50, 60)
    assert TM.sample_window(0.001, 0.0505, None) == (51, None)
    issues = S.validate(api.new_spec(copy.deepcopy(SERVO), objective={"kind": "hard_bound", "tolerance": 1.0, "deadline": 0.0155},
                                     t_end=0.0158, **HARD))
    assert any(i["code"] == "timing_conflict" for i in issues)


def test_F2_kalman_level_judged_on_the_samples_of_the_record():
    pf, P = [], 0.1
    for k in range(6):
        Pf = P / (P + 1); pf.append(Pf); P = Pf + 1
    target = 0.5 * (pf[3] + pf[4])
    spec = api.new_spec(copy.deepcopy(WALK), objective={"kind": "mse_bound", "horizon": "finite", "mse_target": target}, target=0, Ts=1.0,
                        Q=[[1.0]], Q_kind="discrete_cov", R=[[1.0]], P0=[[0.1]], x0_hat=[0.0], distribution="gaussian", white=True, t_end=3.5)
    d = api.decide(spec)
    assert d["disposition"] == "RELEASE" and abs(d["claim"]["value"] - pf[3]) < 1e-9


def test_F2_two_signal_limit_needs_a_window_sample_after_its_time():
    spec = api.new_spec(copy.deepcopy(SERVO), objective={"kind": "hard_bound", "tolerance": 0.04, "deadline": 0.01},
                        target=1, Ts=0.01, V=1e-4, D=6.0, E=[2e-3, 3.0], t_end=0.0165)
    assert api.decide(spec)["disposition"] == "NO_VERIFIED_CANDIDATE"
    spec2 = api.new_spec(copy.deepcopy(SERVO), objective={"kind": "hard_bound", "tolerance": 0.04, "deadline": 0.01},
                         target=1, Ts=0.01, V=1e-4, D=6.0, E=[2e-3, 3.0], t_end=0.03)
    d2 = api.decide(spec2)
    assert d2["disposition"] == "PROVED_IMPOSSIBLE" and d2["impossibility"]["sample_time"] == 0.02


def test_F2_certificate_on_a_finite_window_uses_its_samples_only():
    M = H.gain_sampled(Ts, 0.004); c = np.array([0.0, 1.0])
    W = peak.worst_case_lti(Ts, M, c, [1e-4, 3.0], 6.0, 1e-4, 200, init="first_measurement")["W"]
    cert = peak.certify_lti(Ts, M, c, [1e-4, 3.0], 6.0, 1e-4, k_deadline=20, init="first_measurement", k_end=40)
    assert abs(cert["post_deadline_upper"] - W[20:41].max()) <= 1e-12 * W[20:41].max()
    assert abs(cert["peak_upper"] - W[:41].max()) <= 1e-12 * W[:41].max()


# ---------------------------------------------------------------- F3 Kalman start-up
def test_F3_exported_covariance_of_the_constant_gain_starts_from_P0():
    spec = api.new_spec(copy.deepcopy(WALK), objective={"kind": "mse_optimal", "horizon": "steady_state"}, target=0, Ts=1.0, Q=[[1.0]],
                        Q_kind="discrete_cov", R=[[1.0]], P0=[[100.0]], x0_hat=[0.0], distribution="gaussian", white=True)
    est = api.solve(spec, np.zeros(30))["estimates"]
    M = (math.sqrt(5) - 1) / 2
    P = np.asarray(est["P_filt"]).reshape(-1)
    assert abs(P[0] - ((1 - M) ** 2 * 100 + M * M)) < 1e-12 and abs(P[0] - 14.9718) < 1e-4
    assert abs(P[-1] - M) < 1e-9 and "including start-up" in est["covariance_basis"]
    Pp = 100.0                                             # independent: constant-gain covariance recursion
    for k in range(30):
        Pf = (1 - M) ** 2 * Pp + M * M
        assert abs(Pf - P[k]) < 1e-10 * max(1, Pf)
        Pp = Pf + 1.0


def test_F3_kalman_hard_bound_release_exports_its_envelope_and_first_reading_start():
    spec = api.new_spec(copy.deepcopy(SERVO), objective={"kind": "hard_bound", "tolerance": 0.2, "deadline": 0.05}, **HARD, **stat(6.0, 1e-4))
    d = api.decide(spec, methods=["kalman_filter"])
    assert d["disposition"] == "RELEASE"
    y, u = servo_data()
    est = R.execute(d, y, u)
    assert np.allclose(np.asarray(est["gain"])[0].ravel(), [1.0, 0.0]) and abs(est["x_filt"][0][0] - y[0]) < 1e-15
    Pf0 = np.asarray(est["P_filt"])[0]
    assert abs(Pf0[0, 0] - 1e-8 / 3) < 1e-20 and abs(Pf0[1, 1] - 3.0) < 1e-12 and abs(Pf0[0, 1]) < 1e-20
    assert "envelope" in est and abs(est["envelope"][0] - 3.0) < 1e-12 and est["uncertainty_type"].startswith("deterministic")


# ---------------------------------------------------------------- F4 all constraints together
def test_F4_tighter_additive_bound_is_used_with_the_total_bound():
    spec = api.new_spec(copy.deepcopy(SERVO), objective={"kind": "hard_bound", "tolerance": 0.045}, target=1, Ts=Ts, V=1e-4, D=6.0, d_bound=1.5,
                        E=[2e-3, 3.0])
    d = api.decide(spec)
    assert d["disposition"] == "RELEASE" and d["config"]["D"] == 1.5
    # independent: the exact ultimate worst case of the released gain with the additive bound
    w = RH.worst_case_exact(Ts, np.array(d["config"]["M"]), np.array([0.0, 1.0]), [0.0, 0.0], 1.5, 1e-4, 1500, sub=4)
    assert w <= d["claim"]["value"] * (1 + 1e-9) <= 0.045
    jr = EV.judge_release(spec, d)
    assert jr["correct"], jr["why"]


def test_F4_no_statistical_claim_for_uncertain_parameters():
    pl = copy.deepcopy(SERVO); pl["second_order"]["param_ranges"] = {"b": [3.6, 4.4]}
    d = api.decide(api.new_spec(pl, objective={"kind": "mse_optimal", "horizon": "steady_state"}, target=1, Ts=Ts, **stat(6.0, 1e-4)))
    assert d["disposition"] != "RELEASE"
    pl2 = copy.deepcopy(SERVO)
    d2 = api.decide(api.new_spec(pl2, objective={"kind": "mse_optimal", "horizon": "steady_state"}, target=1, Ts=Ts, **stat(6.0, 1e-4)))
    assert d2["disposition"] == "RELEASE"
    assert not EV.judge_release(api.new_spec(pl, objective={"kind": "mse_optimal", "horizon": "steady_state"}, target=1, Ts=Ts, **stat(6.0, 1e-4)), d2)["correct"]


def test_F4_two_signal_limit_respects_speed_and_input_limits():
    V, D = 1e-4, 6.0
    low = 2 * math.sqrt(D * V)
    base = dict(target=1, Ts=Ts, V=V, D=D, E=[2e-3, 3.0])
    obj = {"kind": "hard_bound", "tolerance": 0.03}
    assert api.decide(api.new_spec(copy.deepcopy(SERVO), objective=obj, **base))["disposition"] == "PROVED_IMPOSSIBLE"
    # speed range that excludes the velocities of the construction
    d = api.decide(api.new_spec(copy.deepcopy(SERVO), objective=obj, domain={"x2": [-0.01, 3.0]}, **base))
    assert d["disposition"] == "NO_VERIFIED_CANDIDATE"
    # speed range that admits them
    d = api.decide(api.new_spec(copy.deepcopy(SERVO), objective=obj, domain={"x2": [-2 * low, 3.0]}, **base))
    assert d["disposition"] == "PROVED_IMPOSSIBLE"
    # an input range that excludes every rest input of the nominal model
    d = api.decide(api.new_spec(copy.deepcopy(SERVO), objective=obj, domain={"u": [0.2, 1.0]}, **base))
    assert d["disposition"] == "NO_VERIFIED_CANDIDATE"


# ---------------------------------------------------------------- F5 valid impossibility arguments only
def test_F5_probability_request_is_never_refused_by_a_worst_case_limit():
    spec = api.new_spec(copy.deepcopy(SERVO), objective={"kind": "probability", "horizon": "steady_state", "tolerance": 0.02, "prob": 0.9},
                        target=1, Ts=Ts, V=1e-4, D=6.0, E=[2e-3, 3.0], **stat(6.0, 1e-4))
    d = api.decide(spec)
    assert d["disposition"] == "NO_VERIFIED_CANDIDATE" and d["display"] == "No verified design found."
    d_h = api.decide(spec, methods=["high_gain_observer"])
    assert d_h["disposition"] != "PROVED_IMPOSSIBLE"
    ok, why = EV._verify_impossibility(spec, dict(d, disposition="PROVED_IMPOSSIBLE"))
    assert ok is not True


def test_F5_benchmark_probability_cases_are_not_refused_by_the_observer_baseline():
    rows = _cases(["test-00121", "test-00200", "test-00290", "test-00455"])
    for cid in ("test-00121", "test-00200", "test-00290", "test-00455"):
        c = rows[cid]
        d = api.decide(c["given"], methods=["high_gain_observer"])
        assert d["disposition"] != "PROVED_IMPOSSIBLE"
        ok, why = EV._verify_impossibility(c["truth"], dict(d, disposition="PROVED_IMPOSSIBLE"))
        assert ok is False, why                    # refuted: the Kalman design for the truth meets the request and is verified


# ---------------------------------------------------------------- F6 first-reading initialisation
def test_F6_first_reading_worst_case_is_exact_and_below_the_box_value():
    c = np.array([0.0, 1.0]); V, D, E = 1e-4, 6.0, [2e-3, 3.0]
    for eps in (0.0006, 0.003):
        M = H.gain_sampled(Ts, eps)
        Wf = peak.worst_case_lti(Ts, M, c, [V, E[1]], D, V, 80, init="first_measurement")["W"]
        Wb = peak.worst_case_lti(Ts, M, c, [V, E[1]], D, V, 80)["W"]
        Wl = peak.worst_case_ltv(Ts, [M] * 81, c, E, D, V, init="first_measurement")
        assert np.all(Wf <= Wb * (1 + 1e-15)) and abs(Wf[0] - E[1]) < 1e-15 and abs(Wb[0] - (E[1] + 2 * M[1] * V)) < 1e-12
        assert np.max(np.abs(Wf - Wl) / Wl) < 1e-12
        for n in (0, 1, 5, 40):
            ref = RH.worst_case_exact(Ts, M, c, E, D, V, n, sub=4, init="first_measurement")
            assert abs(ref - Wf[n]) <= 1e-12 * ref
        # attained: sign-aligned admissible signals (independent simulation with the first-reading start)
        n = 25
        w = peak.worst_inputs_lti(Ts, M, c, [V, E[1]], D, V, n, init="first_measurement")
        assert abs(w["v"][0]) == V

        def delta(t, dist=w["dist"]):
            t = np.asarray(t, float); k = np.minimum((t // Ts).astype(int), n - 1); s = t - k * Ts
            out = np.empty_like(t)
            for i in range(t.size):
                b_, a_, sw = dist[k.flat[i]]
                out.flat[i] = D * (b_ if s.flat[i] < sw else a_)
            return out
        sim = RH.simulate_observer(Ts, M, (0.0, w["e0"][1]), (0.0, 0.0), w["v"], delta, None, n, substeps=400, init="first_measurement")
        err = sim["x"][n, 1] - sim["xp"][n, 1]
        assert abs(abs(err) - Wf[n]) <= 2e-3 * Wf[n]


def test_F6_released_envelope_starts_at_the_initial_velocity_bound():
    spec = api.new_spec(copy.deepcopy(SERVO), objective={"kind": "hard_bound", "tolerance": 0.09, "deadline": 0.05}, **HARD)
    y, u = servo_data()
    res = api.solve(spec, y, u)
    assert abs(res["estimates"]["envelope"][0] - 3.0) < 1e-12 and "first-reading" in res["estimates"]["envelope_kind"]


# ---------------------------------------------------------------- F7 cap below the later tolerance
def test_F7_cap_below_tolerance_is_valid_input_and_decided_by_the_first_sample():
    obj = {"kind": "hard_bound", "tolerance": 0.09, "deadline": 0.05, "peak_cap": 0.045}
    spec = api.new_spec(copy.deepcopy(SERVO), objective=obj, **HARD)
    assert not S.validate(spec)
    d = api.decide(spec)
    assert d["disposition"] == "PROVED_IMPOSSIBLE" and d["impossibility"]["kind"] == "initial_sample" and abs(d["impossibility"]["bound"] - 3.0) < 1e-12
    assert EV._verify_impossibility(spec, d)[0] is True
    # a cap above the initial velocity uncertainty is a transient requirement like any other
    spec2 = api.new_spec(copy.deepcopy(SERVO), objective=dict(obj, peak_cap=3.5), **HARD)
    d2 = api.decide(spec2)
    assert d2["disposition"] == "RELEASE" and d2["claim"]["peak_value"] <= 3.5 if "peak_value" in (d2["claim"] or {}) else d2["disposition"] == "RELEASE"


def test_F7_relabel_file_matches_the_first_sample_rule():
    rel = json.loads((ROOT / "data" / "revision2_relabel.json").read_text())["cases"]
    rows = _cases(sorted(rel))
    assert len(rel) == 20
    for cid, r in rel.items():
        c = EV.with_revisions(rows[cid])
        assert c["final"]["disposition"] == r["new_final"] == "PROVED_IMPOSSIBLE"
        assert api.decide(c["given"])["disposition"] == "PROVED_IMPOSSIBLE"


def test_display_sentences():
    spec = api.new_spec(copy.deepcopy(SERVO), objective={"kind": "hard_bound", "tolerance": 0.055}, target=1, Ts=Ts, V=1e-4, D=6.0, E=[2e-3, 3.0])
    d = api.decide(spec)
    assert d["disposition"] == "NO_VERIFIED_CANDIDATE" and d["display"] == "No verified design found."
    assert SV.summary(d).splitlines()[0] == "No verified design found."
