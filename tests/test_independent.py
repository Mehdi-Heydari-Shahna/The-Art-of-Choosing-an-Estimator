"""Cross-checks of both specialists against the independently written references (independent/).
Tolerances were fixed from precision and conditioning before any comparative result was examined:
  Kalman estimates / covariances vs batch conditioning: 1e-8 relative (float64 batch conditioning loses digits by
  cancellation; the mpmath variant is used at 1e-10 for one case); discretisation: 1e-11; worst case: 1e-9."""
import math
import numpy as np
import pytest
from heros import api
from heros.core import peak
from heros.kalman import specialist as K
from heros.khalil import specialist as H
from heros.sim import plants as PL, loop
from independent import ref_kalman as RK, ref_hgo as RH


def _boom_spec(horizon="finite", N=40):
    fam = PL.hydraulic_boom(nominal=True); sc = fam["scales"]
    return api.new_spec(fam["plant"], target=1, Ts=sc["Ts"], Q=sc["Qc"], Q_kind="continuous_psd", G=sc["G"],
                        R=np.diag(np.square(sc["r_std"])).tolist(), P0=np.diag(np.square(sc["P0_std"])).tolist(),
                        distribution="gaussian", white=True, t_end=N * sc["Ts"], objective={"kind": "mse_optimal", "horizon": horizon})


def test_discretisation_matches_quadrature_reference():
    sp = _boom_spec(); cfg = K.design(sp)
    lin = sp["plant"]["linear"]
    Ad, Bd = RK.zoh_quadrature(np.array(lin["A"]), np.array(lin["B"]), cfg["Ts"])
    Qd = RK.process_cov_quadrature(np.array(lin["A"]), np.array(sp["stochastic"]["G"]), np.array(sp["stochastic"]["Q"]), cfg["Ts"])
    assert np.allclose(cfg["A"], Ad, rtol=1e-11, atol=1e-13 * abs(Ad).max())
    assert np.allclose(cfg["B"], Bd, rtol=1e-11, atol=1e-13 * abs(Bd).max())
    assert np.allclose(cfg["Qd"], Qd, rtol=1e-10, atol=1e-12 * abs(Qd).max())


@pytest.mark.parametrize("family", ["servo_axis", "thermal", "dc_motor"])
def test_public_run_matches_batch_conditioning(family):
    rng = np.random.default_rng(11)
    from heros import datagen as DG
    b = DG.base_case(family, rng)
    sp = DG._spec(b, {"kind": "mse_optimal", "horizon": "finite"}, {"target": 1, "t_end": 30 * b["fields"]["Ts"]})
    cfg = K.design(sp); N = 30
    x0 = rng.multivariate_normal(np.zeros(cfg["A"].shape[0]), cfg["P0"])
    u = rng.normal(size=(N + 1, cfg["B"].shape[1]))
    X, Y = loop.simulate_linear(cfg["A"], cfg["B"], cfg["C"], cfg["Qd"], cfg["R"], x0, u, N, rng)
    res = api.solve(sp, Y, u)
    assert res["decision"]["disposition"] == "RELEASE" and res["decision"]["selected"] == K.METHOD
    mean, cov = RK.batch_filtered(cfg["A"], cfg["B"], cfg["C"], cfg["Qd"], cfg["R"], cfg["x0_hat"], cfg["P0"], Y, u, dps=40)
    est = res["estimates"]
    scale = np.sqrt(np.diag(cfg["P0"])) + 1e-300
    assert np.max(np.abs(est["x_filt"] - mean) / scale) < 1e-9
    assert np.max(np.abs(est["P_filt"] - cov) / np.outer(scale, scale)) < 1e-9


def test_steady_state_matches_riccati_iteration_and_lyapunov():
    sp = _boom_spec("steady_state"); cfg = K.design(sp)
    P, Pf, M = RK.riccati_fixed_point(cfg["A"], cfg["C"], cfg["Qd"], cfg["R"])
    assert np.allclose(cfg["P_ss"], P, rtol=1e-8, atol=1e-12 * abs(P).max())
    assert np.allclose(cfg["M_ss"], M, rtol=1e-7, atol=1e-10 * abs(M).max())
    Pl = RK.lyapunov_mse_of_gain(cfg["A"], cfg["C"], cfg["Qd"], cfg["R"], cfg["M_ss"])
    assert np.allclose(K.steady_cov_of_constant_gain(cfg["A"], cfg["C"], cfg["Qd"], cfg["R"], cfg["M_ss"]), Pl, rtol=1e-8, atol=1e-12 * abs(Pl).max())


def test_tutorial_variance_ambiguity():
    # y = x + v: (prior, noise) variances (0.5, 1.5) and (1, 1) give the same measurement variance but means y/4 and y/2
    for (p, r, g) in ((0.5, 1.5, 0.25), (1.0, 1.0, 0.5)):
        plant = {"family": "scalar", "kind": "linear", "states": [{"name": "x", "unit": "-"}], "inputs": [], "outputs": [{"name": "y", "unit": "-"}],
                 "linear": {"clock": "discrete", "A": [[1.0]], "B": [[0.0]], "C": [[1.0]], "Ts_native": 1.0}, "second_order": None, "flags": []}
        sp = api.new_spec(plant, target=0, Q=[[0.0]], Q_kind="discrete_cov", R=[[r]], P0=[[p]], distribution="gaussian", objective={"kind": "mse_optimal", "horizon": "finite"})
        est = api.solve(sp, np.array([[2.0]]))["estimates"]
        assert est["x_filt"][0, 0] == pytest.approx(2.0 * g, rel=1e-14)
        assert RK.batch_gain_first(np.eye(1), np.eye(1), np.zeros((1, 1)), np.array([[r]]), np.array([[p]]))[0, 0] == pytest.approx(g, rel=1e-14)


@pytest.mark.parametrize("form,eps_mult,tgt", [("current", 8.0, 1), ("current", 1.5, 1), ("predictor", 6.0, 1), ("current", 5.0, 0)])
def test_exact_worst_case_matches_independent_enumeration(form, eps_mult, tgt):
    T = 0.002; E = [2e-3, 1.5]; D = 4.0; V = 3e-4; N = 60
    M = H.gain_sampled(T, eps_mult * T) if form == "current" else H.gain_euler_predictor(T, eps_mult * T)
    c = np.eye(2)[tgt]
    W = peak.worst_case_lti(T, M, c, E, D, V, N, form)["W"]
    for n in (0, 1, 7, 60):
        ref = RH.worst_case_exact(T, M, c, E, D, V, n, form=form)
        assert W[n] == pytest.approx(ref, rel=1e-9, abs=1e-15)
    lo, hi = RH.worst_case_by_enumeration_bounds(T, M, c, E, D, V, 60, form=form)[:2]
    assert lo * (1 - 1e-9) <= W[60] <= hi * (1 + 1e-9)


def test_worst_case_attained_in_original_coordinates_and_certificate_dominates():
    T = 0.001; E = [1e-3, 2.0]; D = 5.0; V = 1e-4; N = 120; c = np.array([0.0, 1.0])
    M = H.gain_sampled(T, 0.008)
    x0err, v, dfn = RH.worst_case_signals(T, M, c, E, D, V, N, sub=64)
    sim = RH.simulate_observer(T, M, np.array([0.3, -0.2]), np.array([0.3, -0.2]) - np.asarray(x0err), v, dfn, np.zeros(N + 1), N, substeps=64)
    err = abs(sim["x"][N, 1] - sim["xp"][N, 1]) if "x" in sim else None
    W = peak.worst_case_lti(T, M, c, E, D, V, N)["W"][N]
    assert err == pytest.approx(W, rel=1e-5)          # piecewise-constant disturbance on 64 sub-intervals: lower bound, O(1/sub^2)
    assert err <= W * (1 + 1e-12)
    cert = peak.certify_lti(T, M, c, E, D, V, k_deadline=N)
    assert cert["stable"] and W * (1 - 1e-12) <= cert["W_at_deadline_upper"] <= W * (1 + 1e-9)   # float evaluation carries round-off; the enclosure is exact
    assert cert["peak_upper"] >= peak.worst_case_lti(T, M, c, E, D, V, 4000)["W"].max() * (1 - 1e-12)


def test_continuous_envelope_closed_form_matches_quadrature_and_sampled_limit():
    g, E, D, V = 80.0, [1e-3, 1.0], 3.0, 2e-4
    for tgt in (0, 1):
        t, Wref = RH.ct_observer_worst_case(g, E, D, V, 0.5, target=tgt, n=20001)
        W = peak.ct_envelope(g, E, D, V, t, target=tgt)
        assert np.max(np.abs(W - Wref) / (np.abs(Wref) + 1e-12)) < 1e-6
    # the sampled exact envelope tends to the continuous one as T -> 0 (same eps, matched initial boxes)
    T = 2e-6
    Wd = peak.worst_case_lti(T, H.gain_sampled(T, 1 / g), np.array([0.0, 1.0]), E, D, V, 50000)["W"]
    Wc = peak.ct_envelope(g, E, D, V, np.arange(50001) * T, target=1)
    assert np.max(np.abs(Wd[5000:] - Wc[5000:]) / Wc[5000:]) < 2e-2


def test_lower_limit_two_signals():
    D, V = 5.0, 1e-4
    info = RH.lower_limit_two_signals(D, V)
    assert peak.lower_limit_velocity(D, V) == pytest.approx(2 * math.sqrt(D * V))
    # the best certified sampled HGO bound must not be below the limit
    eps, val, _ = H.search_eps(0.001, np.array([0.0, 1.0]), [V, 0.0], D, V, 0, None, None, ultimate_only=True)
    assert val >= 2 * math.sqrt(D * V)


def test_mse_of_hgo_gain_matches_monte_carlo():
    T = 0.001; G = np.array([[T * T / 2], [T]]); Qd = 9.0 * G @ G.T; R = np.array([[1e-8]])
    A = peak.Ad_of(T); C = np.array([[1.0, 0.0]]); M = H.gain_sampled(T, 0.004).reshape(2, 1)
    Pf = K.steady_cov_of_constant_gain(A, C, Qd, R, M)
    mc = RK.fixed_gain_mse_montecarlo(A, np.zeros((2, 1)), C, Qd, R, M, np.zeros(2), Pf, 400, 20000, 5, dist="uniform")
    assert np.allclose(mc[-50:].mean(axis=0), np.diag(Pf), rtol=0.03)
