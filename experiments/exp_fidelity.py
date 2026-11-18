"""Specialist fidelity: numerical agreement of the public paths with the independently written references."""
import numpy as np
from common import save
from heros import api, datagen as DG
from heros.core import peak
from heros.kalman import specialist as K
from heros.khalil import specialist as H
from heros.sim import plants as PL, loop
from independent import ref_kalman as RK, ref_hgo as RH

rng = np.random.default_rng(42)
out = {"tolerances": {"kalman_estimates_rel_to_prior_std": 1e-9, "discretisation_rel": 1e-10, "worst_case_rel": 1e-9}, "kalman": [], "hgo": []}
for fam in ("servo_axis", "gantry_axis", "mass_spring_damper", "dc_motor", "thermal", "two_mass", "hydraulic_boom"):
    for rep in range(3):
        b = DG.base_case(fam, rng)
        sp = DG._spec(b, {"kind": "mse_optimal", "horizon": "finite"}, {"target": 1, "t_end": 30 * b["fields"]["Ts"]}, stoch=True,
                      drop=[k for k in ("V", "D", "d_bound", "domain", "domain_evidence", "E") if k in b["fields"]])
        cfg = K.design(sp); N = 30
        x0 = rng.multivariate_normal(np.zeros(cfg["A"].shape[0]), cfg["P0"])
        u = rng.normal(size=(N + 1, cfg["B"].shape[1]))
        X, Y = loop.simulate_linear(cfg["A"], cfg["B"], cfg["C"], cfg["Qd"], cfg["R"], x0, u, N, rng)
        res = api.solve(sp, Y, u)
        mean, cov = RK.batch_filtered(cfg["A"], cfg["B"], cfg["C"], cfg["Qd"], cfg["R"], cfg["x0_hat"], cfg["P0"], Y, u, dps=40)
        sc = np.sqrt(np.diag(cfg["P0"])) + 1e-300
        lin = sp["plant"]["linear"]
        Ad, Bd = RK.zoh_quadrature(np.array(lin["A"]), np.array(lin["B"]), cfg["Ts"])
        row = {"family": fam, "n_states": int(cfg["A"].shape[0]), "max_estimate_diff_rel_prior_std": float(np.max(np.abs(res["estimates"]["x_filt"] - mean) / sc)),
               "max_cov_diff_rel": float(np.max(np.abs(res["estimates"]["P_filt"] - cov) / np.outer(sc, sc))),
               "max_Ad_diff_rel": float(np.max(np.abs(cfg["A"] - Ad)) / np.max(np.abs(Ad))), "max_Bd_diff_rel": float(np.max(np.abs(cfg["B"] - Bd)) / np.max(np.abs(Bd)))}
        if sp["stochastic"]["Q_kind"] == "continuous_psd":
            Qd = RK.process_cov_quadrature(np.array(lin["A"]), np.array(sp["stochastic"]["G"]), np.array(sp["stochastic"]["Q"]), cfg["Ts"])
            row["max_Qd_diff_rel"] = float(np.max(np.abs(cfg["Qd"] - Qd)) / np.max(np.abs(Qd)))
        sps = DG._spec(b, {"kind": "mse_optimal", "horizon": "steady_state"}, {"target": 1}, stoch=True,
                       drop=[k for k in ("V", "D", "d_bound", "domain", "domain_evidence", "E") if k in b["fields"]])
        try:
            cs = K.design(sps)
            P, Pf, M = RK.riccati_fixed_point(cs["A"], cs["C"], cs["Qd"], cs["R"])
            row["steady_gain_diff_rel"] = float(np.max(np.abs(cs["M_ss"] - M)) / np.max(np.abs(M)))
            row["steady_cov_diff_rel"] = float(np.max(np.abs(cs["Pf_ss"] - Pf)) / np.max(np.abs(Pf)))
        except Exception as ex:  # noqa
            row["steady"] = f"not evaluated: {type(ex).__name__}"
        out["kalman"].append(row)
for trial in range(20):
    T = float(rng.choice([2.5e-4, 5e-4, 1e-3, 2e-3, 5e-3])); eps = T * float(np.exp(rng.uniform(np.log(0.6), np.log(40))))
    E = [float(np.exp(rng.uniform(-9, -4))), float(np.exp(rng.uniform(-2, 1.5)))]; D = float(np.exp(rng.uniform(-1, 3))); V = float(np.exp(rng.uniform(-11, -7)))
    form = "current" if trial % 4 else "predictor"
    M = H.gain_sampled(T, eps) if form == "current" else H.gain_euler_predictor(T, max(eps, 0.7 * T))
    tgt = 1 if trial % 5 else 0
    c = np.eye(2)[tgt]; N = int(rng.integers(5, 150))
    W = peak.worst_case_lti(T, M, c, E, D, V, N, form)["W"][N]
    ref = RH.worst_case_exact(T, M, c, E, D, V, N, form=form, sub=4)
    cert = peak.certify_lti(T, M, c, E, D, V, k_deadline=N, form=form)
    out["hgo"].append({"Ts": T, "eps": eps, "form": form, "target": tgt, "N": N, "W_float": float(W), "W_independent": float(ref), "rel_diff": float(abs(W - ref) / ref),
                       "certified_at_N": cert.get("W_at_deadline_upper"), "certificate_minus_float_rel": float((cert["W_at_deadline_upper"] - W) / W) if cert["stable"] else None,
                       "stable": cert["stable"]})
# first-reading initialisation (revision 2): closed form against the independent simulation-based reference, current form
out["hgo_first_reading"] = []
for trial in range(20):
    T = float(rng.choice([2.5e-4, 5e-4, 1e-3, 2e-3, 5e-3])); eps = T * float(np.exp(rng.uniform(np.log(0.6), np.log(40))))
    E = [float(np.exp(rng.uniform(-9, -4))), float(np.exp(rng.uniform(-2, 1.5)))]; D = float(np.exp(rng.uniform(-1, 3))); V = float(np.exp(rng.uniform(-11, -7)))
    M = H.gain_sampled(T, eps); tgt = 1 if trial % 5 else 0
    c = np.eye(2)[tgt]; N = int(rng.integers(0, 150))
    Wf = peak.worst_case_lti(T, M, c, [V, E[1]], D, V, N, init="first_measurement")["W"]
    ref = RH.worst_case_exact(T, M, c, E, D, V, N, sub=4, init="first_measurement")
    Wb = peak.worst_case_lti(T, M, c, [V, E[1]], D, V, N)["W"][N]
    cert = peak.certify_lti(T, M, c, [V, E[1]], D, V, k_deadline=N, init="first_measurement")
    out["hgo_first_reading"].append({"Ts": T, "eps": eps, "target": tgt, "N": N, "W_float": float(Wf[N]), "W_independent": float(ref),
                                     "rel_diff": float(abs(Wf[N] - ref) / ref), "independent_box_value": float(Wb),
                                     "box_excess_rel": float((Wb - Wf[N]) / Wf[N]),
                                     "certificate_minus_float_rel": float((cert["W_at_deadline_upper"] - Wf[N]) / Wf[N]) if cert["stable"] else None})
# tightness: admissible signals that attain the bound, simulated in original coordinates by the independent simulator
T, E, D, V, N = 0.001, [1e-3, 2.0], 5.0, 1e-4, 150
M = H.gain_sampled(T, 0.006); c = np.array([0.0, 1.0])
x0err, v, dfn = RH.worst_case_signals(T, M, c, E, D, V, N, sub=64)
sim = RH.simulate_observer(T, M, np.array([0.2, 0.1]), np.array([0.2, 0.1]) - np.asarray(x0err), v, dfn, np.zeros(N + 1), N, substeps=64)
W = peak.worst_case_lti(T, M, c, E, D, V, N)["W"][N]
out["tightness"] = {"attained_error": float(abs(sim["x"][N, 1] - sim["xp"][N, 1])), "exact_worst_case": float(W),
                    "ratio": float(abs(sim["x"][N, 1] - sim["xp"][N, 1]) / W), "note": "piecewise-constant disturbance on 64 sub-intervals per sample (lower bound, O(1/64^2))"}
# continuous-time limit
g = 100.0
Wd = peak.worst_case_lti(2e-6, H.gain_sampled(2e-6, 1 / g), c, E, D, V, 60000)["W"]
Wc = peak.ct_envelope(g, E, D, V, np.arange(60001) * 2e-6, target=1)
t, Wq = RH.ct_observer_worst_case(g, E, D, V, 0.12, target=1, n=20001)
out["continuous_limit"] = {"max_rel_diff_sampled_vs_closed_form_after_10ms": float(np.max(np.abs(Wd[5000:] - Wc[5000:]) / Wc[5000:])), "Ts": 2e-6,
                           "max_rel_diff_closed_form_vs_quadrature": float(np.max(np.abs(peak.ct_envelope(g, E, D, V, t, 1) - Wq) / Wq))}
# tightness of the first-reading formula: sign-aligned signals simulated with the observer started from the first reading
Mt = H.gain_sampled(T, 0.006)
wi = peak.worst_inputs_lti(T, Mt, c, [V, E[1]], D, V, N, init="first_measurement")
sw = wi["dist"]


def dfm(tq, sw=sw, T=T, D=D, N=N):
    tq = np.asarray(tq, float); k = np.minimum((tq // T).astype(int), N - 1); sq = tq - k * T
    return np.array([D * (sw[kk][0] if ss < sw[kk][2] else sw[kk][1]) for kk, ss in zip(k.ravel(), sq.ravel())]).reshape(tq.shape)


simf = RH.simulate_observer(T, Mt, np.array([0.2, 0.1 + wi["e0"][1]]), np.array([0.0, 0.1]), wi["v"], dfm, np.zeros(N + 1), N, substeps=400, init="first_measurement")
Wt = peak.worst_case_lti(T, Mt, c, [V, E[1]], D, V, N, init="first_measurement")["W"][N]
out["tightness_first_reading"] = {"attained_error": float(abs(simf["x"][N, 1] - simf["xp"][N, 1])), "exact_worst_case": float(Wt),
                                  "ratio": float(abs(simf["x"][N, 1] - simf["xp"][N, 1]) / Wt),
                                  "note": "switching disturbance of worst_inputs_lti resolved on 400 sub-steps per sample"}
out["summary"] = {"kalman_max_estimate_diff": max(r["max_estimate_diff_rel_prior_std"] for r in out["kalman"]),
                  "kalman_max_cov_diff": max(r["max_cov_diff_rel"] for r in out["kalman"]),
                  "kalman_max_discretisation_diff": max(max(r["max_Ad_diff_rel"], r["max_Bd_diff_rel"], r.get("max_Qd_diff_rel", 0)) for r in out["kalman"]),
                  "kalman_max_steady_gain_diff": max(r.get("steady_gain_diff_rel", 0) for r in out["kalman"]),
                  "n_kalman_cases": len(out["kalman"]), "hgo_max_rel_diff": max(r["rel_diff"] for r in out["hgo"]), "n_hgo_cases": len(out["hgo"]),
                  "hgo_certificate_below_float": sum(1 for r in out["hgo"] if r["stable"] and r["certificate_minus_float_rel"] < -1e-12),
                  "hgo_first_reading_max_rel_diff": max(r["rel_diff"] for r in out["hgo_first_reading"]),
                  "hgo_first_reading_box_excess_max": max(r["box_excess_rel"] for r in out["hgo_first_reading"]),
                  "n_hgo_first_reading_cases": len(out["hgo_first_reading"])}
save("fidelity.json", out)
print(out["summary"], out["tightness"], out["continuous_limit"])
