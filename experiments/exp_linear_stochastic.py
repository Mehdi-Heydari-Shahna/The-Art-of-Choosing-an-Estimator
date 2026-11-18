"""Linear stochastic case: Kalman specialist on the simulated hydraulic boom (4 states, 2 sensors; SYNTHETIC parameters).

200 independent noise realisations (Gaussian) and 200 with uniform noise of the same covariances.  Checks:
actual state estimates through the public path, covariance calibration (empirical error covariance vs reported P,
NEES, NIS), agreement of one run with the independent batch reference, and the scope of the claim (conditional
mean under Gaussian noise; best linear estimate otherwise)."""
import json
import numpy as np
from common import save, RES
from heros import api, record as R
from heros.kalman import specialist as K
from heros.sim import plants as PL, loop
from independent import ref_kalman as RK

fam = PL.hydraulic_boom(nominal=True); sc = fam["scales"]
N_RUNS, T_END = 200, 3.0
out = {"n_runs": N_RUNS, "plant": "hydraulic boom (simulated, synthetic parameters)", "states": [s["name"] for s in fam["plant"]["states"]]}
traces = {}
for dist in ("gaussian", "bounded"):
    spec = api.new_spec(fam["plant"], target=1, Ts=sc["Ts"], Q=sc["Qc"], Q_kind="continuous_psd", G=sc["G"], R=np.diag(np.square(sc["r_std"])).tolist(),
                        P0=np.diag(np.square(sc["P0_std"])).tolist(), distribution=dist, white=True, t_end=T_END, objective={"kind": "mse_optimal", "horizon": "finite"})
    dec = api.decide(spec)
    assert dec["disposition"] == "RELEASE" and dec["selected"] == K.METHOD
    cfg = K.design(spec); N = cfg["N"]; n = 4
    u = (0.08 * np.sign(np.sin(2 * np.pi * 0.7 * np.arange(N + 1) * cfg["Ts"]))).reshape(-1, 1)      # known valve command
    E = np.empty((N_RUNS, N + 1, n)); nees = np.empty((N_RUNS, N + 1)); nis = np.empty((N_RUNS, N + 1))
    for r in range(N_RUNS):
        rng = np.random.default_rng(5000 + r + (0 if dist == "gaussian" else 10000))
        x0 = rng.multivariate_normal(np.zeros(n), cfg["P0"]) if dist == "gaussian" else np.sqrt(np.diag(cfg["P0"])) * rng.uniform(-np.sqrt(3), np.sqrt(3), n)
        X, Y = loop.simulate_linear(cfg["A"], cfg["B"], cfg["C"], cfg["Qd"], cfg["R"], x0, u, N, rng, "gaussian" if dist == "gaussian" else "uniform")
        est = K.run(cfg, Y, u)
        e = X - est["x_filt"]; E[r] = e
        nees[r] = np.einsum("ki,kij,kj->k", e, np.linalg.inv(est["P_filt"]), e); nis[r] = est["nis"]
        if r == 0:
            res = api.solve(spec, Y, u)
            mean, cov = RK.batch_filtered(cfg["A"], cfg["B"], cfg["C"], cfg["Qd"], cfg["R"], cfg["x0_hat"], cfg["P0"], Y[:61], u[:61], dps=40)
            ind = {"max_estimate_diff": float(np.max(np.abs(res["estimates"]["x_filt"][:61] - mean))),
                   "max_cov_diff_rel": float(np.max(np.abs(res["estimates"]["P_filt"][:61] - cov) / np.abs(cov).max()))}
            if dist == "gaussian":
                (RES / "boom_record.json").write_text(R.dumps(R.make_record(spec, res["decision"], Y, u, res["estimates"], store_inputs=True)))
                traces.update(t=est["t"], x=X, xf=est["x_filt"], sd=np.sqrt(np.einsum("kii->ki", est["P_filt"])), y=Y, u=u[:, 0])
    P = np.array(K.gains(cfg, N)[2])
    emp = np.einsum("rki,rkj->kij", E, E) / N_RUNS
    k0 = 50
    ratio = np.array([emp[k0:, i, i].mean() / P[k0:, i, i].mean() for i in range(n)])
    se = 1.96 * np.sqrt(2.0 / (N_RUNS * (N + 1 - k0) / 20.0))     # rough band: errors are correlated over ~20 samples
    out[dist] = {"claim_type": dec["claim"]["type"], "claim": dec["claim"]["statement"], "independent_reference_run0": ind,
                 "rmse": np.sqrt((E[:, k0:] ** 2).mean(axis=(0, 1))).tolist(), "reported_std": np.sqrt(np.array([P[k0:, i, i].mean() for i in range(n)])).tolist(),
                 "empirical_over_reported_variance": ratio.tolist(), "approx_95_band_half_width": float(se),
                 "mean_nees": float(nees[:, k0:].mean()), "expected_nees": n, "mean_nis": float(nis[:, k0:].mean()), "expected_nis": 2,
                 "coverage_2sigma_target": float(np.mean(np.abs(E[:, k0:, 1]) <= 2 * np.sqrt(P[k0:, 1, 1]))), "gaussian_2sigma": 0.9545}
    traces[f"err_{dist}"] = E[:, :, 1]
    traces[f"emp_std_{dist}"] = np.sqrt(emp[:, 1, 1]); traces[f"rep_std_{dist}"] = np.sqrt(P[:, 1, 1])
    # sub-optimal comparison on the same model: a mis-tuned filter (R ten times too large) has a larger exact MSE
    cfg2 = dict(cfg); cfg2["R"] = cfg["R"] * 10
    M2 = K.gains(cfg2, N)[0]
    Pmis = K.error_cov_of_gain(cfg, M2, N + 1)[1]
    out[dist]["exact_mse_ratio_mistuned_R_x10_target"] = float(np.mean([p[1, 1] for p in Pmis[k0:]]) / P[k0:, 1, 1].mean())
np.savez_compressed(RES / "boom_traces.npz", **traces)
save("linear_stochastic.json", out)
for d in ("gaussian", "bounded"):
    print(d, {k: out[d][k] for k in ("claim_type", "empirical_over_reported_variance", "mean_nees", "mean_nis", "coverage_2sigma_target", "independent_reference_run0")})
