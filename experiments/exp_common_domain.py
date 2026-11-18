"""Common-domain case: one sampled servo axis, one set of records, two specialists, several definitions of success.

Plant: x1' = x2, x2' = b u + a(t), y_k = x1(t_k) + v_k, period Ts.  The unmodelled acceleration a is constant within a
sample and uniform on [-D, D]; the measurement error is uniform on [-V, V]; the initial error is uniform in its box.
So the SAME records satisfy (i) the second-order statistical model used by the Kalman filter (white, known
covariances; non-Gaussian, so the Kalman claim is the linear-estimator one) and (ii) the hard bounds used by the
high-gain observer certificate.  Both estimators receive the same measurements and inputs.
For each criterion the coordinator's decision is recorded together with exact values (covariance recursion, exact
worst case) and empirical values over the common records.  Whether the preferred design changes with the criterion
is an outcome of the experiment."""
import math
import numpy as np
from common import save, RES
from heros import api
from heros.core import peak
from heros.kalman import specialist as K
from heros.khalil import specialist as H
from heros.sim import plants as PL
from independent import ref_kalman as RK, ref_hgo as RH

Ts, b = 0.001, 4.0
V, D, E = 1e-4, 6.0, [2e-3, 3.0]
sa, sv = D / math.sqrt(3), V / math.sqrt(3)
G = np.array([[Ts * Ts / 2], [Ts]])
Qd = sa * sa * (G @ G.T); Rm = np.array([[sv * sv]]); P0 = np.diag([E[0] ** 2 / 3, E[1] ** 2 / 3])
plant = {"family": "servo_axis", "kind": "second_order", "states": [{"name": "shaft angle", "unit": "rad"}, {"name": "shaft speed", "unit": "rad/s"}],
         "inputs": [{"name": "motor torque", "unit": "N m"}], "outputs": [{"name": "encoder angle", "unit": "rad"}],
         "linear": {"clock": "continuous", "A": [[0.0, 1.0], [0.0, 0.0]], "B": [[0.0], [b]], "C": [[1.0, 0.0]], "Ts_native": None},
         "second_order": {"f": "b*u", "f0": "b*u", "params": {"b": b}, "param_ranges": {}, "output": "x1"}, "flags": []}
common = dict(target=1, Ts=Ts, V=V, D=D, E=E, Q=Qd.tolist(), Q_kind="discrete_cov", R=Rm.tolist(), P0=P0.tolist(), distribution="bounded", white=True, t_end=1.0)
DEADLINE = 0.05
kd = int(round(DEADLINE / Ts))
FM = "first_measurement"           # every design starts from the first encoder reading (exact worst case, Proposition 1 corollary)
c = np.array([0.0, 1.0])

# ---- designs
kcfg_tv = K.design(api.new_spec(plant, objective={"kind": "mse_optimal", "horizon": "finite"}, **common))
kcfg_ss = K.design(api.new_spec(plant, objective={"kind": "mse_optimal", "horizon": "steady_state"}, **common))
N = 1000
M_tv = [m.reshape(2) for m in K.gains(kcfg_tv, N)[0]]
M_ss = kcfg_ss["M_ss"].reshape(2)
E_eff = [V, E[1]]            # both estimators start from the first measurement (same initial information)
eps_h, bound_h, _ = H.search_eps(Ts, c, E_eff, D, V, kd, None, init=FM)
M_h = H.gain_sampled(Ts, eps_h)
eps_m = min(np.exp(np.linspace(math.log(0.3 * Ts), math.log(300 * Ts), 400)),
            key=lambda e: K.steady_cov_of_constant_gain(kcfg_ss["A"], kcfg_ss["C"], Qd, Rm, H.gain_sampled(Ts, e).reshape(2, 1))[1, 1])
M_hm = H.gain_sampled(Ts, eps_m)
designs = {"kalman_time_varying": M_tv, "kalman_steady_state": [M_ss] * (N + 1), "hgo_worst_case_tuned": [M_h] * (N + 1), "hgo_mse_tuned": [M_hm] * (N + 1)}

# ---- exact criteria
A, C = kcfg_ss["A"], kcfg_ss["C"]
P0 = np.diag([E[0] ** 2 / 3, E[1] ** 2 / 3])        # prior of the stated initial estimate (zero); its position part is replaced below
exact = {}
for name, Ms in designs.items():
    # first-reading initialisation: the first update has the gain [1, 0]', so e_0 = (-v_0, e_0,2) and P_0|0 = diag(R, P0_22)
    P = P0.copy(); var = []
    for k, M in enumerate(Ms):                     # covariance recursion written here independently of the specialists
        M = np.array([1.0, 0.0]) if k == 0 else M
        J = np.eye(2) - np.outer(M, [1.0, 0.0]); Pf = J @ P @ J.T + np.outer(M, M) * sv * sv; var.append(Pf[1, 1]); P = A @ Pf @ A.T + Qd
    W = (peak.worst_case_ltv(Ts, Ms[: 400], c, E_eff, D, V, init=FM) if name == "kalman_time_varying"
         else peak.worst_case_lti(Ts, Ms[0], c, E_eff, D, V, N, init=FM)["W"])
    row = {"mse_steady": float(var[-1]), "mse_at_deadline": float(var[kd]), "worst_case_at_deadline": float(W[kd]),
           "worst_case_post_deadline_max_evaluated": float(W[kd:].max()), "worst_case_peak_evaluated": float(W.max())}
    if name != "kalman_time_varying":
        cert = peak.certify_lti(Ts, Ms[0], c, E_eff, D, V, k_deadline=kd, init=FM)
        row.update(certified_post_deadline=cert["post_deadline_upper"], certified_ultimate=cert["ultimate_upper"], certified_peak=cert["peak_upper"])
        row["independent_worst_case_at_deadline"] = float(RH.worst_case_exact(Ts, Ms[0], c, [V, E[1]], D, V, kd, sub=4, init=FM))
        Wbox = peak.worst_case_lti(Ts, Ms[0], c, E_eff, D, V, N)["W"]   # Proposition 1 with an independent box |e1(0)| <= V (upper bound)
        row["independent_box_upper_bound_at_k0"] = float(Wbox[0]); row["exact_at_k0"] = float(W[0])
        row["independent_box_max_rel_excess"] = float(np.max((Wbox - W) / W))
        row["independent_mse_steady"] = float(RK.lyapunov_mse_of_gain(A, C, Qd, Rm, Ms[0].reshape(2, 1))[1, 1])
    else:
        row["note"] = "time-varying gain: worst case evaluated in floating point over 400 steps (no interval certificate)"
    exact[name] = row

# ---- common records
N_REC = 200
errs = {k: np.empty((N_REC, N + 1)) for k in designs}
for r in range(N_REC):
    rng = np.random.default_rng(9000 + r)
    x = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-E[1], E[1])])
    acc = rng.uniform(-D, D, N + 1); v = rng.uniform(-V, V, N + 1)
    u = 0.5 * np.sin(2 * np.pi * 2.0 * np.arange(N + 1) * Ts)
    X = np.empty((N + 1, 2)); Y = np.empty(N + 1)
    for k in range(N + 1):
        X[k] = x; Y[k] = x[0] + v[k]
        a = b * u[k] + acc[k]
        x = np.array([x[0] + Ts * x[1] + 0.5 * Ts * Ts * a, x[1] + Ts * a])      # exact for piecewise-constant acceleration
    for name, Ms in designs.items():
        xm = np.array([Y[0], 0.0])               # same initialisation for all: first measurement, zero velocity
        for k in range(N + 1):
            xp = xm + Ms[k] * (Y[k] - xm[0]); errs[name][r, k] = X[k, 1] - xp[1]
            xm = A @ xp + G[:, 0] * (b * u[k])
emp = {}
for name in designs:
    e = errs[name]
    emp[name] = {"mse_steady_empirical": float((e[:, 300:] ** 2).mean()), "max_abs_error_post_deadline": float(np.abs(e[:, kd:]).max()),
                 "max_abs_error_overall": float(np.abs(e).max()), "rmse_at_deadline": float(np.sqrt((e[:, kd] ** 2).mean()))}

# ---- coordinator decisions on the same information, different definitions of success
kf_mse = exact["kalman_steady_state"]["mse_steady"]
tol_feasible = round(1.6 * exact["hgo_worst_case_tuned"]["certified_post_deadline"], 4)
requests = {
    "smallest mean-square error": {"kind": "mse_optimal", "horizon": "steady_state"},
    "mean-square error below a level": {"kind": "mse_bound", "horizon": "steady_state", "mse_target": round(3 * kf_mse, 6)},
    "error within r with probability 0.9": {"kind": "probability", "horizon": "steady_state", "tolerance": round(4 * math.sqrt(kf_mse), 4), "prob": 0.9},
    "guaranteed error limit after the deadline": {"kind": "hard_bound", "tolerance": tol_feasible, "deadline": DEADLINE},
    "guaranteed ultimate error limit": {"kind": "hard_bound", "tolerance": tol_feasible},
    "guaranteed limit below the two-signal limit": {"kind": "hard_bound", "tolerance": round(0.8 * peak.lower_limit_velocity(D, V), 4)},
}
decisions = {}
for name, obj in requests.items():
    d = api.decide(api.new_spec(plant, objective=obj, **common))
    decisions[name] = {"objective": obj, "disposition": d["disposition"], "selected": d["selected"], "qualifying": d["qualifying"],
                       "criterion_values": (d.get("tradeoff") or {}).get("values"), "ranking_supported": (d.get("tradeoff") or {}).get("ranking_supported"),
                       "claim": (d.get("claim") or {}).get("statement"), "explanation": d["explanations"][:2]}
    print(f"{name:45s} -> {d['disposition']:22s} {d['selected']} {decisions[name]['criterion_values']}")

# ---- sweep: how the two criteria rank the constant-gain designs as the noise-to-disturbance ratio changes
sweep = []
for Vs in (2e-5, 5e-5, 1e-4, 3e-4, 1e-3):
    for Ds in (1.0, 6.0, 30.0):
        Qs = (Ds * Ds / 3) * (G @ G.T); Rs = np.array([[Vs * Vs / 3]])
        Mk = K.steady_state(A, C, Qs, Rs)[2].reshape(2)
        eh, bh, _ = H.search_eps(Ts, c, [Vs, E[1]], Ds, Vs, kd, None, init=FM)
        Mh = H.gain_sampled(Ts, eh)
        wk = peak.worst_case_lti(Ts, Mk, c, [Vs, E[1]], Ds, Vs, 4000, init=FM)["W"]; wh = peak.worst_case_lti(Ts, Mh, c, [Vs, E[1]], Ds, Vs, 4000, init=FM)["W"]
        mk = K.steady_cov_of_constant_gain(A, C, Qs, Rs, Mk.reshape(2, 1))[1, 1]; mh = K.steady_cov_of_constant_gain(A, C, Qs, Rs, Mh.reshape(2, 1))[1, 1]
        sweep.append({"V": Vs, "D": Ds, "kalman_mse": float(mk), "hgo_mse": float(mh), "mse_ratio_hgo_over_kalman": float(mh / mk),
                      "kalman_worst_post_deadline": float(wk[kd:].max()), "hgo_worst_post_deadline": float(wh[kd:].max()),
                      "worst_ratio_kalman_over_hgo": float(wk[kd:].max() / wh[kd:].max()), "kalman_worst_ultimate": float(wk[-1]), "hgo_worst_ultimate": float(wh[-1]),
                      "two_signal_limit": peak.lower_limit_velocity(Ds, Vs)})
out = {"setting": {"Ts": Ts, "b": b, "V": V, "D": D, "E": E, "sigma_a": sa, "sigma_v": sv, "deadline": DEADLINE, "n_records": N_REC, "n_steps": N,
                   "noise": "uniform (bounded), white; identical records for all designs",
        "initialisation": "first measurement, zero velocity (all designs); worst cases are exact for this initialisation (e_1(0) = -v_0)"},
       "designs": {"hgo_worst_case_tuned": {"eps": eps_h, "M": M_h.tolist()}, "hgo_mse_tuned": {"eps": float(eps_m), "M": M_hm.tolist()}, "kalman_steady_state": {"M": M_ss.tolist()}},
       "exact": exact, "empirical": emp, "decisions": decisions, "sweep": sweep, "two_signal_limit": peak.lower_limit_velocity(D, V)}
r1 = exact["hgo_worst_case_tuned"]; r2 = exact["kalman_steady_state"]
out["finding"] = {"mse_preferred": min(("kalman_steady_state", "hgo_worst_case_tuned", "hgo_mse_tuned"), key=lambda k: exact[k]["mse_steady"]),
                  "hard_bound_preferred": min(("kalman_steady_state", "hgo_worst_case_tuned"), key=lambda k: exact[k]["certified_post_deadline"]),
                  "mse_ratio_hgo_worstcase_tuned_over_kalman": r1["mse_steady"] / r2["mse_steady"],
                  "worst_case_ratio_kalman_over_hgo": r2["certified_post_deadline"] / r1["certified_post_deadline"]}
np.savez_compressed(RES / "common_traces.npz", t=np.arange(N + 1) * Ts, **{f"err_{k}": v[:40] for k, v in errs.items()},
                    **{f"rms_{k}": np.sqrt((v ** 2).mean(0)) for k, v in errs.items()}, **{f"max_{k}": np.abs(v).max(0) for k, v in errs.items()},
                    W_hgo=peak.worst_case_lti(Ts, M_h, c, E_eff, D, V, N, init=FM)["W"], W_kss=peak.worst_case_lti(Ts, M_ss, c, E_eff, D, V, N, init=FM)["W"],
                    W_ktv=np.concatenate([peak.worst_case_ltv(Ts, M_tv[:400], c, E_eff, D, V, init=FM), np.full(N + 1 - 400, np.nan)]),
                    W_hgo_box=peak.worst_case_lti(Ts, M_h, c, E_eff, D, V, N)["W"], W_kss_box=peak.worst_case_lti(Ts, M_ss, c, E_eff, D, V, N)["W"],
                    cert_hgo=exact["hgo_worst_case_tuned"]["certified_post_deadline"], cert_kss=exact["kalman_steady_state"]["certified_post_deadline"])
save("common_domain.json", out)
print(out["finding"])
for k in exact: print(k, {a: round(b_, 6) if isinstance(b_, float) else b_ for a, b_ in exact[k].items() if a != "note"}, emp[k])
