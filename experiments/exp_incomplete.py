"""Incomplete and conflicting specifications: the coordinator's decision and an INDEPENDENT check of why it is justified.

For every case: the missing or conflicting fact, its consequence, the returned derivation or question, and a check that
does not use the coordinator (independent references or a direct construction)."""
import copy, math
import numpy as np
from common import save
from heros import api
from heros.core import peak, spec as S
from heros.coordinator.coordinator import Session
from heros.khalil import specialist as H
from heros.kalman import specialist as K
from heros.sim import plants as PL
from independent import ref_kalman as RK, ref_hgo as RH

rows = []
Ts, V, D, E = 0.001, 1e-4, 5.0, [1e-3, 2.0]
G = np.array([[Ts * Ts / 2], [Ts]])
plant = {"family": "servo_axis", "kind": "second_order", "states": [{"name": "shaft angle", "unit": "rad"}, {"name": "shaft speed", "unit": "rad/s"}],
         "inputs": [{"name": "motor torque", "unit": "N m"}], "outputs": [{"name": "encoder angle", "unit": "rad"}],
         "linear": {"clock": "continuous", "A": [[0.0, 1.0], [0.0, 0.0]], "B": [[0.0], [1.0]], "C": [[1.0, 0.0]], "Ts_native": None},
         "second_order": {"f": "b*u", "f0": "b*u", "params": {"b": 1.0}, "param_ranges": {}, "output": "x1"}, "flags": []}
hard = dict(target=1, Ts=Ts, V=V, D=D, E=E, objective={"kind": "hard_bound", "tolerance": 0.2, "deadline": 0.1})
c = np.array([0.0, 1.0])


def row(case, missing, consequence, dec, check, ok):
    q = dec.get("question") or {}
    rows.append({"case": case, "missing_or_conflicting": missing, "consequence": consequence, "disposition": dec["disposition"], "selected": dec.get("selected"),
                 "question": q.get("question"), "ask_path": q.get("path"), "threshold": q.get("threshold"),
                 "harmless": [h["path"] for h in dec.get("harmless", [])], "independent_check": check, "check_passed": bool(ok)})
    print(f"{case:38s} {dec['disposition']:10s} ask={q.get('path')} check_passed={ok}")


# 1 Gaussian variance ambiguity (tutorial): y = x + v, only var(y) = 2 known
sc = {"family": "scalar", "kind": "linear", "states": [{"name": "x", "unit": "-"}], "inputs": [], "outputs": [{"name": "y", "unit": "-"}],
      "linear": {"clock": "discrete", "A": [[1.0]], "B": [[0.0]], "C": [[1.0]], "Ts_native": 1.0}, "second_order": None, "flags": []}
d = api.decide(api.new_spec(sc, target=0, Q=[[0.0]], Q_kind="discrete_cov", distribution="gaussian", t_end=1.0, objective={"kind": "mse_optimal", "horizon": "finite"},
                            ranges={"P0": [0.5, 1.0], "R": [1.0, 1.5]}, constraints=[{"paths": ["P0", "R"], "sum": 2.0}]))
g = [RK.batch_gain_first(np.eye(1), np.eye(1), np.zeros((1, 1)), np.array([[r]]), np.array([[p]]))[0, 0] for p, r in ((0.5, 1.5), (1.0, 1.0))]
row("Gaussian variance ambiguity", "prior and noise variances: their sum (2) and a range for the noise variance (1 to 1.5) are known", "conditional mean between y/4 and y/2", d,
    f"independent batch conditioning gives gains {g[0]:.3f} and {g[1]:.3f} for the two admissible completions; both have measurement variance 2", abs(g[0] - .25) < 1e-12 and abs(g[1] - .5) < 1e-12 and d["disposition"] == "CLARIFY")

# 1b the same information, a mean-square-error LEVEL: one gain meets it for every admissible completion
d = api.decide(api.new_spec(sc, target=0, Q=[[0.0]], Q_kind="discrete_cov", distribution="gaussian", t_end=1.0,
                            objective={"kind": "mse_bound", "horizon": "finite", "mse_target": 0.65},
                            ranges={"P0": [0.5, 1.0], "R": [1.0, 1.5]}, constraints=[{"paths": ["P0", "R"], "sum": 2.0}]))
# first gain of the released configuration, recomputed independently from the covariances stored in it
K0 = float(RK.batch_gain_first(np.eye(1), np.eye(1), np.zeros((1, 1)), np.atleast_2d(d["config"]["R"]), np.atleast_2d(d["config"]["P0"]))[0, 0]) if d.get("config") else None
mse_v = [(1 - K0) ** 2 * p + K0 ** 2 * r for p, r in ((0.5, 1.5), (1.0, 1.0))] if K0 is not None else [float("nan")]
row("variance ambiguity, error level requested", "same as above; a mean-square error below 0.65 is requested", "a single gain can meet the level for every split", d,
    f"independent: the released first gain {K0:.3f} gives error {mse_v[0]:.3f} and {mse_v[-1]:.3f} at the two extreme admissible splits; the error is affine in the split, so these are the largest values",
    d["disposition"] == "RELEASE" and max(mse_v) <= 0.65 + 1e-12)

# 2 harmless: initial error with an ultimate-bound request
d = api.decide(api.new_spec(plant, **dict(hard, E=None, objective={"kind": "hard_bound", "tolerance": 0.2})))
M = np.array(d["config"]["M"])
w = [RH.worst_case_exact(Ts, M, c, e0, D, V, 400, sub=4, init="first_measurement") for e0 in ([0.0, 0.0], [1.0, 50.0])]
row("no initial-error bound, ultimate limit", "initial error bounds", "none: the initial-error terms vanish in the limit", d,
    f"independent worst case after 400 steps: {w[0]:.6f} (zero initial error) vs {w[1]:.6f} (initial error 1 rad, 50 rad/s); claimed ultimate bound {d['claim']['value']:.6f}",
    d["disposition"] == "RELEASE" and abs(w[0] - w[1]) < 1e-9 and w[1] <= d["claim"]["value"] * (1 + 1e-9))

# 3 harmless: P0 with a steady-state request
stoch = dict(target=1, Ts=Ts, Q=(D * D / 3 * G @ G.T).tolist(), Q_kind="discrete_cov", R=[[V * V / 3]], distribution="bounded")
d = api.decide(api.new_spec(plant, objective={"kind": "mse_optimal", "horizon": "steady_state"}, **stoch))
A = np.array([[1, Ts], [0, 1.0]]); C_ = np.array([[1.0, 0]])
Pa = RK.riccati_fixed_point(A, C_, np.array(stoch["Q"]), np.array(stoch["R"]))
gains = []
for P0 in (np.diag([1e-8, 1e-2]), np.diag([1.0, 100.0])):
    P = P0.copy()
    for _ in range(20000):
        Mk = P @ C_.T / (C_ @ P @ C_.T + stoch["R"][0][0]); P = A @ (P - Mk @ C_ @ P) @ A.T + np.array(stoch["Q"])
    gains.append((P @ C_.T / (C_ @ P @ C_.T + stoch["R"][0][0])).ravel())
row("no initial covariance, steady-state filter", "initial error covariance", "none for a steady-state (k -> infinity) statement: the gain and its limiting covariance do not depend on it", d,
    f"Riccati iteration from two very different initial covariances converges to the same gain (difference {np.abs(gains[0] - gains[1]).max():.1e}); equals the released gain to {np.abs(gains[0] - np.array(d['config']['M_ss']).ravel()).max():.1e}",
    d["disposition"] == "RELEASE" and np.abs(gains[0] - gains[1]).max() < 1e-9 and np.abs(gains[0] - np.array(d["config"]["M_ss"]).ravel()).max() < 1e-7)

# 4 harmless: distribution not stated
d = api.decide(api.new_spec(plant, objective={"kind": "mse_optimal", "horizon": "steady_state"}, **dict(stoch, distribution=None)))
mc_g = RK.fixed_gain_mse_montecarlo(A, np.zeros((2, 1)), C_, np.array(stoch["Q"]), np.array(stoch["R"]), np.array(d["config"]["M_ss"]), np.zeros(2), np.array(d["config"]["Pf_ss"]), 300, 20000, 1, dist="gaussian")[-50:].mean(0)[1]
mc_u = RK.fixed_gain_mse_montecarlo(A, np.zeros((2, 1)), C_, np.array(stoch["Q"]), np.array(stoch["R"]), np.array(d["config"]["M_ss"]), np.zeros(2), np.array(d["config"]["Pf_ss"]), 300, 20000, 2, dist="uniform")[-50:].mean(0)[1]
row("noise distribution not stated", "distribution of the noises", "none for the estimate; only the weaker (linear-estimator) claim is made", d,
    f"Monte Carlo error variance of the released gain: {mc_g:.3e} (Gaussian) and {mc_u:.3e} (uniform), reported {d['claim']['value']:.3e}; claim type {d['claim']['type']}",
    d["disposition"] == "RELEASE" and d["claim"]["type"] == "lmmse" and abs(mc_g / d["claim"]["value"] - 1) < 0.05 and abs(mc_u / d["claim"]["value"] - 1) < 0.05)

# 5 missing hard noise bound
d = api.decide(api.new_spec(plant, **dict(hard, V=None)))
M1 = np.array(api.decide(api.new_spec(plant, **hard))["config"]["M"])
w_small, w_big = RH.worst_case_exact(Ts, M1, c, [V, 2.0], D, V, 200, sub=4, init="first_measurement"), RH.worst_case_exact(Ts, M1, c, [20 * V, 2.0], D, 20 * V, 200, sub=4, init="first_measurement")
row("missing hard bound on the measurement error", "hard measurement-error bound", "the guaranteed limit depends on it", d,
    f"the design certified for V = {V:g} has independent worst case {w_small:.4f} <= 0.2, but {w_big:.4f} > 0.2 if the true bound is {20 * V:g}",
    d["disposition"] == "CLARIFY" and d["question"]["path"] == "deterministic.V" and w_small <= 0.2 < w_big)

# 6 RMS supplied where a hard bound is needed
d = api.decide(api.new_spec(plant, **dict(hard, V=None, R=[[(V / math.sqrt(3)) ** 2]])))
row("RMS noise level given, guaranteed limit requested", "hard measurement-error bound (an RMS value is not one)", "no hard bound follows from an RMS value", d,
    "a zero-mean two-point noise with values +-a/sqrt(p), probability p/2 each, has RMS a for every p in (0, 1] and magnitude a/sqrt(p), which is unbounded as p -> 0",
    d["disposition"] == "CLARIFY" and d["question"]["path"] == "deterministic.V")

# 7 range of the bound: threshold question
d = api.decide(api.new_spec(plant, **dict(hard, V=None), ranges={"V": [5e-5, 2e-3]}))
thr = d["question"]["threshold"]
dy, dn = api.decide(api.new_spec(plant, **dict(hard, V=thr))), api.decide(api.new_spec(plant, **dict(hard, V=1.05 * thr)))
wy = RH.worst_case_exact(Ts, np.array(dy["config"]["M"]), c, [thr, 2.0], D, thr, 140, sub=4, init="first_measurement")
row("measurement bound known only as a range", "value of the bound inside [5e-5, 2e-3] rad", "certified for the lower part of the range only", d,
    f"at the threshold {thr:.3e} the released design has independent worst case {wy:.4f} <= 0.2 at step 140; at 1.05 x threshold the disposition is {dn['disposition']}",
    d["disposition"] == "CLARIFY" and dy["disposition"] == "RELEASE" and wy <= 0.2 and dn["disposition"] != "RELEASE")

# 8 ambiguous sample period
d = api.decide(api.new_spec(plant, **dict(hard, Ts=None), ranges={"Ts": [0.001, 0.002]}))
Ma = H.gain_sampled(0.001, 0.004)
e_mis = []
for Ttrue in (0.001, 0.002):       # observer built for 1 ms, plant sampled at Ttrue, constant velocity 2 rad/s, no noise, no disturbance
    x = np.array([0.0, 2.0]); xm = np.array([0.0, 2.0])
    for k in range(3000):
        xp = xm + Ma * (x[0] - xm[0]); xm = np.array([[1, 0.001], [0, 1]]) @ xp; x = np.array([x[0] + Ttrue * x[1], x[1]])
    e_mis.append(abs(x[1] - xp[1]))
row("sample period stated as a range", "exact update period", "model, gains and certificate are period-specific", d,
    f"an observer built for 1 ms, driven by error-free data of a 2 rad/s motion: velocity error {e_mis[0]:.2e} when the true period is 1 ms, {e_mis[1]:.2f} rad/s when it is 2 ms",
    d["disposition"] == "CLARIFY" and d["question"]["path"] == "time.Ts" and e_mis[0] < 1e-9 and e_mis[1] > 0.5)

# 9 scenario that introduces a deadline
s = Session(api.new_spec(plant, **dict(hard, E=None, objective={"kind": "hard_bound", "tolerance": 0.2})))
d0 = s.decide(); s.update({"objective.deadline": 0.005}, source="scenario"); d = s.decide()
M0 = np.array(d0["config"]["M"])
wi = RH.worst_case_exact(Ts, M0, c, [V, 100.0], D, V, 5, sub=4, init="first_measurement")
s.update({"initial.E": [1e-3, 100.0]}); d2 = s.decide()
wi2 = RH.worst_case_exact(Ts, np.array(d2["config"]["M"]), c, [V, 100.0], D, V, 5, sub=4, init="first_measurement") if d2["disposition"] == "RELEASE" else float("nan")
row("deadline added after an omission was harmless", "initial velocity-error bound (now consequential)", "with a deadline the initial error matters", d,
    f"first decision {d0['disposition']} (initial error irrelevant for an ultimate limit); after a 5 ms deadline is added the coordinator asks; the earlier design has "
    f"independent worst case {wi:.3g} rad/s at 5 ms for an admissible initial error of 100 rad/s; once that bound is supplied a different (faster) design is "
    f"released ({d2['disposition']}), with independent worst case {wi2:.3g} rad/s at 5 ms",
    d0["disposition"] == "RELEASE" and d["disposition"] == "CLARIFY" and d["question"]["path"] == "initial.E" and wi > 0.2
    and (d2["disposition"] != "RELEASE" or wi2 <= 0.2))

# 10 missing domain evidence (nonlinear joint)
fam = PL.rotary_joint(nominal=True); jsc = fam["scales"]
jkw = dict(target=1, Ts=jsc["Ts"], V=jsc["V"], d_bound=jsc["d_bound"], domain=jsc["domain"], E=jsc["E"], objective={"kind": "hard_bound", "tolerance": 0.25, "deadline": 0.1})
d = api.decide(api.new_spec(fam["plant"], **jkw))
so = fam["plant"]["second_order"]
from heros.sim import loop
ft = loop.chain_rhs(so["f"], {k: v[1] if k in ("J", "bv", "Fc") else v[0] for k, v in so["param_ranges"].items()}); f0 = loop.chain_rhs(so["f0"].replace("y", "x1"), so["params"])
mm_in, mm_out = abs(ft(1.5, 4.0, 6.0) - f0(1.5, 4.0, 6.0)), abs(ft(1.5, 40.0, 6.0) - f0(1.5, 40.0, 6.0))
row("operating range without evidence", "evidence that the trajectory stays in the stated ranges", "the mismatch bound holds only inside the ranges", d,
    f"model mismatch at the edge of the range (4 rad/s): {mm_in:.1f} rad/s^2; at 40 rad/s: {mm_out:.1f} rad/s^2, above the bound used inside the range",
    d["disposition"] == "CLARIFY" and d["question"]["path"] == "deterministic.domain_evidence" and mm_out > 2 * mm_in)

# 11 process noise without its kind
tf = PL.thermal(np.random.default_rng(0)); tsc = tf["scales"]
tkw = dict(target=1, Ts=tsc["Ts"], G=tsc["G"], R=[[tsc["r_std"][0] ** 2]], P0=np.diag(np.square(tsc["P0_std"])).tolist(), objective={"kind": "mse_optimal", "horizon": "finite"})
d = api.decide(api.new_spec(tf["plant"], Q=tsc["Qc"], **tkw))
Ac = np.array(tf["plant"]["linear"]["A"]); Gm = np.array(tsc["G"]); Qc = np.array(tsc["Qc"])
Qd_psd = RK.process_cov_quadrature(Ac, Gm, Qc, tsc["Ts"]); Qd_dis = Gm @ Qc @ Gm.T
row("process noise without its meaning", "whether the value is a spectral density or a per-step covariance", "the two readings differ by a factor of order Ts", d,
    f"independent quadrature: trace of the discrete covariance is {np.trace(Qd_psd):.3e} if the value is a spectral density and {np.trace(Qd_dis):.3e} if it is a per-step covariance (Ts = {tsc['Ts']} s)",
    d["disposition"] == "CLARIFY" and d["question"]["path"] == "stochastic.Q_kind" and abs(np.trace(Qd_psd) / np.trace(Qd_dis) - 1) > 0.3)

# 12 conflicting timing
plant_d = copy.deepcopy(plant); plant_d["second_order"] = None; plant_d["kind"] = "linear"
plant_d["linear"] = {"clock": "discrete", "A": [[1.0, 0.1], [0.0, 1.0]], "B": [[0.005], [0.1]], "C": [[1.0, 0.0]], "Ts_native": 0.1}
d = api.decide(api.new_spec(plant_d, target=1, Ts=0.05, Q=(1e-4 * np.eye(2)).tolist(), Q_kind="discrete_cov", R=[[1e-4]], P0=np.eye(2).tolist(), objective={"kind": "mse_optimal", "horizon": "finite"}))
row("discrete model and a different stated period", "(conflict) native period 0.1 s, stated 0.05 s", "a discrete model cannot be re-sampled without the continuous one", d,
    "A = [[1, 0.1], [0, 1]] is the 0.1 s transition; at 0.05 s the transition would be [[1, 0.05], [0, 1]], which is not the supplied matrix",
    d["disposition"] == "INVALID_INPUT")
# 13 recorded timestamps that disagree with the stated period (revision 2, finding 1)
spec13 = api.new_spec(plant, **hard)
rng13 = np.random.default_rng(13)
x = np.array([0.0, 1.0]); ys = []
for k in range(400):
    ys.append(x[0] + rng13.uniform(-V, V)); a = rng13.uniform(-D, D); x = np.array([x[0] + 0.002 * x[1] + 2e-6 * a, x[1] + 0.002 * a])
ys = np.array(ys)
r13 = api.solve(spec13, ys, np.zeros(400), t=np.arange(400) * 0.002)
d13 = r13["decision"]
Mr = np.array(api.decide(spec13)["config"]["M"])
xm = np.array([ys[0], 0.0]); xs_true = np.array([0.0, 1.0])
row("timestamps disagree with the stated period", "(conflict) data recorded every 2 ms, request states 1 ms", "the gains and the certificate are period-specific", d13,
    f"the release for 1 ms is withdrawn for these data ({d13['issues'][0]['code'] if d13.get('issues') else '-'}); the same records with matching 1 ms timestamps are processed",
    d13["disposition"] == "INVALID_INPUT" and r13["estimates"] is None
    and api.solve(spec13, ys, np.zeros(400), t=np.arange(400) * 0.001)["decision"]["disposition"] == "RELEASE")

# 14 a transient cap below the later tolerance (revision 2, finding 7): compatible requirements, decided at the first sample
d14 = api.decide(api.new_spec(plant, **dict(hard, objective={"kind": "hard_bound", "tolerance": 0.2, "deadline": 0.1, "peak_cap": 0.1})))
row("transient cap below the later tolerance", "none (the cap applies at every sample, the tolerance after the deadline)",
    "at t = 0 one position reading carries no velocity information", d14,
    f"independent: the admissible initial velocities span +-{E[1]} rad/s around the initial estimate, so every estimate is at least {E[1]} rad/s from one of them at t = 0, above the cap 0.1",
    d14["disposition"] == "PROVED_IMPOSSIBLE" and E[1] > 0.1)

# 15 a total disturbance bound and a tighter additive bound for an exactly known model (revision 2, finding 4)
d15 = api.decide(api.new_spec(plant, **dict(hard, D=4 * D, d_bound=D)))
w15 = RH.worst_case_exact(Ts, np.array(d15["config"]["M"]), c, [V, E[1]], D, V, 100, sub=4, init="first_measurement") if d15["disposition"] == "RELEASE" else float("nan")
row("total and additive disturbance bounds both stated", "none (two statements bound the same acceleration error)", "the smaller bound applies", d15,
    f"with the exact nominal model the acceleration error is the additive disturbance, bounded by {D:g}; the released design has independent worst case {w15:.4f} rad/s at 0.1 s with that bound",
    d15["disposition"] == "RELEASE" and d15["config"]["D"] == D and w15 <= 0.2)
save("incomplete.json", {"rows": rows, "all_checks_passed": all(r["check_passed"] for r in rows)})
print("all passed:", all(r["check_passed"] for r in rows))
