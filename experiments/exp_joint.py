"""Nonlinear high-gain-observer case and observer-based closed loop on the simulated rotary joint.

ALL PHYSICAL PARAMETERS ARE SYNTHETIC.  The estimator is selected by the coordinator BEFORE any run from the
specification below; the true plant uses the unfavourable ends of the parameter ranges.
Part A (estimation only): the joint tracks the reference with ideal state feedback; the released observer runs on the
        recorded measurements and inputs for 100 noise realisations.  Simulated errors are compared with the certified
        envelope (a comparison, not a proof).
Part B (closed loop): three complete loops under identical initial condition, reference, disturbance and noise:
        feedback from the released observer, from a conventional filtered difference (independent design, no certificate),
        and from the true state (ideal reference).  Because the estimates change the applied input, these are three
        different closed-loop experiments, compared as such.
Part C (sampling): pole-matched and forward-Euler gain mappings over a range of period/eps ratios.
"""
import math
import numpy as np
from common import save, RES
from heros import api, record as R
from heros.core import peak
from heros.khalil import specialist as H
from heros.sim import plants as PL, loop

fam = PL.rotary_joint(nominal=True)
sc, so = fam["scales"], fam["plant"]["second_order"]
Ts, V = sc["Ts"], sc["V"]
DEADLINE, TOL = 0.1, 0.25
spec = api.new_spec(fam["plant"], target=1, Ts=Ts, V=V, d_bound=sc["d_bound"], domain=sc["domain"], E=sc["E"],
                    domain_evidence={"kind": "mechanical_limit", "statement": "speed limiter at 4 rad/s; torque saturated at 6 N m"},
                    objective={"kind": "hard_bound", "tolerance": TOL, "deadline": DEADLINE})
dec = api.decide(spec)
assert dec["disposition"] == "RELEASE", dec["explanations"]
cfg = dec["config"]
cert = dec["assessments"]["high_gain_observer"]["metrics"]["worst_case"]
print("selected:", dec["selected"], "eps", cfg["eps"], "claim:", dec["claim"]["statement"])

# true plant: unfavourable ends of the ranges
pr = so["param_ranges"]
true_params = {"J": pr["J"][1], "mgl": pr["mgl"][0], "bv": pr["bv"][1], "Fc": pr["Fc"][1], "vs": so["params"]["vs"]}
f_true = loop.chain_rhs(so["f"], true_params)
nom = so["params"]
T_END = 2.6; N = int(round(T_END / Ts)); t = np.arange(N + 1) * Ts
U_MAX = sc["domain"]["u"][1]
WN, ZETA = 12.0, 1.0
K1, K2 = WN * WN, 2 * ZETA * WN


def minjerk(t, t0, T, a, b):
    s = np.clip((t - t0) / T, 0, 1)
    p = a + (b - a) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)
    v = (b - a) * (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T
    acc = (b - a) * (60 * s - 180 * s ** 2 + 120 * s ** 3) / T ** 2
    return p, v, acc


def reference(tt):
    p1, v1, a1 = minjerk(tt, 0.3, 0.6, 0.0, 1.0)
    p2, v2, a2 = minjerk(tt, 1.5, 0.6, 0.0, -0.6)
    return p1 + p2, v1 + v2, a1 + a2


def dist(tt):          # additive acceleration disturbance, switched on at 1.1 s (bounded by d_bound = 0.5)
    return 0.0 if tt < 1.1 else 0.5 * math.copysign(1.0, math.sin(2 * math.pi * 3.0 * tt))


def law(tt, y, vel):
    r, rd, rdd = reference(tt)
    v = rdd + K2 * (rd - vel) + K1 * (r - y)
    return float(np.clip(nom["J"] * v + nom["mgl"] * math.sin(y), -U_MAX, U_MAX))


class DirtyDerivative:
    """Conventional design written independently of the specialists: backward difference of the encoder followed by a
    first-order low-pass at 40 Hz (Tustin).  No guarantee is attached."""
    def __init__(self, Ts, fc=40.0):
        w = 2 * math.pi * fc
        self.a = (2 - w * Ts) / (2 + w * Ts); self.b = w * Ts / (2 + w * Ts)
        self.prev_y = None; self.prev_raw = 0.0; self.v = 0.0; self.Ts = Ts
    def step(self, k, y):
        raw = 0.0 if self.prev_y is None else (y - self.prev_y) / self.Ts
        self.v = self.a * self.v + self.b * (raw + self.prev_raw)
        self.prev_y, self.prev_raw = y, raw
        return np.array([y, self.v])
    def feed(self, k, u):
        pass


X0 = [0.0, 1.2]          # the joint is already moving when the observer starts (initial velocity error 1.2 rad/s <= E2 = 2)
out = {"spec_hash": dec["spec_hash"], "selected": dec["selected"], "eps": cfg["eps"], "M": cfg["M"], "Ts": Ts, "V": V, "D": cfg["D"],
       "D_detail": {k: v for k, v in cfg["D_detail"].items() if k != "domain_evidence"}, "tolerance": TOL, "deadline": DEADLINE,
       "certified_post_deadline": cert["post_deadline_upper"], "certified_peak": cert["peak_upper"], "certified_ultimate": cert["ultimate_upper"],
       "lower_limit_additive": peak.lower_limit_velocity(sc["d_bound"], V), "true_params": true_params, "nominal_params": nom, "x0": X0,
       "controller": {"law": "u = sat(J0 (r'' + k2 (r' - v) + k1 (r - y)) + mgl0 sin(y))", "wn": WN, "zeta": ZETA, "u_max": U_MAX},
       "simulation_note": "simulated plant, synthetic parameters; RK4 with 20 substeps per sample"}

# ---------------- Part A: estimation on recorded data under ideal state feedback
kd = int(round(DEADLINE / Ts))
errs, post, peaks, maxvel, viol = [], [], [], [], 0
traces = {}
for run in range(100):
    rng = np.random.default_rng(1000 + run)
    v = rng.uniform(-V, V, N + 1)
    x_holder = {}
    def ctrl(k, tt, y, xh, holder=x_holder):
        return law(tt, holder["x"][0], holder["x"][1])
    # state feedback needs the true state: run the loop manually
    x = np.array(X0, float); Xs = np.empty((N + 1, 2)); Ys = np.empty(N + 1); Us = np.empty(N + 1)
    for k in range(N + 1):
        Xs[k] = x; Ys[k] = x[0] + v[k]
        u = law(k * Ts, x[0], x[1]); Us[k] = u
        if k == N: break
        h = Ts / 20
        for j in range(20):
            tt = k * Ts + j * h
            g = lambda s, tq: np.array([s[1], f_true(s[0], s[1], u) + dist(tq)])
            k1 = g(x, tt); k2 = g(x + h / 2 * k1, tt + h / 2); k3 = g(x + h / 2 * k2, tt + h / 2); k4 = g(x + h * k3, tt + h)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    est = H.run(cfg, Ys, Us)
    e = np.abs(Xs[:, 1] - est["x_hat"][:, 1])
    viol += int(np.sum(e > est["envelope"] * (1 + 1e-12)))
    errs.append(e); post.append(e[kd:].max()); peaks.append(e.max()); maxvel.append(np.abs(Xs[:, 1]).max())
    if run == 0:
        res0 = api.solve(spec, Ys, Us)
        (RES / "joint_record.json").write_text(R.dumps(res0["record"]))
        traces.update(t=t, x=Xs, y=Ys, u=Us, xh=est["x_hat"], envelope=est["envelope"], env_init=est["envelope_parts"]["initial"],
                      envelope_box=peak.worst_case_lti(Ts, cfg["M"], np.array([0.0, 1.0]), cfg["E_eff"], cfg["D"], V, N)["W"],
                      env_dist=est["envelope_parts"]["disturbance"], env_noise=est["envelope_parts"]["noise"])
        out["monitor_run0"] = est["assumption_monitor"]["status"]
errs = np.array(errs)
out["estimation"] = {"n_runs": 100, "envelope_violations": viol, "simulated_post_deadline_max": float(max(post)), "simulated_post_deadline_median": float(np.median(post)),
                     "simulated_peak_max": float(max(peaks)), "max_abs_velocity": float(max(maxvel)), "max_abs_input": float(np.abs(traces["u"]).max()),
                     "conservatism_post_deadline": float(cert["post_deadline_upper"] / max(post)),
                     "domain_respected": bool(max(maxvel) <= sc["domain"]["x2"][1])}
traces.update(err_q50=np.quantile(errs, 0.5, axis=0), err_max=errs.max(axis=0), err_q95=np.quantile(errs, 0.95, axis=0))
# continuous-time idealisation with the same eps and bounds (reference curve)
g = 1.0 / cfg["eps"]
traces["ct_envelope"] = peak.ct_envelope(g, cfg["E_eff"], cfg["D"], V, t, target=1)
out["ct_ultimate_same_eps"] = peak.ct_ultimate(g, cfg["D"], V)

# ---------------- Part B: three closed loops, identical conditions
loops = {}
rng = np.random.default_rng(77)
vcl = rng.uniform(-V, V, N + 1)
for name in ("observer", "filtered_difference", "state_feedback"):
    if name == "observer":
        estm = loop.OnlineHGO(cfg)
    elif name == "filtered_difference":
        estm = DirtyDerivative(Ts)
    else:
        estm = None
    x = np.array(X0, float); Xs = np.empty((N + 1, 2)); Us = np.empty(N + 1); XH = np.full((N + 1, 2), np.nan); sat = 0
    for k in range(N + 1):
        Xs[k] = x; y = x[0] + vcl[k]
        if estm is None:
            vel, pos = x[1], x[0]
        else:
            xh = estm.step(k, y); XH[k] = xh; vel, pos = xh[1], y
        r, rd, rdd = reference(k * Ts)
        vraw = nom["J"] * (rdd + K2 * (rd - vel) + K1 * (r - pos)) + nom["mgl"] * math.sin(pos)
        u = float(np.clip(vraw, -U_MAX, U_MAX)); sat += int(abs(vraw) > U_MAX); Us[k] = u
        if estm is not None:
            estm.feed(k, u)
        if k == N: break
        h = Ts / 20
        for j in range(20):
            tt = k * Ts + j * h
            gg = lambda s, tq: np.array([s[1], f_true(s[0], s[1], u) + dist(tq)])
            k1 = gg(x, tt); k2 = gg(x + h / 2 * k1, tt + h / 2); k3 = gg(x + h / 2 * k2, tt + h / 2); k4 = gg(x + h * k3, tt + h)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    r_all = np.array([reference(tt)[0] for tt in t])
    e_tr = Xs[:, 0] - r_all
    ev = np.abs(Xs[:, 1] - XH[:, 1]) if estm is not None else np.zeros(N + 1)
    loops[name] = {"rms_tracking_error": float(np.sqrt(np.mean(e_tr[kd:] ** 2))), "max_tracking_error_after_deadline": float(np.abs(e_tr[kd:]).max()),
                   "rms_input": float(np.sqrt(np.mean(Us ** 2))), "input_total_variation": float(np.abs(np.diff(Us)).sum()),
                   "saturated_samples": sat, "max_abs_velocity": float(np.abs(Xs[:, 1]).max()),
                   "velocity_estimate_error_post_deadline_max": float(ev[kd:].max()), "velocity_estimate_error_rms": float(np.sqrt(np.mean(ev[kd:] ** 2)))}
    traces[f"cl_{name}_x"] = Xs; traces[f"cl_{name}_u"] = Us; traces[f"cl_{name}_xh"] = XH
traces["cl_ref"] = np.array([reference(tt)[0] for tt in t]); traces["cl_dist"] = np.array([dist(tt) for tt in t])
out["closed_loop"] = loops
out["closed_loop_errors"] = {name: {"tracking_rms_after_deadline": v["rms_tracking_error"], "tracking_max_after_deadline": v["max_tracking_error_after_deadline"],
                                    "velocity_estimation_max_after_deadline": v["velocity_estimate_error_post_deadline_max"] if name != "state_feedback" else 0.0,
                                    "velocity_estimation_rms_after_deadline": v["velocity_estimate_error_rms"] if name != "state_feedback" else 0.0,
                                    "within_certified_limit": (v["velocity_estimate_error_post_deadline_max"] <= cert["post_deadline_upper"]) if name == "observer" else None}
                             for name, v in loops.items()}
out["closed_loop_note"] = ("three different closed-loop experiments under matched initial condition, reference, disturbance and noise; "
                           "no closed-loop stability guarantee is claimed; the observer certificate applies while the trajectory stays in the domain")
out["closed_loop_domain_respected"] = bool(loops["observer"]["max_abs_velocity"] <= sc["domain"]["x2"][1])

# ---------------- Part C: sampling behaviour of two gain mappings (double-integrator error model, same bounds)
rows = []
c = np.array([0.0, 1.0]); E = cfg["E_eff"]
for ratio in (0.05, 0.2, 0.5, 1.0, 1.5, 1.9, 2.1, 3.0):
    eps = Ts / ratio
    row = {"Ts_over_eps": ratio}
    for nm, M, form in (("pole_matched", H.gain_sampled(Ts, eps), "current"), ("forward_euler", H.gain_euler_predictor(Ts, eps), "predictor")):
        # the pole-matched observer starts from the first reading (exact); the forward-Euler predictor has no update at k = 0, so its
        # initial position error is bounded by an independent box |e1(0)| <= V (Proposition 1 as stated)
        ct = peak.certify_lti(Ts, M, c, E, cfg["D"], V, k_deadline=kd, form=form, init="first_measurement" if form == "current" else "box")
        row[nm] = {"stable": ct["stable"], "spectral_radius": ct["spectral_radius_float"], "post_deadline_bound": ct.get("post_deadline_upper"),
                   "ultimate_bound": ct.get("ultimate_upper")}
    rows.append(row)
out["sampling"] = rows
# gain trade-off (ultimate bound and its parts versus eps) for the same bounds: disturbance part grows with eps, noise part with 1/eps
tr = {"eps": [], "post_deadline": [], "ultimate": [], "disturbance_part": [], "noise_part": []}
for eps in np.exp(np.linspace(math.log(0.35 * Ts), math.log(40 * Ts), 70)):
    Mg = H.gain_sampled(Ts, eps)
    nst = int(kd + max(400, 40 * eps / Ts))
    wc = peak.worst_case_lti(Ts, Mg, c, E, cfg["D"], V, nst, init="first_measurement")
    tr["eps"].append(float(eps)); tr["post_deadline"].append(float(wc["W"][kd:].max())); tr["ultimate"].append(float(wc["W"][-1] - wc["init"][-1]))
    tr["disturbance_part"].append(float(wc["dist"][-1])); tr["noise_part"].append(float(wc["noise"][-1]))
out["tradeoff"] = tr
np.savez_compressed(RES / "joint_traces.npz", **traces)
save("joint.json", out)
print({k: out["estimation"][k] for k in ("envelope_violations", "simulated_post_deadline_max", "conservatism_post_deadline", "max_abs_velocity")})
print({k: (v["rms_tracking_error"], v["velocity_estimate_error_post_deadline_max"], v["saturated_samples"], v["input_total_variation"]) for k, v in loops.items()})
