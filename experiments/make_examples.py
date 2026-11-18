"""Write the example inputs used by the quick demonstration, the CLI and the web page (examples/)."""
import json, math
import numpy as np
from common import ROOT
from heros.core.assess import jsonable
from heros.sim import plants as PL, loop

EX = ROOT / "examples"; EX.mkdir(exist_ok=True)
rng = np.random.default_rng(2026)

# ---- rotary joint (natural-language request + model + data)
fam = PL.rotary_joint(nominal=True); sc = fam["scales"]; so = fam["plant"]["second_order"]
model = {"plant": fam["plant"], "deterministic": {"domain": {"x1": sc["domain"]["x1"], "u": sc["domain"]["u"]}}}
(EX / "rotary_joint_model.json").write_text(json.dumps(jsonable(model), indent=1))
answers = {"title": "Rotary joint: guaranteed velocity-error limit (high-gain observer)",
           "answers": ["I need the joint velocity.",
                       "We only have the joint encoder, sampled at 1 kHz; the model is attached. The encoder error never exceeds 0.1 mrad. "
                       "Unmodelled disturbance accelerations are bounded by 0.5 rad/s^2. The joint never moves faster than 4 rad/s, "
                       "which is enforced by a hardware limiter in the drive. At start-up the angle is known to within 1 mrad.",
                       "The velocity estimate must be guaranteed to stay within 0.25 rad/s of the true velocity from 100 ms after start-up onward, "
                       "for every admissible disturbance and noise."],
           "clarifications": {"initial.E": [0.001, 2.0]}, "replies": ["[0.001, 2.0]"],
           "note": "the request deliberately omits the bound on the initial velocity error; the coordinator asks for it because a deadline is stated"}
(EX / "rotary_joint_answers.json").write_text(json.dumps(answers, indent=1))
f = loop.chain_rhs(so["f"], so["params"])
Ts, N = sc["Ts"], 1500
nom = so["params"]
x = np.array([0.0, 1.2]); rows = ["t,y,u"]
v = rng.uniform(-sc["V"], sc["V"], N + 1)
for k in range(N + 1):
    t = k * Ts
    r = 0.8 * (1 - math.cos(2 * math.pi * 0.5 * t)) / 2
    u = float(np.clip(nom["J"] * (144.0 * (r - x[0]) - 24.0 * x[1]) + nom["mgl"] * math.sin(x[0]), -6, 6))
    rows.append(f"{t:.4f},{x[0] + v[k]:.9f},{u:.9f}")
    h = Ts / 20
    for j in range(20):
        g = lambda s: np.array([s[1], f(s[0], s[1], u) + 0.3 * math.sin(40 * (t + j * h))])
        k1 = g(x); k2 = g(x + h / 2 * k1); k3 = g(x + h / 2 * k2); k4 = g(x + h * k3); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
(EX / "rotary_joint_data.csv").write_text("\n".join(rows) + "\n")

# ---- servo axis (natural-language request, both specialists eligible) and the structured variant used by the demo
b, Ts, V, D, E = 4.0, 0.001, 1e-4, 6.0, [2e-3, 3.0]
G = np.array([[Ts * Ts / 2], [Ts]]); sa, sv = 3.46, 5.77e-5
plant = {"family": "servo_axis", "kind": "second_order", "states": [{"name": "shaft angle", "unit": "rad"}, {"name": "shaft speed", "unit": "rad/s"}],
         "inputs": [{"name": "motor torque", "unit": "N m"}], "outputs": [{"name": "encoder angle", "unit": "rad"}],
         "linear": {"clock": "continuous", "A": [[0.0, 1.0], [0.0, 0.0]], "B": [[0.0], [b]], "C": [[1.0, 0.0]], "Ts_native": None},
         "second_order": {"f": "b*u", "f0": "b*u", "params": {"b": b}, "param_ranges": {}, "output": "x1"}, "flags": []}
(EX / "servo_axis_model.json").write_text(json.dumps({"plant": plant}, indent=1))
(EX / "servo_axis_answers.json").write_text(json.dumps({
    "title": "Servo axis: smallest mean-square error (Kalman filter); change the third answer to a guaranteed limit to see the selection change",
    "answers": ["The shaft speed.",
                "Encoder angle only, sampled every 1 ms, model attached. The encoder noise is white with standard deviation 57.7 µrad, and the encoder error never exceeds 0.1 mrad. "
                "The unmodelled acceleration is white with standard deviation 3.46 rad/s^2, constant within a sample, and bounded by 6 rad/s^2. "
                "The noises are bounded, not Gaussian; only their variances are reliable. Initial angle error at most 2 mrad, initial speed error at most 3 rad/s.",
                "I want the estimate with the smallest mean-square error; steady state is what matters."], "replies": []}, indent=1))
x = np.array([0.2, 1.0]); rows = ["t,y,u"]
for k in range(1001):
    u = 0.5 * math.sin(2 * math.pi * 2 * k * Ts); a = b * u + rng.uniform(-D, D)
    rows.append(f"{k * Ts:.4f},{x[0] + rng.uniform(-V, V):.9f},{u:.9f}")
    x = np.array([x[0] + Ts * x[1] + 0.5 * Ts * Ts * a, x[1] + Ts * a])
(EX / "servo_axis_data.csv").write_text("\n".join(rows) + "\n")
spec = {"plant": plant, "target": 1, "Ts": Ts, "V": V, "D": D, "E": E, "Q": (sa * sa * G @ G.T).tolist(), "Q_kind": "discrete_cov", "R": [[sv * sv]],
        "P0": [[E[0] ** 2 / 3, 0.0], [0.0, E[1] ** 2 / 3]], "distribution": "bounded", "white": True, "_hard_tolerance": 0.09, "_deadline": 0.05}
(EX / "servo_axis_spec.json").write_text(json.dumps(jsonable(spec), indent=1))

# ---- hydraulic boom (structured request + data)
fam = PL.hydraulic_boom(nominal=True); sc = fam["scales"]
spec = {"plant": fam["plant"], "target": 1, "Ts": sc["Ts"], "Q": sc["Qc"], "Q_kind": "continuous_psd", "G": sc["G"], "R": np.diag(np.square(sc["r_std"])).tolist(),
        "P0": np.diag(np.square(sc["P0_std"])).tolist(), "distribution": "gaussian", "white": True, "t_end": 3.0, "objective": {"kind": "mse_optimal", "horizon": "finite"}}
(EX / "hydraulic_boom_spec.json").write_text(json.dumps(jsonable(spec), indent=1))
from heros import api
from heros.kalman import specialist as K
cfg = K.design(api.new_spec(spec["plant"], objective=spec["objective"], **{k: v for k, v in spec.items() if k not in ("plant", "objective")}))
Nb = cfg["N"]; u = (0.08 * np.sign(np.sin(2 * np.pi * 0.7 * np.arange(Nb + 1) * cfg["Ts"]))).reshape(-1, 1)
X, Y = loop.simulate_linear(cfg["A"], cfg["B"], cfg["C"], cfg["Qd"], cfg["R"], rng.multivariate_normal(np.zeros(4), cfg["P0"]), u, Nb, rng)
(EX / "hydraulic_boom_data.csv").write_text("t,y1,y2,u\n" + "\n".join(f"{k * cfg['Ts']:.3f},{Y[k, 0]:.9f},{Y[k, 1]:.9f},{u[k, 0]:.4f}" for k in range(Nb + 1)) + "\n")
print("examples written:", sorted(p.name for p in EX.iterdir()))
