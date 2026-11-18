"""Model families used by the examples, the data generator and the evaluation.

ALL PARAMETER VALUES ARE SYNTHETIC (chosen to be physically plausible); none is identified from a machine.
Each constructor returns a dict:
    plant   block for the common specification (see heros.core.spec)
    scales  typical magnitudes used by the case generator (Ts, V, D or noise levels, initial errors, tolerances)
    truth   parameters of the simulated "true" plant
Families marked HELD_OUT are never used for training or model selection.
The linear plant matrices of dc_motor, thermal, two_mass and hydraulic_boom are taken from the supplied Kalman
package (kalman_agent/plant/library.py); the rotary joint follows the supplied Khalil package (hgo_agent/plants.py)
with a measurement-driven nominal model.
"""
from __future__ import annotations

import math

import numpy as np

TRAIN_FAMILIES = ["servo_axis", "mass_spring_damper", "rotary_joint", "dc_motor", "thermal", "two_mass", "bearing_only"]
HELD_OUT_FAMILIES = ["gantry_axis", "duffing_stage", "hydraulic_boom", "cart_pole"]
BOTH_METHODS = ["servo_axis", "gantry_axis"]            # sampled double integrators: both specialists apply
KALMAN_ONLY = ["dc_motor", "thermal", "two_mass", "hydraulic_boom"]
KHALIL_ONLY = ["rotary_joint", "duffing_stage"]
MIXED = ["mass_spring_damper"]                          # linear model (Kalman) and chain form (HGO) both available
UNSUPPORTED = ["bearing_only", "cart_pole"]


def _u(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _r(x, k=3):
    """Round to k significant digits (so that stated numbers are exactly the model numbers)."""
    if x == 0:
        return 0.0
    return float(f"{x:.{k}g}")


def _sig(name, unit):
    return {"name": name, "unit": unit}


def servo_axis(rng):
    b = _r(_u(rng, 0.5, 20.0))          # input gain [rad/s^2 per N m] = 1/J
    plant = {"family": "servo_axis", "kind": "second_order",
             "states": [_sig("shaft angle", "rad"), _sig("shaft speed", "rad/s")], "inputs": [_sig("motor torque", "N m")],
             "outputs": [_sig("encoder angle", "rad")],
             "linear": {"clock": "continuous", "A": [[0.0, 1.0], [0.0, 0.0]], "B": [[0.0], [b]], "C": [[1.0, 0.0]], "Ts_native": None},
             "second_order": {"f": "b*u", "f0": "b*u", "params": {"b": b}, "param_ranges": {}, "output": "x1"}, "flags": []}
    Ts = float(rng.choice([0.0005, 0.001, 0.002, 0.005]))
    V = _r(_u(rng, 2e-5, 5e-4)); D = _r(_u(rng, 0.5, 20.0))
    return {"plant": plant, "scales": {"Ts": Ts, "V": V, "D": D, "E": [_r(_u(rng, 1e-3, 2e-2)), _r(_u(rng, 0.2, 5.0))], "u_max": 1.0},
            "truth": {"b": b}}


def gantry_axis(rng):
    m = _r(_u(rng, 2.0, 40.0))
    b = _r(1.0 / m)
    plant = {"family": "gantry_axis", "kind": "second_order",
             "states": [_sig("carriage position", "m"), _sig("carriage velocity", "m/s")], "inputs": [_sig("drive force", "N")],
             "outputs": [_sig("linear scale reading", "m")],
             "linear": {"clock": "continuous", "A": [[0.0, 1.0], [0.0, 0.0]], "B": [[0.0], [b]], "C": [[1.0, 0.0]], "Ts_native": None},
             "second_order": {"f": "b*u", "f0": "b*u", "params": {"b": b}, "param_ranges": {}, "output": "x1"}, "flags": []}
    Ts = float(rng.choice([0.00025, 0.0005, 0.001, 0.004]))
    V = _r(_u(rng, 1e-6, 5e-5)); D = _r(_u(rng, 0.05, 3.0))
    return {"plant": plant, "scales": {"Ts": Ts, "V": V, "D": D, "E": [_r(_u(rng, 1e-4, 2e-3)), _r(_u(rng, 0.02, 0.5))], "u_max": 50.0},
            "truth": {"b": b}}


def mass_spring_damper(rng):
    m = _r(_u(rng, 0.5, 5.0)); k = _r(_u(rng, 20.0, 400.0)); c = _r(_u(rng, 0.2, 4.0))
    plant = {"family": "mass_spring_damper", "kind": "second_order",
             "states": [_sig("mass position", "m"), _sig("mass velocity", "m/s")], "inputs": [_sig("actuator force", "N")],
             "outputs": [_sig("position sensor", "m")],
             "linear": {"clock": "continuous", "A": [[0.0, 1.0], [_r(-k / m, 6), _r(-c / m, 6)]], "B": [[0.0], [_r(1.0 / m, 6)]],
                        "C": [[1.0, 0.0]], "Ts_native": None},
             "second_order": {"f": "(u - k*x1 - c*x2)/m", "f0": "(u - k*y)/m", "params": {"m": m, "k": k, "c": c},
                              "param_ranges": {"m": [m, m], "k": [k, k], "c": [c, c]}, "output": "x1"}, "flags": []}
    Ts = float(rng.choice([0.0005, 0.001, 0.002]))
    V = _r(_u(rng, 2e-6, 5e-5)); w = _r(_u(rng, 0.2, 1.0))
    return {"plant": plant, "scales": {"Ts": Ts, "V": V, "d_bound": _r(_u(rng, 0.1, 2.0)), "domain": {"x1": [-0.05, 0.05], "x2": [-w, w], "u": [-20.0, 20.0]},
                                        "E": [_r(_u(rng, 1e-4, 2e-3)), _r(_u(rng, 0.05, 0.5))], "u_max": 20.0},
            "truth": {"m": m, "k": k, "c": c}}


def rotary_joint(rng=None, nominal=False):
    """Single rotary joint with gravity load, viscous and Coulomb-like friction (synthetic parameters).
        J x2' = u - mgl sin(x1) - bv x2 - Fc tanh(x2/vs) + J d
    The observer's nominal model uses the measured angle and the applied torque only: f0 = (u - mgl sin(y)) / J."""
    if nominal or rng is None:
        J, mgl, bv, Fc = 0.05, 2.45, 0.10, 0.10
    else:
        J = _r(_u(rng, 0.02, 0.2)); mgl = _r(_u(rng, 0.5, 5.0)); bv = _r(_u(rng, 0.02, 0.3)); Fc = _r(_u(rng, 0.02, 0.3))
    vs = 0.05
    tolJ, tolm, tolf = 0.05, 0.05, 0.5
    plant = {"family": "rotary_joint", "kind": "second_order",
             "states": [_sig("joint angle", "rad"), _sig("joint velocity", "rad/s")], "inputs": [_sig("joint torque", "N m")],
             "outputs": [_sig("joint encoder", "rad")], "linear": None,
             "second_order": {"f": "(u - mgl*sin(x1) - bv*x2 - Fc*tanh(x2/vs))/J", "f0": "(u - mgl*sin(y))/J",
                              "params": {"J": J, "mgl": mgl, "bv": bv, "Fc": Fc, "vs": vs},
                              "param_ranges": {"J": [_r(J * (1 - tolJ), 6), _r(J * (1 + tolJ), 6)], "mgl": [_r(mgl * (1 - tolm), 6), _r(mgl * (1 + tolm), 6)],
                                               "bv": [_r(bv * (1 - tolf), 6), _r(bv * (1 + tolf), 6)], "Fc": [_r(Fc * (1 - tolf), 6), _r(Fc * (1 + tolf), 6)],
                                               "vs": [vs, vs]},
                              "output": "x1"}, "flags": []}
    u_max = _r(3.0 * mgl) if not nominal else 6.0
    w = 4.0
    Ts = 0.001 if (nominal or rng is None) else float(rng.choice([0.0005, 0.001, 0.002]))
    V = 1e-4 if (nominal or rng is None) else _r(_u(rng, 2e-5, 3e-4))
    return {"plant": plant, "scales": {"Ts": Ts, "V": V, "d_bound": 0.5 if nominal or rng is None else _r(_u(rng, 0.1, 2.0)),
                                        "domain": {"x1": [-3.2, 3.2], "x2": [-w, w], "u": [-u_max, u_max]},
                                        "E": [1e-3, 2.0] if nominal or rng is None else [_r(_u(rng, 1e-3, 1e-2)), _r(_u(rng, 0.5, 4.0))], "u_max": u_max},
            "truth": {"J": J, "mgl": mgl, "bv": bv, "Fc": Fc, "vs": vs}}


def duffing_stage(rng):
    """Flexure-guided positioning stage with a hardening spring (HELD OUT):  m x2' = u - k1 x1 - k3 x1^3 - c x2 + m d."""
    m = _r(_u(rng, 0.2, 2.0)); k1 = _r(_u(rng, 50.0, 500.0)); k3 = _r(_u(rng, 1e4, 1e6)); c = _r(_u(rng, 0.2, 3.0))
    plant = {"family": "duffing_stage", "kind": "second_order",
             "states": [_sig("stage displacement", "m"), _sig("stage velocity", "m/s")], "inputs": [_sig("voice-coil force", "N")],
             "outputs": [_sig("capacitive probe", "m")], "linear": None,
             "second_order": {"f": "(u - k1*x1 - k3*x1**3 - c*x2)/m", "f0": "(u - k1*y - k3*y**3)/m",
                              "params": {"m": m, "k1": k1, "k3": k3, "c": c},
                              "param_ranges": {"m": [_r(m * 0.97, 6), _r(m * 1.03, 6)], "k1": [_r(k1 * 0.95, 6), _r(k1 * 1.05, 6)],
                                               "k3": [_r(k3 * 0.9, 6), _r(k3 * 1.1, 6)], "c": [_r(c * 0.5, 6), _r(c * 1.5, 6)]},
                              "output": "x1"}, "flags": []}
    Ts = float(rng.choice([0.00025, 0.0005, 0.001]))
    V = _r(_u(rng, 2e-7, 5e-6)); w = _r(_u(rng, 0.1, 0.5))
    return {"plant": plant, "scales": {"Ts": Ts, "V": V, "d_bound": _r(_u(rng, 0.05, 1.0)), "domain": {"x1": [-0.01, 0.01], "x2": [-w, w], "u": [-10.0, 10.0]},
                                        "E": [_r(_u(rng, 1e-5, 2e-4)), _r(_u(rng, 0.02, 0.2))], "u_max": 10.0},
            "truth": {"m": m, "k1": k1, "k3": k3, "c": c}}


def _lin(family, states, inputs, outputs, Ac, Bc, C, Ts, q_psd, G, r_std, P0std):
    plant = {"family": family, "kind": "linear", "states": states, "inputs": inputs, "outputs": outputs,
             "linear": {"clock": "continuous", "A": np.asarray(Ac, float).tolist(), "B": np.asarray(Bc, float).tolist(),
                        "C": np.asarray(C, float).tolist(), "Ts_native": None},
             "second_order": None, "flags": []}
    return {"plant": plant, "scales": {"Ts": Ts, "Qc": q_psd, "G": G, "r_std": r_std, "P0_std": P0std}, "truth": {}}


def dc_motor(rng):
    J = _u(rng, 0.005, 0.05); b = _u(rng, 0.01, 0.2); Kt = _u(rng, 0.05, 0.5); Rm = _u(rng, 0.5, 5); Lm = _u(rng, 0.2, 0.8)
    Ac = np.round([[0, 1, 0], [0, -b / J, Kt / J], [0, -Kt / Lm, -Rm / Lm]], 5)
    Bc = np.round([[0], [0], [1 / Lm]], 5)
    return _lin("dc_motor", [_sig("shaft angle", "rad"), _sig("shaft speed", "rad/s"), _sig("armature current", "A")],
                [_sig("armature voltage", "V")], [_sig("shaft encoder", "rad"), _sig("current sensor", "A")],
                Ac, Bc, [[1.0, 0, 0], [0, 0, 1.0]], 0.01, [[_r(_u(rng, 1e-3, 1e-1))]], [[0.0], [1.0], [0.0]],
                [_r(_u(rng, 1e-4, 5e-3)), _r(_u(rng, 1e-3, 5e-2))], [0.05, 1.0, 0.2])


def thermal(rng):
    a1, a2, a12 = _u(rng, 0.02, 0.1), _u(rng, 0.02, 0.1), _u(rng, 0.05, 0.2); g = _u(rng, 0.05, 0.3)
    Ac = np.round([[-(a1 + a12), a12, a1], [a12, -(a2 + a12), a2], [0, 0, 0.0]], 5)
    Bc = np.round([[g], [0], [0.0]], 5)
    return _lin("thermal", [_sig("zone one temperature", "K"), _sig("zone two temperature", "K"), _sig("ambient offset", "K")],
                [_sig("heater power", "kW")], [_sig("zone one thermocouple", "K")],
                Ac, Bc, [[1.0, 0, 0]], 2.0, [[_r(_u(rng, 1e-4, 1e-2)), 0.0], [0.0, _r(_u(rng, 1e-6, 1e-4))]], [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
                [_r(_u(rng, 0.02, 0.5))], [2.0, 2.0, 5.0])


def two_mass(rng):
    m1, m2 = _u(rng, 0.5, 2), _u(rng, 0.5, 2); k = _u(rng, 5, 50); c = _u(rng, 0.1, 1)
    Ac = np.round([[0, 1, 0, 0], [-k / m1, -c / m1, k / m1, c / m1], [0, 0, 0, 1], [k / m2, c / m2, -k / m2, -c / m2]], 5)
    Bc = np.round([[0], [1 / m1], [0], [0.0]], 5)
    return _lin("two_mass", [_sig("motor position", "m"), _sig("motor velocity", "m/s"), _sig("load position", "m"), _sig("load velocity", "m/s")],
                [_sig("motor force", "N")], [_sig("motor encoder", "m"), _sig("load encoder", "m")],
                Ac, Bc, [[1.0, 0, 0, 0], [0, 0, 1.0, 0]], 0.02, [[_r(_u(rng, 1e-3, 1e-1))]], [[0.0], [0.0], [0.0], [1.0]],
                [_r(_u(rng, 1e-4, 2e-3)), _r(_u(rng, 1e-4, 2e-3))], [0.01, 0.1, 0.01, 0.1])


def hydraulic_boom(rng=None, nominal=False):
    """Single-axis hydraulic boom (valve-controlled cylinder) linearised about a hold point (HELD OUT; synthetic values).
    States: boom angle deviation [rad], angular rate [rad/s], load pressure deviation [MPa], load-torque bias [kN m]."""
    if nominal or rng is None:
        J, b, Ap, r, beta, Vt, Kq, Kc = 2.0e4, 5.0e3, 8.0e-3, 0.5, 1.2e9, 4.0e-3, 1.0e-3, 2.0e-11
    else:
        J = _u(rng, 1.0e4, 4.0e4); b = _u(rng, 2e3, 1e4); Ap = _u(rng, 5e-3, 1.2e-2); r = _u(rng, 0.35, 0.7)
        beta = _u(rng, 0.8e9, 1.5e9); Vt = _u(rng, 2e-3, 6e-3); Kq = _u(rng, 5e-4, 2e-3); Kc = _u(rng, 1e-11, 5e-11)
    MPa, kNm = 1e6, 1e3
    Ac = np.array([[0, 1, 0, 0], [0, -b / J, Ap * r * MPa / J, kNm / J],
                   [0, -(4 * beta / Vt) * Ap * r / MPa, -(4 * beta / Vt) * Kc, 0], [0, 0, 0, 0.0]])
    Bc = np.array([[0], [0], [(4 * beta / Vt) * Kq / MPa], [0.0]])
    Ac = np.array([[_r(v, 6) for v in row] for row in Ac]); Bc = np.array([[_r(v[0], 6)] for v in Bc])
    q = [[4.0, 0.0], [0.0, 25.0]] if nominal or rng is None else [[_r(_u(rng, 0.5, 10.0)), 0.0], [0.0, _r(_u(rng, 5.0, 50.0))]]
    out = _lin("hydraulic_boom", [_sig("boom angle", "rad"), _sig("boom rate", "rad/s"), _sig("load pressure", "MPa"), _sig("load torque bias", "kN m")],
               [_sig("valve command", "-")], [_sig("inclinometer", "rad"), _sig("pressure transducer", "MPa")],
               Ac, Bc, [[1.0, 0, 0, 0], [0, 0, 1.0, 0]], 0.01, q, [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
               [2e-3, 0.05] if nominal or rng is None else [_r(_u(rng, 5e-4, 5e-3)), _r(_u(rng, 0.02, 0.2))], [0.02, 0.05, 1.0, 5.0])
    return out


def bearing_only(rng):
    plant = {"family": "bearing_only", "kind": "other",
             "states": [_sig("target east position", "m"), _sig("target north position", "m"), _sig("target east velocity", "m/s"), _sig("target north velocity", "m/s")],
             "inputs": [], "outputs": [_sig("bearing angle", "rad")], "linear": None, "second_order": None,
             "flags": ["nonlinear_measurement"],
             "out_of_scope_reason": "the measurement is a nonlinear function (bearing) of the state; the register contains no nonlinear filter"}
    return {"plant": plant, "scales": {"Ts": 1.0}, "truth": {}}


def cart_pole(rng):
    plant = {"family": "cart_pole", "kind": "other",
             "states": [_sig("cart position", "m"), _sig("cart velocity", "m/s"), _sig("pole angle", "rad"), _sig("pole rate", "rad/s")],
             "inputs": [_sig("cart force", "N")], "outputs": [_sig("cart encoder", "m"), _sig("pole encoder", "rad")],
             "linear": None, "second_order": None, "flags": ["multi_output_nonlinear"],
             "out_of_scope_reason": "a nonlinear fourth-order plant with two outputs is outside the second-order single-output observer class, "
                                    "and no linear model is supplied for the Kalman filter"}
    return {"plant": plant, "scales": {"Ts": 0.005}, "truth": {}}


FAMILIES = {f.__name__: f for f in (servo_axis, gantry_axis, mass_spring_damper, rotary_joint, duffing_stage, dc_motor, thermal, two_mass,
                                    hydraulic_boom, bearing_only, cart_pole)}


def make(family: str, rng):
    return FAMILIES[family](rng)
