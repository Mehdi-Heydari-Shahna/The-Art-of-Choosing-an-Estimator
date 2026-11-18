"""Reproduce the documented defects of the SUPPLIED packages (development baselines) and save the observed outcomes.

Usage:  python reproduce_legacy_defects.py /path/to/src   (directory that contains heros_kalman/ and heros_khalil/)
Writes  legacy_defects.json  next to this file.  The corrected behaviour of the new public path is asserted in
tests/test_regressions.py; this script only documents what the legacy code does.
"""
import json, math, sys, traceback, pathlib
import numpy as np

SRC = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "../../src").resolve()
out = {}

def attempt(key, fn):
    try:
        out[key] = fn()
    except Exception as ex:          # noqa
        out[key] = {"exception": type(ex).__name__, "message": str(ex)[:300], "trace_tail": traceback.format_exc().splitlines()[-3:]}

# ------------------------------------------------------------------ Khalil package
sys.path.insert(0, str(SRC / "heros_khalil" / "repo" / "src")); sys.path.insert(0, str(SRC / "heros_khalil" / "repo"))
from hgo_agent.decision import FieldValue, Request, decide          # noqa
from hgo_agent.plants import double_integrator, rotary_joint, PlantModel, analyze   # noqa

def R(**kw):
    flags = kw.pop("flags", {}); sc = kw.pop("scenarios", None); pat = kw.pop("pattern", "repeated")
    gmin = kw.pop("g_min", None); gmax = kw.pop("g_max", None)
    f = {k: (FieldValue("range", lo=v[0], hi=v[1]) if isinstance(v, tuple) else FieldValue("value", value=v)) for k, v in kw.items()}
    return Request(fields=f, flags=flags, scenarios=sc, pattern=pat, g_min=gmin, g_max=gmax)

DI = double_integrator()

def k1():
    d = decide(DI, R(D=0.5, V=0.0, E1=0.0, E2=2.0, rho=1.0, g_min=0.5, g_max=1.0))
    des = d.get("design") or {}
    g = des.get("g"); t0 = des.get("earliest_certified_time")
    w = lambda t: 1 + (1 + 1.5 * t) * math.exp(-t)         # admissible error for d = 0.5, e2(0) = 2, g = 1
    return {"decision": d.get("decision"), "gain": g, "earliest_certified_time": t0,
            "admissible_error_at_t": {str(t): w(t) for t in (0.0, 1.0, 10.0, 30.0)}, "tolerance": 1.0,
            "defect": "a finite-time claim is displayed although the admissible error exceeds the tolerance at every finite time"}
attempt("khalil_timing_claim", k1)

def k2():
    d0 = decide(DI, R(D=0.0, V=0.0, rho=0.5, g_max=5.0))
    d1 = decide(DI, R(D=0.0, V=0.0, rho=0.5, g_max=5.0, scenarios=[{"T": 1.0}]))
    g = (d1.get("design") or {}).get("g")
    err = None
    if g:
        tau = g * 1.0
        err = (1 + tau) * math.exp(-tau) * 100.0             # e2(1 s) for e2(0) = 100, e1(0) = 0, no disturbance
    return {"without_scenario": d0.get("decision"), "with_deadline_scenario": d1.get("decision"), "gain": g,
            "question_asked": (d1.get("question") or {}).get("field"),
            "error_at_deadline_for_initial_error_100": err, "tolerance": 0.5,
            "defect": "the deadline introduced by the scenario is not followed by a re-evaluation of the missing initial-velocity bound"}
attempt("khalil_scenario_deadline", k2)

def k3():
    d = decide(DI, R(D=1.0, V=0.01, E1=0.001, E2=1.0, rho=2.0, T=1.0))
    des = d.get("design") or {}
    return {"decision": d.get("decision"), "E1_supplied": 0.001, "V": 0.01, "initialisation": des.get("initialisation") or des.get("init"),
            "params_E1": (d.get("params") or des.get("params") or {}).get("E1"),
            "e1_from_measurement": (d.get("params") or des.get("params") or {}).get("e1_from_measurement"),
            "keys": sorted(des.keys())[:30]}
attempt("khalil_initialisation_premise", k3)

def k4():
    d = decide(DI, R(D=1.0, V=0.001, E1=0.0, E2=1.0, rho=2.0, T=1.0, sample_period=(0.001, 0.002)))
    return {"decision": d.get("decision")}
attempt("khalil_sample_period_range", k4)

def k5():
    pm = PlantModel(name="feedthrough", states=["x1", "x2"], input="u", dynamics=["x2", "u"], output="x1 + 0.1*u", base_unit="rad")
    a = analyze(pm)
    return {"supported": a.get("supported"), "reason": a.get("reason"), "code": a.get("code"), "keys": sorted(a.keys())[:20]}
attempt("khalil_feedthrough_output", k5)

# ------------------------------------------------------------------ Kalman package
sys.path.insert(0, str(SRC / "heros_kalman"))
def ka1():
    from kalman_agent import agent as AG, verifier as V
    from kalman_agent.schema import Spec
    from kalman_agent.plant import library as PL
    rng = np.random.default_rng(0)
    plant, meta = PL.generic(rng)                       # discrete model with native sample time 0.1 s
    n, m = plant.n, plant.m
    spec = Spec("lqr", "policy", 10, 0.05, np.ones(n), np.ones(m), np.ones(n))
    req = AG.spec_to_request(spec, plant)
    dec = V.verify(req)
    return {"dt_native": plant.dt_native, "dt_requested": 0.05, "disposition": dec.disposition, "category": dec.category,
            "issue": str(getattr(dec, "issue", ""))[:200],
            "defect_if_executed": "the native 0.1 s matrices are used although the request states 0.05 s"}
attempt("kalman_timing_conflict", ka1)

def ka2():
    from kalman_agent import agent as AG, verifier as V
    from kalman_agent.schema import Spec, UNRESOLVED
    from kalman_agent.plant import library as PL
    rng = np.random.default_rng(0)
    plant, meta = PL.double_integrator(rng)
    spec = Spec("lqr", "policy", 20, 0.05, np.array([10.0, 1.0]), np.array([5.5]), np.array([10.0, 1.0]))
    req = AG.spec_to_request(spec, plant)
    req.set("r[0]", None, UNRESOLVED, "declared interval", candidates=("interval", 5.0, 6.0))
    dec = V.verify(req)
    w = (dec.question or {}).get("witness") or {}
    return {"disposition": dec.disposition, "declared_interval": [5.0, 6.0], "witness_names": w.get("theta_names"),
            "witness_a": w.get("theta_a"), "witness_b": w.get("theta_b"),
            "defect": "returned ambiguity witnesses lie outside the declared admissible interval"}
attempt("kalman_inadmissible_witness", ka2)

pathlib.Path(__file__).with_name("legacy_defects.json").write_text(json.dumps(out, indent=1, default=str))
for k, v in out.items():
    print(k, "->", json.dumps(v, default=str)[:420])
