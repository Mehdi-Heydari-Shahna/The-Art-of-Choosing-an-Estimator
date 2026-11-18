"""Reproduce the seven review findings through the public interface and record the outcome of the code under test.

    python experiments/review_findings.py --label current            # this checkout
    PYTHONPATH=<checkout of the previous revision> python experiments/review_findings.py --label previous

Only the public interface is used: heros.api (new_spec, decide, solve) and the command line (python -m heros solve).
The same script runs unchanged against the previous revision (commit dbc415a, the delivered version before revision 2)
and the current one; features that the previous revision does not have (a timestamp argument) are detected, not assumed.
Results go to results/review_findings_<label>.json; experiments/review_findings_compare.py tabulates both runs.

The review that accompanied the request was not attached, so the inputs below are reconstructed from the request text,
one or more cases per finding (F1 to F7).  The script takes further cases in the same format with --cases FILE.
"""
from __future__ import annotations

import argparse
import copy
import csv
import inspect
import io
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

try:
    from heros import api                      # the checkout on PYTHONPATH (e.g. the previous revision)
except ModuleNotFoundError:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from heros import api

CODE_ROOT = pathlib.Path(api.__file__).resolve().parents[1]     # checkout whose code is under test
HERE = pathlib.Path(__file__).resolve().parents[1]               # this checkout (data, examples, output)


# ----------------------------------------------------------------------------------------------------------------------
# plants and helpers
# ----------------------------------------------------------------------------------------------------------------------
def servo_plant(b=4.0, param_ranges=None, f=None):
    return {"family": "servo_axis", "kind": "second_order",
            "states": [{"name": "shaft angle", "unit": "rad"}, {"name": "shaft speed", "unit": "rad/s"}],
            "inputs": [{"name": "motor torque", "unit": "N m"}], "outputs": [{"name": "encoder angle", "unit": "rad"}],
            "linear": {"clock": "continuous", "A": [[0.0, 1.0], [0.0, 0.0]], "B": [[0.0], [b]], "C": [[1.0, 0.0]], "Ts_native": None},
            "second_order": {"f": f or "b*u", "f0": "b*u", "params": {"b": b}, "param_ranges": param_ranges or {}, "output": "x1"},
            "flags": []}


def random_walk_plant():
    return {"family": "random_walk", "kind": "linear", "states": [{"name": "level", "unit": "m"}], "inputs": [{"name": "none", "unit": "-"}],
            "outputs": [{"name": "level gauge", "unit": "m"}],
            "linear": {"clock": "discrete", "A": [[1.0]], "B": [[0.0]], "C": [[1.0]], "Ts_native": 1.0}, "second_order": None, "flags": []}


def brief(dec):
    """The decision as a user sees it: disposition, method, claim value, first explanation."""
    out = {"disposition": dec["disposition"], "selected": dec.get("selected")}
    if dec.get("display"):
        out["display"] = dec["display"]
    if dec.get("claim"):
        out["claim"] = dec["claim"]["statement"]
        out["claim_value"] = dec["claim"].get("value")
    if dec.get("question"):
        out["question"] = dec["question"].get("question")
    if dec["disposition"] not in ("RELEASE", "CLARIFY"):
        out["explanation"] = (dec.get("explanations") or [""])[0]
    if dec.get("impossibility"):
        out["impossibility"] = dec["impossibility"].get("argument")
    return out


def has_time_argument():
    return "t" in inspect.signature(api.solve).parameters


def cli_solve(spec_obj, data_text):
    """python -m heros solve --spec S --data D --out R   (the code under test, in a fresh directory)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / "spec.json").write_text(json.dumps(spec_obj))
        (tmp / "data.csv").write_text(data_text)
        env = dict(os.environ, PYTHONPATH=str(CODE_ROOT))
        p = subprocess.run([sys.executable, "-m", "heros", "solve", "--spec", str(tmp / "spec.json"), "--data", str(tmp / "data.csv"),
                            "--out", str(tmp / "record.json")], cwd=tmp, env=env, capture_output=True, text=True)
        rec = json.loads((tmp / "record.json").read_text()) if (tmp / "record.json").exists() else None
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    first = lines[0] if lines else ""
    est = [l for l in lines if l.startswith("Estimates:")]
    return {"returncode": p.returncode, "first_line": first, "estimates_line": est[0] if est else None,
            "n_estimates": (len((rec.get("outputs") or {}).get("t", [])) if rec else None),
            "record_has_timestamps": bool(rec and (rec.get("inputs") or {}).get("t") is not None),
            "stderr_tail": p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ""}


# ----------------------------------------------------------------------------------------------------------------------
# F1  stated period versus timestamps of the data
# ----------------------------------------------------------------------------------------------------------------------
def finding_1():
    ex = json.loads((HERE / "examples" / "servo_axis_spec.json").read_text())
    tol, dl = ex.pop("_hard_tolerance"), ex.pop("_deadline")
    for k in ("Q", "Q_kind", "R", "P0", "distribution", "white"):
        ex.pop(k, None)
    ex["objective"] = {"kind": "hard_bound", "tolerance": tol, "deadline": dl}
    rows = list(csv.DictReader(io.StringIO((HERE / "examples" / "servo_axis_data.csv").read_text())))

    def text(rws, dt_scale=1.0, drop=None):
        buf = io.StringIO()
        buf.write("t,y,u\n")
        for i, r in enumerate(rws):
            if drop is not None and i == drop:
                continue
            buf.write(f"{float(r['t']) * dt_scale:.6f},{r['y']},{r['u']}\n")
        return buf.getvalue()
    cases = {
        "timestamps agree with the stated 1 ms period": text(rows),
        "data recorded every 2 ms (every second sample kept), request states 1 ms": text(rows[::2]),
        "one sample missing (a 2 ms gap at t = 0.5 s)": text(rows, drop=500),
    }
    out = []
    for name, data in cases.items():
        r = cli_solve(ex, data)
        out.append({"case": name, "command": "python -m heros solve --spec servo.json --data data.csv", "result": r})
    return {"id": "F1", "title": "A fixed-period certificate is released only when the stated period and the timestamps agree",
            "cases": out}


# ----------------------------------------------------------------------------------------------------------------------
# F2  every sample used to establish or disprove a requirement lies in the requested window
# ----------------------------------------------------------------------------------------------------------------------
def finding_2():
    out = []
    # (a) Kalman filter, record of 3.5 s sampled every 1 s: the samples are t = 0, 1, 2, 3 s.  The optimal error variance
    # rises from P0 = 0.1 towards its limit; the requested level lies between the values at t = 3 s and t = 4 s.
    rw = dict(target=0, Ts=1.0, Q=[[1.0]], Q_kind="discrete_cov", R=[[1.0]], P0=[[0.1]], x0_hat=[0.0], distribution="gaussian", white=True, t_end=3.5)
    P = 0.1
    pf = []
    for k in range(6):
        Pf = P / (P + 1.0); pf.append(Pf); P = Pf + 1.0
    target = round(0.5 * (pf[3] + pf[4]), 6)
    s = api.new_spec(random_walk_plant(), objective={"kind": "mse_bound", "horizon": "finite", "mse_target": target}, **rw)
    out.append({"case": f"Kalman filter, record ends at t = 3.5 s (samples t = 0..3 s), mean-square-error level {target}",
                "independent": {"filtered_variance_by_sample": [round(v, 6) for v in pf[:5]], "max_over_t_0_to_3": round(max(pf[:4]), 6),
                                "value_at_t_4_outside_window": round(pf[4], 6)},
                "result": brief(api.decide(s))})
    # (b) high-gain observer, requirement only on the recorded window [0.05 s, 0.06 s]
    ex = json.loads((HERE / "examples" / "servo_axis_spec.json").read_text())
    for k in ("plant", "Q", "Q_kind", "R", "P0", "distribution", "white", "_hard_tolerance", "_deadline"):
        ex.pop(k, None)
    for tol in (0.075,):
        s = api.new_spec(servo_plant(), objective={"kind": "hard_bound", "tolerance": tol, "deadline": 0.05}, t_end=0.06, **ex)
        out.append({"case": f"observer, velocity error <= {tol} rad/s on the record from 0.05 s to its end at 0.06 s",
                    "result": brief(api.decide(s))})
    # (c) two-signal limit: the record ends between two samples, before any sample at or after 4 sqrt(V/D)
    V, D = 1e-4, 6.0
    tstar = 4 * math.sqrt(V / D)
    s = api.new_spec(servo_plant(), objective={"kind": "hard_bound", "tolerance": 0.04, "deadline": 0.01},
                     target=1, Ts=0.01, V=V, D=D, E=[0.002, 3.0], t_end=0.0165)
    out.append({"case": f"100 Hz encoder, requirement on t in [0.01 s, 0.0165 s]; the two-signal argument needs a sample at or after {tstar:.5f} s",
                "result": brief(api.decide(s))})
    return {"id": "F2", "title": "Samples used to establish or disprove a requirement lie inside the requested window", "cases": out}


# ----------------------------------------------------------------------------------------------------------------------
# F3  Kalman filter: estimate, exported covariance and explanation agree at every sample, including start-up
# ----------------------------------------------------------------------------------------------------------------------
def finding_3():
    spec = api.new_spec(random_walk_plant(), objective={"kind": "mse_optimal", "horizon": "steady_state"},
                        target=0, Ts=1.0, Q=[[1.0]], Q_kind="discrete_cov", R=[[1.0]], P0=[[100.0]], x0_hat=[0.0], distribution="gaussian", white=True)
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 10.0); ys = []
    for k in range(20):
        ys.append(x + rng.normal()); x = x + rng.normal()
    res = api.solve(spec, np.array(ys))
    est = res["estimates"]
    M = (math.sqrt(5.0) - 1.0) / 2.0                     # steady-state gain of the scalar random walk with Q = R = 1
    first = (1 - M) ** 2 * 100.0 + M ** 2 * 1.0          # variance of x0 - M (x0 + v0) with var x0 = 100, var v0 = 1
    # Monte Carlo of the released estimator at k = 0 (independent of the package)
    g = float(np.asarray(est["gain"])[0].reshape(-1)[0])
    x0 = rng.normal(0.0, 10.0, 400000); v0 = rng.normal(0.0, 1.0, 400000)
    emp = float(np.var(x0 - (0.0 + g * (x0 + v0 - 0.0))))
    return {"id": "F3", "title": "Kalman filter start-up: estimate, exported covariance and explanation agree",
            "cases": [{"case": "scalar random walk, Q = R = 1, P0 = 100, constant (steady-state) gain",
                       "independent": {"first_filtered_variance_closed_form": round(first, 6), "monte_carlo_400k": round(emp, 4)},
                       "result": dict(brief(res["decision"]), exported_P_filt_first=float(np.asarray(est["P_filt"])[0].reshape(-1)[0]),
                                      exported_P_filt_last=float(np.asarray(est["P_filt"])[-1].reshape(-1)[0]),
                                      gain_first=g, uncertainty_type=est.get("uncertainty_type"),
                                      covariance_basis=est.get("covariance_basis"))}]}


# ----------------------------------------------------------------------------------------------------------------------
# F4  all supplied constraints together: tighter disturbance bound, nominal versus true parameters
# ----------------------------------------------------------------------------------------------------------------------
def finding_4():
    out = []
    V = 1e-4
    s = api.new_spec(servo_plant(), objective={"kind": "hard_bound", "tolerance": 0.045}, target=1, Ts=0.001, V=V, D=6.0, d_bound=1.5, E=[0.002, 3.0])
    out.append({"case": "total disturbance bound D = 6 and, for the exactly known model, additive bound d = 1.5; ultimate velocity limit 0.045 rad/s "
                        f"(two-signal limits 2 sqrt(DV) = {2 * math.sqrt(6.0 * V):.4f}, 2 sqrt(dV) = {2 * math.sqrt(1.5 * V):.4f})",
                "result": brief(api.decide(s))})
    stoch = dict(Q=[[2.9929e-12, 5.9858e-09], [5.9858e-09, 1.19716e-05]], Q_kind="discrete_cov", R=[[3.32929e-09]],
                 P0=[[1.33333e-06, 0.0], [0.0, 3.0]], distribution="bounded", white=True)
    s = api.new_spec(servo_plant(param_ranges={"b": [3.6, 4.4]}), objective={"kind": "mse_optimal", "horizon": "steady_state"},
                     target=1, Ts=0.001, **stoch)
    out.append({"case": "smallest mean-square error of the speed; nominal input gain b = 4 but the true value lies in [3.6, 4.4]",
                "result": brief(api.decide(s))})
    return {"id": "F4", "title": "The decision uses all supplied constraints together", "cases": out}


# ----------------------------------------------------------------------------------------------------------------------
# F5  an impossibility claim needs a valid argument for the request
# ----------------------------------------------------------------------------------------------------------------------
def finding_5():
    stoch = dict(Q=[[2.9929e-12, 5.9858e-09], [5.9858e-09, 1.19716e-05]], Q_kind="discrete_cov", R=[[3.32929e-09]],
                 P0=[[1.33333e-06, 0.0], [0.0, 3.0]], distribution="bounded", white=True)
    out = []
    for r in (0.04, 0.02):
        s = api.new_spec(servo_plant(), objective={"kind": "probability", "horizon": "steady_state", "tolerance": r, "prob": 0.9},
                         target=1, Ts=0.001, V=1e-4, D=6.0, E=[0.002, 3.0], **stoch)
        out.append({"case": f"speed error within {r} rad/s with probability 0.9 (bounded noise; worst-case limit 2 sqrt(DV) = 0.049)",
                    "result": brief(api.decide(s))})
    # Monte Carlo of the steady-state Kalman filter on the same model (uniform noise), independent of the package's claim
    rng = np.random.default_rng(5)
    Ts, D, V = 0.001, 6.0, 1e-4
    from scipy.linalg import solve_discrete_are
    A = np.array([[1.0, Ts], [0.0, 1.0]]); G = np.array([[Ts * Ts / 2], [Ts]]); C = np.array([[1.0, 0.0]])
    Qd = (D * D / 3) * (G @ G.T); Rm = np.array([[V * V / 3]])
    P = solve_discrete_are(A.T, C.T, Qd, Rm); M = P @ C.T @ np.linalg.inv(C @ P @ C.T + Rm)
    n, steps = 4000, 3000
    x = np.zeros((n, 2)); xh = np.zeros((n, 2))
    for k in range(steps):
        y = x[:, 0] + rng.uniform(-V, V, n)
        xh = xh + (y - xh[:, 0])[:, None] * M[:, 0][None, :]
        e = x[:, 1] - xh[:, 1]
        a = rng.uniform(-D, D, n)
        x = np.column_stack([x[:, 0] + Ts * x[:, 1] + 0.5 * Ts * Ts * a, x[:, 1] + Ts * a]); xh = xh @ A.T
    out[-1]["independent"] = {"monte_carlo_share_within_0.02_after_3_s": float(np.mean(np.abs(e) <= 0.02)), "records": n}
    return {"id": "F5", "title": "A worst-case limit does not show that a probability requirement is impossible", "cases": out}


# ----------------------------------------------------------------------------------------------------------------------
# F6  exported worst-case envelope with initialisation from the first encoder reading
# ----------------------------------------------------------------------------------------------------------------------
def finding_6():
    ex = json.loads((HERE / "examples" / "servo_axis_spec.json").read_text())
    tol, dl = ex.pop("_hard_tolerance"), ex.pop("_deadline")
    for k in ("plant", "Q", "Q_kind", "R", "P0", "distribution", "white"):
        ex.pop(k, None)
    spec = api.new_spec(servo_plant(), objective={"kind": "hard_bound", "tolerance": tol, "deadline": dl}, **ex)
    rows = list(csv.DictReader(io.StringIO((HERE / "examples" / "servo_axis_data.csv").read_text())))
    y = np.array([float(r["y"]) for r in rows]); u = np.array([float(r["u"]) for r in rows])
    res = api.solve(spec, y, u)
    cfg, env = res["decision"]["config"], np.asarray(res["estimates"]["envelope"])
    M2, V, E2 = cfg["M"][1], cfg["V"], cfg["E_eff"][1]
    return {"id": "F6", "title": "Worst-case envelope with initialisation from the first encoder reading",
            "cases": [{"case": "servo axis, velocity estimate initialised from the first reading (position) and zero (velocity)",
                       "independent": {"exact_worst_case_at_k0": E2, "independent_box_value_at_k0": E2 + 2 * M2 * V},
                       "result": {"disposition": res["decision"]["disposition"], "exported_envelope_k0": float(env[0]),
                                  "exported_envelope_k1": float(env[1]), "exported_envelope_at_deadline": float(env[int(round(dl / cfg['Ts']))]),
                                  "claim": res["decision"]["claim"]["statement"]}}]}


# ----------------------------------------------------------------------------------------------------------------------
# F7  transient cap below the later tolerance: the 20 test requests
# ----------------------------------------------------------------------------------------------------------------------
def finding_7():
    rows = [json.loads(l) for l in (HERE / "data" / "test_structured.jsonl").read_text().splitlines() if l.strip()]
    cases = [r for r in rows if r["category"] == "invalid_cap_below_tol"]
    out, counts = [], {}
    for c in cases:
        d = api.decide(c["given"])
        counts[d["disposition"]] = counts.get(d["disposition"], 0) + 1
        out.append({"case": c["id"], "family": c["family"], "result": brief(d)})
    return {"id": "F7", "title": "Transient cap tighter than the later tolerance (20 test requests)", "counts": counts, "cases": out}


FINDINGS = [finding_1, finding_2, finding_3, finding_4, finding_5, finding_6, finding_7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="current")
    ap.add_argument("--out")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    res = {"code_root": CODE_ROOT.name, "label": a.label, "has_timestamp_argument": has_time_argument(), "findings": []}
    try:
        res["code_commit"] = subprocess.run(["git", "-C", str(CODE_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa
        res["code_commit"] = None
    for f in FINDINGS:
        if a.only and f.__name__.split("_")[1] not in a.only:
            continue
        r = f()
        res["findings"].append(r)
        print(json.dumps(r, indent=1, default=str)[:4000])
    out = pathlib.Path(a.out) if a.out else HERE / "results" / f"review_findings_{a.label}.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    print("written", out)


if __name__ == "__main__":
    main()
