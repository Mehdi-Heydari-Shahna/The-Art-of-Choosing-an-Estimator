"""Template wording families (written by the developer) for training and development requests.

A request answers the three opening questions:
    Q1  What do you want to estimate?
    Q2  What do you measure, and what model and uncertainty information do you have?
    Q3  What does successful estimation mean?
Styles A, B, C are used for training, style D for the development set.  The acceptance test uses requests written
independently by other authors (see data/README.md); no test wording is generated from this file.
Each realised request records the character span and role of every stated quantity (the supervision of the reader).
"""
from __future__ import annotations

import math

import numpy as np

from .quantities import fmt_quantity

ROLES = ["other", "Ts", "fs", "meas_bound", "meas_std", "meas_var", "acc_bound", "acc_std", "init_pos_bound", "init_vel_bound",
         "init_pos_std", "init_vel_std", "vel_limit", "tolerance", "deadline", "peak_cap", "prob", "mse_target", "rms_target", "window"]
OBJECTIVES = ["unspecified", "hard_bound", "mse_optimal", "mse_bound", "probability"]
DISTRIBUTIONS = ["unstated", "gaussian", "bounded"]
HORIZONS = ["unstated", "steady_state", "finite"]
EVIDENCE = ["unstated", "mechanical_limit", "assumed"]
FLAGS = ["hidden_dynamics", "correlated_noise", "coloured_noise", "irregular_sampling"]

# role -> dimension used when the value is displayed
ROLE_DIM = {"Ts": "time", "fs": "freq", "meas_bound": "pos", "meas_std": "pos", "meas_var": "pos2", "acc_bound": "acc", "acc_std": "acc",
            "init_pos_bound": "pos", "init_vel_bound": "vel", "init_pos_std": "pos", "init_vel_std": "vel", "vel_limit": "vel",
            "tolerance": "vel", "deadline": "time", "peak_cap": "vel", "prob": "frac", "mse_target": "vel2", "rms_target": "vel", "window": "time"}

P = {}
P["A"] = {
    "Ts": ["The observer is executed with a sample period of {Ts}.", "Measurements arrive every {Ts}.", "The sampling interval is {Ts}."],
    "fs": ["The controller samples the sensor at {fs}.", "The sampling rate is {fs}.", "Data are acquired at a rate of {fs}."],
    "meas_bound": ["The measurement error never exceeds {meas_bound} in magnitude.", "The {sensor} error is bounded by {meas_bound}.",
                   "The worst-case error of the {sensor} is {meas_bound}."],
    "meas_std": ["The {sensor} noise has a standard deviation of {meas_std}.", "The measurement noise is zero-mean with standard deviation {meas_std}.",
                 "The RMS value of the {sensor} noise is {meas_std}."],
    "meas_var": ["The measurement-noise variance is {meas_var}.", "The {sensor} noise has variance {meas_var} per sample."],
    "acc_bound": ["Unmodelled accelerations are bounded by {acc_bound}.", "The disturbance acceleration does not exceed {acc_bound}.",
                  "The magnitude of the unknown acceleration is at most {acc_bound}."],
    "acc_std": ["The unmodelled acceleration is a white sequence with standard deviation {acc_std}.",
                "The disturbance acceleration has standard deviation {acc_std} and is held over each sample."],
    "init_pos_bound": ["The initial {x1} estimate is accurate to within {init_pos_bound}.", "The initial {x1} error is bounded by {init_pos_bound}."],
    "init_vel_bound": ["The initial {x2} is known to within {init_vel_bound}.", "The initial {x2} error is at most {init_vel_bound}."],
    "init_pos_std": ["The initial {x1} uncertainty has standard deviation {init_pos_std}."],
    "init_vel_std": ["The initial {x2} uncertainty has standard deviation {init_vel_std}."],
    "vel_limit": ["The {x2} magnitude does not exceed {vel_limit}.", "Operation is restricted to {x2} magnitudes below {vel_limit}."],
    "tolerance": ["The {target} estimation error must not exceed {tolerance}.", "The error of the {target} estimate shall stay within {tolerance}."],
    "deadline": ["This must hold from {deadline} after start-up onward.", "The requirement applies after a settling time of {deadline}."],
    "peak_cap": ["During the transient the error may not exceed {peak_cap}.", "The transient estimation error is capped at {peak_cap}."],
    "prob": ["The requirement must be met with probability {prob}.", "A confidence level of {prob} is required."],
    "mse_target": ["The mean-square error of the {target} estimate must be below {mse_target}."],
    "rms_target": ["The RMS error of the {target} estimate must be below {rms_target}."],
    "window": ["The record of interest lasts {window}."],
    "obj:hard_bound": ["A guaranteed error limit is required for every admissible disturbance and noise realisation.",
                       "The limit must hold in the worst case, not on average."],
    "obj:mse_optimal": ["The estimate with the smallest mean-square error is required.", "A minimum-variance estimate is requested."],
    "obj:mse_bound": ["A mean-square accuracy requirement applies."],
    "obj:probability": ["A probabilistic accuracy requirement applies."],
    "obj:unspecified": ["", "The estimate should be as good as possible."],
    "dist:gaussian": ["All noises are Gaussian.", "The noises and the initial error are jointly Gaussian."],
    "dist:bounded": ["The noises are bounded and uniformly distributed, with the stated means and variances.",
                     "Only the means and variances of the noises are known; they are not Gaussian."],
    "hor:steady_state": ["Only the steady-state behaviour matters.", "A constant-gain (steady-state) estimator is acceptable."],
    "hor:finite": ["The transient from the initial uncertainty matters as well.", "The time-varying optimal gain is wanted."],
    "ev:mechanical_limit": ["A hardware limiter enforces this range.", "The drive electronics enforce this limit."],
    "ev:assumed": ["We assume the range is respected and accept that the guarantee is conditional on it."],
    "flag:hidden_dynamics": ["The actuator has its own dynamics, which are not included in the model.",
                             "There is an unmodelled lag between the commanded and the applied input."],
    "flag:correlated_noise": ["The process noise and the measurement noise are correlated with each other."],
    "flag:coloured_noise": ["The measurement noise is strongly correlated in time (it is not white)."],
    "flag:irregular_sampling": ["Samples arrive at irregular times and some are missing."],
    "q1": ["The quantity to be estimated is the {target}.", "An estimate of the {target} is required."],
    "model": ["The model is attached.", "The plant model is given in the attached file."],
    "sensor": ["The measured signal is the {sensor}.", "Only the {sensor} is available."],
    "other": ["The {sensor} has {n} bits of resolution.", "The mechanism was commissioned in {year}.", "The supply voltage is {volt} V."],
}
P["B"] = {
    "Ts": ["Ts = {Ts}", "sample time {Ts}", "update period: {Ts}"],
    "fs": ["fs = {fs}", "sampling: {fs}", "loop rate {fs}"],
    "meas_bound": ["{sensor} error <= {meas_bound} (hard)", "max {sensor} error {meas_bound}", "|v| <= {meas_bound}"],
    "meas_std": ["{sensor} noise sigma {meas_std}", "noise std {meas_std}", "{sensor} noise {meas_std} rms"],
    "meas_var": ["noise variance {meas_var}", "R = {meas_var}"],
    "acc_bound": ["disturbance accel <= {acc_bound}", "|d| <= {acc_bound}", "unmodelled accel max {acc_bound}"],
    "acc_std": ["accel disturbance sigma {acc_std}, white, piecewise constant", "process noise: accel std {acc_std} per step"],
    "init_pos_bound": ["initial {x1} error <= {init_pos_bound}", "{x1}(0) known to {init_pos_bound}"],
    "init_vel_bound": ["initial {x2} error <= {init_vel_bound}", "{x2}(0) within {init_vel_bound}"],
    "init_pos_std": ["initial {x1} std {init_pos_std}"],
    "init_vel_std": ["initial {x2} std {init_vel_std}"],
    "vel_limit": ["|{x2}| <= {vel_limit}", "{x2} range +/- {vel_limit}"],
    "tolerance": ["{target} error <= {tolerance}", "need |error| <= {tolerance} on {target}", "accuracy: {tolerance}"],
    "deadline": ["after {deadline}", "from t = {deadline} on", "settling allowed: {deadline}"],
    "peak_cap": ["transient error cap {peak_cap}", "never above {peak_cap} during transient"],
    "prob": ["probability {prob}", "confidence {prob}"],
    "mse_target": ["MSE <= {mse_target}", "mean-square error under {mse_target}"],
    "rms_target": ["RMS error <= {rms_target}"],
    "window": ["run length {window}"],
    "obj:hard_bound": ["guaranteed bound, worst case", "hard guarantee needed"],
    "obj:mse_optimal": ["minimum MSE estimate", "want min-variance estimate"],
    "obj:mse_bound": ["MSE spec"], "obj:probability": ["probabilistic spec"],
    "obj:unspecified": ["", "best possible accuracy"],
    "dist:gaussian": ["Gaussian noise", "all noises Gaussian"],
    "dist:bounded": ["noise bounded/uniform, known variance", "non-Gaussian, second moments known"],
    "hor:steady_state": ["steady state only", "constant gain OK"], "hor:finite": ["include transient", "time-varying gain"],
    "ev:mechanical_limit": ["enforced by hardware limiter", "(mechanical stop / drive limit)"],
    "ev:assumed": ["(assumed, guarantee conditional)"],
    "flag:hidden_dynamics": ["NB actuator dynamics not modelled", "unmodelled actuator lag present"],
    "flag:correlated_noise": ["process and measurement noise correlated"], "flag:coloured_noise": ["sensor noise coloured (not white)"],
    "flag:irregular_sampling": ["irregular sampling, dropouts"],
    "q1": ["estimate: {target}", "{target}", "target = {target}"],
    "model": ["model: attached", "see attached model"],
    "sensor": ["measured: {sensor}", "sensor: {sensor}"],
    "other": ["{n}-bit {sensor}", "rig built {year}", "bus voltage {volt} V"],
}
P["C"] = {
    "Ts": ["we run the loop every {Ts}", "the thing updates once per {Ts}", "our cycle time is {Ts}"],
    "fs": ["we sample at {fs}", "the loop runs at {fs}", "it's a {fs} system"],
    "meas_bound": ["the {sensor} is never off by more than {meas_bound}", "I can promise the {sensor} error stays under {meas_bound}",
                   "worst case the {sensor} is wrong by {meas_bound}"],
    "meas_std": ["the {sensor} noise is about {meas_std} RMS", "I see roughly {meas_std} of noise (one sigma) on the {sensor}",
                 "typical {sensor} noise is {meas_std} standard deviation"],
    "meas_var": ["the noise variance on the {sensor} came out as {meas_var}"],
    "acc_bound": ["disturbances can push it with at most {acc_bound}", "whatever we haven't modelled is below {acc_bound} of acceleration",
                  "the unknown acceleration can't be more than {acc_bound}"],
    "acc_std": ["the random acceleration disturbance is about {acc_std} one-sigma, new value every sample",
                "process noise looks like white acceleration with {acc_std} standard deviation"],
    "init_pos_bound": ["at start we know the {x1} to {init_pos_bound}", "the starting {x1} is good to {init_pos_bound}"],
    "init_vel_bound": ["at start the {x2} could be off by up to {init_vel_bound}", "we only know the starting {x2} to within {init_vel_bound}"],
    "init_pos_std": ["our starting {x1} guess has a sigma of {init_pos_std}"],
    "init_vel_std": ["our starting {x2} guess has a sigma of {init_vel_std}"],
    "vel_limit": ["it never moves faster than {vel_limit}", "the {x2} stays below {vel_limit}"],
    "tolerance": ["I need the {target} to within {tolerance}", "the {target} estimate has to be good to {tolerance}",
                  "anything worse than {tolerance} on the {target} is useless to us"],
    "deadline": ["once {deadline} have passed", "after the first {deadline}", "we can wait {deadline} for it to settle"],
    "peak_cap": ["and it must never be off by more than {peak_cap} even at the start", "even while settling, no more than {peak_cap} error"],
    "prob": ["{prob} of the time is enough", "with {prob} probability"],
    "mse_target": ["a mean-square error below {mse_target} would do"],
    "rms_target": ["an RMS error below {rms_target} would do"],
    "window": ["a typical run is {window} long"],
    "obj:hard_bound": ["and I need that guaranteed, not just on average", "this has to hold no matter what the noise does"],
    "obj:mse_optimal": ["just give me the best estimate in the mean-square sense", "I want the statistically optimal estimate"],
    "obj:mse_bound": ["on average is fine"], "obj:probability": ["it doesn't have to hold always"],
    "obj:unspecified": ["", "make it as accurate as you can"],
    "dist:gaussian": ["the noise looks Gaussian", "everything is Gaussian as far as we can tell"],
    "dist:bounded": ["the noise is not Gaussian, more like uniform between hard limits", "we only trust the variances, not the bell curve"],
    "hor:steady_state": ["we only care once it has settled", "a fixed gain is fine"],
    "hor:finite": ["the start-up phase matters too", "we want the proper time-varying filter"],
    "ev:mechanical_limit": ["there is a hardware limiter that makes sure of that", "the drive physically cannot exceed it"],
    "ev:assumed": ["I'll take that as an assumption, on my head be it"],
    "flag:hidden_dynamics": ["by the way the actuator is not instantaneous and that lag is not in the model",
                             "note the valve/drive dynamics are missing from the model"],
    "flag:correlated_noise": ["one caveat: the process and sensor noise are correlated"],
    "flag:coloured_noise": ["the sensor noise drifts slowly, it is clearly not white"],
    "flag:irregular_sampling": ["samples come in irregularly and we lose some"],
    "q1": ["I'd like to know the {target}", "what I'm after is the {target}", "we need the {target}"],
    "model": ["the model is in the attachment", "I've attached our model"],
    "sensor": ["all we measure is the {sensor}", "we have the {sensor} and nothing else"],
    "other": ["the {sensor} is a {n}-bit unit", "the rig dates from {year}", "it runs off {volt} V"],
}
P["D"] = {   # development wording family
    "Ts": ["Estimator step: {Ts}.", "One update per {Ts}.", "Period of the digital loop: {Ts}."],
    "fs": ["Acquisition frequency: {fs}.", "The digital loop executes at {fs}.", "Update frequency {fs}."],
    "meas_bound": ["Absolute accuracy of the {sensor}: {meas_bound} (never exceeded).", "The {sensor} reading deviates by no more than {meas_bound}.",
                   "Hard limit on the {sensor} error: {meas_bound}."],
    "meas_std": ["Noise of the {sensor}: {meas_std} (1-sigma).", "The {sensor} shows {meas_std} RMS noise.", "Std. dev. of the reading noise: {meas_std}."],
    "meas_var": ["Variance of the reading noise: {meas_var}.", "Sensor-noise variance {meas_var}."],
    "acc_bound": ["Unknown acceleration: no more than {acc_bound}.", "Load disturbances stay below {acc_bound}.", "Bound on unmodelled acceleration: {acc_bound}."],
    "acc_std": ["Random acceleration, white, std. dev. {acc_std}, constant within a step.", "Acceleration noise {acc_std} (1-sigma) per update."],
    "init_pos_bound": ["Start-up {x1} uncertainty: at most {init_pos_bound}.", "The {x1} at power-on is trusted to {init_pos_bound}."],
    "init_vel_bound": ["Start-up {x2} uncertainty: at most {init_vel_bound}.", "The {x2} at power-on may differ from our guess by {init_vel_bound}."],
    "init_pos_std": ["Std. dev. of the start-up {x1} guess: {init_pos_std}."],
    "init_vel_std": ["Std. dev. of the start-up {x2} guess: {init_vel_std}."],
    "vel_limit": ["Maximum {x2}: {vel_limit}.", "The {x2} is confined to plus/minus {vel_limit}."],
    "tolerance": ["Required accuracy of the {target}: {tolerance}.", "Allowed {target} error: {tolerance}.", "The {target} must be reconstructed to {tolerance}."],
    "deadline": ["Valid {deadline} after switch-on.", "Convergence time allowed: {deadline}.", "From {deadline} onward."],
    "peak_cap": ["Transient error limit: {peak_cap}.", "At no time may the error be larger than {peak_cap}."],
    "prob": ["Required probability: {prob}.", "It is sufficient that this holds with probability {prob}."],
    "mse_target": ["Mean-square error allowed: {mse_target}."],
    "rms_target": ["RMS error allowed: {rms_target}."],
    "window": ["Duration of a run: {window}."],
    "obj:hard_bound": ["The accuracy must be certified for all admissible conditions.", "A worst-case guarantee is mandatory."],
    "obj:mse_optimal": ["Deliver the least-mean-square-error estimate.", "The optimal estimate in the mean-square sense is wanted."],
    "obj:mse_bound": ["The requirement is on the mean-square error."], "obj:probability": ["The requirement is probabilistic."],
    "obj:unspecified": ["", "High accuracy is desired."],
    "dist:gaussian": ["Noise statistics: Gaussian.", "Normal distributions may be assumed."],
    "dist:bounded": ["Noise statistics: uniform within hard limits; only variances are reliable.", "Distributions are not normal; second moments are known."],
    "hor:steady_state": ["Stationary operation only.", "Use the stationary gain."], "hor:finite": ["Include the start-up transient.", "Use the time-varying gain."],
    "ev:mechanical_limit": ["This is enforced by a hardware limiter.", "A physical limit guarantees it."],
    "ev:assumed": ["This is an operating assumption that we accept."],
    "flag:hidden_dynamics": ["Caution: actuator dynamics exist that the model omits.", "The model ignores a lag in the actuation path."],
    "flag:correlated_noise": ["Process and measurement disturbances are mutually correlated."],
    "flag:coloured_noise": ["The reading noise is time-correlated rather than white."],
    "flag:irregular_sampling": ["Sampling instants are irregular, with occasional dropouts."],
    "q1": ["Wanted: the {target}.", "Estimate the {target}.", "The unknown of interest is the {target}."],
    "model": ["Model: see attachment.", "The attached file contains the model."],
    "sensor": ["Available measurement: {sensor}.", "Instrumentation: {sensor} only."],
    "other": ["Resolution of the {sensor}: {n} bits.", "Installed in {year}.", "Rated voltage: {volt} V."],
}
RANGE = {"A": ["The {what} is only known to lie between {lo} and {hi}.", "The {what} is uncertain: it is somewhere between {lo} and {hi}."],
         "B": ["{what}: between {lo} and {hi} (not known exactly)", "{what} in [{lo}, {hi}]"],
         "C": ["not sure about the {what}, somewhere between {lo} and {hi}", "the {what} is anywhere from {lo} to {hi}, we haven't pinned it down"],
         "D": ["{what}: uncertain, from {lo} to {hi}.", "Range of the {what}: {lo} to {hi}."]}
WHAT = {"meas_bound": ["bound on the {sensor} error", "worst-case {sensor} error"], "Ts": ["sample period", "update period"],
        "meas_var": ["measurement-noise variance", "variance of the {sensor} noise"]}
TRAIN_STYLES, DEV_STYLES = ["A", "B", "C"], ["D"]


def case_facts(case: dict, rng) -> dict:
    """Facts that a request for this case must convey (from the GIVEN specification)."""
    from ..core import spec as S
    from ..sim import plants as PL
    g = case["given"]
    plant = g["plant"]
    fam = case["family"]
    so = plant["kind"] == "second_order"
    lin_unit = (plant["states"][0]["unit"] if plant["states"] else "rad")
    family_unit = "lin" if lin_unit == "m" else "ang"
    facts = []

    def add(role, value, dim=None, fam_=None):
        facts.append({"role": role, "value": float(value), "dim": dim or ROLE_DIM[role], "family": family_unit if fam_ is None else fam_})

    Ts = S.get(g, "time.Ts")
    rngs = g.get("ranges") or {}
    if Ts is not None and not (plant.get("linear") or {}).get("clock") == "discrete":
        if rng.random() < 0.5:
            add("Ts", Ts, fam_="")
        else:
            add("fs", 1.0 / Ts, fam_="")
    elif Ts is not None:
        add("Ts", Ts, fam_="")
    if "time.Ts" in rngs:
        add("Ts", rngs["time.Ts"][0], fam_=""); add("Ts", rngs["time.Ts"][1], fam_="")
    nl_stats = fam in PL.BOTH_METHODS or (so and fam != "mass_spring_damper")
    if so:
        V = S.get(g, "deterministic.V")
        if V is not None:
            add("meas_bound", V)
        if "deterministic.V" in rngs:
            add("meas_bound", rngs["deterministic.V"][0]); add("meas_bound", rngs["deterministic.V"][1])
        for path in ("deterministic.D", "deterministic.d_bound"):
            v = S.get(g, path)
            if v is not None:
                add("acc_bound", v)
        E = S.get(g, "initial.E")
        if E is not None:
            add("init_pos_bound", E[0]); add("init_vel_bound", E[1])
        dom = S.get(g, "deterministic.domain") or {}
        if dom.get("x2"):
            add("vel_limit", dom["x2"][1])
    if nl_stats:
        R = S.get(g, "stochastic.R")
        if R is not None:
            if rng.random() < 0.7:
                add("meas_std", math.sqrt(R[0][0]))
            else:
                add("meas_var", R[0][0])
        if "stochastic.R" in rngs:
            add("meas_var", rngs["stochastic.R"][0]); add("meas_var", rngs["stochastic.R"][1])
        Q = S.get(g, "stochastic.Q")
        if Q is not None and S.get(g, "stochastic.Q_kind") == "discrete_cov" and Ts:
            add("acc_std", math.sqrt(Q[1][1]) / Ts)
        P0 = S.get(g, "initial.P0")
        if P0 is not None and fam in PL.BOTH_METHODS:
            add("init_pos_std", math.sqrt(P0[0][0])); add("init_vel_std", math.sqrt(P0[1][1]))
    ob = g["objective"]
    tunit = plant["states"][case["target"]]["unit"] if plant["states"] else ""
    tfam = family_unit if so else tunit
    tdim = "vel" if (so and case["target"] == 1) else ("pos" if so else "other")
    if ob.get("tolerance") is not None:
        add("tolerance", ob["tolerance"], tdim, tfam)
    if ob.get("deadline") is not None:
        add("deadline", ob["deadline"], fam_="")
    if ob.get("peak_cap") is not None:
        add("peak_cap", ob["peak_cap"], tdim, tfam)
    if ob.get("prob") is not None:
        add("prob", ob["prob"], fam_="")
    if ob.get("mse_target") is not None:
        if rng.random() < 0.5:
            add("rms_target", math.sqrt(ob["mse_target"]), tdim, tfam)
        else:
            add("mse_target", ob["mse_target"], tdim + "2" if tdim != "other" else "other2", tfam)
    te = S.get(g, "time.t_end")
    if te is not None and case["category"] == "invalid_deadline_after_window":
        add("window", te, fam_="")
    lab = case["labels"]
    return {"facts": facts, "objective": ob.get("kind") or "unspecified", "distribution": S.get(g, "stochastic.distribution") if nl_stats or True else None,
            "horizon": ob.get("horizon"), "flags": [f for f in (lab.get("flags") or []) if f in FLAGS],
            "evidence": (S.get(g, "deterministic.domain_evidence") or {}).get("kind"),
            "target": plant["states"][case["target"]]["name"] if plant["states"] else "state",
            "sensor": plant["outputs"][0]["name"] if plant["outputs"] else "sensor",
            "x1": plant["states"][0]["name"].split()[-1] if plant["states"] else "position",
            "x2": plant["states"][1]["name"].split()[-1] if len(plant["states"]) > 1 else "velocity",
            "stats_in_attachment": not nl_stats}


def _sq_unit(unit):
    """Squared form of a unit string: compound units are bracketed, so that 'm/s' squared is '(m/s)^2' and not 'm/s^2'."""
    return f"({unit})^2" if "/" in unit or " " in unit else unit + "^2"


def _fmt_other(value, unit):
    s = f"{value:.6g}"
    return (s + " " + unit).strip()


def realise(case: dict, style: str, rng) -> dict:
    """One natural-language request for a case in the given style, with gold spans."""
    F = case_facts(case, rng)
    bank = P[style]
    ctx = {"target": F["target"], "sensor": F["sensor"], "x1": F["x1"], "x2": F["x2"]}
    parts = {"q1": [], "q2": [], "q3": []}
    spans = []          # (section, local_start, local_end, role, value)

    def emit(section, tmpl, role=None, value=None, dim=None, fam=None):
        txt = tmpl
        for k, v in ctx.items():
            txt = txt.replace("{" + k + "}", v)
        if role is not None and "{" + role + "}" in txt:
            if dim in ("other", "other2"):
                disp = _fmt_other(value, _sq_unit(fam) if dim == "other2" else fam)
            else:
                disp, _ = fmt_quantity(value, dim, fam, rng)
            i = txt.index("{" + role + "}")
            txt = txt.replace("{" + role + "}", disp)
            parts[section].append((txt, [(i, i + len(disp), role, value)]))
        else:
            if txt:
                parts[section].append((txt, []))

    emitted = set()

    def pick(key):
        t = bank[key][int(rng.integers(0, len(bank[key])))]
        if t:
            emitted.add(key)
        return t
    emit("q1", pick("q1"))
    emit("q2", pick("sensor"))
    emit("q2", pick("model"))
    q3_roles = {"tolerance", "deadline", "peak_cap", "prob", "mse_target", "rms_target"}
    # a role stated twice is a range: one sentence with both ends
    by_role = {}
    for f in F["facts"]:
        by_role.setdefault(f["role"], []).append(f)
    singles = []
    for role, fl in by_role.items():
        if len(fl) == 2 and role in WHAT:
            lo, hi = sorted(fl, key=lambda z: z["value"])
            w = WHAT[role][int(rng.integers(0, len(WHAT[role])))]
            tmpl = RANGE[style][int(rng.integers(0, len(RANGE[style])))].replace("{what}", w)
            for k, v in ctx.items():
                tmpl = tmpl.replace("{" + k + "}", v)
            dlo, u = fmt_quantity(lo["value"], lo["dim"], lo["family"], rng)
            dhi, _ = fmt_quantity(hi["value"], hi["dim"], hi["family"], rng)
            i = tmpl.index("{lo}"); tmpl = tmpl.replace("{lo}", dlo)
            j = tmpl.index("{hi}"); tmpl = tmpl.replace("{hi}", dhi)
            parts["q2"].append((tmpl, [(i, i + len(dlo), role, lo["value"]), (j, j + len(dhi), role, hi["value"])]))
        else:
            singles += fl
    for f in singles:
        sec = "q3" if f["role"] in q3_roles else "q2"
        if f["role"] == "deadline" and rng.random() < 0.15:
            sec = "q2"
        emit(sec, pick(f["role"]), f["role"], f["value"], f["dim"], f["family"])
        if f["role"] == "vel_limit" and F["evidence"] in ("mechanical_limit", "assumed"):
            emit("q2", pick("ev:" + F["evidence"]))
    if F["distribution"] in ("gaussian", "bounded") and not F["stats_in_attachment"]:
        emit("q2", pick("dist:" + F["distribution"]))
    elif F["distribution"] in ("gaussian", "bounded") and rng.random() < 0.5:
        emit("q2", pick("dist:" + F["distribution"]))
    for fl in F["flags"]:
        emit("q2", pick("flag:" + fl))
    if F["horizon"] in ("steady_state", "finite"):
        emit("q3", pick("hor:" + F["horizon"]))
    emit("q3", pick("obj:" + F["objective"]))
    # distractors
    for _ in range(int(rng.integers(0, 3))):
        t = pick("other").replace("{n}", str(int(rng.integers(12, 24)))).replace("{year}", str(int(rng.integers(1995, 2026)))).replace("{volt}", str(int(rng.choice([24, 48, 230, 400]))))
        emit("q2", t)
    out = {"id": case["id"], "style": style, "sections": {}, "gold_quantities": []}
    sep = {"A": " ", "B": "; ", "C": ", ", "D": " "}[style]
    for sec in ("q1", "q2", "q3"):
        items = parts[sec]
        if sec != "q1":
            order = rng.permutation(len(items))
            items = [items[i] for i in order]
        text = ""
        for txt, sp in items:
            if style == "C" and text:
                txt = txt[0].lower() + txt[1:] if txt[:2] != "I " and not txt[:2].startswith("I'") else txt
            off = len(text) + (len(sep) if text else 0)
            text = text + (sep if text else "") + txt
            for a, b_, role, value in sp:
                out["gold_quantities"].append({"section": sec, "start": off + a, "end": off + b_, "role": role, "value": value, "text": text[off + a: off + b_]})
        out["sections"][sec] = text
    dist = F["distribution"] if ("dist:" + str(F["distribution"])) in emitted else "unstated"
    ev = F["evidence"] if ("ev:" + str(F["evidence"])) in emitted else "unstated"
    out["labels"] = {"objective": F["objective"], "distribution": dist, "horizon": F["horizon"] or "unstated", "flags": F["flags"],
                     "evidence": ev, "target": F["target"]}
    return out


def full_text(req: dict) -> str:
    """Single text with section markers, as seen by the reader."""
    s = req["sections"]
    return "[Q1] " + s["q1"] + " [Q2] " + s["q2"] + " [Q3] " + s["q3"]


SECTION_OFFSETS = lambda req: {"q1": 5, "q2": 5 + len(req["sections"]["q1"]) + 6, "q3": 5 + len(req["sections"]["q1"]) + 6 + len(req["sections"]["q2"]) + 6}
