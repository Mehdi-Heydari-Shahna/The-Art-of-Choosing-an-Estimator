"""Sample instants of a requirement window and agreement of the stated period with recorded timestamps.

One convention is used by every specialist, the coordinator and the run-time path:

    t_k = k Ts (k = 0, 1, ...), the first measurement at t_0 = 0;
    a requirement "from the deadline on" refers to the samples with t_k >= deadline;
    a record ending at t_end contains the samples with t_k <= t_end.

So a window [deadline, t_end] contains the samples k_dead <= k <= n_end with k_dead = ceil(deadline / Ts) and
n_end = floor(t_end / Ts).  The ratio of two stated times carries the rounding of decimal input (0.3 / 0.1 =
2.9999999999999996 must count as 3), so both use an allowance of a few units in the last place of the ratio
(ratio_allowance); a stated endpoint that lies before a sample by more than that, such as 4 - 2^-31 s with Ts = 1 s,
is taken as exact and the sample after it is outside the window (review case C of revision 2b; revision 2b used an
absolute allowance of 1e-9, which admitted that sample).  A sample outside the window is never used to establish or
to refute a requirement.
"""
from __future__ import annotations

import math

import numpy as np

EPS = float(np.finfo(float).eps)


def ratio_allowance(r: float) -> float:
    """Allowance on the ratio r of two stated times when it is compared with an integer: 8 units of float64 rounding
    relative to |r| (each stated time and the division contribute at most half a unit each)."""
    return 8.0 * EPS * max(1.0, abs(float(r)))


def sample_window(Ts: float, deadline: float | None = None, t_end: float | None = None):
    """(k_dead, n_end): the first and the last sample index of the window; n_end is None for an open window."""
    if deadline is None:
        k_dead = 0
    else:
        r = deadline / Ts
        k_dead = max(0, int(math.ceil(r - ratio_allowance(r))))
    if t_end is None:
        n_end = None
    else:
        r = t_end / Ts
        n_end = int(math.floor(r + ratio_allowance(r)))
    return k_dead, n_end


def first_sample_at_or_after(Ts: float, t: float) -> int:
    r = t / Ts
    return max(0, int(math.ceil(r - ratio_allowance(r))))


def describe_window(Ts: float, k_dead: int, n_end: int | None) -> str:
    if n_end is None:
        return f"at every sample instant t_k >= {k_dead * Ts:.6g} s"
    return f"at every sample instant from t = {k_dead * Ts:.6g} s to t = {n_end * Ts:.6g} s"


def label_precision(t, Ts: float | None) -> float:
    """Precision with which float64 time labels place a sample: 4 units in the last place of the largest |time|
    (the rounding of each label, of k Ts, of t_0 + k Ts and the representation of Ts together stay below it).  It
    grows with the clock origin: absolute times near 2^30 s are resolved only to 2.4e-7 s, so the check is the same
    for any origin up to that resolution (review cases A1 to A3 of revision 2b; revision 2b allowed 64 eps times the
    largest time, 32 to 64 such units, which accepted a sampling deviation of 32 units at that origin)."""
    t = np.asarray(t, float).reshape(-1)
    if not t.size:
        return 0.0
    scale = max(abs(float(t[0])), abs(float(t[-1])), abs(float(Ts)) if Ts else 0.0)
    return 4.0 * float(np.spacing(scale)) if scale > 0 else 0.0


def label_tolerance(t, Ts: float | None, label_resolution: float | None = None) -> float:
    """Largest difference between a recorded time and its sample instant t_0 + k Ts that is attributed to the labels.

    Without a statement about the labels this is their float64 precision (label_precision): the recorded times are
    taken to be the sample instants, so any larger difference is a sampling deviation (review case 01c).  When the
    request states that the time labels are rounded to a resolution r (time.label_resolution), differences up to r / 2
    are attributed to that rounding as well."""
    tol = label_precision(t, Ts)
    if label_resolution:
        tol += 0.5 * float(label_resolution)
    return tol


# The labels confirm the instants only to within their precision.  Above PRECISION_NOTE * Ts this precision is stated
# as a condition of a released claim; above PRECISION_LIMIT * Ts the labels are too coarse to confirm the period at
# all and nothing is released for the data.
PRECISION_NOTE = 1e-9
PRECISION_LIMIT = 1e-2


def check_timestamps(t, Ts: float | None, n_samples: int | None = None, label_resolution: float | None = None) -> dict:
    """Compare recorded sample times with the stated period.

    Returns {"ok": bool, "code": None | "timing_conflict" | "irregular_sampling" | "bad_timestamps", "message": str, ...}.
      timing_conflict      the samples are uniform but at a different period
      irregular_sampling   the sample instants deviate from a uniform grid (jitter, a missing or a repeated sample)
      bad_timestamps       not increasing, not finite, or a different number of rows than measurements
    A certificate for one fixed period is computed for samples exactly at t_0 + k Ts; it is valid for the data only when
    the result is ok, i.e. every recorded time equals its instant to within the tolerance of label_tolerance."""
    t = np.asarray(t, float).reshape(-1)
    out = {"ok": False, "code": None, "n": int(t.size), "Ts_stated": Ts, "label_resolution": label_resolution}
    if n_samples is not None and t.size != n_samples:
        out.update(code="bad_timestamps", message=f"{t.size} timestamps for {n_samples} measurements")
        return out
    if t.size < 2:
        out.update(ok=True, message="fewer than two timestamps: nothing to compare")
        return out
    if not np.all(np.isfinite(t)):
        out.update(code="bad_timestamps", message="the timestamps contain non-finite values")
        return out
    dt = np.diff(t)
    out.update(dt_min=float(dt.min()), dt_max=float(dt.max()), dt_median=float(np.median(dt)), t0=float(t[0]))
    if dt.min() <= 0:
        out.update(code="bad_timestamps", message="the timestamps are not strictly increasing")
        return out
    k = np.arange(t.size)
    prec = label_precision(t, Ts)
    out["label_precision"] = prec
    if Ts and prec > PRECISION_LIMIT * float(Ts):
        out.update(code="bad_timestamps",
                   message=(f"the stored times resolve instants only to {prec:.3g} s (times near {max(abs(t[0]), abs(t[-1])):.6g} s), too coarse to "
                            f"confirm samples of the stated period {Ts:.6g} s; record the times relative to the first sample"))
        return out
    tol = label_tolerance(t, Ts, label_resolution)
    # period of the record as a uniform grid: the end-to-end average, whose rounding error shrinks with the number of
    # intervals (the median of the differences carries the rounding of single labels, which k Ts would multiply)
    med = float((t[-1] - t[0]) / (t.size - 1))
    dev_med = t - (t[0] + k * med)
    uniform = float(np.max(np.abs(dev_med))) <= tol
    out["tolerance"] = tol
    if Ts is None:
        if uniform:
            out.update(ok=True, message=f"uniform recorded period {med:.6g} s; no period is stated to compare with")
        else:
            out.update(code="irregular_sampling", max_deviation=float(np.max(np.abs(dev_med))),
                       message="the recorded sample instants are not uniform; a certificate for one fixed period does not apply")
        return out
    dev = t - (t[0] + k * float(Ts))
    m = int(np.argmax(np.abs(dev)))
    out["max_deviation"] = float(abs(dev[m]))
    if abs(dev[m]) <= tol:
        out.update(ok=True, message=(f"the recorded times equal the instants t_0 + k Ts of the stated period {Ts:.6g} s "
                                     + (f"to within the stated label resolution {label_resolution:.6g} s" if label_resolution else
                                        f"to within the precision of the stored times ({prec:.3g} s)")))
        return out
    if uniform:
        out.update(code="timing_conflict",
                   message=f"the data are recorded every {med:.9g} s but the request states a period of {Ts:.9g} s; the gains, the model "
                           "and the certificate are computed for one exact period, so state the period with which the data were recorded")
        return out
    out.update(code="irregular_sampling",
               message=(f"the recorded sample instants deviate from the instants t_0 + k Ts of the stated period by up to {abs(dev[m]):.3g} s "
                        f"(largest at sample {m}); the certificate is computed for samples exactly at those instants and does not cover "
                        "sampling deviations" + ("" if label_resolution else
                                                 "; if the deviations are only the rounding of the time labels, state the label resolution")))
    return out


def claim_statement(bound, peak_value, Ts, k_dead, n_end, deadline):
    """Text of a hard-bound claim on the requested window (shared by both specialists and the coordinator)."""
    from .assess import fmt_up
    if deadline is not None:
        stmt = f"|error| <= {fmt_up(bound)} " + describe_window(Ts, k_dead, n_end)
    else:
        stmt = f"lim sup |error| <= {fmt_up(bound)} at the sample instants (an ultimate bound; no finite-time claim is made)"
    if peak_value is not None:
        stmt += f", and |error| <= {fmt_up(peak_value)} " + (describe_window(Ts, 0, n_end) if n_end is not None else "at every sample instant")
    return stmt
