"""Assessment record returned by each specialist for one specification, and small shared helpers."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import math

import numpy as np


@dataclass
class Need:
    path: str            # missing or unresolved field
    why: str             # what it affects (the claim that cannot be made without it)
    question: str        # the literal question for the user
    threshold: Optional[float] = None      # if set: the question is "is <field> <= threshold?" and that value certifies
    threshold_unit: str = ""
    witness: Optional[dict] = None         # two admissible completions with different justified outcomes

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None and v != ""}


@dataclass
class Assessment:
    method: str
    status: str                      # "qualifies" | "needs" | "fails" | "not_applicable"
    reasons: list = field(default_factory=list)
    needs: list = field(default_factory=list)           # list[Need]
    config: Optional[dict] = None
    claim: Optional[dict] = None
    metrics: dict = field(default_factory=dict)
    limits: list = field(default_factory=list)          # stated scope of the guarantee / of a failed sufficient test
    harmless: list = field(default_factory=list)        # omissions demonstrated irrelevant: {"path", "why"}
    impossibility: Optional[dict] = None                # a separate valid argument that no estimator in a stated class can meet the request
    checks: list = field(default_factory=list)          # checks performed: {"name", "passed", "detail"}

    def to_dict(self):
        d = asdict(self)
        d["needs"] = [n.to_dict() if isinstance(n, Need) else n for n in self.needs]
        return jsonable(d)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None                                   # strict JSON: non-finite numbers are stored as null
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Need):
        return o.to_dict()
    return o


def is_double_integrator(Ad, C, Ts, tol=1e-12) -> bool:
    Ad = np.asarray(Ad, float)
    C = np.atleast_2d(np.asarray(C, float))
    return (Ad.shape == (2, 2) and C.shape == (1, 2)
            and np.allclose(Ad, [[1.0, Ts], [0.0, 1.0]], rtol=0, atol=tol * max(1.0, Ts))
            and np.allclose(C, [[1.0, 0.0]], rtol=0, atol=tol))


def fmt_up(x: float, sig: int = 6) -> str:
    """Decimal text of x with `sig` significant digits, rounded UP (an upper bound printed this way stays an upper bound)."""
    from decimal import Decimal, ROUND_CEILING
    if x is None or not math.isfinite(x):
        return str(x)
    if x == 0:
        return "0"
    d = Decimal(x)
    q = Decimal(1).scaleb(d.adjusted() - sig + 1)
    return f"{d.quantize(q, rounding=ROUND_CEILING).normalize():g}" if abs(d.adjusted()) < 5 else f"{float(d.quantize(q, rounding=ROUND_CEILING)):.{sig}g}"


def fmt_down(x: float, sig: int = 6) -> str:
    """As fmt_up, rounded DOWN (for guaranteed lower levels such as probabilities)."""
    from decimal import Decimal, ROUND_FLOOR
    if x is None or not math.isfinite(x):
        return str(x)
    if x == 0:
        return "0"
    d = Decimal(x)
    q = Decimal(1).scaleb(d.adjusted() - sig + 1)
    return f"{d.quantize(q, rounding=ROUND_FLOOR).normalize():g}" if abs(d.adjusted()) < 5 else f"{float(d.quantize(q, rounding=ROUND_FLOOR)):.{sig}g}"
