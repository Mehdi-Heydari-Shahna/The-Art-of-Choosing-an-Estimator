"""Outward-rounded bounds of symbolic expressions over boxes (mpmath interval arithmetic).

Used to derive the total-disturbance bound of the high-gain observer from a symbolic model.  Every operation
is carried out on intervals; a binary float is an exact rational, decimal strings are enclosed, and the final
number handed on is the upper endpoint rounded up, so conservatism of the source bounds is preserved.
"""
from __future__ import annotations

import math

import sympy as sp
from mpmath import iv, mp

_ALLOWED = {"sin": sp.sin, "cos": sp.cos, "tanh": sp.tanh, "exp": sp.exp, "sqrt": sp.sqrt, "Abs": sp.Abs, "pi": sp.pi}


class BoundError(Exception):
    pass


def parse(expr: str, names) -> sp.Expr:
    ns = dict(_ALLOWED)
    ns.update({n: sp.Symbol(n, real=True) for n in names})
    try:
        e = sp.sympify(expr, locals=ns)
    except Exception as ex:  # noqa
        raise BoundError(f"cannot parse expression {expr!r}: {ex}")
    extra = {str(s) for s in e.free_symbols} - set(names)
    if extra:
        raise BoundError(f"expression {expr!r} uses undeclared symbols {sorted(extra)}")
    return e


def _iv_tanh(x):
    # tanh is increasing
    lo = mp.tanh(x.a); hi = mp.tanh(x.b)
    return iv.mpf([lo - abs(lo) * mp.mpf(2) ** (-(iv.prec - 4)) - mp.mpf(2) ** (-(iv.prec + 20)),
                   hi + abs(hi) * mp.mpf(2) ** (-(iv.prec - 4)) + mp.mpf(2) ** (-(iv.prec + 20))])


def iv_eval(e: sp.Expr, box: dict):
    """Interval enclosure of e over the box {name: mpmath interval}."""
    if e.is_Symbol:
        return box[str(e)]
    if e.is_Integer:
        return iv.mpf(int(e))
    if e.is_Rational:
        return iv.mpf(int(e.p)) / iv.mpf(int(e.q))
    if e.is_Float:
        return iv.mpf(float(e))            # a binary float is exact
    if e is sp.pi:
        return iv.pi
    if e.is_Add:
        r = iv.mpf(0)
        for a in e.args:
            r = r + iv_eval(a, box)
        return r
    if e.is_Mul:
        r = iv.mpf(1)
        for a in e.args:
            r = r * iv_eval(a, box)
        return r
    if e.is_Pow:
        b, x = e.args
        bv = iv_eval(b, box)
        if x.is_Integer:
            k = int(x)
            if k >= 0:
                return bv ** k
            if bv.a <= 0 <= bv.b:
                raise BoundError("division by an interval containing zero")
            return 1 / (bv ** (-k))
        if x == sp.Rational(1, 2):
            if bv.a < 0:
                raise BoundError("square root of a possibly negative quantity")
            return iv.sqrt(bv)
        raise BoundError(f"unsupported power {x}")
    f = e.func
    if f == sp.sin:
        return iv.sin(iv_eval(e.args[0], box))
    if f == sp.cos:
        return iv.cos(iv_eval(e.args[0], box))
    if f == sp.exp:
        return iv.exp(iv_eval(e.args[0], box))
    if f == sp.tanh:
        return _iv_tanh(iv_eval(e.args[0], box))
    if f == sp.Abs:
        return abs(iv_eval(e.args[0], box))
    raise BoundError(f"unsupported function {f}")


def up(x) -> float:
    f = float(x.b)
    if mp.mpf(f) < x.b:
        f = math.nextafter(f, math.inf)
    return f


def sup_abs(e: sp.Expr, box: dict, split_vars=None, budget: int = 256) -> float:
    """Upper bound of sup |e| over the box ({name: (lo, hi)} floats), by interval evaluation with adaptive bisection.
    The result is a valid upper bound for every budget; a larger budget only tightens it."""
    names = sorted(str(s) for s in e.free_symbols)
    for n in names:
        if n not in box:
            raise BoundError(f"no range for {n}")
        lo, hi = box[n]
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise BoundError(f"unbounded range for {n}")
        if lo > hi:
            raise BoundError(f"reversed range for {n}")
    cells = [{n: (float(box[n][0]), float(box[n][1])) for n in names}]
    split_vars = [v for v in (names if split_vars is None else split_vars) if v in names and box[v][1] > box[v][0]]

    def ub(cell):
        return up(abs(iv_eval(e, {n: iv.mpf([cell[n][0], cell[n][1]]) for n in names})))

    vals = [ub(cells[0])]
    count = 1
    while split_vars and count < budget:
        i = max(range(len(cells)), key=lambda j: vals[j])
        cell = cells[i]
        v = max(split_vars, key=lambda n: (cell[n][1] - cell[n][0]) / (box[n][1] - box[n][0]))
        lo, hi = cell[v]
        if hi - lo <= 1e-6 * (box[v][1] - box[v][0]):
            break
        mid = 0.5 * (lo + hi)
        c1, c2 = dict(cell), dict(cell)
        c1[v], c2[v] = (lo, mid), (mid, hi)
        cells[i], vals[i] = c1, ub(c1)
        cells.append(c2); vals.append(ub(c2))
        count += 1
    return max(vals)


def sup_value(e: sp.Expr, box: dict, budget: int = 32) -> float:
    """Upper bound of sup e (not |e|) over the box ({name: (lo, hi)} floats), by interval evaluation with adaptive bisection.
    Valid for every budget.  inf e is -sup_value(-e)."""
    names = sorted(str(s) for s in e.free_symbols)
    if not names:
        return up(iv_eval(e, {}))
    for n in names:
        if n not in box:
            raise BoundError(f"no range for {n}")
        lo, hi = box[n]
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise BoundError(f"unbounded range for {n}")
        if lo > hi:
            raise BoundError(f"reversed range for {n}")
    cells = [{n: (float(box[n][0]), float(box[n][1])) for n in names}]
    split_vars = [v for v in names if box[v][1] > box[v][0]]

    def ub(cell):
        return up(iv_eval(e, {n: iv.mpf([cell[n][0], cell[n][1]]) for n in names}))

    vals = [ub(cells[0])]
    count = 1
    while split_vars and count < budget:
        i = max(range(len(cells)), key=lambda j: vals[j])
        cell = cells[i]
        v = max(split_vars, key=lambda n: (cell[n][1] - cell[n][0]) / (box[n][1] - box[n][0]))
        lo, hi = cell[v]
        if hi - lo <= 1e-6 * (box[v][1] - box[v][0]):
            break
        mid = 0.5 * (lo + hi)
        c1, c2 = dict(cell), dict(cell)
        c1[v], c2[v] = (lo, mid), (mid, hi)
        cells[i], vals[i] = c1, ub(c1)
        cells.append(c2); vals.append(ub(c2))
        count += 1
    return max(vals)


def collect_for_bounds(e: sp.Expr, state_names) -> sp.Expr:
    """Algebraic rearrangement that reduces interval dependency: expand and collect by the state/input symbols and
    the functions of them, so that parameter differences such as (1/J - 1/J0) are evaluated once."""
    ex = sp.expand(e)
    syms = [sp.Symbol(n, real=True) for n in state_names]
    atoms = [a for a in ex.atoms(sp.Function) if any(s in a.free_symbols for s in syms)]
    return sp.collect(ex, atoms + syms)
