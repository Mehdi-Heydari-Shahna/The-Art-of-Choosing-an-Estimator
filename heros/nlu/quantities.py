"""Deterministic reading of numbers and units (no learned component).

find_quantities(text) returns every numeric mention with its SI value and physical dimension.  Dimensions:
    time, freq, pos (angle or length), vel, acc, pos2, vel2, acc2, frac (percent / probability), count, other
Angles and lengths share the dimension "pos"; the unit family ("ang" / "lin") is kept so that a request in
degrees for a linear axis can be reported as a unit conflict instead of being converted silently.
Number words ("half a millisecond", "two hundred hertz", "one in a thousand") and scientific notation
("5e-5", "5 x 10^-5", "5·10⁻⁵") are read; anything that cannot be read is left out and therefore never bound.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict

DEG = math.pi / 180.0
# unit -> (dimension, factor to SI, family)
_UNITS = {
    "s": ("time", 1.0, ""), "sec": ("time", 1.0, ""), "secs": ("time", 1.0, ""), "second": ("time", 1.0, ""), "seconds": ("time", 1.0, ""),
    "ms": ("time", 1e-3, ""), "msec": ("time", 1e-3, ""), "millisecond": ("time", 1e-3, ""), "milliseconds": ("time", 1e-3, ""),
    "us": ("time", 1e-6, ""), "µs": ("time", 1e-6, ""), "μs": ("time", 1e-6, ""), "microsecond": ("time", 1e-6, ""), "microseconds": ("time", 1e-6, ""),
    "min": ("time", 60.0, ""), "minute": ("time", 60.0, ""), "minutes": ("time", 60.0, ""),
    "hz": ("freq", 1.0, ""), "hertz": ("freq", 1.0, ""), "khz": ("freq", 1e3, ""), "kilohertz": ("freq", 1e3, ""),
    "rad": ("pos", 1.0, "ang"), "radian": ("pos", 1.0, "ang"), "radians": ("pos", 1.0, "ang"),
    "mrad": ("pos", 1e-3, "ang"), "millirad": ("pos", 1e-3, "ang"), "milliradian": ("pos", 1e-3, "ang"), "milliradians": ("pos", 1e-3, "ang"),
    "urad": ("pos", 1e-6, "ang"), "µrad": ("pos", 1e-6, "ang"), "μrad": ("pos", 1e-6, "ang"), "microrad": ("pos", 1e-6, "ang"),
    "microradian": ("pos", 1e-6, "ang"), "microradians": ("pos", 1e-6, "ang"),
    "deg": ("pos", DEG, "ang"), "degree": ("pos", DEG, "ang"), "degrees": ("pos", DEG, "ang"), "°": ("pos", DEG, "ang"),
    "mdeg": ("pos", 1e-3 * DEG, "ang"), "millidegree": ("pos", 1e-3 * DEG, "ang"), "millidegrees": ("pos", 1e-3 * DEG, "ang"),
    "arcsec": ("pos", DEG / 3600.0, "ang"), "arcmin": ("pos", DEG / 60.0, "ang"),
    "m": ("pos", 1.0, "lin"), "meter": ("pos", 1.0, "lin"), "meters": ("pos", 1.0, "lin"), "metre": ("pos", 1.0, "lin"), "metres": ("pos", 1.0, "lin"),
    "cm": ("pos", 1e-2, "lin"), "mm": ("pos", 1e-3, "lin"), "millimeter": ("pos", 1e-3, "lin"), "millimeters": ("pos", 1e-3, "lin"),
    "millimetre": ("pos", 1e-3, "lin"), "millimetres": ("pos", 1e-3, "lin"),
    "um": ("pos", 1e-6, "lin"), "µm": ("pos", 1e-6, "lin"), "μm": ("pos", 1e-6, "lin"), "micron": ("pos", 1e-6, "lin"), "microns": ("pos", 1e-6, "lin"),
    "micrometer": ("pos", 1e-6, "lin"), "micrometers": ("pos", 1e-6, "lin"), "micrometre": ("pos", 1e-6, "lin"), "micrometres": ("pos", 1e-6, "lin"),
    "nm": ("pos", 1e-9, "lin"), "nanometer": ("pos", 1e-9, "lin"), "nanometers": ("pos", 1e-9, "lin"), "nanometre": ("pos", 1e-9, "lin"), "nanometres": ("pos", 1e-9, "lin"),
    "rpm": ("vel", 2 * math.pi / 60.0, "ang"),
    "%": ("frac", 1e-2, ""), "percent": ("frac", 1e-2, ""), "pct": ("frac", 1e-2, ""),
    "k": ("other", 1.0, "K"), "kelvin": ("other", 1.0, "K"), "°c": ("other", 1.0, "K"), "mpa": ("other", 1.0, "MPa"), "bar": ("other", 0.1, "MPa"),
    "a": ("other", 1.0, "A"), "ma": ("other", 1e-3, "A"), "amp": ("other", 1.0, "A"), "amps": ("other", 1.0, "A"),
    "v": ("other", 1.0, "V"), "volt": ("other", 1.0, "V"), "volts": ("other", 1.0, "V"),
    "n": ("other", 1.0, "N"), "nm_torque": ("other", 1.0, "N m"), "kn": ("other", 1e3, "N"), "kg": ("other", 1.0, "kg"), "kw": ("other", 1.0, "kW"), "w": ("other", 1e-3, "kW"),
    "bit": ("count", 1.0, ""), "bits": ("count", 1.0, ""), "counts": ("count", 1.0, ""), "samples": ("count", 1.0, ""), "steps": ("count", 1.0, ""),
}
_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
          "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
          "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALE = {"hundred": 100, "thousand": 1000, "million": 10 ** 6}
_FRACW = {"half": 0.5, "quarter": 0.25, "third": 1.0 / 3.0, "tenth": 0.1, "hundredth": 0.01, "thousandth": 0.001}
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")

_NUM = r"(?<![\w.])(?:[-+−]|±\s?)?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:\s?(?:[eE]\s?[-+−]?\d+|(?:[x×·*]\s?)?10\s?(?:\^|\*\*)?\s?[-+−]?\d+|(?:[x×·*]\s?)?10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+))?"
_WORDNUM = r"(?:(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|half|quarter|third|tenth|hundredth|thousandth)\b(?:[\s-]+(?:and\s+|of\s+)?)?)+"
_UNIT_TOK = r"[A-Za-zµμ°%]+(?:\s?/\s?[A-Za-zµμ]+)?(?:\s?(?:\^|\*\*)\s?-?\d|[²³])?"
_RE = re.compile(rf"(?P<num>{_NUM})|(?P<word>\b{_WORDNUM}\b)", re.IGNORECASE)


@dataclass
class Quantity:
    start: int
    end: int
    text: str
    value: float          # SI value (positive magnitude; sign kept in `negative`)
    dim: str
    family: str
    unit: str
    negative: bool = False
    plusminus: bool = False

    def to_dict(self):
        return asdict(self)


def _parse_number(tok: str):
    t = tok.strip().translate(_SUP)
    neg = t[:1] in "-−"
    pm = t[:1] == "±"
    t = t.lstrip("+-−± ").replace("−", "-")
    m = re.match(r"^(\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)\s?(?:[eE]\s?([-+]?\d+)|(?:[x×·*]\s?)?10\s?(?:\^|\*\*)?\s?([-+]?\d+))?$", t)
    if not m:
        return None
    val = float(m.group(1).replace(",", ""))
    ex = m.group(2) or m.group(3)
    if ex:
        val *= 10.0 ** int(ex)
    return val, neg, pm


def _parse_words(tok: str):
    """'two hundred' -> 200, 'half a' -> 0.5, 'a quarter of a' -> 0.25, 'one and a half' -> 1.5, 'three quarters' is not read."""
    ws = [w for w in re.split(r"[\s-]+", tok.lower()) if w]
    while ws and ws[-1] in ("a", "an", "and", "of"):
        ws.pop()
    if not ws or all(w in ("a", "an", "and", "of") for w in ws):
        return None
    total, cur, seen, additive = 0.0, 0.0, False, False
    for w in ws:
        if w == "and":
            additive = seen
            continue
        if w in ("a", "an", "of"):
            continue
        if w in _WORDS:
            cur += _WORDS[w]; seen = True
        elif w in _SCALE:
            cur = (cur or 1.0) * _SCALE[w]; seen = True
            if _SCALE[w] >= 1000:
                total += cur; cur = 0.0
        elif w in _FRACW:
            if additive:
                cur += _FRACW[w]
            else:
                cur = _FRACW[w] * (cur if cur else 1.0)
            seen = True
        else:
            return None
    return (total + cur) if seen else None


def _unit_after(text: str, pos: int):
    """Longest unit expression starting at `pos` (after optional space / hyphen).  Returns (dim, factor, family, unit_text, end)."""
    m = re.match(r"[\s-]{0,2}(" + _UNIT_TOK + ")", text[pos:])
    if not m:
        return None
    raw = m.group(1)
    end = pos + m.end()
    u = raw.replace(" ", "")
    power = 1
    pm = re.search(r"(?:\^|\*\*)(-?\d)$|([²³])$", u)
    if pm:
        power = int(pm.group(1)) if pm.group(1) else {"²": 2, "³": 3}[pm.group(2)]
        u = u[: pm.start()]
    num, _, den = u.partition("/")
    key = num.lower()
    if num == "Nm" or num == "N·m":
        key = "nm_torque"
    base = _UNITS.get(key if key in _UNITS else num)
    if base is None:
        # try a shorter token (e.g. "ms." or a following ordinary word)
        return None
    dim, fac, fam = base
    if den:
        d = den.lower()
        if d in ("s", "sec", "second"):
            if dim == "pos":
                dim = "acc" if power == 2 and not pm else "vel"
                if pm and power == 2 and raw.rstrip().endswith(("^2", "**2", "²")):
                    # "rad/s^2": the power applies to the denominator
                    dim = "acc"
                power_applied = True
            elif dim == "vel":            # rpm/s
                dim = "acc"
            else:
                return None
        elif d in ("s2", "s^2", "s²"):
            dim = "acc" if dim == "pos" else None
        elif d in ("sqrthz", "√hz", "rthz"):
            return None
        else:
            return None
        if dim is None:
            return None
        return dim, fac, fam, raw, end
    if power == 2 and dim in ("pos", "vel", "acc", "other"):
        return dim + "2", fac * fac, fam, raw, end
    if power != 1:
        return None
    return dim, fac, fam, raw, end


def find_quantities(text: str) -> list:
    out = []
    for m in _RE.finditer(text):
        s, e = m.span()
        tok = m.group(0)
        if m.group("num"):
            pn = _parse_number(tok)
            if pn is None:
                continue
            val, neg, pm = pn
        else:
            val = _parse_words(tok)
            if val is None:
                continue
            neg = pm = False
            stripped = re.sub(r"(?:[\s-]+(?:a|an|and|of))*[\s-]*$", "", tok, flags=re.IGNORECASE)
            if tok.lower().strip() in ("a", "an") or not stripped:
                continue
            # keep a trailing article that belongs to the phrase ("half a millisecond")
            e = s + len(tok.rstrip())
            tok = text[s:e]
        # "(rad/s)^2" style
        mm = re.match(r"\s?\(\s?(" + _UNIT_TOK + r")\s?\)\s?(?:\^|\*\*)?\s?([2²])", text[e:])
        ua = None
        if mm:
            inner = _unit_after("x " + mm.group(1), 1)
            if inner and inner[0] in ("pos", "vel", "acc"):
                ua = (inner[0] + "2", inner[1] ** 2, inner[2], mm.group(0).strip(), e + mm.end())
        if ua is None:
            ua = _unit_after(text, e)
        if ua is None:
            # bare number: dimensionless (probability if within (0,1)), unless it is a list index like "1)" or "Q2"
            if m.group("word") and val in (1.0,) and tok.lower().strip() in ("a", "an", "one"):
                continue
            out.append(Quantity(s, e, text[s:e], abs(val), "frac" if 0 < abs(val) < 1 else "count", "", "", neg, pm))
            continue
        dim, fac, fam, utext, uend = ua
        # the squared form of a compound unit written "rad/s^2" is an acceleration, handled in _unit_after
        out.append(Quantity(s, uend, text[s:uend], abs(val) * fac, dim, fam, utext, neg, pm))
    # drop nested / overlapping matches (keep the earliest, longest)
    out.sort(key=lambda q: (q.start, -(q.end - q.start)))
    res, last = [], -1
    for q in out:
        if q.start >= last:
            res.append(q); last = q.end
    return res


def fmt_quantity(value_si: float, dim: str, family: str, rng=None, style: int = 0):
    """Human display of an SI value in a randomly chosen but exactly invertible unit (used by the template generator)."""
    import numpy as np
    rng = rng or np.random.default_rng(0)
    opts = {
        ("time", ""): [("s", 1.0), ("ms", 1e-3), ("s", 1.0), ("ms", 1e-3), ("µs", 1e-6)],
        ("freq", ""): [("Hz", 1.0), ("kHz", 1e3)],
        ("pos", "ang"): [("rad", 1.0), ("mrad", 1e-3), ("µrad", 1e-6), ("urad", 1e-6)],
        ("pos", "lin"): [("m", 1.0), ("mm", 1e-3), ("µm", 1e-6), ("um", 1e-6)],
        ("vel", "ang"): [("rad/s", 1.0), ("mrad/s", 1e-3)],
        ("vel", "lin"): [("m/s", 1.0), ("mm/s", 1e-3)],
        ("acc", "ang"): [("rad/s^2", 1.0), ("rad/s²", 1.0)],
        ("acc", "lin"): [("m/s^2", 1.0), ("m/s²", 1.0), ("mm/s^2", 1e-3)],
        ("pos2", "ang"): [("rad^2", 1.0), ("rad²", 1.0), ("mrad^2", 1e-6)],
        ("pos2", "lin"): [("m^2", 1.0), ("mm^2", 1e-6), ("µm^2", 1e-12)],
        ("vel2", "ang"): [("(rad/s)^2", 1.0)], ("vel2", "lin"): [("(m/s)^2", 1.0), ("(mm/s)^2", 1e-6)],
        ("frac", ""): [("", 1.0), ("%", 1e-2)],
    }
    cands = opts.get((dim, family)) or opts.get((dim, "")) or [("", 1.0)]
    # prefer a unit that gives a mantissa between 0.1 and 1000
    good = [(u, f) for u, f in cands if 0.1 <= value_si / f < 1000] or cands
    u, f = good[int(rng.integers(0, len(good)))]
    x = value_si / f
    s = f"{x:.6g}"
    if "e" in s:
        mant, ex = s.split("e")
        s = f"{mant}e{int(ex)}"
    return (s + (" " if u and u != "%" else "") + u).strip(), u
