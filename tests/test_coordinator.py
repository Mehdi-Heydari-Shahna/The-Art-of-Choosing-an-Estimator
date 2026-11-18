"""Coordinator: the five required cases, invalid input, proved impossibility, confirmation binding, record replay."""
import copy, json, pathlib
import numpy as np
import pytest
from heros import api
from heros.core import spec as S
from heros.sim import plants as PL
from heros.datagen import make_case
from heros.interface import service as SV

EX = pathlib.Path(__file__).resolve().parents[1] / "examples"
BASE = json.loads((EX / "servo_axis_spec.json").read_text())
FIELDS = {k: v for k, v in BASE.items() if not k.startswith("_") and k != "plant"}
HARD = {"kind": "hard_bound", "tolerance": BASE["_hard_tolerance"], "deadline": BASE["_deadline"]}


def mk(obj, drop=(), ranges=None, plant=None, **over):
    f = {**copy.deepcopy(FIELDS), **over}
    for d in drop:
        f.pop(d, None)
    return api.new_spec(copy.deepcopy(plant or BASE["plant"]), objective=obj, ranges=ranges, **f)


def test_both_qualify_and_criterion_decides():
    d = api.decide(mk(HARD))
    assert d["disposition"] == "RELEASE" and set(d["qualifying"]) == {"high_gain_observer", "kalman_filter"}
    assert d["selected"] == "high_gain_observer"            # smaller certified limit on the requested criterion
    d = api.decide(mk({"kind": "mse_optimal", "horizon": "steady_state"}))
    assert d["disposition"] == "RELEASE" and d["selected"] == "kalman_filter"


def test_one_qualifies():
    d = api.decide(mk({"kind": "mse_optimal", "horizon": "steady_state"}))
    assert d["qualifying"] == ["kalman_filter"]


def test_clarification_names_the_missing_fact():
    d = api.decide(mk(HARD, drop=("V",)))
    assert d["disposition"] == "CLARIFY" and d["question"]["path"] == "deterministic.V"
    d = api.decide(mk(None))
    assert d["disposition"] == "CLARIFY" and d["question"]["path"] == "objective.kind"


def test_proved_impossible_below_two_signal_limit():
    lim = 2 * np.sqrt(FIELDS["D"] * FIELDS["V"])
    d = api.decide(mk({"kind": "hard_bound", "tolerance": 0.8 * lim}))
    assert d["disposition"] == "PROVED_IMPOSSIBLE"
    d = api.decide(mk({"kind": "hard_bound", "tolerance": 2.0 * lim}))
    assert d["disposition"] == "RELEASE"


def test_invalid_input():
    d = api.decide(mk(HARD, V=-1.0))
    assert d["disposition"] == "INVALID_INPUT" and d["issues"][0]["path"] == "deterministic.V"


def test_out_of_scope_family():
    rng = np.random.default_rng(3)
    for fam in PL.UNSUPPORTED:
        case = make_case(fam, "unsupported_family", rng, "t-" + fam)
        d = api.decide(case["given"])
        assert d["disposition"] == "OUT_OF_SCOPE", fam


def test_in_scope_without_verified_candidate():
    # a tolerance between the two-signal limit and what the supported designs can certify
    lim = 2 * np.sqrt(FIELDS["D"] * FIELDS["V"])
    d = api.decide(mk({"kind": "hard_bound", "tolerance": 1.02 * lim, "deadline": 0.002}))
    assert d["disposition"] in ("NO_VERIFIED_CANDIDATE", "PROVED_IMPOSSIBLE")
    assert d["disposition"] == "NO_VERIFIED_CANDIDATE"


def test_confirmation_is_bound_to_the_specification_version():
    model = json.loads((EX / "rotary_joint_model.json").read_text())
    script = json.loads((EX / "rotary_joint_answers.json").read_text())
    conv = SV.Conversation(model, script["answers"])
    rb = conv.readback()
    assert conv.step()["decision"]["disposition"] == "CONFIRM"       # nothing is released before confirmation
    conv.correct({"deterministic.V": 2e-4})                           # the specification changes ...
    with pytest.raises(Exception):
        conv.confirm(rb["spec_hash"])                                 # ... so the old confirmation is refused
    conv.confirm(conv.readback()["spec_hash"])
    assert conv.step()["decision"]["disposition"] in ("CLARIFY", "RELEASE")


def test_record_replays_exactly_and_detects_tampering():
    spec = json.loads((EX / "hydraulic_boom_spec.json").read_text())
    spec = api.new_spec(spec.pop("plant"), objective=spec.pop("objective"), **spec)
    y, u = SV.read_data((EX / "hydraulic_boom_data.csv").read_text())
    res = api.solve(spec, y[:300], u[:300])
    rec = json.loads(json.dumps(res["record"], default=float))
    assert api.replay(rec)["match"]
    bad = copy.deepcopy(rec)
    key = "x_filt" if "x_filt" in bad["outputs"] else "x_hat"
    bad["outputs"][key][10][0] += 1e-6
    assert not api.replay(bad)["match"]
