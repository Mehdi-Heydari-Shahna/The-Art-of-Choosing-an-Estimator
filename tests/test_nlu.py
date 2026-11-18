"""Language layer: deterministic number reading, assembly discipline, interface behaviour (no trained artifact required)."""
import json, math, pathlib
import numpy as np
import pytest
from heros.nlu.quantities import find_quantities
from heros.nlu.readers import RuleReader, NullReader
from heros.nlu import assemble as AS
from heros.core import spec as S
from heros.interface import service as SV

EX = pathlib.Path(__file__).resolve().parents[1] / "examples"


def values(text):
    return [(round(q.value, 12), q.dim) for q in find_quantities(text)]


@pytest.mark.parametrize("text,value,dim", [
    ("sampled at 1 kHz", 1000.0, "freq"),
    ("every half a millisecond", 5e-4, "time"),
    ("error below 0.4mm", 4e-4, "pos"),
    ("within 0.25 rad/s", 0.25, "vel"),
    ("noise variance 5e-5 rad^2", 5e-5, "pos2"),
    ("5 x 10^-5 rad", 5e-5, "pos"),
    ("one and a half seconds", 1.5, "time"),
    ("with probability 95 %", 0.95, "frac"),
])
def test_number_and_unit_reader(text, value, dim):
    qs = find_quantities(text)
    assert any(abs(q.value - value) <= 1e-9 * abs(value) and q.dim == dim for q in qs), [(q.value, q.dim) for q in qs]


def test_unreadable_numbers_are_never_bound():
    assert find_quantities("the tolerance is roughly small") == []


def test_null_reader_binds_nothing_and_nothing_is_released():
    model = json.loads((EX / "rotary_joint_model.json").read_text())
    script = json.loads((EX / "rotary_joint_answers.json").read_text())
    att = S.empty_spec(); att["plant"].update(model if "plant" not in model else model["plant"])
    text = "[Q1] " + script["answers"][0] + " [Q2] " + script["answers"][1] + " [Q3] " + script["answers"][2]
    asm = AS.assemble(att, text, NullReader().read(text, att["plant"]))
    assert S.get(asm["spec"], "deterministic.V") is None and S.get(asm["spec"], "objective.kind") is None


def test_interpreted_facts_need_confirmation_before_any_release():
    model = json.loads((EX / "rotary_joint_model.json").read_text())
    script = json.loads((EX / "rotary_joint_answers.json").read_text())
    conv = SV.Conversation(model, script["answers"])
    rb = conv.readback()
    interpreted = [r for r in rb["rows"] if r["status"] == "interpreted"]
    assert interpreted, "natural-language facts must be marked as interpreted"
    assert conv.step()["decision"]["disposition"] == "CONFIRM"
    conv.confirm(rb["spec_hash"])
    d = conv.step()["decision"]
    assert d["disposition"] != "CONFIRM"


def test_full_request_with_data_and_replay(tmp_path):
    model = json.loads((EX / "rotary_joint_model.json").read_text())
    script = json.loads((EX / "rotary_joint_answers.json").read_text())
    y, u = SV.read_data((EX / "rotary_joint_data.csv").read_text())
    conv = SV.Conversation(model, script["answers"], y, u)
    conv.confirm(conv.readback()["spec_hash"])
    out = conv.step()
    for _ in range(4):
        if out["decision"]["disposition"] != "CLARIFY":
            break
        path = out["decision"]["question"]["path"]
        conv.answer(path, script["clarifications"][path])
        out = conv.step()
    assert out["decision"]["disposition"] == "RELEASE" and out["estimates"] is not None
    from heros import record as R, api
    rec = json.loads(R.dumps(out["record"]))
    assert api.replay(rec)["match"]
    assert rec["interpretation_model"]["used"] is True and "request_text" in rec["interpretation_model"]
