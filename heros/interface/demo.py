"""Quick demonstration: one request per specialist and one coordinated decision, run through the public pipeline."""
import json, pathlib
import numpy as np
from .. import api, record as R
from ..sim import plants as PL, loop
from . import service as SV

EX = pathlib.Path(__file__).resolve().parents[2] / "examples"


def run_demo(outdir="demo_output"):
    out = pathlib.Path(outdir); out.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    print("=" * 78 + "\n1. Rotary joint, guaranteed velocity-error limit (natural-language request)\n" + "=" * 78)
    model = json.loads((EX / "rotary_joint_model.json").read_text())
    script = json.loads((EX / "rotary_joint_answers.json").read_text())
    for q, a in zip(SV.QUESTIONS, script["answers"]):
        print(f"Q: {q}\nA: {a}")
    y, u, t = SV.read_data_t((EX / "rotary_joint_data.csv").read_text())
    conv = SV.Conversation(model, script["answers"], y, u, t=t)
    rb = conv.readback()
    ri = rb["reader"]                                  # which reader produced the readback (review C-11)
    print(f"\nReader: {ri.get('kind')} {ri.get('id') or ''}" + (f"  WARNING: {ri['note']}" if ri.get("note") else ""))
    print("Readback:")
    for r in rb["rows"]:
        print(f"  {r['path']:32s} = {json.dumps(r['value'] if r['value'] is not None else r['range'])[:60]:60s} [{r['status']}]")
    conv.confirm(rb["spec_hash"])
    res = conv.step()
    if res["decision"]["disposition"] == "CLARIFY":
        print("\n" + SV.summary(res["decision"]))
        path = res["decision"]["question"]["path"]
        print(f"(scripted answer: {script['clarifications'][path]})")
        conv.answer(path, script["clarifications"][path])
        res = conv.step()
    print("\n" + SV.summary(res["decision"]))
    if res["estimates"] is not None:
        (out / "joint.png").write_bytes(SV.plot_png(res["decision"], res["estimates"], "Rotary joint: sampled high-gain observer"))
        print("Assumption monitor:", res["estimates"]["assumption_monitor"]["status"])
    (out / "joint_record.json").write_text(R.dumps(res["record"]))
    print("replay of the exported record:", api.replay(json.loads((out / "joint_record.json").read_text()))["match"])

    print("\n" + "=" * 78 + "\n2. Hydraulic boom (4 states, 2 sensors), minimum mean-square-error estimate (structured request)\n" + "=" * 78)
    spec = json.loads((EX / "hydraulic_boom_spec.json").read_text())
    spec = api.new_spec(spec.pop("plant"), objective=spec.pop("objective"), **spec)
    y, u, t = SV.read_data_t((EX / "hydraulic_boom_data.csv").read_text())
    res = api.solve(spec, y, u, t=t)
    print(SV.summary(res["decision"]))
    (out / "boom.png").write_bytes(SV.plot_png(res["decision"], res["estimates"], "Hydraulic boom: Kalman filter, boom rate"))
    (out / "boom_record.json").write_text(R.dumps(res["record"]))
    print("replay of the exported record:", api.replay(json.loads((out / "boom_record.json").read_text()))["match"])

    print("\n" + "=" * 78 + "\n3. Servo axis: the same information, two definitions of success\n" + "=" * 78)
    base = json.loads((EX / "servo_axis_spec.json").read_text())
    for name, obj in (("smallest mean-square error", {"kind": "mse_optimal", "horizon": "steady_state"}),
                      ("guaranteed error limit", {"kind": "hard_bound", "tolerance": base["_hard_tolerance"], "deadline": base["_deadline"]})):
        b = {k: v for k, v in base.items() if not k.startswith("_")}
        spec = api.new_spec(b.pop("plant"), objective=obj, **b)
        d = api.decide(spec)
        print(f"-- success = {name}\n" + SV.summary(d) + "\n")
    print(f"Outputs written to {out}/")
