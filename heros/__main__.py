"""Command line:  python -m heros <command>

  demo                      quick demonstration (no training, no data needed)
  ask --model M.json [--answers A.json] [--data D.csv] [--out record.json]
                            the three opening questions, readback, confirmation, clarification, result
  solve --spec S.json [--data D.csv] [--out record.json] [--plot P.png]
                            structured specification (explicit fields count as confirmed)
  replay record.json        recompute a stored record and compare every quantity
  serve [--port 8765]       local web page with the same pipeline
"""
import argparse, json, pathlib, sys
import numpy as np
from . import api, record as R
from .core import spec as S
from .interface import service as SV


def _data(path):
    if not path:
        return None, None, None
    return SV.read_data_t(pathlib.Path(path).read_text())


def _finish(out, args):
    dec = out["decision"]
    print(SV.summary(dec))
    if out.get("timing") is not None:
        print("Timestamps: " + out["timing"]["message"])
    if out.get("estimates") is not None:
        est = out["estimates"]
        key = "x_hat" if "x_hat" in est else "x_filt"
        tgt = dec["config"].get("target") or 0
        print(f"Estimates: {len(est['t'])} samples; last value of the target = {np.asarray(est[key])[-1, tgt]:.6g}")
        print("Uncertainty type: " + est["uncertainty_type"])
        if est.get("assumption_monitor"):
            print("Assumption monitor: " + est["assumption_monitor"]["status"])
        if getattr(args, "plot", None):
            pathlib.Path(args.plot).write_bytes(SV.plot_png(dec, est)); print("plot written to", args.plot)
    if getattr(args, "out", None):
        pathlib.Path(args.out).write_text(R.dumps(out["record"])); print("record written to", args.out)


def cmd_solve(args):
    spec = json.loads(pathlib.Path(args.spec).read_text())
    if "schema" not in spec:
        spec = api.new_spec(spec.pop("plant"), objective=spec.pop("objective", None), ranges=spec.pop("ranges", None), **spec)
    y, u, t = _data(args.data)
    _finish(api.solve(spec, y, u, t=t), args)


def cmd_ask(args):
    model = json.loads(pathlib.Path(args.model).read_text())
    if args.answers:
        script = json.loads(pathlib.Path(args.answers).read_text())
        answers, replies = script["answers"], list(script.get("replies", []))
    else:
        answers, replies = [input(q + "\n> ") for q in SV.QUESTIONS], None
    y, u, t = _data(args.data)
    conv = SV.Conversation(model, answers, y, u, t=t)
    rb = conv.readback()
    ri = rb["reader"]
    print(f"Reader: {ri.get('kind')} {ri.get('id') or ''}" + (f"  WARNING: {ri['note']}" if ri.get("note") else ""))
    print("\nInterpreted specification (please review):")
    for r in rb["rows"]:
        v = r["value"] if r["value"] is not None else f"range {r['range']}"
        print(f"  {r['path']:32s} = {json.dumps(v)[:70]:70s} [{r['status']}] {r.get('source') or ''}")
    for i in rb["issues"]:
        print("  ! " + i["message"])
    if replies is not None:
        print("(scripted answers: the readback is confirmed by the script, not reviewed by a person)")
    ok = "y" if replies is not None else input("Confirm these facts? [y/N] ").strip().lower()
    if ok != "y":
        print("Not confirmed; nothing is released. Edit the answers or use a structured specification."); return
    conv.confirm(rb["spec_hash"])
    for _ in range(8):
        out = conv.step()
        dec = out["decision"]
        if dec["disposition"] != "CLARIFY":
            break
        print("\n" + SV.summary(dec))
        raw = replies.pop(0) if replies else (input("> (a value, or 'unknown') ") if replies is None else None)
        if raw is None:
            break
        if isinstance(raw, str) and raw.strip().lower() in ("unknown", "i do not know", "i don't know", "?"):
            conv.unknown(dec["question"]["path"])
            continue
        try:
            val = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            val = raw
        if dec["question"]["path"] == "stochastic.R" and isinstance(val, (int, float)):
            val = [[val]]
        conv.answer(dec["question"]["path"], val)
    print()
    _finish(out, args)


def cmd_replay(args):
    rep = R.replay(json.loads(pathlib.Path(args.record).read_text()))
    print(json.dumps({k: rep[k] for k in ("match", "schema_ok", "spec_hash_ok", "n_compared", "n_differences", "outputs_stored", "unverified")}, indent=1))
    for d in rep["differences"][:10]:
        print("  difference:", d)
    sys.exit(0 if rep["match"] else 1)


def cmd_demo(args):
    from .interface.demo import run_demo
    run_demo(args.outdir)


def cmd_serve(args):
    from .interface.web import serve
    serve(args.port)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="heros", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("demo"); p.add_argument("--outdir", default="demo_output"); p.set_defaults(f=cmd_demo)
    p = sub.add_parser("ask"); p.add_argument("--model", required=True); p.add_argument("--answers"); p.add_argument("--data"); p.add_argument("--out"); p.add_argument("--plot"); p.set_defaults(f=cmd_ask)
    p = sub.add_parser("solve"); p.add_argument("--spec", required=True); p.add_argument("--data"); p.add_argument("--out"); p.add_argument("--plot"); p.set_defaults(f=cmd_solve)
    p = sub.add_parser("replay"); p.add_argument("record"); p.set_defaults(f=cmd_replay)
    p = sub.add_parser("serve"); p.add_argument("--port", type=int, default=8765); p.set_defaults(f=cmd_serve)
    args = ap.parse_args(argv)
    args.f(args)


if __name__ == "__main__":
    main()
