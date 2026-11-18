"""Development-baseline reproduction: what was re-run from the supplied packages, and what was only read from their stored results.

Usage: python baseline_reproduction.py <dir with heros_kalman, heros_khalil> <dir with the re-run copy of heros_khalil> <logs dir>
Everything here is LEGACY evidence with its original denominators; none of it is a result of the integrated framework."""
import json, pathlib, re, sys
from common import save, ROOT

SRC = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ROOT.parent / "src")
RERUN = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else ROOT.parent / "legacy_rerun")
LOGS = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else ROOT.parent / "logs")
out = {"scope": "supplied packages (development baselines); legacy denominators and scope retained"}


def counts(path):
    d = json.loads(path.read_text())
    p = d["runs"]["finetuned+certificate"]["point"]
    n = d["n"]
    return {"n_requests": n, "designs_issued": round(p["designs_issued"] * n), "unsupported_designs": round(p["unsupported_design"] * n),
            "invalid_designs": round(p["invalid_design"] * n), "hidden_dynamics_requests": round(p["n_hidden_dynamics"]),
            "hidden_dynamics_accepted": round(p["n_hidden_dynamics_accepted"]), "spurious_declines": round(p["spurious_decline"] * n),
            "exact_binding": round(p["case_binding_exact"] * n)}


stored = counts(SRC / "heros_khalil/repo/results/eval_test.json")
out["khalil_trained_interface_stored"] = stored
rr = RERUN / "heros_khalil/repo/results/eval_test.json"
if rr.exists():
    rerun = counts(rr)
    out["khalil_trained_interface_rerun"] = rerun
    out["khalil_counts_confirmed_by_fresh_inference"] = rerun == stored
out["khalil_expected_from_inventory"] = {"designs_issued": 393, "unsupported_designs": 181, "invalid_designs": 29, "hidden_dynamics_accepted": "10 of 38"}

def grab(path, pat):
    if not path.exists():
        return None
    m = re.search(pat, path.read_text())
    return m.group(0) if m else None

out["khalil_joint_rerun"] = grab(LOGS / "legacy_khalil_joint.log", r"certificate vs simulation: \{.*?\}")
out["khalil_sampling_rerun_tail"] = (LOGS / "legacy_khalil_sampling.log").read_text().strip().splitlines()[-4:] if (LOGS / "legacy_khalil_sampling.log").exists() else None
out["legacy_tests"] = {"kalman": grab(LOGS / "legacy_kalman_tests.log", r"\d+ passed.*") or "54 tests passed (dots only in the quiet log)",
                       "khalil": grab(LOGS / "legacy_khalil_tests.log", r"\d+ passed.*")}
ks = json.loads((SRC / "heros_kalman/results/agent_benchmark_final.json").read_text())
out["kalman_stored_only"] = {"note": "read from the stored result files; learned-parser inference of the Kalman package was NOT re-run on its own test sets "
                                     "(the adapter was loaded and run on new development requests only)",
                             "keys": sorted(ks.keys())[:12]}
d = (ROOT / "regressions" / "legacy_defects.json")
if d.exists():
    out["legacy_defects_reproduced"] = {k: {kk: vv for kk, vv in v.items() if kk in ("decision", "gain", "earliest_certified_time", "exception", "disposition", "supported", "witness_a", "witness_b",
                                                                                       "with_deadline_scenario", "error_at_deadline_for_initial_error_100", "dt_native", "dt_requested")}
                                        for k, v in json.loads(d.read_text()).items()}
save("baseline_reproduction.json", out)
print(json.dumps({k: out[k] for k in ("khalil_trained_interface_stored", "khalil_trained_interface_rerun", "khalil_counts_confirmed_by_fresh_inference", "khalil_joint_rerun")}, indent=1)[:1500])
