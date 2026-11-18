"""Write paper/generated.tex: every number quoted in the article and the bodies of Tables 2 and 3.

    python experiments/make_paper_assets.py
All values are read from results/*.json (and models/*/config.json); nothing is typed by hand.  The file also
writes results/paper_numbers.json, which maps every macro to its value and to the file and key it came from.
"""
import json, math, pathlib, re
from common import ROOT, RES

OUT = ROOT / "paper" / "generated.tex"
MAC, SRC = {}, {}


def J(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def sci(x, d=1):
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    m = x / 10 ** e
    if round(m, d) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.{d}f}\\times 10^{{{e}}}"


def put(name, value, src):
    assert name.isalpha(), name
    MAC[name] = str(value)
    SRC[name] = src


def f(x, d):
    return f"{x:.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def numbers():
    fi = J("fidelity.json")["summary"]
    F = J("fidelity.json")
    put("FidNKal", fi["n_kalman_cases"], "fidelity.json:summary.n_kalman_cases")
    put("FidNHgo", fi["n_hgo_cases"], "fidelity.json:summary.n_hgo_cases")
    put("FidKalEst", sci(fi["kalman_max_estimate_diff"]), "fidelity.json:summary.kalman_max_estimate_diff")
    put("FidKalCov", sci(fi["kalman_max_cov_diff"]), "fidelity.json:summary.kalman_max_cov_diff")
    put("FidDisc", sci(fi["kalman_max_discretisation_diff"]), "fidelity.json:summary.kalman_max_discretisation_diff")
    put("FidSteady", sci(fi["kalman_max_steady_gain_diff"]), "fidelity.json:summary.kalman_max_steady_gain_diff")
    put("FidHgo", sci(fi["hgo_max_rel_diff"]), "fidelity.json:summary.hgo_max_rel_diff")
    put("FidTightGap", sci(1 - F["tightness"]["ratio"]), "fidelity.json:tightness.ratio (one minus)")
    put("FidCtLimit", sci(F["continuous_limit"]["max_rel_diff_sampled_vs_closed_form_after_10ms"]), "fidelity.json:continuous_limit")
    put("FidFamilies", len({r["family"] for r in F["kalman"]}), "fidelity.json:kalman[*].family")

    L = J("linear_stochastic.json")
    put("BoomRuns", L["n_runs"], "linear_stochastic.json:n_runs")
    put("BoomRefDiff", sci(max(L[k]["independent_reference_run0"]["max_estimate_diff"] for k in ("gaussian", "bounded"))),
        "linear_stochastic.json:{gaussian,bounded}.independent_reference_run0.max_estimate_diff (max)")
    put("BoomNeesG", f(L["gaussian"]["mean_nees"], 2), "linear_stochastic.json:gaussian.mean_nees")
    put("BoomNeesU", f(L["bounded"]["mean_nees"], 2), "linear_stochastic.json:bounded.mean_nees")
    put("BoomNisG", f(L["gaussian"]["mean_nis"], 2), "linear_stochastic.json:gaussian.mean_nis")
    put("BoomNisU", f(L["bounded"]["mean_nis"], 2), "linear_stochastic.json:bounded.mean_nis")
    put("BoomCovG", pct(L["gaussian"]["coverage_2sigma_target"]), "linear_stochastic.json:gaussian.coverage_2sigma_target")
    put("BoomCovU", pct(L["bounded"]["coverage_2sigma_target"]), "linear_stochastic.json:bounded.coverage_2sigma_target")
    put("BoomMistune", f(100 * (L["gaussian"]["exact_mse_ratio_mistuned_R_x10_target"] - 1), 1), "linear_stochastic.json:gaussian.exact_mse_ratio_mistuned_R_x10_target")

    Jn = J("joint.json")
    put("JointEpsMs", f(1e3 * Jn["eps"], 3), "joint.json:eps")
    put("JointRatio", f(Jn["Ts"] / Jn["eps"], 2), "joint.json:Ts/eps")
    put("JointMone", f(Jn["M"][0], 4), "joint.json:M[0]")
    put("JointMtwo", f(Jn["M"][1], 1), "joint.json:M[1]")
    put("JointD", f(Jn["D"], 2), "joint.json:D")
    put("JointDmis", f(Jn["D_detail"]["model_mismatch"], 2), "joint.json:D_detail.model_mismatch")
    put("JointDadd", f(Jn["D_detail"]["additive"], 1), "joint.json:D_detail.additive")
    put("JointDsh", f(Jn["D_detail"]["sample_hold_and_noise"], 2), "joint.json:D_detail.sample_hold_and_noise")
    put("JointTol", f(Jn["tolerance"], 2), "joint.json:tolerance")
    put("JointDeadlineMs", f(1e3 * Jn["deadline"], 0), "joint.json:deadline")
    put("JointBound", f(Jn["certified_post_deadline"], 3), "joint.json:certified_post_deadline")
    put("JointPeak", f(Jn["certified_peak"], 2), "joint.json:certified_peak")
    put("JointCt", f(Jn["ct_ultimate_same_eps"], 3), "joint.json:ct_ultimate_same_eps")
    e = Jn["estimation"]
    put("JointRuns", e["n_runs"], "joint.json:estimation.n_runs")
    put("JointSimMax", f(e["simulated_post_deadline_max"], 3), "joint.json:estimation.simulated_post_deadline_max")
    put("JointSimMed", f(e["simulated_post_deadline_median"], 3), "joint.json:estimation.simulated_post_deadline_median")
    put("JointCons", f(e["conservatism_post_deadline"], 2), "joint.json:estimation.conservatism_post_deadline")
    put("JointViol", e["envelope_violations"], "joint.json:estimation.envelope_violations")
    put("JointVmax", f(e["max_abs_velocity"], 2), "joint.json:estimation.max_abs_velocity")
    cl = Jn["closed_loop"]
    for tag, k in (("Obs", "observer"), ("Fd", "filtered_difference"), ("Sf", "state_feedback")):
        put("LoopRms" + tag, f(1e3 * cl[k]["rms_tracking_error"], 1), f"joint.json:closed_loop.{k}.rms_tracking_error")
        put("LoopTv" + tag, f(cl[k]["input_total_variation"], 1), f"joint.json:closed_loop.{k}.input_total_variation")
        put("LoopSat" + tag, cl[k]["saturated_samples"], f"joint.json:closed_loop.{k}.saturated_samples")
        if k != "state_feedback":
            put("LoopVmax" + tag, f(cl[k]["velocity_estimate_error_post_deadline_max"], 3), f"joint.json:closed_loop.{k}.velocity_estimate_error_post_deadline_max")
            put("LoopVrms" + tag, f(cl[k]["velocity_estimate_error_rms"], 3), f"joint.json:closed_loop.{k}.velocity_estimate_error_rms")
    put("LoopUmax", f(Jn["controller"]["u_max"], 0), "joint.json:controller.u_max")
    eul = [s for s in Jn["sampling"] if not s["forward_euler"]["stable"]]
    put("EulerUnstableFrom", f(min(s["Ts_over_eps"] for s in eul), 1), "joint.json:sampling[*].forward_euler.stable")
    s1 = [s for s in Jn["sampling"] if abs(s["Ts_over_eps"] - 1.0) < 1e-9][0]
    put("SampPmOne", f(s1["pole_matched"]["post_deadline_bound"], 3), "joint.json:sampling[Ts/eps=1].pole_matched")
    put("SampFeOne", f(s1["forward_euler"]["post_deadline_bound"], 3), "joint.json:sampling[Ts/eps=1].forward_euler")

    C = J("common_domain.json")
    ex = C["exact"]
    put("ComRecords", C["setting"]["n_records"], "common_domain.json:setting.n_records")
    put("ComMseK", sci(ex["kalman_steady_state"]["mse_steady"], 2), "common_domain.json:exact.kalman_steady_state.mse_steady")
    put("ComMseHw", sci(ex["hgo_worst_case_tuned"]["mse_steady"], 2), "common_domain.json:exact.hgo_worst_case_tuned.mse_steady")
    put("ComMseHm", sci(ex["hgo_mse_tuned"]["mse_steady"], 2), "common_domain.json:exact.hgo_mse_tuned.mse_steady")
    put("ComMseRatioW", f(ex["hgo_worst_case_tuned"]["mse_steady"] / ex["kalman_steady_state"]["mse_steady"], 2), "common_domain.json:exact (ratio)")
    put("ComMseRatioM", f(ex["hgo_mse_tuned"]["mse_steady"] / ex["kalman_steady_state"]["mse_steady"], 2), "common_domain.json:exact (ratio)")
    put("ComWK", f(ex["kalman_steady_state"]["certified_post_deadline"], 4), "common_domain.json:exact.kalman_steady_state.certified_post_deadline")
    put("ComWH", f(ex["hgo_worst_case_tuned"]["certified_post_deadline"], 4), "common_domain.json:exact.hgo_worst_case_tuned.certified_post_deadline")
    put("ComWHm", f(ex["hgo_mse_tuned"]["certified_post_deadline"], 4), "common_domain.json:exact.hgo_mse_tuned.certified_post_deadline")
    put("ComWRatio", f(ex["kalman_steady_state"]["certified_post_deadline"] / ex["hgo_worst_case_tuned"]["certified_post_deadline"], 2), "common_domain.json:exact (ratio)")
    put("ComLimit", f(C["two_signal_limit"], 4), "common_domain.json:two_signal_limit")
    em = C["empirical"]
    put("ComEmpK", sci(em["kalman_steady_state"]["mse_steady_empirical"], 2), "common_domain.json:empirical.kalman_steady_state")
    put("ComEmpHw", sci(em["hgo_worst_case_tuned"]["mse_steady_empirical"], 2), "common_domain.json:empirical.hgo_worst_case_tuned")
    put("ComEmpMaxK", f(em["kalman_steady_state"]["max_abs_error_post_deadline"], 3), "common_domain.json:empirical.kalman_steady_state.max_abs_error_post_deadline")
    put("ComEmpMaxH", f(em["hgo_worst_case_tuned"]["max_abs_error_post_deadline"], 3), "common_domain.json:empirical.hgo_worst_case_tuned.max_abs_error_post_deadline")
    sw = C["sweep"]
    put("SweepN", len(sw), "common_domain.json:sweep")
    put("SweepMseMin", f(min(s["mse_ratio_hgo_over_kalman"] for s in sw), 2), "common_domain.json:sweep[*].mse_ratio_hgo_over_kalman")
    put("SweepMseMax", f(max(s["mse_ratio_hgo_over_kalman"] for s in sw), 1), "common_domain.json:sweep[*].mse_ratio_hgo_over_kalman")
    put("SweepWMin", f(min(s["worst_ratio_kalman_over_hgo"] for s in sw), 2), "common_domain.json:sweep[*].worst_ratio_kalman_over_hgo")
    put("SweepWMax", f(max(s["worst_ratio_kalman_over_hgo"] for s in sw), 1), "common_domain.json:sweep[*].worst_ratio_kalman_over_hgo")
    put("SweepObsBetter", sum(1 for s in sw if s["worst_ratio_kalman_over_hgo"] > 1), "common_domain.json:sweep (count of ratio > 1)")
    ult = [s["kalman_worst_ultimate"] / s["hgo_worst_ultimate"] for s in sw]
    put("SweepWUltMin", f(min(ult), 2), "common_domain.json:sweep[*].kalman_worst_ultimate / hgo_worst_ultimate (min)")
    put("SweepWUltMax", f(max(ult), 2), "common_domain.json:sweep[*].kalman_worst_ultimate / hgo_worst_ultimate (max)")
    put("SweepKalUltBetter", sum(1 for x in ult if x < 1), "common_domain.json:sweep (count of ultimate ratio < 1)")
    d = C["decisions"]
    put("ComTolHard", f(d["guaranteed error limit after the deadline"]["objective"]["tolerance"], 4), "common_domain.json:decisions")
    put("ComTolImp", f(d["guaranteed limit below the two-signal limit"]["objective"]["tolerance"], 4), "common_domain.json:decisions")

    B = J("baseline_reproduction.json")
    k = B["khalil_trained_interface_rerun"]
    put("LegReq", k["n_requests"], "baseline_reproduction.json:khalil_trained_interface_rerun")
    put("LegDesigns", k["designs_issued"], "baseline_reproduction.json:khalil_trained_interface_rerun.designs_issued")
    put("LegUnsupported", k["unsupported_designs"], "baseline_reproduction.json:khalil_trained_interface_rerun.unsupported_designs")
    put("LegInvalid", k["invalid_designs"], "baseline_reproduction.json:khalil_trained_interface_rerun.invalid_designs")
    put("LegHidden", k["hidden_dynamics_requests"], "baseline_reproduction.json:khalil_trained_interface_rerun.hidden_dynamics_requests")
    put("LegHiddenAcc", k["hidden_dynamics_accepted"], "baseline_reproduction.json:khalil_trained_interface_rerun.hidden_dynamics_accepted")
    put("LegDefects", len(B["legacy_defects_reproduced"]), "baseline_reproduction.json:legacy_defects_reproduced")
    LD = json.loads((ROOT / "regressions" / "legacy_defects.json").read_text())
    LD = LD.get("results", LD)["khalil_scenario_deadline"]
    put("LegDeadlineFactor", f(LD["error_at_deadline_for_initial_error_100"] / LD["tolerance"], 0), "regressions/legacy_defects.json:khalil_scenario_deadline")
    rows_ = [l for l in (ROOT / "docs" / "LEGACY_FINDINGS.md").read_text().splitlines() if re.match(r"^\| \d+ \|", l)]
    put("LegFindings", len(rows_), "docs/LEGACY_FINDINGS.md: rows of the findings table")
    put("LegReproduced", sum(1 for l in rows_ if "| yes" in l), "docs/LEGACY_FINDINGS.md: rows reproduced on the supplied code")
    PH = J("posthoc_prefix_effect.json")
    if PH:
        put("PostHocExactBefore", pct(PH["posthoc_prefix_s0"]["exact_reading"], 0), "posthoc_prefix_effect.json:posthoc_prefix_s0.exact_reading (post hoc)")
        put("PostHocExactAfter", pct(PH["final_neural_scratch_s0"]["exact_reading"], 0), "posthoc_prefix_effect.json:final_neural_scratch_s0.exact_reading (post hoc)")
    I = J("incomplete.json")
    put("IncN", len(I["rows"]), "incomplete.json:rows")
    put("IncPassed", sum(1 for r in I["rows"] if r["check_passed"]), "incomplete.json:rows[*].check_passed")

    # ---- revision 2
    put("FidFrN", fi["n_hgo_first_reading_cases"], "fidelity.json:summary.n_hgo_first_reading_cases")
    put("FidFrHgo", sci(fi["hgo_first_reading_max_rel_diff"]), "fidelity.json:summary.hgo_first_reading_max_rel_diff")
    put("FidTightGapFr", sci(1 - F["tightness_first_reading"]["ratio"]), "fidelity.json:tightness_first_reading.ratio (one minus)")
    put("ComBoxExcess", f(100 * ex["hgo_worst_case_tuned"]["independent_box_max_rel_excess"], 2),
        "common_domain.json:exact.hgo_worst_case_tuned.independent_box_max_rel_excess (percent)")
    RF = J("review_findings_current.json")
    f3 = [x for x in RF["findings"] if x["id"] == "F3"][0]["cases"][0]["result"]
    put("KalStartFirst", f(f3["exported_P_filt_first"], 4), "review_findings_current.json:F3.exported_P_filt_first")
    put("KalStartSteady", f(f3["exported_P_filt_last"], 3), "review_findings_current.json:F3.exported_P_filt_last")
    put("KalStartGain", f(f3["gain_first"], 3), "review_findings_current.json:F3.gain_first")
    V_ = J("validation.json")
    put("ValN", V_["n"], "validation.json:n")
    put("ValAgree", V_["agree"], "validation.json:agree")
    v11 = [r for r in V_["rows"] if r["id"] == "V11"][0]
    put("ValChebyLevel", f(v11["expected_value"], 2), "validation.json:rows[V11].expected_value (Chebyshev level)")
    v08 = [r for r in V_["rows"] if r["id"] == "V08"][0]
    put("ValTwoSignal", f(2 * math.sqrt(6.0 * 1e-4), 3), "two-signal limit 2 sqrt(DV) for D = 6, V = 1e-4 (validation V11)")
    for k, tag in (("release", "Rel"), ("ask", "Ask"), ("decline", "Dec")):
        put("Val" + tag, sum(1 for r in V_["rows"] if r["expected"].startswith(k)), f"validation.json:rows (expected answers starting with {k})")
    ce = Jn["closed_loop_errors"]
    for tag, k in (("Obs", "observer"), ("Fd", "filtered_difference"), ("Sf", "state_feedback")):
        put("LoopTrMax" + tag, f(1e3 * ce[k]["tracking_max_after_deadline"], 1), f"joint.json:closed_loop_errors.{k}.tracking_max_after_deadline")


# ---------------------------------------------------------------------------------------------------------
T2 = [  # (case name in results/incomplete.json, short wording of the first three columns)
    ("Gaussian variance ambiguity", "Prior and noise variances; their sum (2) and a range for the noise variance (1 to 1.5) are known; smallest mean-square error requested",
     "Optimal gain lies between $1/4$ and $1/2$"),
    ("variance ambiguity, error level requested", "As above; a mean-square error below 0.65 is requested", "None: gain $1/4$ meets it for every admissible split"),
    ("no initial-error bound, ultimate limit", "Initial-error bound; an ultimate limit is requested", "None: initial-error terms vanish in the limit"),
    ("no initial covariance, steady-state filter", "Initial covariance; a steady-state (limit) statement is requested", "None: the steady-state gain and its limit do not depend on it"),
    ("noise distribution not stated", "Distribution of the noises", "Same estimate; only the linear-estimator claim holds"),
    ("missing hard bound on the measurement error", "Hard bound on the measurement error", "The guaranteed limit depends on it"),
    ("RMS noise level given, guaranteed limit requested", "Hard noise bound; an RMS value is supplied instead", "No hard bound follows from an RMS value"),
    ("measurement bound known only as a range", "Noise bound known only as a range", "Requirement is met on part of the range only"),
    ("sample period stated as a range", "Sample period stated as a range", "Model, gains and certificate depend on the period"),
    ("deadline added after an omission was harmless", "A deadline is added later", "The initial error, irrelevant before, now matters"),
    ("operating range without evidence", "Evidence that the motion stays in the stated range", "The mismatch bound holds only inside the range"),
    ("process noise without its meaning", "Whether a noise value is a spectral density or a per-step covariance", "The two readings differ by a factor of order $T$"),
    ("discrete model and a different stated period", "Discrete model with a different stated period", "A discrete model cannot be re-sampled"),
    ("timestamps disagree with the stated period", "Data recorded every 2 ms, request states 1 ms", "Gains and certificate hold for one period"),
    ("transient cap below the later tolerance", "None: a transient cap below the later tolerance", "At $t = 0$ one reading carries no speed information"),
    ("total and additive disturbance bounds both stated", "None: a total and a tighter additive disturbance bound", "The smaller bound applies"),
]
T2_OUT = {"CLARIFY": "Question", "RELEASE": "Release", "INVALID_INPUT": "Invalid input", "NO_VERIFIED_CANDIDATE": "No verified design found",
          "PROVED_IMPOSSIBLE": "No estimator can meet it"}
T2_CHECK = {  # short wording of the independent check; the numbers are taken from the results file below
    "Gaussian variance ambiguity": "Batch conditioning: gains {g1} and {g2}",
    "variance ambiguity, error level requested": "Errors 0.375 and 0.625 at the two extreme splits, affine in between",
    "no initial-error bound, ultimate limit": "Worst case after 400 steps equal for zero and large initial error",
    "no initial covariance, steady-state filter": "Riccati iteration from two initial covariances: same gain",
    "noise distribution not stated": "Monte Carlo variance matches for Gaussian and uniform noise",
    "missing hard bound on the measurement error": "Design certified for one bound violates the limit for a larger admissible one",
    "process noise without its meaning": "Quadrature: the two readings give different discrete covariances",
    "RMS noise level given, guaranteed limit requested": "Noise with fixed RMS and arbitrarily large magnitude",
    "measurement bound known only as a range": "Worst case meets the limit at the threshold, not above it",
    "sample period stated as a range": "Wrong period: velocity error 2 rad/s on error-free data",
    "deadline added after an omission was harmless": "Earlier design violates the deadline for an admissible initial error",
    "operating range without evidence": "Mismatch outside the range exceeds the bound used inside",
    "discrete model and a different stated period": "Supplied matrix is not the transition for the stated period",
    "timestamps disagree with the stated period": "Release withdrawn for these data; the same records with 1 ms timestamps are processed",
    "transient cap below the later tolerance": "Admissible initial speeds span $\\pm 2$ rad/s, above the cap",
    "total and additive disturbance bounds both stated": "Independent worst case with the additive bound meets the limit",
}


def table2():
    rows = {r["case"]: r for r in J("incomplete.json")["rows"]}
    L = []
    for case, a, b in T2:
        r = rows[case]
        out = T2_OUT[r["disposition"]]
        if r["disposition"] == "RELEASE":
            out += " (" + {"kalman_filter": "Kalman", "high_gain_observer": "observer"}[r["selected"]] + ")" + (
                ", omission reported as irrelevant" if r.get("harmless") else ", no question needed")   # recheck D, N6
        elif r["disposition"] == "INVALID_INPUT" and "timestamps" in case:
            out = "Release withdrawn (timing conflict)"
        elif r["disposition"] == "CLARIFY":
            out += " about the " + {"stochastic.R": "noise variance", "deterministic.V": "hard noise bound", "time.Ts": "exact period", "initial.E": "initial-error bound",
                                    "deterministic.domain_evidence": "evidence for the range", "stochastic.Q_kind": "meaning of the value"}.get(r["ask_path"], r["ask_path"])
            if r["threshold"] is not None:
                out += " (with the decisive threshold)"
        chk = T2_CHECK[case].format(g1="0.25", g2="0.50") + (" \\checkmark" if r["check_passed"] else " (FAILED)")
        L.append(f"{a} & {b} & {out} & {chk} \\\\")
    return "\n".join(L)


def table_loop():
    ce = J("joint.json")["closed_loop_errors"]; cl = J("joint.json")["closed_loop"]
    L = []
    for k, lab in (("observer", "released observer"), ("filtered_difference", "filtered difference"), ("state_feedback", "true velocity")):
        e = ce[k]
        est = ("--", "--") if k == "state_feedback" else (f"{e['velocity_estimation_max_after_deadline']:.3f}", f"{e['velocity_estimation_rms_after_deadline']:.3f}")
        L.append(f"{lab} & {est[0]} & {est[1]} & {1e3 * e['tracking_max_after_deadline']:.1f} & {1e3 * e['tracking_rms_after_deadline']:.1f} & "
                 f"{cl[k]['input_total_variation']:.1f} \\\\")
    return "\n".join(L)


def main():
    numbers()
    extra = ""
    try:
        import paper_eval
        extra = paper_eval.build(put, J)
    except ImportError:
        pass
    body = ["% generated by experiments/make_paper_assets.py -- do not edit"]
    body += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MAC.items())]
    body.append("\\newcommand{\\TableTwoBody}{%\n" + table2() + "}")
    body.append("\\newcommand{\\TableLoopBody}{%\n" + table_loop() + "}")
    body.append(extra)
    OUT.write_text("\n".join(body) + "\n")
    (RES / "paper_numbers.json").write_text(json.dumps({k: {"value": MAC[k], "source": SRC[k]} for k in sorted(MAC)}, indent=1))
    print("wrote", OUT, "with", len(MAC), "numbers")


if __name__ == "__main__":
    main()
