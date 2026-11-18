"""Numbers and Table 3 of the article that come from model selection, the acceptance test and the cost measurements.
Called by make_paper_assets.py.  Every value is read from results/*.json or models/*/config.json."""
import json
from common import ROOT, RES


def pc(x, d=1):
    return "--" if x is None else f"{100 * x:.{d}f}"


def ci(c, d=1):
    return "--" if not c or c[0] is None else f"{100 * c[0]:.{d}f}--{100 * c[1]:.{d}f}"


NAMES = {"null": "Null", "rules": "Rules", "select_linear_s0": "Linear", "select_neural_pretrained_s0": "Pre", "select_neural_scratch_s0": "Scratch",
         "select_neural_frozen_s0": "Frozen", "supplied_khalil_spacy_finetuned": "LegKh", "supplied_kalman_roberta_lora": "LegKa"}


def build(put, J):
    out = []
    # ---- revision 2: relabelled cases, changes from run 5 to run 6, run 5 re-scored with the revised labels and judge
    import collections
    put("RelabelRestrictedN", sum(1 for c in json.loads((ROOT / "data" / "revision2_relabel.json").read_text())["cases"].values()
                                       if c.get("travel_or_speed_range_used")), "data/revision2_relabel.json: cases decided with a travel or speed range")
    RL = json.loads((ROOT / "data" / "revision2_relabel.json").read_text())
    put("RelabelN", RL["summary"]["cases"], "data/revision2_relabel.json:summary.cases")
    RH6 = J("run_history/run5_to_run6.json")
    if RH6:
        st6 = RH6["files"]["test_structured_rows.jsonl"]
        put("RunChangedStruct", st6["changed"], "run_history/run5_to_run6.json:files.test_structured_rows.jsonl.changed")
        put("RunChangedProb", sum(1 for r in st6["rows"] if r["policy"] == "khalil_only" and r["before"][1] == "PROVED_IMPOSSIBLE"),
            "run_history/run5_to_run6.json: khalil_only rows that were PROVED_IMPOSSIBLE in run 5")
        if "test_nl_rows.jsonl" in RH6["files"]:
            put("RunChangedNl", RH6["files"]["test_nl_rows.jsonl"]["changed"], "run_history/run5_to_run6.json:files.test_nl_rows.jsonl.changed")
    RS = ROOT / "results" / "run5" / "rescored_revision2" / "rescore_report.json"
    if RS.exists():
        rs = json.loads(RS.read_text())["changed"]
        put("RescoreFiStruct", sum(1 for c in rs if c["file"].startswith("test_structured") and c["after"] == "false_impossibility"),
            "run5/rescored_revision2/rescore_report.json: structured rows re-scored as false impossibility")
        put("RescoreFiNl", sum(1 for c in rs if c["file"].startswith("test_nl") and c["after"] == "false_impossibility"),
            "run5/rescored_revision2/rescore_report.json: language rows re-scored as false impossibility")
        put("RescoreOtherStruct", sum(1 for c in rs if c["file"].startswith("test_structured") and c["after"] == "nonrelease_other_reason"),
            "run5/rescored_revision2/rescore_report.json: structured rows re-scored as refusals for another reason")
        # run 5 language rows under the revised scoring: false impossibility claims without review, per seed of the trained reader
        sel5 = json.loads((ROOT / "models" / "selection.json").read_text())["selected_architecture"]
        tot = json.loads(RS.read_text())["totals"]
        cnt = collections.Counter({t["reader"]: t["n"] for t in tot if t["file"] == "test_nl_rows.jsonl" and t["policy"] == "coordinator"
                                   and t["user"] == "blind" and t["outcome"] == "false_impossibility"})
        put("NlTrBlFiListRunFive", ", ".join(str(cnt[f"final_{sel5}_s{s}"]) for s in (0, 1, 2)),
            "run5/rescored_revision2/rescore_report.json:totals (final_*|blind|coordinator, false impossibility, per seed)")
    # ---- review cases of 18 November 2026: changes from run 6 to run 7 (the reported run)
    RH7 = J("run_history/run6_to_run7.json")
    if RH7:
        put("RunSevenChangedStruct", RH7["files"]["test_structured_rows.jsonl"]["changed"],
            "run_history/run6_to_run7.json:files.test_structured_rows.jsonl.changed")
        nl7 = RH7["files"]["test_nl_rows.jsonl"]
        put("RunSevenChangedNl", nl7["changed"], "run_history/run6_to_run7.json:files.test_nl_rows.jsonl.changed")
        put("RunSevenFiToRefusal", sum(1 for r in nl7["rows"] if r["before"][0] == "false_impossibility" and r["after"][0] == "unnecessary_refusal"),
            "run_history/run6_to_run7.json: language rows from false impossibility to unneeded refusal")
    # ---- review cases of revision 2b: changes from run 7 to run 8 (the reported run)
    RH8 = J("run_history/run7_to_run8.json")
    if RH8:
        put("RunEightChangedStruct", RH8["files"]["test_structured_rows.jsonl"]["changed"],
            "run_history/run7_to_run8.json:files.test_structured_rows.jsonl.changed")
        put("RunEightChangedNl", RH8["files"]["test_nl_rows.jsonl"]["changed"], "run_history/run7_to_run8.json:files.test_nl_rows.jsonl.changed")
    S6 = J("run6/test_summary.json")          # shipped in both archives (the light archive leaves out the rows of run 6)
    if S6:
        sel6 = json.loads((ROOT / "models" / "selection.json").read_text())["selected_architecture"]
        put("NlTrBlFiListRunSix", ", ".join(str(S6["nl"][f"final_{sel6}_s{s}|blind|coordinator"]["false_impossibility"]) for s in (0, 1, 2)),
            "run6/test_summary.json:nl.final_*|blind|coordinator.false_impossibility")
    AU = J("blind_impossibility_audit.json")
    if AU:
        put("NlFiAudN", AU["n_false_impossibility"], "blind_impossibility_audit.json:n_false_impossibility")
        put("NlFiFirstCap", AU["first_sample_with_misread_cap"], "blind_impossibility_audit.json:first_sample_with_misread_cap")
        put("NlFiCapTol", AU["first_sample_cap_equals_stated_tolerance"], "blind_impossibility_audit.json:first_sample_cap_equals_stated_tolerance")
    D = J("dev_selection.json")
    if D:
        for k, v in D.items():
            if not (isinstance(v, dict) and "templated_D" in v):
                continue
            tag = NAMES.get(k) or ("Leg" + "".join(ch for ch in k.title() if ch.isalpha())[:10])
            put("Sel" + tag + "Score", f"{v['selection_score']:.2f}", f"dev_selection.json:{k}.selection_score")
            put("Sel" + tag + "ExactT", pc(v["templated_D"]["exact_reading"], 0), f"dev_selection.json:{k}.templated_D.exact_reading")
            put("Sel" + tag + "ExactA", pc(v["authored_dev"]["exact_reading"], 0), f"dev_selection.json:{k}.authored_dev.exact_reading")
            put("Sel" + tag + "RoleT", pc(v["templated_D"]["role_accuracy"], 0), f"dev_selection.json:{k}.templated_D.role_accuracy")
            put("Sel" + tag + "RoleA", pc(v["authored_dev"]["role_accuracy"], 0), f"dev_selection.json:{k}.authored_dev.role_accuracy")
            if v.get("n_trainable_parameters"):
                put("Sel" + tag + "Params", f"{v['n_trainable_parameters']:,}".replace(",", "\\,"), f"dev_selection.json:{k}.n_trainable_parameters")
            if v.get("train_seconds"):
                ts = v["train_seconds"]
                put("Sel" + tag + "Train", (f"{ts:.0f}\\unit{{s}}" if ts < 120 else f"{ts / 60:.0f}\\unit{{min}}"), f"dev_selection.json:{k}.train_seconds")
        put("SelNT", D["rules"]["templated_D"]["n_requests"], "dev_selection.json:rules.templated_D.n_requests")
        au = D["rules"]["authored_dev"]
        put("SelQPerReq", f"{au['n_quantities'] / au['n_requests']:.0f}", "dev_selection.json:rules.authored_dev.n_quantities / n_requests")
        put("SelScratchMs", f"{D['select_neural_scratch_s0']['authored_dev']['ms_per_request']:.0f}", "dev_selection.json:select_neural_scratch_s0.authored_dev.ms_per_request")
        put("SelNA", D["rules"]["authored_dev"]["n_requests"], "dev_selection.json:rules.authored_dev.n_requests")
        cfg = json.loads((ROOT / "models" / "final_neural_scratch_s0" / "config.json").read_text())   # deployed reader, seed 0
        put("ReaderTrainReq", f"{cfg['n_train_requests']:,}".replace(",", "\\,"), "models/final_neural_scratch_s0/config.json:n_train_requests")
        put("ReaderTrainQty", f"{cfg['n_train_quantities']:,}".replace(",", "\\,"), "models/final_neural_scratch_s0/config.json:n_train_quantities")
        put("ReaderHidden", cfg["hidden"], "models/final_neural_scratch_s0/config.json:hidden")
        EXPR = {"supplied_khalil_spacy_finetuned": ("LegKh", {"meas_bound", "meas_std", "acc_bound", "init_pos_bound", "init_vel_bound", "deadline", "tolerance", "peak_cap", "Ts", "vel_limit", "other"}),
                "supplied_kalman_roberta_lora": ("LegKa", {"Ts", "fs", "meas_std", "meas_var", "acc_std", "init_pos_std", "init_vel_std", "other"})}
        for k, (tag, roles) in EXPR.items():
            if k in D:
                for part, P in (("templated_D", "T"), ("authored_dev", "A")):
                    prr = D[k][part]["per_role"]
                    tot = sum(v["n"] for v in prr.values())
                    n = sum(v["n"] for r, v in prr.items() if r in roles); ok = sum(v["n"] * v["acc"] for r, v in prr.items() if r in roles)
                    put("Sel" + tag + "Expr" + P, pc(n / tot, 0), f"dev_selection.json:{k}.{part}.per_role (share of numbers whose role the model can output, 'other' included)")
                    put("Sel" + tag + "ExprAcc" + P, pc(ok / n, 0), f"dev_selection.json:{k}.{part}.per_role (role accuracy on those numbers)")
    S = J("test_summary.json")
    if not S:
        return ""
    st, nl = S["structured"], S["nl"]
    co = st["coordinator"]
    put("EvalNStruct", co["n"], "test_summary.json:structured.coordinator.n")
    put("EvalNReleasable", co["n_releasable"], "test_summary.json:structured.coordinator.n_releasable")
    put("EvalNNotReleasable", co["n_not_releasable"], "test_summary.json:structured.coordinator.n_not_releasable")
    for pol, tag in (("coordinator", "Co"), ("selector", "Sel"), ("kalman_only", "Ka"), ("khalil_only", "Kh"), ("ask_all", "Ask"), ("default_fill", "Def")):
        m = st[pol]
        put("St" + tag + "Inc", m["incorrect_releases"], f"test_summary.json:structured.{pol}.incorrect_releases")
        put("St" + tag + "Rel", m["released"], f"test_summary.json:structured.{pol}.released")
        put("St" + tag + "Cov", pc(m["useful_coverage"]), f"test_summary.json:structured.{pol}.useful_coverage")
        put("St" + tag + "Q", f"{m['questions_per_request']:.2f}", f"test_summary.json:structured.{pol}.questions_per_request")
        put("St" + tag + "Unnec", m["unnecessary_refusals"], f"test_summary.json:structured.{pol}.unnecessary_refusals")
        put("St" + tag + "Other", m["nonrelease_other_reason"], f"test_summary.json:structured.{pol}.nonrelease_other_reason")
        put("St" + tag + "Err", m["errors"], f"test_summary.json:structured.{pol}.errors")
        put("St" + tag + "Unres", m["unresolved"], f"test_summary.json:structured.{pol}.unresolved")
    put("StCoIncCiHi", pc(co["incorrect_per_request_ci"][1], 2), "test_summary.json:structured.coordinator.incorrect_per_request_ci[1]")
    if co["incorrect_releases"] or co["false_impossibility"]:
        raise SystemExit("paper_eval: the article states no incorrect release and no false impossibility claim on the structured set; it must change")
    put("StCoCorrectNon", co["correct_nonrelease"], "test_summary.json:structured.coordinator.correct_nonrelease")
    h = co["held_out_families"]
    put("StCoHeldN", h["n"], "test_summary.json:structured.coordinator.held_out_families.n")
    put("StCoHeldInc", h["incorrect_releases"], "test_summary.json:structured.coordinator.held_out_families.incorrect_releases")
    put("StCoHeldCov", pc(h["useful_coverage"]), "test_summary.json:structured.coordinator.held_out_families.useful_coverage")
    qs = S["structured_question_saving"]
    put("StQSaving", pc(qs["relative_reduction"], 0), "test_summary.json:structured_question_saving.relative_reduction")
    CE = J("criterion_effect.json")
    if CE:
        rat = [c["ratio_selector_over_coordinator"] for c in CE["cases"] if c["ratio_selector_over_coordinator"]]
        put("CritNDiff", CE["n_different_choice"], "criterion_effect.json:n_different_choice")
        put("CritNHard", len(rat), "criterion_effect.json:cases with a ratio (hard-bound objective)")
        put("CritMin", f"{100 * (min(rat) - 1):.0f}", "criterion_effect.json:ratio_selector_over_coordinator (min, percent above 1)")
        put("CritMax", f"{100 * (max(rat) - 1):.0f}", "criterion_effect.json:ratio_selector_over_coordinator (max, percent above 1)")
        prob = [c for c in CE["cases"] if c["objective"] == "probability"]
        if len(prob) + len(rat) != CE["n_different_choice"] or len(prob) > 1:
            raise SystemExit("paper_eval: the selector cases are no longer one probability case plus hard-bound cases; the article's wording must change")
        for c in prob:   # review D, m19: the probability-objective case (criterion = one minus the provable level)
            put("CritProbSel", f"{1 - c['criterion_values'][c['selector']]:.2f}", f"criterion_effect.json:{c['id']} (one minus the selector's criterion value)")
            put("CritProbCo", f"{1 - c['criterion_values'][c['coordinator']]:.2f}", f"criterion_effect.json:{c['id']} (one minus the coordinator's criterion value)")
    tr = sorted({k.split("|")[0] for k in nl if k.startswith("final_")})
    any_nl = nl[f"{tr[0]}|attentive|coordinator"]
    put("EvalNNl", any_nl["n"], "test_summary.json:nl.*.n")
    put("EvalNNlAuthored", json.loads((ROOT / "data" / "manifest.json").read_text())["counts"]["test_nl"], "data/manifest.json:counts.test_nl")
    put("EvalNNlReleasable", any_nl["n_releasable"], "test_summary.json:nl.*.n_releasable")
    sv = S["nl_seed_variability"]
    for us, U in (("attentive", "Att"), ("blind", "Bl")):
        v = sv[us]
        put("NlTr" + U + "CovMean", pc(v["useful_coverage"]["mean"]), f"test_summary.json:nl_seed_variability.{us}.useful_coverage.mean")
        put("NlTr" + U + "CovMin", pc(v["useful_coverage"]["min"]), f"test_summary.json:nl_seed_variability.{us}.useful_coverage.min")
        put("NlTr" + U + "CovMax", pc(v["useful_coverage"]["max"]), f"test_summary.json:nl_seed_variability.{us}.useful_coverage.max")
        put("NlTr" + U + "EditsMean", f"{v['edits_per_request']['mean']:.2f}", f"test_summary.json:nl_seed_variability.{us}.edits_per_request.mean")
        put("NlTr" + U + "ZeroEdit", pc(v["zero_edit_requests"]["mean"], 0), f"test_summary.json:nl_seed_variability.{us}.zero_edit_requests.mean")
        put("NlTr" + U + "ZeroEditMin", pc(v["zero_edit_requests"]["min"], 0), f"test_summary.json:nl_seed_variability.{us}.zero_edit_requests.min")
        put("NlTr" + U + "ZeroEditMax", pc(v["zero_edit_requests"]["max"], 0), f"test_summary.json:nl_seed_variability.{us}.zero_edit_requests.max")
        if "corrected_requests_share" in v:
            put("NlTr" + U + "Corrected", pc(v["corrected_requests_share"]["mean"], 0), f"test_summary.json:nl_seed_variability.{us}.corrected_requests_share.mean")
            put("NlTr" + U + "EditsCorr", f"{v['edits_per_corrected_request']['mean']:.1f}", f"test_summary.json:nl_seed_variability.{us}.edits_per_corrected_request.mean")
        incs = [nl[f"{t}|{us}|coordinator"]["incorrect_releases"] for t in tr]
        put("NlTr" + U + "IncList", ", ".join(str(i) for i in incs), f"test_summary.json:nl.final_*|{us}|coordinator.incorrect_releases")
        put("NlTr" + U + "IncMax", max(incs), f"test_summary.json:nl.final_*|{us}|coordinator.incorrect_releases (max)")
        put("NlTr" + U + "IncSum", sum(incs), f"test_summary.json:nl.final_*|{us}|coordinator.incorrect_releases (sum over seeds)")
    # the first run (kept in results/before_change/), scored with the earlier judge
    import json as _json
    first = ROOT / "results" / "before_change" / "test_summary.json"
    if first.exists():
        F = _json.loads(first.read_text())["nl"]
        put("NlTrBlIncListFirst", ", ".join(str(F[f"{t}|blind|coordinator"]["incorrect_releases"]) for t in tr),
            "results/before_change/test_summary.json:nl.final_*|blind|coordinator.incorrect_releases (first run, earlier judge)")
    # categories of the failures without review (review B9) and overlap between seeds (review B10)
    ov = S.get("nl_seed_overlap")
    if ov:
        put("NlTrBlCatSeedZero", len(ov["blind_incorrect_categories"][tr[0]]), f"test_summary.json:nl_seed_overlap.blind_incorrect_categories.{tr[0]} (count)")
        put("NlTrBlCatUnion", len(ov["blind_incorrect_categories_union"]), "test_summary.json:nl_seed_overlap.blind_incorrect_categories_union (count)")
        put("NlCatTotal", ov["n_categories_total"], "test_summary.json:nl_seed_overlap.n_categories_total")
        put("NlSeedExactAll", ov["exact_reading_all_seeds"], "test_summary.json:nl_seed_overlap.exact_reading_all_seeds")
        put("NlSeedExactAny", ov["exact_reading_any_seed"], "test_summary.json:nl_seed_overlap.exact_reading_any_seed")
        put("NlSeedFailUnion", ov["blind_incorrect_union"], "test_summary.json:nl_seed_overlap.blind_incorrect_union")
        put("NlSeedFailAll", ov["blind_incorrect_all_seeds"], "test_summary.json:nl_seed_overlap.blind_incorrect_all_seeds")
        put("NlSeedFailOne", ov["blind_incorrect_one_seed_only"], "test_summary.json:nl_seed_overlap.blind_incorrect_one_seed_only")
    # held-out versus training families without review (review B7)
    fs = S.get("nl_family_split")
    if fs:
        k0 = f"{tr[0]}|blind|coordinator"
        put("NlHeldN", fs[k0]["held_out"]["n"], f"test_summary.json:nl_family_split.{k0}.held_out.n")
        put("NlTrainN", fs[k0]["training"]["n"], f"test_summary.json:nl_family_split.{k0}.training.n")
        put("NlHeldBlIncList", ", ".join(str(fs[f"{t}|blind|coordinator"]["held_out"]["incorrect_releases"]) for t in tr), "test_summary.json:nl_family_split.final_*|blind|coordinator.held_out.incorrect_releases")
        put("NlTrainBlIncList", ", ".join(str(fs[f"{t}|blind|coordinator"]["training"]["incorrect_releases"]) for t in tr), "test_summary.json:nl_family_split.final_*|blind|coordinator.training.incorrect_releases")
        put("NlHeldBlPctList", ", ".join(pc(fs[f"{t}|blind|coordinator"]["held_out"]["incorrect_per_request"]) for t in tr), "test_summary.json:nl_family_split.final_*|blind|coordinator.held_out.incorrect_per_request")
        put("NlTrainBlPctList", ", ".join(pc(fs[f"{t}|blind|coordinator"]["training"]["incorrect_per_request"]) for t in tr), "test_summary.json:nl_family_split.final_*|blind|coordinator.training.incorrect_per_request")
        for rd, tag in (("rules", "Rules"), ("null", "Null"), ("select_neural_frozen_s0", "Frozen")):
            kk = f"{rd}|blind|coordinator"
            if kk in fs:
                put("Nl" + tag + "HeldBlInc", fs[kk]["held_out"]["incorrect_releases"], f"test_summary.json:nl_family_split.{kk}.held_out.incorrect_releases")
                put("Nl" + tag + "TrainBlInc", fs[kk]["training"]["incorrect_releases"], f"test_summary.json:nl_family_split.{kk}.training.incorrect_releases")
                put("Nl" + tag + "HeldBlPct", pc(fs[kk]["held_out"]["incorrect_per_request"]), f"test_summary.json:nl_family_split.{kk}.held_out.incorrect_per_request")
                put("Nl" + tag + "TrainBlPct", pc(fs[kk]["training"]["incorrect_per_request"]), f"test_summary.json:nl_family_split.{kk}.training.incorrect_per_request")
    # paired comparisons on the same requests, cluster bootstrap over (family, author) (review B7)
    pr = S.get("nl_paired")
    if pr:
        # paired differences, trained reader minus baseline; the article states them as reductions, so all must be negative
        def lower(vals, what):
            if max(vals) >= 0:
                raise SystemExit(f"paper_eval: {what}: not all paired differences are negative ({vals}); the article's wording must change")
            return [-v for v in vals]
        def sigtext(k):
            # "point estimates" unless an interval excludes zero (review B7c)
            return "no interval excludes zero" if k == 0 else f"intervals excluding zero for {k} of {len(tr)} seeds"
        for base, tag in (("null", "Null"), ("rules", "Rules")):
            ks = [f"{t}-{base}|blind" for t in tr]
            d = lower([pr[k]["d_incorrect_per_request"] for k in ks], f"incorrect vs {base}"); c = [pr[k]["ci_incorrect"] for k in ks]
            put("Pair" + tag + "IncLessMin", f"{100 * min(d):.1f}", f"test_summary.json:nl_paired.final_*-{base}|blind.d_incorrect_per_request (smallest reduction, percentage points)")
            put("Pair" + tag + "IncLessMax", f"{100 * max(d):.1f}", f"test_summary.json:nl_paired.final_*-{base}|blind.d_incorrect_per_request (largest reduction, percentage points)")
            put("Pair" + tag + "IncSig", sum(1 for lo, hi in c if hi < 0 or lo > 0), f"test_summary.json:nl_paired.final_*-{base}|blind.ci_incorrect (seeds whose interval excludes zero)")
            put("Pair" + tag + "IncSigText", sigtext(sum(1 for lo, hi in c if hi < 0 or lo > 0)), f"test_summary.json:nl_paired.final_*-{base}|blind.ci_incorrect (wording)")
            ka = [f"{t}-{base}|attentive" for t in tr]
            e = lower([pr[k]["d_edits_per_request"] for k in ka], f"edits vs {base}"); ce = [pr[k]["ci_edits"] for k in ka]
            put("Pair" + tag + "EditsLessMin", f"{min(e):.2f}", f"test_summary.json:nl_paired.final_*-{base}|attentive.d_edits_per_request (smallest reduction)")
            put("Pair" + tag + "EditsLessMax", f"{max(e):.2f}", f"test_summary.json:nl_paired.final_*-{base}|attentive.d_edits_per_request (largest reduction)")
            put("Pair" + tag + "EditsSig", sum(1 for lo, hi in ce if hi < 0 or lo > 0), f"test_summary.json:nl_paired.final_*-{base}|attentive.ci_edits (seeds whose interval excludes zero)")
            put("Pair" + tag + "EditsSigText", sigtext(sum(1 for lo, hi in ce if hi < 0 or lo > 0)), f"test_summary.json:nl_paired.final_*-{base}|attentive.ci_edits (wording)")
        ks = [f"{t}-null|blind" for t in tr]
        dc = lower([pr[k]["d_useful_coverage"] for k in ks], "coverage vs null"); cc = [pr[k]["ci_coverage"] for k in ks]
        put("PairNullCovLessMin", f"{100 * min(dc):.1f}", "test_summary.json:nl_paired.final_*-null|blind.d_useful_coverage (smallest reduction, points)")
        put("PairNullCovLessMax", f"{100 * max(dc):.1f}", "test_summary.json:nl_paired.final_*-null|blind.d_useful_coverage (largest reduction, points)")
        put("PairNullCovSig", sum(1 for lo, hi in cc if hi < 0 or lo > 0), "test_summary.json:nl_paired.final_*-null|blind.ci_coverage (seeds whose interval excludes zero)")
    # false impossibility claims (review B, recheck N1) and answers per request (review B4)
    put("StCoFi", co["false_impossibility"], "test_summary.json:structured.coordinator.false_impossibility")
    put("StFiAll", sum(m["false_impossibility"] for m in st.values()), "test_summary.json:structured.*.false_impossibility (sum over policies)")
    put("NlTrBlFiList", ", ".join(str(nl[f"{t}|blind|coordinator"]["false_impossibility"]) for t in tr), "test_summary.json:nl.final_*|blind|coordinator.false_impossibility")
    put("NlTrAttFiSum", sum(nl[f"{t}|attentive|coordinator"]["false_impossibility"] for t in tr), "test_summary.json:nl.final_*|attentive|coordinator.false_impossibility (sum)")
    for rd, tag in (("null", "Null"), ("rules", "Rules"), ("select_neural_frozen_s0", "Frozen")):
        for us, U in (("attentive", "Att"), ("blind", "Bl")):
            m = nl.get(f"{rd}|{us}|coordinator")
            if m:
                put("Nl" + tag + U + "Fi", m["false_impossibility"], f"test_summary.json:nl.{rd}|{us}|coordinator.false_impossibility")
                put("Nl" + tag + U + "Ans", f"{m['answers_per_request']:.2f}", f"test_summary.json:nl.{rd}|{us}|coordinator.answers_per_request")
    for us, U in (("attentive", "Att"), ("blind", "Bl")):
        m = nl.get(f"{tr[0]}|{us}|single_model")
        if m:
            put("NlSingle" + U + "Fi", m["false_impossibility"], f"test_summary.json:nl.{tr[0]}|{us}|single_model.false_impossibility")
    put("NlFiTotal", sum(m["false_impossibility"] for m in nl.values()), "test_summary.json:nl.*.false_impossibility (sum over configurations)")
    att = {k: m for k, m in nl.items() if k.endswith("|attentive|coordinator")}
    if any(m["incorrect_releases"] or m["false_impossibility"] for m in att.values()):
        raise SystemExit("paper_eval: the article states that every reader reaches the oracle's outcome with attentive review; it must change")
    # a scripted user that knows only the ranges it stated (review B4)
    lim = st.get("coordinator_limited_user")
    if lim:
        put("StLimCov", pc(lim["useful_coverage"]), "test_summary.json:structured.coordinator_limited_user.useful_coverage")
        put("StLimInc", lim["incorrect_releases"], "test_summary.json:structured.coordinator_limited_user.incorrect_releases")
        put("StLimUnnec", lim["unnecessary_refusals"], "test_summary.json:structured.coordinator_limited_user.unnecessary_refusals")
        put("StLimUnknown", lim.get("answers_unknown_total", 0), "test_summary.json:structured.coordinator_limited_user.answers_unknown_total")
    # family-level result of the structured set (review B6) and the time per request in the test (review B12)
    fam = co["by_family"]
    put("StCoFamN", len(fam), "test_summary.json:structured.coordinator.by_family (count)")
    put("StCoFamNoInc", sum(1 for f in fam.values() if f["incorrect_releases"] == 0), "test_summary.json:structured.coordinator.by_family (families without an incorrect release)")
    put("StCoSecMax", f"{co['seconds_per_request_max']:.1f}", "test_summary.json:structured.coordinator.seconds_per_request_max")
    put("StCoOverOneS", co["requests_over_1s"], "test_summary.json:structured.coordinator.requests_over_1s")
    put("StAskUnknown", st["ask_all"].get("answers_unknown_total", 0), "test_summary.json:structured.ask_all.answers_unknown_total")
    put("StAskQTotal", int(round(st["ask_all"]["questions_per_request"] * st["ask_all"]["n"])), "test_summary.json:structured.ask_all.questions_per_request * n")
    for rd, tag in (("null", "Null"), ("rules", "Rules"), ("select_neural_frozen_s0", "Frozen")):
        for us, U in (("attentive", "Att"), ("blind", "Bl")):
            m = nl.get(f"{rd}|{us}|coordinator")
            if m:
                put("Nl" + tag + U + "Inc", m["incorrect_releases"], f"test_summary.json:nl.{rd}|{us}|coordinator.incorrect_releases")
                put("Nl" + tag + U + "Cov", pc(m["useful_coverage"]), f"test_summary.json:nl.{rd}|{us}|coordinator.useful_coverage")
                put("Nl" + tag + U + "Edits", f"{m['edits_per_request']:.2f}", f"test_summary.json:nl.{rd}|{us}|coordinator.edits_per_request")
    for us, U in (("attentive", "Att"), ("blind", "Bl")):
        m = nl.get(f"{tr[0]}|{us}|single_model")
        if m:
            put("NlSingle" + U + "Inc", m["incorrect_releases"], f"test_summary.json:nl.{tr[0]}|{us}|single_model.incorrect_releases")
            put("NlSingle" + U + "Rel", m["released"], f"test_summary.json:nl.{tr[0]}|{us}|single_model.released")
            put("NlSingle" + U + "Cov", pc(m["useful_coverage"]), f"test_summary.json:nl.{tr[0]}|{us}|single_model.useful_coverage")
    ho = nl[f"{tr[0]}|attentive|coordinator"]["held_out_families"]
    put("NlTrHeldN", ho["n"], f"test_summary.json:nl.{tr[0]}|attentive|coordinator.held_out_families.n")
    put("NlTrHeldInc", ho["incorrect_releases"], f"test_summary.json:nl.{tr[0]}|attentive|coordinator.held_out_families.incorrect_releases")
    put("NlTrHeldEdits", f"{ho['edits_per_request']:.2f}", f"test_summary.json:nl.{tr[0]}|attentive|coordinator.held_out_families.edits_per_request")
    put("NlTrAttQ", f"{nl[tr[0] + '|attentive|coordinator']['questions_per_request']:.2f}", f"test_summary.json:nl.{tr[0]}|attentive|coordinator.questions_per_request")
    put("NlAskAttQ", f"{nl[tr[0] + '|attentive|ask_all']['questions_per_request']:.2f}", f"test_summary.json:nl.{tr[0]}|attentive|ask_all.questions_per_request")
    put("NlNullBlQ", f"{nl['null|blind|coordinator']['questions_per_request']:.2f}", "test_summary.json:nl.null|blind|coordinator.questions_per_request")
    put("NlTrBlQ", f"{nl[tr[0] + '|blind|coordinator']['questions_per_request']:.2f}", f"test_summary.json:nl.{tr[0]}|blind|coordinator.questions_per_request")
    orc = S.get("nl_oracle_structured")
    if orc:
        put("NlOracleCov", pc(orc["useful_coverage"]), "test_summary.json:nl_oracle_structured.useful_coverage")
        put("NlOracleInc", orc["incorrect_releases"], "test_summary.json:nl_oracle_structured.incorrect_releases")
    Rg = J("test_reading.json")
    if Rg:
        for rd, tag in (("null", "Null"), ("rules", "Rules"), ("select_neural_frozen_s0", "Frozen")):
            put("Rd" + tag + "Role", pc(Rg[rd]["role_accuracy"], 0), f"test_reading.json:{rd}.role_accuracy")
            put("Rd" + tag + "Exact", pc(Rg[rd]["exact_reading"], 0), f"test_reading.json:{rd}.exact_reading")
        fr = [Rg[t] for t in tr if t in Rg]
        if fr:
            import numpy as np
            put("RdTrRole", pc(float(np.mean([m["role_accuracy"] for m in fr])), 0), "test_reading.json:final_*.role_accuracy (mean over seeds)")
            put("RdTrExact", pc(float(np.mean([m["exact_reading"] for m in fr])), 0), "test_reading.json:final_*.exact_reading (mean over seeds)")
            put("RdTrExactMin", pc(min(m["exact_reading"] for m in fr), 0), "test_reading.json:final_*.exact_reading (min)")
            put("RdTrExactMax", pc(max(m["exact_reading"] for m in fr), 0), "test_reading.json:final_*.exact_reading (max)")
            put("RdTrHeldExact", pc(float(np.mean([m["held_out_families"]["exact_reading"] for m in fr])), 0), "test_reading.json:final_*.held_out_families.exact_reading (mean)")
            put("RdTrExactSeedZero", pc(fr[0]["exact_reading"], 0), f"test_reading.json:{tr[0]}.exact_reading")
            put("RdTrHeldExactSeedZero", pc(fr[0]["held_out_families"]["exact_reading"], 0), f"test_reading.json:{tr[0]}.held_out_families.exact_reading")
            put("RdTrProposal", pc(float(np.mean([m["proposal_accuracy"] for m in fr if m["proposal_accuracy"] is not None])), 0), "test_reading.json:final_*.proposal_accuracy (mean)")
            a0 = fr[0]["by_author"]
            for a in sorted(a0):
                put("RdTrExactAuthor" + {"author1": "One", "author2": "Two", "author3": "Three"}[a],
                    pc(float(np.mean([m["by_author"][a]["exact_reading"] for m in fr])), 0), f"test_reading.json:final_*.by_author.{a}.exact_reading (mean)")
    Lt = J("latency.json")
    if Lt:
        rdn = Lt["reading"]
        put("LatRead", f"{rdn['deployed_numpy']['median_ms']:.0f}", "latency.json:reading.deployed_numpy.median_ms")
        put("LatReadRules", f"{rdn['rules']['median_ms']:.1f}", "latency.json:reading.rules.median_ms")
        key = f"{tr[0]}"
        if key in rdn:
            put("LatReadBytes", f"{rdn[key]['file_bytes'] / 1e6:.1f}", f"latency.json:reading.{key}.file_bytes")
        put("LatDecide", f"{Lt['decision_complete_spec']['median_ms']:.0f}", "latency.json:decision_complete_spec.median_ms")
        put("LatDecideP", f"{Lt['decision_complete_spec']['p95_ms']:.0f}", "latency.json:decision_complete_spec.p95_ms")
        put("LatKalStep", f"{Lt['kalman_runtime']['microseconds_per_step']:.0f}", "latency.json:kalman_runtime.microseconds_per_step")
        put("LatObsStep", f"{Lt['observer_runtime']['microseconds_per_step']:.0f}", "latency.json:observer_runtime.microseconds_per_step")
        put("LatJoint", f"{Lt['deployed_fresh_process']['seconds']:.1f}", "latency.json:deployed_fresh_process.seconds")
        put("LatRss", f"{Lt['deployed_fresh_process']['peak_rss_mb']:.0f}", "latency.json:deployed_fresh_process.peak_rss_mb")
        put("LatReplay", f"{Lt['replay_seconds']:.1f}", "latency.json:replay_seconds")
        put("LatReplayN", f"{Lt['replay_quantities_compared']:,}".replace(",", "\\,"), "latency.json:replay_quantities_compared")

    # ---- Table 3 -------------------------------------------------------------------------------------------
    def row(label, m, edits=None, intervals=True, cluster=False):
        inc = m["incorrect_releases"]
        w = ci(m["incorrect_per_request_ci"]) if intervals else "--"
        cl = ci(m.get("incorrect_per_request_cluster_ci")) if (intervals and cluster) else "--"
        inc = f"{inc}/{m['released']}"                                  # incorrect releases / released designs
        qa = f"{m['questions_per_request']:.2f} ({m['answers_per_request']:.2f})"       # questions (answers) per request, review B4
        return (f"{label} & {m['n']} & {inc} & {w} & {cl} & {pc(m['useful_coverage'])} & {m['false_impossibility']} & {m['unnecessary_refusals']} & {m['unresolved']} & "
                f"{qa} & {'--' if edits is None else f'{edits:.2f}'} \\\\")

    NC_ = 11
    L = [f"\\multicolumn{{{NC_}}}{{@{{}}l}}{{\\emph{{Structured specifications (correct fields, no language layer)}}}} \\\\"]
    for pol, lab in (("coordinator", "Full framework (also the oracle for the rows below)"), ("coordinator_limited_user", "Full framework, user knows only the stated ranges"),
                     ("selector", "Selector with fixed priority, same checks"), ("kalman_only", "Kalman specialist alone"),
                     ("khalil_only", "Khalil specialist alone"), ("ask_all", "Ask for every unresolved field"), ("default_fill", "Never ask (defaults and midpoints)")):
        if pol in st:
            L.append(row(lab, st[pol]))
    L.append(f"\\addlinespace\\multicolumn{{{NC_}}}{{@{{}}l}}{{\\emph{{Worded requests, readback reviewed attentively by the scripted user (consistency check)}}}} \\\\")

    def nlrow(lab, key, show_edits=True, intervals=True):
        m = nl.get(key)
        if m:
            L.append(row(lab, m, m["edits_per_request"] if show_edits else None, intervals=intervals or m["incorrect_releases"] > 0, cluster=True))

    if orc:
        L.append(row("Oracle: correct structured specification of the same requests", orc, intervals=False))
    for t in tr:
        nlrow(f"Trained reader, seed {t[-1]}", f"{t}|attentive|coordinator", intervals=False)
    nlrow("Frozen vectors, linear heads (no trained encoder)", "select_neural_frozen_s0|attentive|coordinator", intervals=False)
    nlrow("Keyword rules", "rules|attentive|coordinator", intervals=False)
    nlrow("Reads nothing (every fact entered at the readback)", "null|attentive|coordinator", intervals=False)
    nlrow("Single model chooses the method (trained reader, seed 0)", f"{tr[0]}|attentive|single_model", intervals=False)
    nlrow("Ask for every unresolved field (trained reader, seed 0)", f"{tr[0]}|attentive|ask_all", intervals=False)
    L.append(f"\\addlinespace\\multicolumn{{{NC_}}}{{@{{}}l}}{{\\emph{{Worded requests, readback confirmed without reading (stress test)}}}} \\\\")
    for t in tr:
        nlrow(f"Trained reader, seed {t[-1]}", f"{t}|blind|coordinator", False)
    nlrow("Frozen vectors, linear heads (no trained encoder)", "select_neural_frozen_s0|blind|coordinator", False)
    nlrow("Keyword rules", "rules|blind|coordinator", False)
    nlrow("Reads nothing", "null|blind|coordinator", False)
    nlrow("Single model chooses the method (trained reader, seed 0)", f"{tr[0]}|blind|single_model", False)
    return "\\newcommand{\\TableThreeBody}{%\n" + "\n".join(L) + "}"
