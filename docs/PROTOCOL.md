# Evaluation protocol (frozen before the acceptance test)

This file was written and committed to the project's version control **before** any reader was run on the
acceptance-test requests and before any policy was run on the structured test set. The commit that freezes it
is tagged `protocol-frozen`. Later changes, if any, are listed at the end under "Deviations" with their reason.

## 1. Data and grouping

| Split | Content | Use |
| --- | --- | --- |
| `data/train.jsonl` | 3,600 cases from the seven training families, wording families A, B, C (developer-written templates) | training |
| `data/dev.jsonl` | 600 cases from the training families (different seed, hence different base specifications), wording family D | model selection, any threshold |
| `data/authors/requests_dev_author0.json` | 60 requests written separately (set 0) for 60 further development cases | model selection |
| `data/test_structured.jsonl` | 600 cases over eleven families, four of them held out (`gantry_axis`, `duffing_stage`, `hydraulic_boom`, `cart_pole`) | acceptance test, structured |
| `data/authors/requests_test_author{1,2,3}.json` | 240 requests (80 per set) in three separately written sets (1 to 3) for a random sample of the structured test cases | acceptance test, natural language |

Grouping. Every case has its own randomly drawn base specification; splits use disjoint seeds; held-out families
occur only in the test set; wording families are disjoint across the three splits. The four request sets
were written separately, each from its fact sheets only, without access to the developer's templates or code. The
earlier packages' evaluations are development evidence only.

Natural-language scoring conventions (fixed before any reader was run on authored requests; they follow from what
the three answers can express, not from any outcome). (i) Cases of the category `invalid_reversed_range` are not
part of the natural-language sets: two stated values are an ordinary range in any wording. Their number is
reported. (ii) The attentive review compares the reading with what the request states (text plus attachment);
the record length and the whiteness flag of text-stated noise levels, which no answer carries, are not compared.
(iii) One test fact sheet (`test-00182`, set 1) contains a unit string produced by a display rule of the sheet
generator that was corrected afterwards ("m/s^2" for a squared velocity); the request is kept and scored as written.

Test requests are never used for retrieval, tuning or model selection. The first run of any reader on the test
requests and of any policy on the structured test set is the reported one.

## 2. Model selection (development data only)

Candidates: keyword rules (deterministic baseline); logistic regression on hashed context features (simple
trained approach); a bidirectional GRU with frozen pretrained static word vectors, the same without pretrained
vectors, and frozen vectors with linear heads only (no trained encoder); the two supplied models, applied without
re-training with their labels mapped where a mapping exists.

Rule. Score = mean of the exact-reading rates on the two development parts. Among trained candidates within
0.02 (absolute) of the best score, the one with the fewest trainable parameters is selected. The selected
architecture is then trained with the three predeclared seeds 0, 1, 2 on the full training set; for an
architecture with a deterministic optimiser the seed selects a bootstrap resample of the training requests.
No confidence threshold is used unless it is chosen on development data and stated here (none is planned).
The interface (`models/selected.json`) uses the seed-0 checkpoint of the selected architecture; seeds 1 and 2 serve
only to measure variability. This choice is fixed here and does not depend on test results.

Outcome of the selection (recorded before freezing; `results/dev_selection.json`, `models/selection.json`): the
recurrent network without pretrained vectors (`neural_scratch`) had the best development score (0.511); no other
trained candidate was within 0.02 (pretrained vectors 0.459, logistic regression 0.380, frozen vectors with linear
heads 0.160), so it was selected. The three final seeds were trained in separate single-threaded processes.

## 3. Policies compared

Structured specifications (correct fields, no language layer):
`coordinator` (the full framework), `kalman_only`, `khalil_only` (one specialist with the same checks),
`selector` (same checks, fixed priority Kalman then high-gain observer, no comparison on the requested criterion),
`ask_all` (asks for every unresolved field of the plant class before deciding), `default_fill` (never asks; fills
defaults and range midpoints).

Natural-language requests: the coordinator with each reader (`null`, `rules`, frozen-vector heads, the selected
trained reader with seeds 0, 1, 2), each with an attentive and a blind scripted user; and `single_model` (the
selected trained reader, seed 0, also chooses the method through its proposal head; only the chosen specialist's
own checks run; ranges are replaced by midpoints; no comparison across methods).

The scripted user is a program, not a person: it answers a question with the hidden true value or "unknown",
and reviews a readback attentively (corrects every field that differs from what the request states; each
correction is counted) or blindly (confirms it as shown).

The comparison covers the readers and policies listed above.

## 4. Outcome classes and metrics

Every request ends in exactly one class, decided by `heros/evaluation.py::judge`:

* `correct_release`: a design was released and its claim is true under the hidden complete specification,
  verified with the independently written references (`independent/`), not with the coordinator's numbers.
* `incorrect_release`: a design was released and the verification fails (false claim, wrong model or period,
  a claim type that does not answer the request, a statistical claim without a statistical model, a hard bound
  without the required bounds, or any release for a request outside the supported scope).
* `correct_nonrelease`: no release, and the final disposition equals the designed one
  (`NO_VERIFIED_CANDIDATE`, `PROVED_IMPOSSIBLE`, `OUT_OF_SCOPE`, `INVALID_INPUT`).
* `nonrelease_other_reason`: no release where none is justified, but with a different disposition.
* `unnecessary_refusal`: no release although the complete specification admits a verified design.
* `unresolved`: the dialogue ended in a question or confirmation.

Reported: incorrect releases per request and per released design; useful coverage (correct releases divided by
the number of requests whose complete specification admits a verified design); unnecessary refusals;
unsupported requests detected; questions asked per request; readback corrections per request; unresolved
dialogues; wall-clock time per request (reading, decision); results per family, per category, per author and,
for the trained reader, per seed.

Layer evaluation (`evaluate.py reading`, written before the first test run). For every reader, on the authored test
requests: accuracy of the role assigned to each parsed quantity (gold roles obtained by matching SI values with the
fact sheets), proportion of requests whose reading needs no correction by the attentive reviewer ("exact reading"),
mean corrections, and agreement of the reader's proposed first outcome with the designed first outcome. The proposal
is never used by the coordinator policy; it is used only by `single_model`.

Oracle. The coordinator's rows on the correct structured specifications of the same requests are reported next to
the natural-language rows (`nl_oracle_structured`).

Uncertainty. Wilson 95 % intervals for proportions over requests, and a cluster bootstrap (2,000 resamples) with
the model family as the cluster for the structured set and (family, author) for the natural-language set.
Intervals describe the sampled families, categories and authors; they are not a statement about other wording.

Verification tolerances (fixed before looking at comparative results): model matrices 1e-8 relative; claimed
mean-square error against the independent value 1e-5 relative; optimality 1e-6 relative; independent worst case
against a claimed hard bound 1e-7 relative; a released disturbance bound must not be below a sampled value of the
true mismatch (3,000 samples including corners of the domain and of the parameter ranges).

## 5. Numerical studies (not part of the acceptance statistics)

Specialist fidelity against the independent references (tolerances in `tests/test_independent.py`); linear
stochastic case on the simulated hydraulic boom with 200 noise realisations; rotary-joint observer case with 100
noise realisations and three closed loops; common-domain case on a sampled double integrator with 200 common
records per setting; incomplete-information cases; latency and memory. Simulated errors below a certified bound
are a comparison, not a proof; universal claims rest on the interval-certified exact worst case only.

## 6. What may still change after freezing

Bug fixes that do not use test outcomes are allowed before the first test run. After the first test run, any
change to readers, coordinator, specialists or judge requires a note under "Deviations", and results obtained
before the change are kept in `results/before_change/`.

## Changes made after the first draft of this file and before freezing (no test outcome was available)

* A negative value read for a quantity that is a magnitude (bound, level, tolerance, time, probability) is reported
  as an interpretation issue and not used (`heros/nlu/assemble.py`); this answers a documented defect of the supplied
  Kalman package ("signs"). The training and development requests contain no negative number, so development results
  are unaffected (checked by counting).
* Candidate training was split over two processes to save time (`train_readers.py --only`); seeds and data unchanged.
* Request files. The first commit (`dab7b09`, 12:55:14 UTC) captured the file of test author 3 one second before
  that author's final revision was saved (12:55:15 UTC): 26 first answers and 7 second answers were reworded, and one
  self-reported extra number ("18-bit", test-00200) was withdrawn together with its wording. The file of test author
  2 was handed back at 12:57 UTC. No reader had read any test request at either time. The frozen versions are the
  final hand-back versions (hashes in `docs/FREEZE_RECORD.json`).
* Data regeneration check (`experiments/verify_data.py`): regenerating all data sets from the seeds reproduces the
  SHA-256 of all four case files exactly; the regenerated fact sheets equal the frozen ones except for the known unit
  string of test-00182.
* Per-request time in the natural-language rows includes the reading time (reading + dialogue; the judge is
  excluded); a `--quick` mode re-runs a subset for reproduction checks.

## Deviations

(none at the time of freezing)

Changes after freezing (none affects a reported test result):

* NumPy inference for the deployed checkpoint (`heros/nlu/numpy_reader.py`, added while the test was running). The
  interface uses it so that PyTorch is not needed at run time. On all 869 development and test requests and for all
  three seeds it gives the same role for every number, the same request labels, flags and proposal as the PyTorch
  implementation used in the test; confidence values differ by at most 3.4e-6 (`results/numpy_reader_check.json`).
  The reported test results were produced with the PyTorch implementation.
* Reporting code written or changed after the test run (`experiments/make_figures.py`, `paper_eval.py`,
  `make_paper_assets.py`, `exp_latency.py`, `criterion_effect.py`): it only reads the saved rows and results; the
  analysis in `criterion_effect.py` (which design the fixed-priority selector released where it differed from the
  coordinator) was not planned in this protocol and is reported as descriptive and post hoc.

### Additional pre-freeze facts (recorded after review B)

* Times of source changes made after the first test fact sheets and requests were on disk (12:55 UTC), from the
  committed bytecode (review B5): 12:56:17 `heros/nlu/templates.py` (display of squared compound units; the defect was
  noticed through test author 1's hand-back note on `test-00182`, so test material influenced a fix of the training
  templates: 18 training and 2 development requests changed); 12:57:30 `heros/evaluation.py` (natural-language scoring
  conventions, from the authors' hand-back notes); 13:00:25 `heros/khalil/specialist.py` (wording of the initial-error
  question when no critical value exists); 13:23:09 `heros/nlu/assemble.py` (sign rule); 14:07:12
  `heros/nlu/readers.py` (single-threaded PyTorch); 15:27:42 `heros/kalman/specialist.py` (claim condition for a
  missing initial estimate). No reader had read any test request at these times. The effect of the template fix on the
  trained reader is measured post hoc in `results/posthoc_prefix_effect.json`.
* The final training of seed 1 started at 14:07 UTC, before the selection record was written (14:50 UTC). At 14:07 the
  development scores of the logistic regression (0.380), of the network without pretrained vectors (0.511) and of the
  frozen-vector heads (0.160) were known; the network with pretrained vectors finished training at 14:43 and scored
  0.459. The outcome of the rule is the same; the order of events is not the one the rule implies.

## Changes after the first test run (review-driven; the first run is kept in `results/before_change/`)

The four internal reviews found defects in the coordinator, the specialists and the judge. They were corrected
and the complete acceptance test was run again (at that stage the article reported the second complete run and
stated the first run's results where they differ; the run it now reports is listed under "Run history"). Changes:

* Coordinator and specialists: the two-signal lower limit is used only for an arbitrary bounded acceleration error
  (supplied D, or a true model that does not depend on the state), with its time and velocity-range conditions (A1);
  mean-square-error claims of both specialists are evaluated on the same quantity, either the steady-state limit (for a
  steady-state request without deadline) or the largest value over the requested samples computed from the initial
  covariance, which is then required (A2, A3); no hidden 200-sample window (A7); the derived disturbance bound uses the
  Lipschitz constant over the range widened by the measurement bound (A4); ranges tied by equality constraints are
  checked at every vertex of the admissible set (A5, A8); the Kalman hard-bound certificate uses the observer's
  initialisation (A6); a range on a field that the selected method does not use is not asked about (A14); input hold
  checked (A15); printed upper bounds rounded up (A10); a question whose answer the user does not know ends the
  dialogue with NO_VERIFIED_CANDIDATE (C-13).
* Judge: a release is incorrect when the truth has irregular samples, a transient cap that the claim does not certify,
  a record that ends before the deadline, or an invalid request (B1, B19); a derived disturbance bound is checked
  against a dense sample with local maximisation of the true mismatch, including the measurement offset (a necessary
  check, B2); PROVED_IMPOSSIBLE outcomes are checked independently where a reference exists (B2); the input matrix of a
  Kalman release is compared as well (B20).
* Scripted user and metrics: answers obtained (including the fields that come with an objective answer) and "unknown"
  answers are counted (B4); a structured variant in which the user cannot give exact values for fields stated only as
  ranges is added (`coordinator_limited_user`, B4); per-family splits and paired comparisons are added to the summary
  (B7). The natural-language evaluation reads with the NumPy implementation of the final checkpoints (identical
  readings, `results/numpy_reader_check.json`).
* Reporting: post-hoc analyses are labelled (`criterion_effect.py`, `posthoc_prefix_training.py`).

### Further changes made while preparing the second run

* Judge (reviews A3, B20): statistical claims at finite times are checked from the true initial covariance over the
  whole requested window, independently of the specialists (`_window_max`, `_released_gain_seq`); a time-varying filter
  without an end time is checked to convergence with the limit included; the stated value of an observer released for a
  mean-square level is checked as well.
* Coordinator (review C-13): when the engineer cannot give the exact sample period of a request that states a range for
  it, the dialogue ends with NO_VERIFIED_CANDIDATE instead of repeating the question. This affects only dialogues with an
  "unknown" answer (the limited-user variant). The command line accepts "unknown" as an answer.
* Example `servo_axis_answers.json` reworded so that the reader assigns the noise standard deviation and the hard bound
  correctly (review C-13).
* The acceptance test was then run completely (run 3, commit `c57f606`). An intermediate run (run 2, commit `c52e4bc`)
  had finished its structured part before the last judge and coordinator changes above; its language part was
  interrupted. Its rows were overwritten by run 3 and not kept at the time, contrary to section 6; this statement
  previously said, inaccurately, that the intermediate run was simply not reported (recheck B, finding N2). The
  structured rows of run 2 were re-created from its commit after recheck B; see "Run history" below.

## Run history (added after recheck B, finding N2)

| Run | Code | Parts | Rows kept in | Status |
| --- | --- | --- | --- | --- |
| 1 | `da09d95` (protocol frozen at `2171a27`) | structured (3,600 rows) and language (3,435 rows) | `results/before_change/` | first run, before the review corrections |
| 2 | `c52e4bc` | structured (4,200 rows, with the new limited-user variant); language part interrupted | `results/run2_structured_reconstructed/` | rows re-created on 17 Nov 2026 by running the structured part again at `c52e4bc` (the code is deterministic; the original rows were not kept) |
| 3 | `c57f606` | complete | `results/run3/`; impossibility outcomes re-scored with the corrected judge in `results/run3/rescored/` | the run examined by the rechecks (snapshot `review-snapshot-2`) |
| 4 | `7677b89` | complete | `results/run4/` | made after the corrections of the rechecks (listed below) |
| 5 | `4d2624e` | complete (its language part was restarted from the beginning after an environment restart) | `results/run5/`; re-scored with the revision-2 labels and judge in `results/run5/rescored_revision2/` | made with the code after the corrections of the second rechecks |
| 6 | `cb8d3e3` (tag `revision-2-frozen`) | complete | `results/run6/` | revision 2 (seven findings) |
| 7 | `afb804d` (tag `revision-2b-frozen`) | complete; the structured part was repeated on an idle machine for its timing fields (outcomes identical row by row) | `results/run7/` | review cases of 18 November 2026 |
| 8 | `6f52718` (tag `revision-2c-frozen`) | complete | `results/reference/` | reported in the article; review cases of revision 2b |

Outcome changes between runs, row by row (`experiments/compare_runs.py`; rows matched on id, policy, reader, user and
author; a row counts as changed if its outcome, disposition or selected method differs). The lists are in
`results/run_history/`.

* Run 1 to run 2, structured: 0 of 3,600 matched rows changed (run 2 adds the 600 rows of the limited-user variant).
* Run 2 to run 3, structured: 18 of 4,200 rows changed, all in the limited-user variant: a dialogue that repeated the
  question for an exact sample period that the user did not know (open) now ends with NO_VERIFIED_CANDIDATE (an
  unneeded refusal); this is the C-13 change above.
* Run 1 to run 3, language: 69 of 3,435 rows changed, all without review (blind confirmation): 53 releases judged
  correct in run 1 were judged incorrect under the stricter judge (review B1); 3 outcomes (1 release, 2 refusals) became
  false impossibility claims and 2 correct releases became unneeded refusals; the other 11 rows are refusals whose
  disposition or classification changed (`results/run_history/run1_to_run3.json`).
* Run 3 re-scored (`experiments/rescore_impossibility.py`, judge correction of recheck B, N1): 17 language rows changed
  from a refusal to a false impossibility claim; no structured row changed.
* Run 3 (re-scored) to run 4: 0 of 4,200 structured and 0 of 3,435 language rows changed (outcome, disposition and method identical). Compared with the stored run 3 as scored then, the 17 language rows of the re-scoring differ; every other row is identical.
* Run 4 to run 5: 0 of 4,200 structured and 0 of 3,435 language rows changed (outcome, disposition, method, question
  count and correction count identical; `compare_with_reference.py` and `results/run_history/run4_to_run5.json`). Run
  times differ.

## Changes after the rechecks (made before run 4)

* Kalman specialist (recheck A, N1): the finite-time window rule judges convergence and monotonicity on the diagonally
  scaled increment of the covariance, so a state whose variance is many orders of magnitude below another's is no
  longer cut off early.
* Coordinator (recheck A, N2, m3): a hard-bound claim under equality constraints states the largest worst case over the
  vertices of the admissible set, with the ranges and the constraint as conditions; constraints that share a field are
  handled by a general vertex enumeration.
* Khalil specialist (recheck A, m4): with a supplied disturbance bound and a stated model whose acceleration depends on
  the state, the two-signal lower limit is not claimed (NO_VERIFIED_CANDIDATE instead of PROVED_IMPOSSIBLE). No test case
  is affected: every impossibility outcome with a supplied bound in the test set has a state-free model.
* Judge (recheck B, N1): a failed independent impossibility check returned `numpy.False_`, which an identity test missed;
  the comparison is corrected, and an impossibility claim for a request for the smallest error variance is scored false
  (an optimal filter always exists). The mean-square impossibility check uses the judge's own discretisation (recheck B,
  N4).
* Judge (recheck B, N3): the sampled lower estimate of the true model mismatch searches faces and edges of the box (each
  coordinate at a bound or inside) and refines the best samples by dense coordinate sweeps before a gradient method; it
  now equals the reviewer's dense supremum on the eight rotary-joint cases examined.
* Reference implementation: the steady-state Riccati iteration (`independent/ref_kalman.py`) judges each entry on its own
  scale; its earlier rule, relative to the largest entry, stopped up to 6.6e-6 (relative) early on hydraulic-boom test
  models. Against the library DARE solution over the 284 distinct linear test models the largest relative difference is
  now 3.5e-9 (two-mass models, where the reference has the smaller Riccati residual).
* Replay (recheck C, N3): a released record without stored outputs is reported as unverified for its outputs.
* Summary and reporting: per-category language results, false impossibility counts and answers per request are
  reported (recheck B, N1, B4, B7); Table 3 has a column for false impossibility claims.
* The judge's settling rule for unbounded windows (`_window_max`, recheck B, N4) is unchanged: a check that does not
  settle within its iteration limit counts as an incorrect release, which can only make the reported results worse; no
  row of runs 3 or 4 is affected.

## Changes after the second rechecks (made before run 5)

* Kalman specialist (second recheck A, New-1): the library solution of the Riccati equation is verified by one Newton
  step and, if that step moves it by more than 1e-10 of an entry's scale, refined by Newton steps in 50-digit
  arithmetic; the stabilisability test is scale-aware; the window rule includes the limit in its non-increasing branch.
* Coordinator (second recheck A, New-2): constrained hard-bound and probability claims are rebuilt from the certified
  vertex values (bound and peak), keep the domain clause of a derived bound, and use a Lipschitz range widened by the
  largest admissible V so that the vertex argument holds.
* Khalil specialist (second recheck A, m4 residual): with a supplied D the lower limit requires no stated model, or a
  stated model equal to the nominal one, state-free and without uncertain parameters.
* Reference and judge (second rechecks A and B): the steady-state Riccati reference uses extended-precision doubling and
  refuses non-stabilising fixed points; finite-window mean-square impossibility claims are checked independently.
* No test case is in the classes these changes address (no equality constraints in the test set; every test-set
  impossibility outcome with a supplied D has f = f0 without parameter ranges; no stabilisability verdict changes; the
  Riccati refinement changes the steady-state covariances of 57 of 313 distinct models by at most 2.3e-9 of their
  scale). Run 5 checks this on every row (below).

### Files changed or added since the freeze (output of `tools/freeze_record.py --check`, 17 Nov 2026, after run 4)

| File | Status | Reason |
| --- | --- | --- |
| `docs/PROTOCOL.md` | changed | deviations and changes recorded (this list); run history and post-recheck changes |
| `experiments/compare_with_reference.py` | changed | review C-12: missing re-run file is a failure; recheck C, N1: wording |
| `experiments/evaluate.py` | changed | reviews B4, B6, B7, B9, B10, B12, C-02, C-07: NumPy reader, limited-user policy, answer counts, family split, paired comparisons, seed overlap, timing aggregates; recheck B: per-category language results, false impossibility counts printed |
| `experiments/exp_incomplete.py` | changed | review A8/D C2: sidebar cases (sum and range; error-level release) |
| `experiments/exp_latency.py` | changed | post hoc timing script: deployed process, conditional memory note (C-08), median of repeated runs |
| `experiments/make_examples.py` | changed | review C-13: servo example reworded |
| `experiments/make_figures.py` | changed | figures: larger type, no panel titles, limited-user bar, labels; recheck D: legend positions |
| `experiments/make_paper_assets.py` | changed | new macros and Table 2 rows (reviews A, B, D); recheck D, N6: Table 2 wording |
| `experiments/paper_eval.py` | changed | new macros, slimmer Table 3 (reviews B, D); rechecks B and D: false impossibility, answers, family split, wording guards, Table 3 columns |
| `experiments/verify_data.py` | changed | review C-06: --install option |
| `experiments/write_selected.py` | changed | records the NumPy files of the deployed reader |
| `heros/__main__.py` | changed | review C-11, C-13: reader identity, scripted-confirmation notice, 'unknown' answers; recheck C, N3: replay prints outputs_stored |
| `heros/coordinator/coordinator.py` | changed | reviews A5, A8, A14, A17, C-13: vertices of constrained sets, verified release, unused-field ranges, unknown answers; recheck A, N2, m3: constrained hard-bound claims, shared-field vertices |
| `heros/core/assess.py` | changed | review A10: outward rounding of printed bounds |
| `heros/core/spec.py` | changed | review A17: tolerance for range membership |
| `heros/evaluation.py` | changed | reviews A3, B1, B2, B4, B19, B20: judge and scripted user; recheck B, N1, N3, N4: impossibility scoring, disturbance search, independent model |
| `heros/interface/demo.py` | changed | recheck C, C-11: the demo prints the reader |
| `heros/interface/service.py` | changed | reviews C-11, C-13: NumPy reader, reader identity, unknown answers |
| `heros/kalman/specialist.py` | changed | reviews A3, A6, A7, A10: finite-time claims from P0, common initialisation, no hidden window, rounding; recheck A, N1: window rule on each state's own scale |
| `heros/khalil/specialist.py` | changed | reviews A1, A2, A4, A10, A15: impossibility conditions, MSE window, Lipschitz range, rounding, input hold; recheck A, m4: no lower limit with a state-dependent model and supplied D |
| `heros/nlu/readers.py` | changed | review C-18: vector hash check; single-threaded PyTorch import; recheck C, C-18: digests checked when the cache is created |
| `heros/record.py` | changed | review C-03: strict replay; recheck C, N3: records without outputs reported as unverified |
| `independent/ref_kalman.py` | changed | steady-state Riccati iteration judges each entry on its own scale (found while fixing recheck A, N1) |
| `models/selected.json` | changed | identity of the deployed NumPy files added |
| `experiments/check_numpy_reader.py` | new | new: NumPy inference identical to PyTorch (results/numpy_reader_check.json) |
| `experiments/compare_runs.py` | new | recheck B, N2: outcome changes between stored runs |
| `experiments/criterion_effect.py` | new | new, post hoc: selector comparison |
| `experiments/posthoc_prefix_training.py` | new | new, post hoc: review B5 |
| `experiments/rescore_impossibility.py` | new | recheck B, N1: re-scoring of stored impossibility outcomes |
| `heros/nlu/numpy_reader.py` | new | new: NumPy inference of the deployed reader |
| `models/final_neural_scratch_s0/reader_numpy.json` | new | NumPy export of the final checkpoint (identical readings) |
| `models/final_neural_scratch_s0/reader_numpy.npz` | new | NumPy export of the final checkpoint (identical readings) |
| `models/final_neural_scratch_s1/reader_numpy.json` | new | NumPy export of the final checkpoint (identical readings) |
| `models/final_neural_scratch_s1/reader_numpy.npz` | new | NumPy export of the final checkpoint (identical readings) |
| `models/final_neural_scratch_s2/reader_numpy.json` | new | NumPy export of the final checkpoint (identical readings) |
| `models/final_neural_scratch_s2/reader_numpy.npz` | new | NumPy export of the final checkpoint (identical readings) |
| `models/vectors_en_core_web_md.sha256.json` | new | not new: committed in the freeze commit, now listed by the freeze tool (review B16) |

No file that defines the cases, the requests, the trained checkpoints or the selection record (`data/`, `models/final_*/reader.pt`, `models/selection.json`) changed after the freeze.

## Revision 2 (18 November 2026): changes before run 6

The request for the next revision listed seven findings. They were reproduced through the public interface on the
previous revision (commit dbc415a) and on the revised code with `experiments/review_findings.py`
(`results/review_findings_previous.json`, `results/review_findings_current.json`; summary in
`docs/responses/RESPONSE_REVISION2.md`). The software was then changed as follows and frozen (git tag
`revision-2-frozen`) before run 6, the studies and the fresh validation set were run.

| Finding | Change | Files |
|---|---|---|
| F1 stated period and timestamps | Recorded sample times are read (column `t`) and compared with the period of the design; a release is withdrawn when they disagree (`INVALID_INPUT`, timing_conflict) or are not uniform (`OUT_OF_SCOPE`, irregular_sampling). The record stores the timestamps and replay repeats the check. | `heros/core/timing.py` (new), `heros/record.py`, `heros/api.py`, `heros/__main__.py`, `heros/interface/service.py`, `heros/interface/web.py`, `heros/interface/demo.py` |
| F2 samples inside the window | One window rule for every component: samples with `t_k >= deadline` and `t_k <= t_end` (`floor`, not `round`, for the last one). Hard-bound certificates use the window samples when a record length is stated. The two-signal limit is claimed only if a sample of the requirement window lies at or after its time. A window without a sample is invalid input. | `heros/core/timing.py`, `heros/core/peak.py`, `heros/core/spec.py`, both specialists, `heros/coordinator/coordinator.py`, `heros/evaluation.py` |
| F3 Kalman start-up | The exported covariance of a constant gain is propagated from the stated initial covariance (14.9718 at the first sample in the scalar case); with the first-reading start the first update is the gain [1, 0]' and is exported as such; a hard-bound release of a Kalman gain exports its deterministic envelope. | `heros/kalman/specialist.py` |
| F4 constraints together | A supplied total disturbance bound and the bound derived from the stated model are both used (the smaller applies); the free part of the acceleration for the two-signal limit is the smallest stated bound on it; the construction must rest at an admissible equilibrium inside the stated travel, speed and input ranges and the initial-error box; no statistical Kalman claim when the true parameters are known only as ranges. | `heros/khalil/specialist.py`, `heros/kalman/specialist.py` |
| F5 valid impossibility arguments | The worst-case two-signal limit is no longer used for a probability requirement; the user-facing sentence for `NO_VERIFIED_CANDIDATE` is "No verified design found."; a first-sample limit (one position reading carries no velocity information) covers requirements at t = 0. | `heros/khalil/specialist.py`, `heros/core/limits.py` (new), `heros/coordinator/coordinator.py`, `heros/interface/service.py` |
| F6 first-reading initialisation | Exact worst case for the observer started from the first reading (e_1(0) = -v_0), float and interval versions; Proposition 1 itself (independent boxes) is unchanged. The independent reference simulates the same start. | `heros/core/peak.py`, `independent/ref_hgo.py`, both specialists, studies |
| F7 cap below the later tolerance | Not a contradiction any more; decided by the certificates and the first-sample limit. | `heros/core/spec.py`, `heros/core/limits.py` |

### Deviation: corrected outcome of 20 test requests

The frozen generator labelled the 20 test requests of category `invalid_cap_below_tol` as `INVALID_INPUT`. The label is
wrong. `experiments/relabel_cap_cases.py` decides each case from its data alone (no package code): in all 20 the cap is
below the half-width of the initial velocities that the stated bound (and speed range) admits, so no estimator can meet
it at t = 0 and the correct outcome is `PROVED_IMPOSSIBLE`. The corrected outcomes are in `data/revision2_relabel.json`;
the evaluation applies them when it loads the cases (`heros.evaluation.with_revisions`). The frozen case files, the
fact sheets and the authored requests are unchanged, so `experiments/verify_data.py` still reproduces them; the
generator keeps the old label (commented) so that it regenerates the frozen files. The development and training sets
contain the same category; their labels are not used by any reported number and were left as they are (the reader's
auxiliary proposal label for those training cases is therefore outdated; it has no effect on any decision of the
framework, which ignores the proposal, but the single-model baseline acts on it and still declares 7 of the 9
relabelled language requests contradictory, which is scored as a refusal for another reason).

### Judge changes

The judge applies the window rule above; evaluates hard-bound releases with the independent reference started from the
first reading when the release is; accepts a released disturbance bound below the supplied total bound only if it is
supported by the stated model (sampled necessary check, as for derived bounds); rejects statistical Kalman claims for
models with parameter ranges; and checks impossibility claims as follows: refuted if a design for the complete truth is
released and verified independently; never confirmed for a probability requirement by a worst-case limit; confirmed by
its own first-sample check or by its own two-signal construction (`independent.ref_hgo.lower_limit_two_signals`, with the
window, speed, travel, input and initial-error conditions).

### Fresh validation set

Twelve requests with answers derived without the package (`validation/expected_answers.py`, outputs `validation/cases.json`
and `validation/expected.json`) were committed with the frozen software of revision 2 (commit `cb8d3e3`, tagged both
`validation-frozen` and `revision-2-frozen`, 2026-11-18 08:04:03 UTC), before they were run
(`experiments/run_validation.py`, `results/validation.json`, `paper/sections/validation_table.tex`). The first run, on
that software 25 s after the commit, is kept unchanged in `results/validation_first_run_revision2.json` (12 of 12 agree);
replayed on the tag `revision-2-frozen` on 18 November 2026 it gives identical rows
(`results/validation_replay_revision2_tag.json`). The set was rerun with revision 2b and revision 2c (12 of 12 each).
The records of those reruns, and of the first rerun with revision 2c, carried the note "software at git tag
revision-2-frozen", which was a fixed string in the script; since revision 2c.1 the record names the software by the
digest of `heros/` and `independent/`, by git when available, and by `RELEASE.json` when the digest matches.

### Run 6 (18 November 2026)

Run 6 used the frozen software (`revision-2-frozen`, commit `cb8d3e3`), the same cases, readers, checkpoints, scripted
user and protocol as run 5, with the corrected outcomes above applied when the cases were loaded
(the steps of `bash reproduce.sh evaluate`, compared with run 5 by
`experiments/compare_runs.py results/run5 results results/run_history/run5_to_run6.json`; log
`results/reference/acceptance_test.log`).
Its rows replace run 5 in `results/reference/`; run 5 is kept in `results/run5/` and, re-scored with the revised
labels and judge without re-running any dialogue, in `results/run5/rescored_revision2/`
(`experiments/rescore_revision2.py`). The changes are listed row by row in `results/run_history/run5_to_run6.json`:
144 structured rows (the 20 relabelled requests under all 7 policies; 4 probability requests under the Khalil-only
baseline, now "No verified design found." instead of a worst-case impossibility claim) and 146 language rows. The
structured totals are unchanged.

Post-hoc analysis (not in the frozen protocol): `experiments/blind_impossibility_audit.py` replays the 23 false
impossibility claims of the trained reader without review (all replays reproduce the stored disposition) and records
the argument and the misread fields (`results/blind_impossibility_audit.json`). In 21 of them the first-sample limit
was applied to a cap that the reader had read where the request states none (19 times the tolerance that applies
after the deadline).

All six numerical studies were rerun with the frozen software after run 6 (`results/logs/studies_revision2.log`).

After the freeze, two tests of `tests/test_revision2.py` were changed so that they also run from the light archive,
which leaves out `data/test_structured.jsonl`: they read the 24 cases they need from
`tests/fixtures/revision2_cases.jsonl`, which must equal the rows of the case file whenever that file is present. Their
assertions are unchanged; the package code is unchanged.

## Review cases of 18 November 2026: changes before run 7

The author supplied six JSON cases from a review of the delivered revision 2 archive. They are stored unchanged in
`tests/fixtures/reviewer_cases_rev2/` and replayed through `heros.api.solve`, with only their `arguments` object, on the
revision 2 software (tag `revision-2-frozen`) and on the revised software (`experiments/reviewer_cases.py`,
`experiments/reviewer_cases_compare.py`; results `results/reviewer_cases_previous.json`,
`results/reviewer_cases_current.json`, `results/reviewer_cases_table.md`; response
`docs/responses/RESPONSE_REVIEW_CASES.md`). The revision 2 replay reproduces the outcome recorded in each case file. The
software was then changed as follows and frozen (tag `revision-2b-frozen`, commit `afb804d`) before run 7, the studies,
the validation set and the seven-finding replay were run again.

| Issue | Change | Files |
|---|---|---|
| Irregular sampling (cases 01a to 01c) | Declared irregular sampling makes the observer not applicable (the Kalman filter already was). Recorded times are the actual sample instants: a fixed-period release is kept only if every instant equals t_0 + k Ts within the binary rounding of the stored time (64 units in the last place of the largest time), or within half of a stated label resolution (`time.label_resolution`), which then becomes a condition of the claim; otherwise out of scope (irregular) or invalid input (uniform at another period). | `heros/khalil/specialist.py`, `heros/core/timing.py`, `heros/record.py`, `heros/core/spec.py`, `heros/api.py`, `heros/coordinator/coordinator.py` |
| Plant and observer gain (case 02) | A one-point parameter range states the true value; the mismatch bound uses it in the true model; the Kalman specialist makes no statistical claim when a stated true value differs from the model. | `heros/khalil/specialist.py`, `heros/kalman/specialist.py` |
| Impossibility under a known travel limit (cases 03a, 03b) | First-sample limit: with a stated travel or speed range, an inner approximation of the initial velocities from which explicit trajectories stay inside every stated range for the whole window (braking with an admissible input and resting with the input then held; or, for a restoring force with damping, one input with zero disturbance and an energy bound), with outward-rounded interval bounds; no position limit with such a range. Two-signal limit: the travel range must contain the braking after the sample (3V on each side of the rest state when the window continues). The judge has separately written versions of both constructions. | `heros/core/limits.py`, `heros/core/bounds.py`, `heros/khalil/specialist.py`, `independent/ref_hgo.py`, `heros/evaluation.py` |

Found while implementing the third row, before the freeze: the first version of the braking construction accepted rest
with any input after a stop between two samples, although the braking input stays in effect until the next sample
instant. Both implementations now require the held input to admit rest, and the energy construction was added (it
decides `test-00343` and `test-00572` of the relabelled cases, two spring stages). Tests: `tests/test_reviewer_cases.py`
(17); the full suite has 116 tests. The judge's impossibility check is cached per distinct truth (it depends on the
complete facts only); this changes no outcome.

### Labels

None of the 600 test labels changed. `experiments/relabel_cap_cases.py` was rerun with the corrected independent
construction: all 20 relabelled requests remain `PROVED_IMPOSSIBLE`; the recorded half-width of the admissible initial
velocities changed for `test-00026`, `test-00178`, `test-00330` and `test-00503` (each still above its cap). The
corrected judge confirms every designed `PROVED_IMPOSSIBLE` outcome of the test set (47) and of the development sets
(26 and 1). No test case states a one-point parameter range that differs from its nominal value, and every test case
with declared irregular sampling asks for the smallest error variance, for which the observer was already not used.

### Run 7 (18 November 2026)

Run 7 used the frozen software (`revision-2b-frozen`), the same cases, readers, checkpoints, scripted user and protocol
as run 6 (the steps of `bash reproduce.sh evaluate`; log `results/reference/acceptance_test.log`),
compared with run 6 by `experiments/compare_runs.py results/run6 results results/run_history/run6_to_run7.json`.
Its rows replace run 6 in `results/reference/`; run 6 is kept in `results/run6/`. Changes, row by row: 0 of 4,200
structured rows and 3 of 3,435 language rows (trained reader, seed 1, without review: three false impossibility claims
on a misread cap became unneeded refusals, because the first-sample limit now counts only initial velocities from which
the joint stays inside its stated travel and speed ranges). Six further language rows asked a different number of
questions and kept their outcome: two with declared irregular sampling, where the observer is now not applicable
before any question, and four where the first-sample limit no longer applied to the request as first read, so the
coordinator asked first. The first pass of the structured part shared the two cores with the numerical studies, which inflated its
per-request times; the structured part was repeated alone (`results/logs/acceptance_run7_structured_repeat.log`), and
its rows, identical to the first pass in outcome, disposition, method and questions, are the reported ones.

The post-hoc audit of the false impossibility claims without review (`experiments/blind_impossibility_audit.py`) was
repeated on run 7 (`results/blind_impossibility_audit.json`; run 6: `results/run6/blind_impossibility_audit.json`).

All six numerical studies were rerun with the frozen software (`results/logs/studies_revision2b.log`); their results
are identical to revision 2 except the timing study, which was repeated on the idle machine after run 7. The validation
set (12 of 12 agree) and the seven-finding replay are unchanged in outcome; the replay's revision 2 results are kept in
`results/review_findings_revision2.json` (only the wording of the first-sample argument in finding 7 differs).

### Review cases of revision 2b and run 8 (18 November 2026)

Seven cases supplied with the review of the revision 2b archive (`tests/fixtures/reviewer_cases_rev2b/`, stored
unchanged; each file's `arguments` object passed to `heros.api.solve`) were replayed on `revision-2b-frozen` and on the
revised software (`experiments/reviewer_cases.py`, tabulated by `experiments/reviewer_cases_rev2b_compare.py` in
`results/reviewer_cases_rev2b_table.md`; response in `docs/responses/RESPONSE_REVIEW_CASES_REV2B.md`). Changes, frozen
as tag `revision-2c-frozen` (`6f52718`), each with a regression test (`tests/test_reviewer_cases_rev2b.py`, 14 tests;
the full suite has 130):

* Timestamps: a recorded instant is compared with `t_0 + k T` to within the float64 precision of the stored times
  (four units in the last place of the largest time), plus half of a stated label resolution; deviations are measured
  from the first sample, so the decision does not depend on the clock origin; a precision coarser than 10^-9 T is
  stated with the claim, and labels coarser than 10^-2 T are not used. The uniform-grid check uses the end-to-end
  period estimate.
* Requirement window: a ratio of two stated times is compared with an integer to within 8 eps max(1, |r|) instead of
  an absolute 10^-9 (package and judge).
* First-sample construction: every velocity on the braking path, from rest to the start, lies inside a stated speed
  range (package and independent reference).
* Inputs: a design whose model uses the applied input runs with the recorded input or, without it, with the fixed
  input that a one-point range states (a condition of the claim); otherwise nothing is run (`missing_input`).

Run 8 used the frozen software, the same cases, readers, checkpoints, scripted user and protocol as run 7 (log
`results/reference/acceptance_test.log`), compared with run 7 by
`experiments/compare_runs.py results/run7 results results/run_history/run7_to_run8.json`: 0 of 4,200 structured and
0 of 3,435 language rows changed. Its rows replace run 7 in `results/reference/`; run 7 is kept in `results/run7/`.
No benchmark or validation specification states a speed range that excludes zero, and no requirement window of the
benchmark is read differently under the new ratio allowance. The numerical studies (log
`results/logs/studies_revision2c.log`) are unchanged except for timing fields; the validation set (12 of 12 agree), the
seven-finding replay, the six review cases of revision 2 and the audit of the false impossibility claims without review
(`results/blind_impossibility_audit.json`) are unchanged in outcome (`results/logs/post_run8.log`).

### Release revision 2c.1 (18 November 2026)

The delivered tree is tagged `revision-2c.1`; `heros/` and `independent/` are identical to `revision-2c-frozen`
(`6f52718`), code digest `431b8c31b526a127ef5c08d02dc855eb16f497913efc625656b2d2a95ed449a5` (`RELEASE.json`). No
software change was made. Records of this release:

* The seven review cases of revision 2b, replayed through `heros.api.solve` on the delivered revision 2c archive
  (`Heros_Estimation_full.zip`, SHA-256 `ac636b45c39bb8c60a2a5f4c00f0e8cb4009db6155511c28e5a8735bfb859275`) by
  `experiments/replay_review_cases.py`: 7 of 7 pass, raw outputs and table in `results/review_cases_rev2b_release/`.
  They are reported separately from the acceptance test.
* The validation requests V01 to V12 rerun on the release (`results/validation.json`): 12 of 12 agree, with the same
  answers as in the first run on revision 2.
* Manuscript: title "The Art of Choosing an Estimator"; the supervised reader in the Summary; the deployed reader named
  (one-layer bidirectional GRU, hidden size 96, 113,620 parameters, seed 0 of three) with one test request read
  correctly and the corrections result; the history of the validation set in the text and the caption of Table 4;
  sidebars near their first citation (switch `\sidebarsatendtrue` in `paper/main.tex` for the submission layout);
  float placement without half-empty pages.

