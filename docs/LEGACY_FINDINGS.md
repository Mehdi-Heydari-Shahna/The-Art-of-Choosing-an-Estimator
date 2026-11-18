# Documented defects of the two supplied packages and their regression tests

Source: the project brief's source inventory (`SOURCE_INVENTORY.md` of the handoff package, which is the author's
document and is not redistributed here); the findings are paraphrased. The defect itself is reproduced on the supplied
code by `regressions/reproduce_legacy_defects.py` where a short script can do so (7 of 15, results in
`regressions/legacy_defects.json`); the corrected behaviour of this project's public path is asserted by the test named
in the last column (`tests/test_regressions.py`). `experiments/make_paper_assets.py` counts the rows of this table.

| # | Package | Finding (paraphrased) | Reproduced on the supplied code | Regression test |
| --- | --- | --- | --- | --- |
| 1 | Kalman | a discrete plant with native period 0.1 s accepts a request at 0.05 s while using the native matrices | yes (`kalman_timing_conflict`) | `test_K1_timing_conflict_is_an_explicit_result` |
| 2 | Kalman | ambiguity witnesses for an LQR weight declared in [5, 6] lie outside the declared interval (1 and 1.25) | yes (`kalman_inadmissible_witness`) | `test_K2_witnesses_and_completions_are_admissible` |
| 3 | Kalman | continuous-plant conversion and the meaning of process noise (spectral density or per-step covariance) are disconnected | no | `test_K3_process_noise_semantics_are_explicit` |
| 4 | Kalman | the command-line trace omits information needed to reconstruct the executed problem | no | `test_K4_K10_record_replays_everything_and_detects_tampering` |
| 5 | Kalman | language failures with signs, variance units, number words, dropped fields, finite versus infinite horizon | no | `test_K5_language_failures_are_visible_not_silent` |
| 6 | Khalil | a finite-time (earliest certified time 0) claim is displayed although the admissible error exceeds the tolerance at every finite time | yes (`khalil_timing_claim`) | `test_H1_finite_time_and_asymptotic_claims_are_distinct` |
| 7 | Khalil | a scenario adds a deadline after a missing initial-velocity bound was treated as irrelevant, without re-evaluation | yes (`khalil_scenario_deadline`) | `test_H2_new_deadline_triggers_full_reevaluation` |
| 8 | Khalil | a position-error bound smaller than the noise bound is accepted while the initialisation uses the noisy first measurement | yes (`khalil_initialisation_premise`) | `test_H3_initialisation_matches_the_certificate_premise` |
| 9 | Khalil | a structured sample-period range raises a TypeError | yes (`khalil_sample_period_range`) | `test_H4_sample_period_range_gets_a_precise_disposition` |
| 10 | Khalil | an output with input feedthrough (and parameter-dependent transformations) is accepted as a verified transformation | yes (`khalil_feedthrough_output`) | `test_H5_output_transformations_are_not_accepted` |
| 11 | Khalil | local model bounds are accepted without evidence that trajectories stay in the domain | no | `test_H6_domain_evidence_is_required_and_monitored` |
| 12 | Khalil | total-disturbance bounds lose conservatism through floating-point steps before interval verification | no | `test_H7_derived_disturbance_bound_is_conservative` |
| 13 | Khalil | scenario overrides, reversed ranges and formatted thresholds conflict with the admissible information; literal questions untrue | no | `test_H8_ranges_thresholds_and_literal_questions`, `test_H8b_initial_error_question_is_literally_true` |
| 14 | Khalil | the readback is advisory and can be stale; a certified design can be returned before confirmation | no | `test_H9_release_requires_confirmation_of_the_final_specification` |
| 15 | Khalil | replay compares mainly the decision and the selected gain | no | `test_K4_K10_record_replays_everything_and_detects_tampering` |
