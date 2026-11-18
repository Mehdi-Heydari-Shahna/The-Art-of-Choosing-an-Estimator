# Review cases of revision 2b on delivered Rev2c archive Heros_Estimation_full.zip (SHA-256 ac636b45...9275)

Code digest (heros/ and independent/): `431b8c31b526a127ef5c08d02dc855eb16f497913efc625656b2d2a95ed449a5`

7 of 7 pass; case files match their SHA-256: True. Raw outputs: `<case>.output.json` in this folder.

| File | Expected | Decision | Numerical result | Pass |
| --- | --- | --- | --- | --- |
| `A1_relative_times_with_jitter.json` | exact-period guarantee declined (same timing deviation, clock origin 0) | OUT_OF_SCOPE | max deviation 7.63e-06 s vs precision 1.73e-18 s | pass |
| `A2_absolute_times_same_jitter.json` | exact-period guarantee declined (same timing deviation, clock origin 2^30 s) | OUT_OF_SCOPE | max deviation 7.63e-06 s vs precision 9.54e-07 s | pass |
| `A3_absolute_times_on_grid.json` | released (samples on the grid) | RELEASE (high_gain_observer) | claim 0.0; label precision 9.54e-07 s | pass |
| `B_forward_only_carriage.json` | no impossibility claim; 'no verified design' acceptable | NO_VERIFIED_CANDIDATE | disposition NO_VERIFIED_CANDIDATE, lower limit None | pass |
| `C_exact_window_ends_before_next_sample.json` | four observations in the window; largest in-window covariance about 0.61589404 <= 0.61680778 | RELEASE (kalman_filter) | observations in window 4; covariances [0.09090909, 0.52173913, 0.60344828, 0.61589404]; largest 0.6158940397 | pass |
| `D1_fixed_input_only_in_specification.json` | velocity 0.5 at t = 0.5 s; same trajectory for D1 and D2 | RELEASE (high_gain_observer) | velocity at t = 0.500 s: 0.49999999999999972 (error 2.8e-16); D1 and D2 trajectories differ by at most 0 | pass |
| `D2_explicit_known_input_control.json` | velocity 0.5 at t = 0.5 s; same trajectory for D1 and D2 | RELEASE (high_gain_observer) | velocity at t = 0.500 s: 0.49999999999999972 (error 2.8e-16); D1 and D2 trajectories differ by at most 0 | pass |
