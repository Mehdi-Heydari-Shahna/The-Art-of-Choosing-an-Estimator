| Finding | Case | Previous revision (dbc415a) | Current revision (afb804d) |
|---|---|---|---|
| F1 | timestamps agree with the stated 1 ms period | RELEASE; Estimates: 1001 samples | Verified design released.; Estimates: 1001 samples |
| F1 | data recorded every 2 ms (every second sample kept), request states 1 ms | RELEASE; Estimates: 501 samples | The timestamps of the data do not match the stated sample period; nothing is released.; no estimates |
| F1 | one sample missing (a 2 ms gap at t = 0.5 s) | RELEASE; Estimates: 1000 samples | The data are not sampled at a fixed period; no fixed-period certificate applies.; no estimates |
| F2 | Kalman filter, record ends at t = 3.5 s (samples t = 0..3 s), mean-square-error level 0.616808 | PROVED_IMPOSSIBLE | RELEASE, claim 0.615894 |
| F2 | observer, velocity error <= 0.075 rad/s on the record from 0.05 s to its end at 0.06 s | RELEASE, claim 0.0566297 | RELEASE, claim 0.0566297 |
| F2 | 100 Hz encoder, requirement on t in [0.01 s, 0.0165 s]; the two-signal argument needs a sample at or after 0.01633 s | PROVED_IMPOSSIBLE | NO_VERIFIED_CANDIDATE |
| F3 | scalar random walk, Q = R = 1, P0 = 100, constant (steady-state) gain | RELEASE, claim 0.618034, P(0|0) exported 0.618034 | RELEASE, claim 0.618034, P(0|0) exported 14.9718 |
| F4 | total disturbance bound D = 6 and, for the exactly known model, additive bound d = 1.5; ultimate velocity limit 0.045 rad/s (two-signal limits 2 sqrt(DV) = 0.0490, 2 sqrt(dV) = 0.0245) | PROVED_IMPOSSIBLE | RELEASE, claim 0.029064 |
| F4 | smallest mean-square error of the speed; nominal input gain b = 4 but the true value lies in [3.6, 4.4] | RELEASE, claim 6.34109e-05 | NO_VERIFIED_CANDIDATE |
| F5 | speed error within 0.04 rad/s with probability 0.9 (bounded noise; worst-case limit 2 sqrt(DV) = 0.049) | RELEASE, claim 0.960368 | RELEASE, claim 0.960368 |
| F5 | speed error within 0.02 rad/s with probability 0.9 (bounded noise; worst-case limit 2 sqrt(DV) = 0.049) | PROVED_IMPOSSIBLE | NO_VERIFIED_CANDIDATE |
| F6 | servo axis, velocity estimate initialised from the first reading (position) and zero (velocity) | RELEASE, envelope at t=0 3.02222 | RELEASE, envelope at t=0 3 |
| F7 | the 20 test requests with a cap below the tolerance | {'INVALID_INPUT': 20} | {'PROVED_IMPOSSIBLE': 20} |
