# Revision 2 review cases, 18 November 2026

These are exact request and result records from a review of the full archive with SHA-256 dbe3651b4d809140dcbf080bcecb1c30c3e61ebfa52589b7f6d519523242606e. Only the JSON object under `arguments` belongs in the call to `heros.api.solve`. The other fields are independent audit notes, true velocities where available and the observed outcomes. No package modification or retraining is needed to replay these cases.

1. Compare 01a and 01b. The user declares irregular sampling in both. In 01b the caller also supplies a timestamp array. File 01c is a separate fine-timing test in which the supplied timestamps are exact.
2. In 02, the plant gain is 2, the observer nominal gain is 1, and the total mismatch is left for the specialist to derive. The analytic trajectory is position t² and velocity 2t. Scoring starts at the declared 0.5 s deadline.
3. In 03a and 03b, the travel restriction is known to hold physically for the whole interval. A universal impossibility claim must respect it from the first sample. There is no requirement to add a new estimator. A justified refusal to certify the registered candidate can be a valid response.

Independent checks that passed on this archive included the 12 supplied structured validation requests, both demo record replays, the original scalar startup covariance (14.97176939), the 0.6 s finite-window example, and the explicit zero-disturbance example. The counterexamples do not determine how many of the 600 benchmark labels will change.
