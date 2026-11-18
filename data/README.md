# Data: construction, splits and wording

All data in this directory are synthetic and were produced for this project. No personal data are included.

| File | Content | How it was made |
| --- | --- | --- |
| `train.jsonl` (3,600), `dev.jsonl` (600), `dev_extra_cases.jsonl` (60), `test_structured.jsonl` (600) | One case per line: hidden complete specification (`truth`), what the user provides (`given`), the designed first outcome (`expected`) and final outcome after truthful answers (`final`), category, family, labels; for train and dev also a templated request | `experiments/make_data.py` with the seeds in `manifest.json`; generator `heros/datagen.py`; the correct outcome follows from how each case is constructed (35 categories), and was cross-checked against the coordinator on development data |
| `authors/sheets_*.json` | Fact sheets from which the worded requests were written: plain statements of the facts a request must convey (values with units), nothing else | `experiments/make_data.py` |
| `authors/requests_dev_author0.json` (60), `authors/requests_test_author{1,2,3}.json` (80 each) | Three answers per case (to the three opening questions) | Four synthetic wording sets were generated from fact sheets and style instructions in separate sessions of the same language model. The writers received neither the code nor the generation templates or other wording sets. No engineers participated in this test; a script supplied confirmations and corrections. Set 0 holds the development requests; sets 1 to 3 hold the test requests |
| `authors/gold_*.json` | For every written request: the facts of its sheet with roles and SI values (used to score the reading layer) | `experiments/make_data.py` |
| `manifest.json` | Seeds, counts, SHA-256 of the case files, family mix, identifiers of the 240 test cases with written requests, notes | `experiments/make_data.py` |
| `train_examples.json` | Cache of tokenised training examples (derived; can be deleted) | `experiments/train_readers.py` |

Splits and grouping. Training and development cases come from seven model families; the test set adds four families
that occur nowhere else (`gantry_axis`, `duffing_stage`, `hydraulic_boom`, `cart_pole`). Supported families use randomly drawn base specifications. In the two unsupported families, twelve test cases in each share a plant model; the bearing-only specification also matches the 106 training cases of that family. Wording families are disjoint:
templates A, B, C (training), template D and set 0 (development), sets 1, 2, 3 (test).

Notes. (i) Eleven of the 240 test cases (`invalid_reversed_range`) belong to a category outside the three answers and
are excluded from the natural-language sets by a rule fixed before any reader was run on them; 229 remain.
(ii) One test fact sheet contains a unit string written with a display rule that was corrected in the templates
before the protocol was frozen; the request is kept and scored as written (see `docs/PROTOCOL.md`).

Reuse. The two supplied packages (`heros_kalman`, `heros_khalil`) are not part of this directory. Their data sets
were not used for training or testing; their stored results and re-runs appear only in
`results/baseline_reproduction.json` as development baselines.
