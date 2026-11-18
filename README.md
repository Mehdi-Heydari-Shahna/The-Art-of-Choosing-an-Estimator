# Heros in Estimation

Research software and simulation results for *The Art of Choosing an Estimator* by Mehdi Heydari Shahna and Jouni Mattila. This distribution contains the numerical implementation from the archived revision 2c.1, the trained readers, the data, and the published benchmark records.

The engineer describes an estimation objective, the model and measurements, and what is known about uncertainty. A supervised bidirectional GRU labels the request. Once the engineer confirms the reading, the coordinator considers a linear Kalman filter and a second-order sampled high-gain observer. A released design includes the assumptions behind its covariance or deterministic error bound.

**Known limitation for coupled uncertainty bounds.** The coordinator can release an invalid bound when constraints link two bounds on the same disturbance. For the included double-integrator example, it reports zero velocity error even though an admissible trajectory has an error of 0.05350 m/s at 0.1 s. The requested limit is 0.001 m/s. The runtime checks the endpoints and misses the interior case. Treat the all-completions claim as unverified for such constraints. The saved request, output summary and replay instructions are in [the coupled-bounds check](results/coupled_bounds_check/README.md).

## Run the software

Use a source checkout with Python 3.12 or later. The original runs used Python 3.13. Install the runtime packages from the repository root, then run a demonstration with the supplied weights.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m heros demo
python -m heros replay demo_output/joint_record.json
```

On Windows, activate the environment with `.venv\Scripts\activate`. The demonstration includes a rotary joint, a hydraulic boom, and a comparison of both estimators on the same servo axis. It saves plots and execution records in `demo_output/`. The deployed reader uses NumPy and runs on a CPU.

For a structured request, run:

```sh
python -m heros solve --spec examples/hydraulic_boom_spec.json --data examples/hydraulic_boom_data.csv --out record.json
```

For an interactive session, run `python -m heros ask --model examples/rotary_joint_model.json --data examples/rotary_joint_data.csv` or start the local interface with `python -m heros serve`. The data format accepts measurement columns `y` or `y1,y2,...`, an optional input `u` and optional timestamps `t` in seconds. A fixed-period design assumes physically periodic sampling. Its timestamp check has a tolerance that depends on clock origin and label resolution.

## Reproduce the results

| Task | Command or archived evidence |
| --- | --- |
| Run the unit and regression tests | `python -m pytest -q tests` |
| Replay the seven separately supplied reviewer cases | `python -I experiments/replay_review_cases.py --code . --out local_replay` |
| Re-evaluate the twelve structured validation requests | `python experiments/run_validation.py` |
| Replay the coupled-bounds counterexample | [Instructions and independent trajectory](results/coupled_bounds_check/README.md) |
| Compare the released policy on 600 structured and 229 worded requests with the archived results | `bash reproduce.sh quick` |
| Recompute the numerical case studies | `bash reproduce.sh studies` |
| Rebuild the five data plots, numerical tables and four explanatory diagrams | `bash reproduce.sh assets` (the diagrams require pdfLaTeX and TikZ) |
| Re-evaluate all acceptance policies and readers using the archived weights | `bash reproduce.sh evaluate` after installing the training dependencies and exporting the external word vectors |
| Recreate and hash-check the generated case sets | `python experiments/verify_data.py` |
| Retrain the selected reader with three seeds | `python experiments/train_readers.py --stage final` after installing the training dependencies |

Commands that rerun studies or validation write to `results/`. Use a fresh checkout for a rerun. The original acceptance rows for comparison are in `results/reference/`. The figures and table fragments go to `paper/figures/` and `paper/sections/`. The lineage diagram uses name labels in this distribution.

## Training and external inputs

The selected reader has 113,620 trainable parameters. Four worded evaluation/development sets and the scripted confirmation process are documented in [the data notes](data/README.md). All plant records are simulated.

Install `requirements-training.txt` for retraining or evaluating all candidate readers. For the two pretrained-vector baselines, also install the versioned `en_core_web_md` wheel listed in that file and run `python experiments/export_vectors.py`. The script exports the external vectors, which the loader checks against `models/vectors_en_core_web_md.sha256.json`. The selected reader needs no external vectors. Historical comparisons with the two earlier specialist projects require their original source trees; the scripts, archived baseline outputs and relevant logs are included here.

## Release and files

- [RELEASE.json](RELEASE.json) identifies the original numerical software tag and the SHA-256 digest of the two implementation directories. [SHA256SUMS](SHA256SUMS) covers the combined public package, except the checksum list itself.
- `heros/` contains the specialists, coordinator, reader, interfaces and simulation code. `independent/` contains the separate numerical reference implementations.
- `data/`, `models/`, `experiments/` and `validation/` contain the case data, labeled requests, training checkpoints, scripts and validation definitions.
- `results/` contains numerical outputs, raw benchmark rows, archived predictions, comparisons across revisions, replay records and run logs. `results/paper_numbers.json` maps reported numbers to their sources.
- [Evaluation protocol](docs/PROTOCOL.md), [freeze record](docs/FREEZE_RECORD.json), [source-to-method register](docs/SOURCE_TO_METHOD.md), [fidelity and method checks](docs/LEGACY_FINDINGS.md). The protocol and freeze record are historical records from the original project and refer to its original tree and tags.
