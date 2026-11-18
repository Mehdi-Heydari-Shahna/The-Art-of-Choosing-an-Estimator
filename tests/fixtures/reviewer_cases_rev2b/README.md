# Review cases of revision 2b (seven JSON files, stored unchanged)

For each file, only its `arguments` object is passed to the public interface `heros.api.solve`; the other fields
record the output of revision 2b and an independent mathematical check. `tests/test_reviewer_cases_rev2b.py` checks
the files against the SHA-256 values below and the revised outcomes; `experiments/reviewer_cases.py` replays them
and `experiments/reviewer_cases_rev2b_compare.py` writes `results/reviewer_cases_rev2b_table.md`.

| File | What the case tests | SHA-256 |
| --- | --- | --- |
| `A1_relative_times_with_jitter.json` | recorded sampling deviation, clock origin 0 | `bcd3146029200c1f4ad079440b56b7133080a889fb6a8e2bf4549e50d5574205` |
| `A2_absolute_times_same_jitter.json` | the same intervals with clock origin 2^30 s | `1ea8898195e9d2d54b2d0af7546a6ec638c8af73725037f6502ef278161ea7e5` |
| `A3_absolute_times_on_grid.json` | regular clock at the same origin (positive control) | `0f16e039d59098833dcc09422e09067542b4486d7101e848147f208cf6d82337` |
| `B_forward_only_carriage.json` | lower limit for a plant whose speed range excludes zero | `792a60c9bdb1206513eee56cfc65e83b81a42aea5a94e30506f5c70efcb44177` |
| `C_exact_window_ends_before_next_sample.json` | requirement window that ends 2^-31 s before the fifth sample | `bdd1c1d4936bc2150f611c0f76fa71937ef5e3003c0fa902e6e8857a1fa1d2dc` |
| `D1_fixed_input_only_in_specification.json` | known input given only as a one-point range | `80fa1f2801e6d699fc9820f72bb0c0caee6add0f85f018e9eb9ae4cf2d173a2d` |
| `D2_explicit_known_input_control.json` | the same input also given as a vector | `2169eb93373461aca41cc0c124062866e5c056f8f3898ec7451580be46dfd5a2` |

The six cases of the revision 2 review (`tests/fixtures/reviewer_cases_rev2/`), the 12 validation requests
(`validation/`) and the two demonstration records keep their outcomes.
