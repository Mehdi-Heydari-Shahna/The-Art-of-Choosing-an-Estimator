"""POST-HOC check requested by review B (finding B5): effect of the display fix of squared compound units on the
trained reader.  Seed 0 is retrained on the training file as it was BEFORE the fix (commit dab7b09) and both readers
are scored on the authored test requests.  This analysis looks at test requests after the test and is labelled post hoc.

    python experiments/posthoc_prefix_training.py /path/to/train_before_fix.jsonl
Writes models/posthoc_prefix_s0/ and results/posthoc_prefix_effect.json."""
import json, sys, time
from common import ROOT, save, load_jsonl
from heros import evaluation as EV
from heros.nlu import readers as RD
import train_readers as TR
import nl_common as NC
import eval_dev as ED

old = load_jsonl(sys.argv[1])
train = [EV.example_from_request(c, c["request"], TR.proposal_of(c)) for c in old]
r = TR.train_one("posthoc_prefix_s0", lambda: RD.NeuralReader("scratch", 0), train, {"note": "post-hoc: training file before the display fix (dab7b09)"})
items = NC.authored("test")
out = {"note": "post hoc (review B, B5); the reported test results use final_neural_scratch_s0", "n_requests": len(items)}
for name, rd in (("final_neural_scratch_s0", ED.load("final_neural_scratch_s0")[0]), ("posthoc_prefix_s0", r)):
    m = NC.reading_metrics(rd, items)
    sq = [i for i in items if "^2" in i[1] or "²" in i[1]]
    ms = NC.reading_metrics(rd, sq) if sq else None
    out[name] = {"role_accuracy": m["role_accuracy"], "exact_reading": m["exact_reading"], "mean_edits": m["mean_edits"],
                 "requests_with_squared_units": len(sq), "exact_reading_on_those": ms["exact_reading"] if ms else None}
    print(name, out[name], flush=True)
save("posthoc_prefix_effect.json", out)
