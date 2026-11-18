import json, pathlib, sys, time, platform
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
from heros.core.assess import jsonable   # noqa: E402


def save(name, obj):
    obj = dict(obj)
    obj["_meta"] = {"script": pathlib.Path(sys.argv[0]).name, "python": platform.python_version(), "numpy": np.__version__,
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hardware": "2 CPU cores, no GPU (build machine)"}
    (RES / name).write_text(json.dumps(jsonable(obj), indent=1, allow_nan=False))
    print("wrote results/" + name)


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]
