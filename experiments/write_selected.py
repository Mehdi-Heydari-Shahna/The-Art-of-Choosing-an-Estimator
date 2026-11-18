"""Write models/selected.json: the checkpoint used by the interface.

Pre-declared (docs/PROTOCOL.md): the interface uses the seed-0 checkpoint of the selected architecture; seeds 1 and 2
are trained only to measure variability.  The choice does not depend on any test result."""
import hashlib, json
from common import ROOT

M = ROOT / "models"
arch = json.loads((M / "selection.json").read_text())["selected_architecture"]
name = f"final_{arch}_s0"
cfg = json.loads((M / name / "config.json").read_text())
f = M / name / cfg["file"]
info = {"kind": "linear" if arch == "linear" else "neural", "id": name, "path": f"{name}/{cfg['file']}",
        "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "architecture": arch,
        "vectors": None if arch in ("linear", "neural_scratch") else "vectors_en_core_web_md.npz",
        "n_trainable_parameters": cfg.get("n_trainable_parameters"), "rule": "seed 0 of the selected architecture (pre-declared)"}
assert info["sha256"] == cfg["sha256"], "checkpoint does not match its configuration record"
npz = M / name / "reader_numpy.npz"
if npz.exists():       # NumPy export of the same checkpoint (experiments/check_numpy_reader.py): the interface needs no PyTorch
    info["numpy_export"] = f"{name}/reader_numpy.npz"
    info["numpy_sha256"] = {f: hashlib.sha256((M / name / f).read_bytes()).hexdigest() for f in ("reader_numpy.npz", "reader_numpy.json")}
(M / "selected.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info, indent=1))
