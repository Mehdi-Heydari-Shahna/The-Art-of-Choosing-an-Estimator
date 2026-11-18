"""Re-create models/vectors_en_core_web_md.npz from the pinned spaCy package and verify it.

    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl
    python experiments/export_vectors.py
Only the pretrained-vector candidates and the frozen-vector baseline need these vectors; the deployed reader does not."""
import hashlib, json, sys
import numpy as np
from common import ROOT
from heros.nlu.readers import load_vectors

npz = ROOT / "models" / "vectors_en_core_web_md.npz"
ref = ROOT / "models" / "vectors_en_core_web_md.sha256.json"


def digest(words, M):
    return {"words_sha256": hashlib.sha256("\n".join(words).encode()).hexdigest(),
            "matrix_sha256": hashlib.sha256(np.ascontiguousarray(M, dtype=np.float32).tobytes()).hexdigest(),
            "n_words": len(words), "dim": int(M.shape[1])}


if __name__ == "__main__":
    if "--record" in sys.argv:                 # used once by the developer to record the reference digests
        w, M = load_vectors(str(npz))
        ref.write_text(json.dumps(dict(digest(w, M), source="en_core_web_md 3.8.0 (spaCy), lower-case entries shorter than 25 characters"), indent=1))
        print(ref.read_text()); sys.exit(0)
    w, M = load_vectors(str(npz))          # creates the file from the spaCy package if it does not exist, else reads it
    d, r = digest(w, M), json.loads(ref.read_text())
    ok = all(d[k] == r[k] for k in d)
    print(json.dumps(d, indent=1), "\nmatches the recorded digests:", ok)
    sys.exit(0 if ok else 1)
