"""Regenerate the data sets into a temporary directory and compare them with the frozen files and data/manifest.json.

    python experiments/verify_data.py            (about 3 minutes on 2 cores)
    python experiments/verify_data.py --install  (light archive: also copy the regenerated case files into data/ when their
                                                  hashes equal the manifest)
Expected: the four case files reproduce their manifest hashes exactly.  Fact sheets are compared field by field; the
one known difference is the unit string of test-00182 (display defect fixed after the sheets were handed out)."""
import hashlib, json, pathlib, subprocess, sys, tempfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
man = json.loads((ROOT / "data" / "manifest.json").read_text())
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run([sys.executable, str(ROOT / "experiments" / "make_data.py"), "--out", tmp], check=True, stdout=subprocess.DEVNULL)
    ok = True
    for name, h in man["sha256"].items():
        got = hashlib.sha256((pathlib.Path(tmp) / name).read_bytes()).hexdigest()
        fz = ROOT / "data" / name
        if fz.exists():
            frozen = hashlib.sha256(fz.read_bytes()).hexdigest()
            print(f"{name:28s} regenerated {'==' if got == h else '!='} manifest; frozen file {'==' if frozen == h else '!='} manifest")
            ok &= got == h and frozen == h
        else:
            print(f"{name:28s} regenerated {'==' if got == h else '!='} manifest; no frozen copy in data/ (light archive)")
            ok &= got == h
            if "--install" in sys.argv and got == h:
                import shutil
                shutil.copy(pathlib.Path(tmp) / name, fz)
                print(f"{'':28s} installed into data/")
    for f in sorted((pathlib.Path(tmp) / "authors").glob("sheets_*.json")):
        a = json.loads(f.read_text()); b = json.loads((ROOT / "data" / "authors" / f.name).read_text())
        diff = [x.get("id") for x, y in zip(a, b) if x != y]
        print(f"{f.name:28s} {len(a)} sheets, {len(diff)} differ" + (f": {diff}" if diff else ""))
sys.exit(0 if ok else 1)
