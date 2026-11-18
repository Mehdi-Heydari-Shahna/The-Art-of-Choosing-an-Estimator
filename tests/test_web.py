"""The local web page drives the same pipeline as the command line (standard library only)."""
import json, threading, urllib.request
from http.server import ThreadingHTTPServer
from heros.interface import web


_OPEN = urllib.request.build_opener(urllib.request.ProxyHandler({}))      # local server: never through a proxy


def _post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with _OPEN.open(req, timeout=60) as r:
        return json.loads(r.read())


def test_web_flow_from_three_answers_to_replayable_record():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with _OPEN.open(base + "/", timeout=20) as r:
            page = r.read().decode()
        assert "What do you want to estimate?" in page
        ex = json.loads(_OPEN.open(base + "/api/example?id=rotary_joint", timeout=20).read())
        out = _post(base, "/api/interpret", {"model": ex["model"], "answers": ex["answers"], "data": ex["data"]})
        sid, rb = out["sid"], out["readback"]
        assert any(r["status"] == "interpreted" for r in rb["rows"])
        res = _post(base, "/api/confirm", {"sid": sid, "spec_hash": rb["spec_hash"]})
        assert res["decision"]["disposition"] == "CLARIFY" and res["decision"]["question"]["path"] == "initial.E"
        res = _post(base, "/api/answer", {"sid": sid, "path": "initial.E", "value": [0.001, 2.0]})
        assert res["decision"]["disposition"] == "RELEASE" and res["decision"]["selected"] == "high_gain_observer"
        rec = json.loads(_OPEN.open(base + "/api/record?sid=" + sid, timeout=60).read())
        from heros import api
        assert api.replay(rec)["match"]
    finally:
        srv.shutdown()
