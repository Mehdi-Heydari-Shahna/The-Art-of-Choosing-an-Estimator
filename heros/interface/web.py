"""Local web page for the three-question workflow (standard library only):  python -m heros serve"""
from __future__ import annotations

import json
import pathlib
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .. import record as R
from ..core.assess import jsonable
from . import service as SV

EX = pathlib.Path(__file__).resolve().parents[2] / "examples"
SESSIONS = {}

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Heros in Estimation</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--ink:#1b2430;--mut:#5d6b7a;--line:#d9dee5;--bg:#f6f7f9;--acc:#1f4e79;--ok:#1d6b43;--warn:#8a5a00;--bad:#8c2f2f}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,Segoe UI,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg)}
main{max-width:980px;margin:0 auto;padding:24px 18px 60px}h1{font-size:22px;margin:0 0 2px}h2{font-size:16px;margin:26px 0 8px}
p.lead{color:var(--mut);margin:0 0 18px}.card{background:#fff;border:1px solid var(--line);border-radius:8px;padding:16px 18px;margin:12px 0}
label{display:block;font-weight:600;margin:12px 0 4px}textarea,input[type=text],select{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:6px;font:inherit}
textarea{min-height:64px}button{font:inherit;padding:8px 16px;border-radius:6px;border:1px solid var(--acc);background:var(--acc);color:#fff;cursor:pointer;margin:10px 8px 0 0}
button.sec{background:#fff;color:var(--acc)}button:disabled{opacity:.5;cursor:default}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600}code{font:12.5px ui-monospace,Menlo,Consolas,monospace}
.tag{display:inline-block;padding:1px 7px;border-radius:10px;font-size:12px;border:1px solid var(--line);color:var(--mut)}
.interpreted{color:var(--warn);border-color:#e2c98f}.supplied{color:var(--ok);border-color:#a9d3bd}.unresolved{color:var(--bad);border-color:#e0b4b4}
.disp{font-weight:700}.RELEASE{color:var(--ok)}.CLARIFY,.CONFIRM{color:var(--warn)}.other{color:var(--bad)}
pre{white-space:pre-wrap;background:#f1f3f6;border-radius:6px;padding:10px 12px;font:12.5px/1.45 ui-monospace,Menlo,Consolas,monospace;margin:8px 0}
img{max-width:100%;border:1px solid var(--line);border-radius:6px}.row{display:flex;gap:14px;flex-wrap:wrap}.row>div{flex:1 1 260px}
small{color:var(--mut)}
</style></head><body><main>
<h1>Heros in Estimation</h1>
<p class="lead">Describe the estimation problem in three answers and attach a model. Numbers and units are read deterministically; the reading is shown for
your confirmation before anything is designed. The numerical methods and all checks are deterministic.</p>
<div class="card">
 <div class="row"><div><label for="ex">Example</label><select id="ex"><option value="">(none)</option></select></div>
 <div><label for="model">Model file (JSON)</label><input type="file" id="model" accept=".json"></div>
 <div><label for="data">Measurements and inputs (CSV: y, u)</label><input type="file" id="data" accept=".csv"></div></div>
 <label for="q1">1. What do you want to estimate?</label><textarea id="q1"></textarea>
 <label for="q2">2. What do you measure, and what model and uncertainty information do you have?</label><textarea id="q2" style="min-height:110px"></textarea>
 <label for="q3">3. What does successful estimation mean for you?</label><textarea id="q3"></textarea>
 <button id="go">Interpret</button><small id="status"></small>
</div>
<div id="readback" class="card" hidden></div>
<div id="result" class="card" hidden></div>
</main><script>
let SID=null, MODEL=null, DATA=null, HASH=null;
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function api(path, body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const j=await r.json();if(!r.ok)throw new Error(j.error||r.statusText);return j;}
fetch('/api/examples').then(r=>r.json()).then(list=>{for(const e of list){const o=document.createElement('option');o.value=e.id;o.textContent=e.title;$('ex').appendChild(o);}});
$('ex').onchange=async()=>{if(!$('ex').value)return;const e=await (await fetch('/api/example?id='+$('ex').value)).json();MODEL=e.model;DATA=e.data;$('q1').value=e.answers[0];$('q2').value=e.answers[1];$('q3').value=e.answers[2];$('status').textContent=' example model and data loaded';};
$('model').onchange=async e=>{MODEL=JSON.parse(await e.target.files[0].text());};
$('data').onchange=async e=>{DATA=await e.target.files[0].text();};
$('go').onclick=async()=>{try{if(!MODEL){$('status').textContent=' please attach a model file or pick an example';return;}
 const r=await api('/api/interpret',{model:MODEL,data:DATA,answers:[$('q1').value,$('q2').value,$('q3').value]});SID=r.sid;showReadback(r.readback);$('result').hidden=true;}catch(err){$('status').textContent=' '+err.message;}};
function showReadback(rb){HASH=rb.spec_hash;let h='<h2 style="margin-top:0">Interpreted specification</h2><table><tr><th>Field</th><th>Value</th><th>Status</th><th>Evidence</th></tr>';
 for(const r of rb.rows){const v=r.value!==null?JSON.stringify(r.value):'range '+JSON.stringify(r.range);
  h+=`<tr><td><code>${esc(r.path)}</code></td><td><input type="text" data-path="${esc(r.path)}" data-orig='${esc(v)}' value='${esc(v)}' ${r.status==='supplied'?'disabled':''}></td><td><span class="tag ${r.status}">${r.status}</span></td><td><small>${esc(r.source||'')}</small></td></tr>`;}
 h+='</table>';for(const i of rb.issues)h+=`<p class="unresolved">! ${esc(i.message)}</p>`;
 h+=`<p><small>Reader: ${esc(rb.reader.kind)} ${esc(rb.reader.id||'')} · its proposal (not binding): ${esc(rb.proposal_of_the_reader||'-')}</small></p>`;
 h+='<button id="conf">Confirm and continue</button><small> Edited values are stored as corrections. Nothing is released before confirmation.</small>';
 $('readback').innerHTML=h;$('readback').hidden=false;
 $('conf').onclick=async()=>{const ch={};document.querySelectorAll('#readback input[data-path]').forEach(i=>{if(!i.disabled&&i.value!==i.dataset.orig){try{ch[i.dataset.path]=JSON.parse(i.value);}catch(e){ch[i.dataset.path]=i.value;}}});
  try{const r=await api('/api/confirm',{sid:SID,spec_hash:HASH,changes:ch});showResult(r);}catch(err){alert(err.message);}};}
function showResult(r){const d=r.decision;const cls=['RELEASE','CLARIFY','CONFIRM'].includes(d.disposition)?d.disposition:'other';
 let h=`<h2 style="margin-top:0">${esc(d.display||d.disposition)} <span class="disp ${cls}">${d.disposition}</span></h2><pre>${esc(r.summary)}</pre>`;
 if(d.disposition==='CLARIFY'){h+=`<label for="ans">${esc(d.question.question)}</label><input type="text" id="ans" placeholder="value in SI units (JSON for lists)"><button id="send">Answer</button><button class="sec" id="unk">I do not know</button>`;}
 if(r.plot)h+=`<img alt="estimate with uncertainty" src="data:image/png;base64,${r.plot}">`;
 if(r.monitor)h+=`<p><small>Assumption monitor: ${esc(r.monitor)}</small></p>`;
 h+=`<button class="sec" id="dl">Download execution record (JSON)</button>`;
 $('result').innerHTML=h;$('result').hidden=false;
 $('dl').onclick=()=>{location.href='/api/record?sid='+SID;};
 if(d.disposition==='CLARIFY'){const send=async(unknown)=>{let v=null;if(!unknown){try{v=JSON.parse($('ans').value);}catch(e){v=$('ans').value;}}
   try{showResult(await api('/api/answer',{sid:SID,path:d.question.path,value:v,unknown:unknown}));}catch(err){alert(err.message);}};
  $('send').onclick=()=>send(false);$('unk').onclick=()=>send(true);}}
</script></body></html>"""


def _examples():
    out = []
    for f in sorted(EX.glob("*_answers.json")):
        stem = f.name[: -len("_answers.json")]
        a = json.loads(f.read_text())
        out.append({"id": stem, "title": a.get("title", stem)})
    return out


def _result(conv):
    out = conv.step()
    dec = out["decision"]
    res = {"decision": jsonable(dec), "summary": SV.summary(dec), "plot": None, "monitor": None}
    if out["estimates"] is not None:
        res["plot"] = SV.plot_b64(dec, out["estimates"])
        mon = out["estimates"].get("assumption_monitor")
        res["monitor"] = mon["status"] if mon else None
    return res


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text") or ctype == "application/json" else ""))
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            if self.path == "/" or self.path.startswith("/index"):
                return self._send(200, PAGE, "text/html")
            if self.path == "/api/examples":
                return self._send(200, _examples())
            if self.path.startswith("/api/example?id="):
                stem = self.path.split("=", 1)[1]
                if not stem.replace("_", "").isalnum():
                    return self._send(400, {"error": "bad example id"})
                a = json.loads((EX / f"{stem}_answers.json").read_text())
                dataf = EX / f"{stem}_data.csv"
                return self._send(200, {"model": json.loads((EX / f"{stem}_model.json").read_text()), "answers": a["answers"],
                                        "data": dataf.read_text() if dataf.exists() else None})
            if self.path.startswith("/api/record?sid="):
                conv = SESSIONS.get(self.path.split("=", 1)[1])
                if conv is None or conv.last is None:
                    return self._send(404, {"error": "no record yet"})
                return self._send(200, R.dumps(conv.last["record"]).encode(), "application/json", {"Content-Disposition": 'attachment; filename="heros_record.json"'})
            self._send(404, {"error": "not found"})
        except Exception as ex:  # noqa
            self._send(500, {"error": f"{type(ex).__name__}: {ex}"})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/api/interpret":
                y = u = None
                if body.get("data"):
                    y, u, t = SV.read_data_t(body["data"])
                conv = SV.Conversation(body["model"], body["answers"], y, u, t=t)
                sid = uuid.uuid4().hex
                SESSIONS[sid] = conv
                return self._send(200, {"sid": sid, "readback": conv.readback()})
            conv = SESSIONS.get(body.get("sid"))
            if conv is None:
                return self._send(404, {"error": "unknown session"})
            if self.path == "/api/confirm":
                if body.get("changes"):
                    conv.correct(body["changes"])
                    conv.confirm(conv.readback()["spec_hash"])        # the user confirmed the corrected readback in the same action
                else:
                    conv.confirm(body["spec_hash"])
                return self._send(200, _result(conv))
            if self.path == "/api/answer":
                if body.get("unknown"):
                    conv.session.spec.setdefault("unavailable", []).append(body["path"])
                else:
                    v = body["value"]
                    if body["path"] == "stochastic.R" and isinstance(v, (int, float)):
                        v = [[v]]
                    conv.answer(body["path"], v)
                return self._send(200, _result(conv))
            self._send(404, {"error": "not found"})
        except Exception as ex:  # noqa
            self._send(400, {"error": f"{type(ex).__name__}: {ex}"})


def serve(port: int = 8765):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Heros in Estimation: open http://127.0.0.1:{port}/  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
