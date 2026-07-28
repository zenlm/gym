"""Vendor-neutral HTTP transport.

Carries canonical BF16 LoRA-delta blobs between workers and coordinator.
HTTP intentional: works over any L3 the user has (Thunderbolt 5 networking,
mDNS .local, plain Ethernet, Tailscale). No vendor SDK required on the
wire — that's the whole point.

Protocol:
    PUT  /v1/round/{round_id}/worker/{name}   raw bytes (delta blob)
    GET  /v1/round/{round_id}/aggregate       raw bytes (aggregated delta)
    POST /v1/round/{round_id}/end             body: {"loss": float, "step": int}
    GET  /v1/topology                         returns the active assignment
    GET  /v1/healthz                          {"ok": true, "backend": ...}

Auth: HMAC-SHA256 over (method | path | body | timestamp). Symmetric token
per node (from lab.yaml). Skipped only when token is None (dev mode).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


# ── auth ─────────────────────────────────────────────────────────────────────


def sign(method: str, path: str, body: bytes, secret: str, ts: int | None = None) -> tuple[str, int]:
    ts = ts or int(time.time())
    msg = f"{method}|{path}|{ts}".encode() + b"|" + hashlib.sha256(body).digest()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    return sig, ts


def verify(method: str, path: str, body: bytes, secret: str,
           sig: str, ts: int, max_skew: int = 300) -> bool:
    if abs(int(time.time()) - ts) > max_skew:
        return False
    expected, _ = sign(method, path, body, secret, ts=ts)
    return hmac.compare_digest(expected, sig)


# ── client ───────────────────────────────────────────────────────────────────


@dataclass
class TransportClient:
    coordinator_url: str         # http://spark.lan:8443
    worker_name: str
    secret: Optional[str] = None
    timeout_s: int = 600

    def healthz(self) -> dict:
        return self._request("GET", "/v1/healthz")

    def put_delta(self, round_id: int, blob: bytes) -> None:
        self._request("PUT", f"/v1/round/{round_id}/worker/{self.worker_name}", body=blob)

    def get_aggregate(self, round_id: int) -> bytes:
        return self._request_raw("GET", f"/v1/round/{round_id}/aggregate")

    def end_round(self, round_id: int, loss: float, step: int) -> None:
        import json
        body = json.dumps({"loss": loss, "step": step}).encode()
        self._request("POST", f"/v1/round/{round_id}/end", body=body)

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _request(self, method: str, path: str, body: bytes = b"") -> dict:
        import json
        raw = self._request_raw(method, path, body)
        if not raw:
            return {}
        return json.loads(raw)

    def _request_raw(self, method: str, path: str, body: bytes = b"") -> bytes:
        import urllib.request
        headers = {"Content-Type": "application/octet-stream"}
        if self.secret:
            sig, ts = sign(method, path, body, self.secret)
            headers["X-Zen-Sig"] = sig
            headers["X-Zen-Ts"] = str(ts)
            headers["X-Zen-Worker"] = self.worker_name
        req = urllib.request.Request(
            self.coordinator_url.rstrip("/") + path,
            data=body if body else None,
            method=method, headers=headers,
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return resp.read()


# ── server ───────────────────────────────────────────────────────────────────


def make_server(state, secrets: dict[str, str], bind: tuple[str, int]):
    """Tiny stdlib HTTP server. state is a CoordinatorState (coordinator.py).

    Kept dependency-free on purpose — no Flask / FastAPI on workers. The
    coordinator can be swapped to a real server when we outgrow this.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *a, **k):  # quiet
            log.debug("http: " + a[0], *a[1:])

        def _auth(self, body: bytes) -> Optional[str]:
            if not secrets:
                return self.headers.get("X-Zen-Worker") or "anon"
            worker = self.headers.get("X-Zen-Worker", "")
            sig = self.headers.get("X-Zen-Sig", "")
            ts  = int(self.headers.get("X-Zen-Ts", "0"))
            secret = secrets.get(worker)
            if not secret or not verify(self.command, self.path, body, secret, sig, ts):
                self.send_response(401); self.end_headers(); return None
            return worker

        def _read_body(self) -> bytes:
            n = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(n) if n else b""

        def do_GET(self):
            body = b""
            # Public read endpoints don't require auth — hanzo desktop + cloud
            # need to poll without juggling secrets. None of these leak data.
            public = {"/", "/v1/healthz", "/v1/metrics", "/v1/topology"}
            if self.path not in public:
                who = self._auth(body)
                if who is None: return
            try:
                if self.path == "/":
                    self._send_html(_dashboard_html())
                elif self.path == "/v1/healthz":
                    self._send_json({"ok": True})
                elif self.path == "/v1/topology":
                    self._send_json(state.topology_view())
                elif self.path == "/v1/metrics":
                    self._send_json(state.metrics())
                elif self.path.startswith("/v1/round/") and self.path.endswith("/aggregate"):
                    rid = int(self.path.split("/")[3])
                    blob = state.get_aggregate(rid)
                    self._send_raw(blob)
                else:
                    self.send_response(404); self.end_headers()
            except Exception as e:
                log.exception("GET %s failed", self.path)
                self.send_response(500); self.end_headers(); self.wfile.write(str(e).encode())

        def do_PUT(self):
            body = self._read_body()
            who = self._auth(body)
            if who is None: return
            try:
                if self.path.startswith("/v1/round/") and "/worker/" in self.path:
                    parts = self.path.split("/")
                    rid = int(parts[3]); name = parts[5]
                    if name != who and secrets:
                        self.send_response(403); self.end_headers(); return
                    state.put_delta(rid, name, body)
                    self._send_json({"ok": True})
                else:
                    self.send_response(404); self.end_headers()
            except Exception as e:
                log.exception("PUT %s failed", self.path)
                self.send_response(500); self.end_headers(); self.wfile.write(str(e).encode())

        def do_POST(self):
            body = self._read_body()
            who = self._auth(body)
            if who is None: return
            try:
                if self.path.endswith("/end"):
                    import json
                    rid = int(self.path.split("/")[3])
                    payload = json.loads(body or b"{}")
                    state.end_round(rid, who, payload)
                    self._send_json({"ok": True})
                else:
                    self.send_response(404); self.end_headers()
            except Exception as e:
                log.exception("POST %s failed", self.path)
                self.send_response(500); self.end_headers(); self.wfile.write(str(e).encode())

        # ── helpers ──
        def _send_json(self, d):
            import json
            blob = json.dumps(d).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers(); self.wfile.write(blob)

        def _send_raw(self, blob: bytes):
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers(); self.wfile.write(blob)

        def _send_html(self, html: str):
            blob = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers(); self.wfile.write(blob)

    return ThreadingHTTPServer(bind, _Handler)


# ── built-in dashboard (no JS framework, fetches /v1/metrics every 2s) ──────


def _dashboard_html() -> str:
    """Minimal HTML — works in any browser, hanzo desktop, cloud-tunnel."""
    return """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><title>zen lab</title>
<style>
  body { font: 13px/1.5 -apple-system, ui-monospace, Menlo, monospace;
         background: #0a0a0a; color: #e8e8e8; padding: 24px; max-width: 1000px; margin: auto; }
  h1 { font-size: 18px; margin: 0 0 4px; font-weight: 600; }
  h2 { font-size: 13px; margin: 24px 0 8px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
  table { border-collapse: collapse; width: 100%; margin: 8px 0; }
  th, td { text-align: left; padding: 6px 12px; border-bottom: 1px solid #1a1a1a; }
  th { color: #888; font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
  .bar { display: inline-block; height: 8px; background: #2da44e; border-radius: 2px; vertical-align: middle; }
  .ok { color: #2da44e; } .warn { color: #d29922; } .err { color: #f85149; }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .pill { display: inline-block; padding: 1px 6px; border-radius: 8px; background: #1a1a1a; font-size: 11px; color: #888; }
  .pill.cuda { background: #1c5b1f; color: #7ed688; }
  .pill.rocm { background: #5b1c1c; color: #e88; }
  .pill.mlx  { background: #1c365b; color: #7eaad6; }
  .pill.mps  { background: #3a3a3a; color: #aaa; }
  small { color: #555; }
</style></head><body>
<h1>zen lab</h1>
<small id="updated">loading...</small>
<h2>workers</h2>
<table id="workers"><thead><tr>
  <th>name</th><th>host</th><th>backend</th><th class="num">mem</th>
  <th class="num">data %</th><th>experts pinned</th>
</tr></thead><tbody></tbody></table>
<h2>recent rounds</h2>
<table id="rounds"><thead><tr>
  <th class="num">round</th><th>workers received</th><th class="num">losses</th>
  <th class="num">duration</th><th>status</th>
</tr></thead><tbody></tbody></table>
<script>
async function tick() {
  try {
    const m = await (await fetch('/v1/metrics')).json();
    const t = m.topology;
    document.getElementById('updated').textContent = new Date().toLocaleTimeString() + ' — current round ' + m.current_round;
    const wb = document.querySelector('#workers tbody');
    wb.innerHTML = '';
    for (const w of t.workers) {
      const dw = t.data_weights[w.name] || 0;
      const pins = (t.expert_pins[w.name] || []).join(', ') || '<small>auto</small>';
      wb.insertAdjacentHTML('beforeend',
        `<tr><td>${w.name}</td><td><small>${w.host}</small></td>
         <td><span class="pill ${w.backend}">${w.backend}</span></td>
         <td class="num">${w.memory_gb} GB</td>
         <td class="num">${(dw*100).toFixed(1)}%</td>
         <td>${pins}</td></tr>`);
    }
    const rb = document.querySelector('#rounds tbody');
    rb.innerHTML = '';
    for (const r of m.rounds.slice(-15).reverse()) {
      const losses = Object.values(r.losses);
      const mean = losses.length ? (losses.reduce((a,b)=>a+b,0)/losses.length).toFixed(4) : '—';
      const status = r.aggregated ? '<span class="ok">aggregated</span>' :
                     `<span class="warn">${r.received.length}/${r.expected.length}</span>`;
      const dur = r.duration_s ? r.duration_s.toFixed(1) + 's' : '—';
      rb.insertAdjacentHTML('beforeend',
        `<tr><td class="num">${r.round_id}</td>
         <td><small>${r.received.join(', ')}</small></td>
         <td class="num">${mean}</td><td class="num">${dur}</td>
         <td>${status}</td></tr>`);
    }
  } catch (e) {
    document.getElementById('updated').innerHTML = '<span class="err">error: ' + e.message + '</span>';
  }
}
tick(); setInterval(tick, 2000);
</script></body></html>
"""
