"""A tiny local web server for driving the limbic arm and browsing run logs.

Pure Python standard library — no Flask, no build step, no npm. That's deliberate:
it starts with one command on any machine, which is what makes it easy to run and
to drive from Claude Code.

    python web/server.py                # then open http://localhost:8765
    python web/server.py --port 9000

Pages:
    GET /            the "Ask" page — type a task (or click a test button) -> run
    GET /runs        the "Logs" page — a scrollable list of every past run

JSON API (what the pages call, and what Claude Code can call directly):
    POST /api/run            body {"task": "...", "mode": "auto|claude|offline"}
                             -> runs the pipeline, returns the structured result
    GET  /api/runs           -> [summary, ...] of every run, newest first
    GET  /api/runs/<run_id>  -> full detail of one run (all three log streams)
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pipeline  # local module (same folder)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


class Handler(BaseHTTPRequestHandler):
    """Routes static pages and the small JSON API."""

    # Quieter, friendlier request log line.
    def log_message(self, fmt: str, *args) -> None:
        print(f"[web] {self.address_string()} {fmt % args}")

    # ----- helpers ------------------------------------------------------- #
    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self._send_json({"error": "not found"}, status=404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ----- GET ----------------------------------------------------------- #
    def do_GET(self) -> None:
        route = urlparse(self.path).path

        if route == "/":
            self._send_file(_STATIC_DIR / "index.html")
        elif route == "/runs":
            self._send_file(_STATIC_DIR / "runs.html")
        elif route == "/api/runs":
            self._send_json({"runs": pipeline.list_runs()})
        elif route.startswith("/api/runs/"):
            run_id = route[len("/api/runs/"):]
            detail = pipeline.get_run(run_id)
            if detail is None:
                self._send_json({"error": f"no such run: {run_id}"}, status=404)
            else:
                self._send_json(detail)
        elif route.startswith("/static/"):
            # Resolve safely under the static dir (no path traversal).
            target = (_STATIC_DIR / route[len("/static/"):]).resolve()
            if _STATIC_DIR.resolve() in target.parents:
                self._send_file(target)
            else:
                self._send_json({"error": "forbidden"}, status=403)
        else:
            self._send_json({"error": "not found"}, status=404)

    # ----- POST ---------------------------------------------------------- #
    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route != "/api/run":
            self._send_json({"error": "not found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return

        task = (payload.get("task") or "").strip()
        mode = payload.get("mode", "auto")
        if not task:
            self._send_json({"error": "missing 'task'"}, status=400)
            return

        try:
            result = pipeline.run_task(task, mode=mode)
        except Exception as exc:  # never crash the server on a bad run
            self._send_json({"status": "error", "task": task, "error": f"{type(exc).__name__}: {exc}"}, status=200)
            return
        self._send_json(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="limbic local web console")
    parser.add_argument("--port", type=int, default=8765, help="port (default 8765)")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{'localhost' if args.host == '127.0.0.1' else args.host}:{args.port}"
    print(f"[web] limbic console running at {url}")
    print(f"[web] open {url}/ to ask the arm to do something, {url}/runs for the logs.")
    print("[web] Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] stopping.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
