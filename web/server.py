"""Static server for the browser demo with cross-origin isolation headers.

Same as `python -m http.server`, but adds Cross-Origin-Opener-Policy and
Cross-Origin-Embedder-Policy headers. Browsers require these before they
allow multi-threaded WASM — without them ONNX Runtime Web silently falls
back to single-threaded, slower inference on machines without WebGPU.

COEP uses `credentialless` (not `require-corp`) so the CDN-hosted runtime
and models keep loading without CORP headers.

Usage:  python web/server.py [port]     (default port 8090)
"""

import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent


class IsolatedHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    handler = partial(IsolatedHandler, directory=str(WEB_DIR))
    server = HTTPServer(("", port), handler)
    print(f"Serving {WEB_DIR} with cross-origin isolation "
          f"on http://localhost:{port}")
    server.serve_forever()
