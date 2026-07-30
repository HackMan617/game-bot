"""Telemetry — the one-way channel from the training loop to the viewer.

This is the piece that makes the new architecture fast. The old trainer drew the
network with pygame *inside* the decision loop, behind a `clock.tick(120)`, so
watching the bot capped it at 120 decisions a second and the expensive part (a
full redraw every single step) ran whether or not anyone was looking.

Here the trainer never renders. It calls `should_capture()`, which is a couple of
integer comparisons and answers False unless a browser is actually connected and
a frame is due at viewer framerate. Only then does it pay for introspection and a
`publish()`, and `publish()` is a JSON encode plus two assignments — it never
touches a socket, never waits on the viewer, and never blocks the handshake.

Two daemon threads do the rest: an asyncio websocket server that pushes the most
recent snapshot to each client, and a static file server for `viewer/`. Clients
poll a sequence number, so a slow or paused browser simply misses frames instead
of applying backpressure to training.
"""

import asyncio
import base64
import functools
import http.server
import json
import os
import threading
import time
from typing import Optional

import numpy as np

try:      # websockets >= 14 (the legacy implementation is gone in 16)
    from websockets.asyncio.server import serve as _ws_serve
except ImportError:                                     # pragma: no cover
    from websockets import serve as _ws_serve

DEFAULT_HTTP_PORT = 8765
DEFAULT_WS_PORT = 8766
VIEWER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "viewer")


def b64(arr: np.ndarray) -> str:
    """uint8 array -> base64. The viewer decodes straight into a Uint8Array."""
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.uint8)).decode("ascii")


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):        # a request line per asset is just noise
        pass


class Telemetry:
    """Publishes training snapshots to any connected viewer."""

    def __init__(self, *, http_port: int = DEFAULT_HTTP_PORT,
                 ws_port: int = DEFAULT_WS_PORT, fps: float = 20.0,
                 viewer_dir: str = VIEWER_DIR, enabled: bool = True):
        self.http_port = http_port
        self.ws_port = ws_port
        self.fps = fps
        self.viewer_dir = viewer_dir
        self.enabled = enabled
        self._period = 1.0 / fps if fps > 0 else 0.0
        self._last = 0.0
        self._clients = 0
        self._seq = 0
        self._latest: Optional[str] = None
        self._meta = "{}"
        self._stopping = threading.Event()
        self._httpd = None

    # --- lifecycle -----------------------------------------------------------
    def start(self, meta: dict) -> str:
        """Start both servers and return the URL to open."""
        self._meta = json.dumps({"type": "meta", **meta})
        if not self.enabled:
            return ""
        threading.Thread(target=self._run_ws, daemon=True, name="gdbot-ws").start()
        threading.Thread(target=self._run_http, daemon=True, name="gdbot-http").start()
        return (f"http://localhost:{self.http_port}/"
                f"?ws=ws://localhost:{self.ws_port}")

    def stop(self) -> None:
        # A plain threading.Event polled from inside the loop, rather than
        # loop.stop() from outside it: stopping a loop mid-await tears down the
        # server task and prints a traceback on the way out of a clean exit.
        self._stopping.set()
        if self._httpd is not None:
            self._httpd.shutdown()

    @property
    def viewers(self) -> int:
        return self._clients

    # --- the trainer-facing API ----------------------------------------------
    def should_capture(self) -> bool:
        """True when a viewer is attached and the next frame is due.

        Call this *before* doing any introspection work: when it answers False —
        which is every step of an unwatched run — the trainer has spent three
        comparisons and nothing else.
        """
        if not self.enabled or self._clients <= 0:
            return False
        now = time.monotonic()
        if now - self._last < self._period:
            return False
        self._last = now
        return True

    def publish(self, payload: dict) -> None:
        """Hand over a snapshot. Encode-and-store only; never sends, never waits."""
        if not self.enabled:
            return
        self._latest = json.dumps({"type": "frame", **payload})
        self._seq += 1        # atomic under the GIL; senders poll it

    # --- servers -------------------------------------------------------------
    def _run_http(self) -> None:
        handler = functools.partial(_QuietHandler, directory=self.viewer_dir)
        try:
            self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", self.http_port),
                                                          handler)
        except OSError as exc:
            print(f"[telemetry] viewer HTTP server disabled: {exc}")
            return
        self._httpd.serve_forever()

    def _run_ws(self) -> None:
        try:
            asyncio.run(self._ws_main())
        except OSError as exc:
            print(f"[telemetry] websocket server disabled: {exc}")

    async def _ws_main(self) -> None:
        async with await _ws_serve(self._client, "127.0.0.1", self.ws_port):
            while not self._stopping.is_set():
                await asyncio.sleep(0.2)

    async def _client(self, ws) -> None:
        self._clients += 1
        try:
            await ws.send(self._meta)
            seen = -1
            poll = max(0.005, self._period / 2)
            while not self._stopping.is_set():
                if self._seq != seen and self._latest is not None:
                    seen = self._seq
                    await ws.send(self._latest)
                await asyncio.sleep(poll)
        except Exception:
            pass            # a viewer closing its tab is not a training problem
        finally:
            self._clients -= 1
