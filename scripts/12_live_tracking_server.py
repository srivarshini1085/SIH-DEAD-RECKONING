"""
Live Tracking Server — Third-Party Remote Viewing
==================================================
Allows a dispatcher, fleet manager, or family member to view the vehicle's
live position remotely, matching the ride-hailing/logistics use case in
ISRO SIH problem statement 26168.

Endpoints:
  GET  /position      → current position as JSON
  WS   /ws/track      → WebSocket stream, pushes update on every pipeline step

How the existing frontend dashboard connects (live mode):
  Instead of replaying sample_data.csv, open index.html and connect via:

      const ws = new WebSocket("ws://localhost:8000/ws/track");
      ws.onmessage = (event) => {
          const pos = JSON.parse(event.data);
          // pos = { time, lat, lng, status, drift_m, velocity_mps, heading_deg }
          updateVehicleMarker(pos.lat, pos.lng);
          updateStatusBadge(pos.status);
      };

  This lets a second person open the SAME dashboard in another browser tab
  and see the same vehicle moving live — fulfilling the "third-party tracking"
  requirement without a separate app.

Usage:
    # Start server (blocking):
    py scripts/12_live_tracking_server.py

    # Start server in background from another script:
    from scripts.12_live_tracking_server import start_server_background, push_update
    start_server_background(port=8000)
    push_update({"time": 0.0, "lat": 52.5, "lng": -1.5, "status": "GPS_LOCK", "drift_m": 0.0})

Required packages:
    pip install fastapi uvicorn websockets
"""

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# ── Shared state (thread-safe via asyncio.Queue) ──────────────────────────────
_current_position: dict[str, Any] = {
    "time": 0.0,
    "lat": 0.0,
    "lng": 0.0,
    "status": "WAITING",
    "drift_m": 0.0,
    "velocity_mps": 0.0,
    "heading_deg": 0.0,
}
_connected_clients: set = set()
_event_loop: asyncio.AbstractEventLoop | None = None
_update_queue: asyncio.Queue | None = None


# ── FastAPI app ───────────────────────────────────────────────────────────────

def _build_app():
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    app = FastAPI(title="IDR Live Tracking Server", version="1.0")

    # Allow the frontend dashboard (any origin) to connect
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/position")
    async def get_position():
        """Returns the latest vehicle position as JSON."""
        return _current_position

    @app.get("/health")
    async def health():
        return {"status": "ok", "clients": len(_connected_clients)}

    @app.websocket("/ws/track")
    async def ws_track(websocket: WebSocket):
        """WebSocket endpoint — pushes position update on every pipeline step."""
        await websocket.accept()
        _connected_clients.add(websocket)
        try:
            # Send current state immediately on connect
            await websocket.send_text(json.dumps(_current_position))
            # Keep connection alive; updates are pushed via push_update()
            while True:
                await asyncio.sleep(30)   # heartbeat — client should not time out
                await websocket.send_text(json.dumps({"heartbeat": True}))
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            _connected_clients.discard(websocket)

    return app


# ── Push update from pipeline (called from 09_run_fusion_pipeline.py) ─────────

def push_update(position: dict[str, Any]):
    """
    Thread-safe: update current position and broadcast to all WS clients.
    Call this from the fusion pipeline on every processed step.

    Args:
        position: dict with keys: time, lat, lng, status, drift_m,
                  velocity_mps (optional), heading_deg (optional)
    """
    global _current_position
    _current_position = {
        "time":         float(position.get("time", 0.0)),
        "lat":          float(position.get("lat", 0.0)),
        "lng":          float(position.get("lng", 0.0)),
        "status":       str(position.get("status", "UNKNOWN")),
        "drift_m":      float(position.get("drift_m", 0.0)),
        "velocity_mps": float(position.get("velocity_mps", 0.0)),
        "heading_deg":  float(position.get("heading_deg", 0.0)),
    }

    if _event_loop is not None and _connected_clients:
        msg = json.dumps(_current_position)
        asyncio.run_coroutine_threadsafe(_broadcast(msg), _event_loop)


async def _broadcast(msg: str):
    dead = set()
    for ws in list(_connected_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    _connected_clients.difference_update(dead)


# ── Background server launcher ────────────────────────────────────────────────

def start_server_background(host: str = "127.0.0.1", port: int = 8000):
    """
    Start the FastAPI server in a daemon thread so the pipeline can continue.
    Call this once at the start of 09_run_fusion_pipeline.py when --live is set.
    """
    global _event_loop

    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn not installed. Run: pip install uvicorn")

    app = _build_app()

    def _run():
        global _event_loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _event_loop = loop
        config = uvicorn.Config(app, host=host, port=port, log_level="warning", loop="asyncio")
        server = uvicorn.Server(config)
        loop.run_until_complete(server.serve())

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Give the server a moment to start
    import time
    time.sleep(1.5)
    print(f"  Live tracking server started → http://{host}:{port}/position")
    print(f"  WebSocket endpoint           → ws://{host}:{port}/ws/track")


# ── Blocking server (direct run) ─────────────────────────────────────────────

def main():
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn not installed. Run: pip install fastapi uvicorn")

    app = _build_app()

    print("=" * 55)
    print("  IDR Live Tracking Server")
    print("=" * 55)
    print("  REST  → http://localhost:8000/position")
    print("  WS    → ws://localhost:8000/ws/track")
    print("  Health→ http://localhost:8000/health")
    print()
    print("  Connect your frontend dashboard:")
    print('  const ws = new WebSocket("ws://localhost:8000/ws/track");')
    print("=" * 55)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
