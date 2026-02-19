"""
Real-time Trading Dashboard — FastAPI + WebSocket server.

Serves a live dashboard showing:
  - Market card with live Polymarket price
  - Signal layer gauges (MTF, OFI, Correlation, Funding, VP)
  - Scan history table with color-coded actions
  - PnL tracker
  - Auto-refreshes every scan cycle via WebSocket
"""
import asyncio
import json
import os
import tempfile
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from data.buffer import buffer

app = FastAPI(title="Polymarket Trading Dashboard")

# ── Shared state for the dashboard ────────────────────────────
scan_history: deque = deque(maxlen=100)
connected_clients: list[WebSocket] = []
latest_scan: dict = {}
pnl_history: deque = deque(maxlen=500)
price_history: deque = deque(maxlen=300)
_last_price_tick: float = 0  # throttle price ticks to max 5/sec
total_scan_count: int = 0

# ── Disk persistence for crash/restart survival ───────────────
_STATE_FILE = Path(__file__).parent / ".dashboard_state.json"


def _load_state():
    """Restore total_scan_count + scan_history from disk on startup."""
    global total_scan_count
    try:
        if _STATE_FILE.exists():
            data = json.loads(_STATE_FILE.read_text())
            total_scan_count = int(data.get("total_scan_count", 0))
            for entry in data.get("scan_history", []):
                scan_history.append(entry)
            for entry in data.get("price_history", []):
                price_history.append(entry)
            if data.get("latest_scan"):
                latest_scan.update(data["latest_scan"])
            logger.info(f"Dashboard state restored: {total_scan_count} scans, {len(scan_history)} history entries")
    except Exception as e:
        logger.warning(f"Could not load dashboard state: {e}")


def _save_state():
    """Atomically persist state to disk (tempfile + os.replace = crash-safe)."""
    try:
        data = {
            "total_scan_count": total_scan_count,
            "scan_history": list(scan_history),
            "price_history": list(price_history)[-100:],
            "latest_scan": dict(latest_scan) if latest_scan else {},
        }
        fd, tmp = tempfile.mkstemp(dir=_STATE_FILE.parent, suffix=".tmp")
        os.write(fd, json.dumps(data, default=str).encode())
        os.close(fd)
        os.replace(tmp, _STATE_FILE)
    except Exception as e:
        logger.warning(f"Could not save dashboard state: {e}")


# Load persisted state on module import
_load_state()


def record_scan(scan_result: dict):
    """Called by main.py after each scan — stores data for the dashboard.
    Thread-safe: pushes to a queue drained by uvicorn's event loop."""
    global total_scan_count
    ts = time.time()
    entry = {
        "timestamp": ts,
        "time_str": time.strftime("%H:%M:%S", time.localtime(ts)),
        **scan_result,
    }

    # Convert numpy types to native Python for JSON serialization
    for k, v in entry.items():
        if hasattr(v, "item"):  # numpy scalar
            entry[k] = v.item()

    scan_history.append(entry)
    latest_scan.update(entry)
    total_scan_count += 1
    entry['_total_scans'] = total_scan_count

    # Persist to disk (atomic write)
    _save_state()

    # Track price
    market_price = scan_result.get("market_price", 0)
    if market_price and market_price > 0:
        price_history.append({
            "time": entry["time_str"],
            "price": float(market_price),
            "ts": ts,
        })

    # Track PnL
    pnl = scan_result.get("daily_pnl", 0)
    pnl_history.append({
        "time": entry["time_str"],
        "pnl": float(pnl) if pnl else 0,
        "ts": ts,
    })

    # Thread-safe broadcast via queue (drained in uvicorn's event loop)
    try:
        _scan_queue.put_nowait(entry)
    except _queue_mod.Full:
        pass  # Drop if full — dashboard will catch up on next scan


# ── Thread-safe queues (main thread → uvicorn thread) ─────────
import queue as _queue_mod
_price_tick_queue: _queue_mod.Queue = _queue_mod.Queue(maxsize=100)
_scan_queue: _queue_mod.Queue = _queue_mod.Queue(maxsize=50)


def record_price_tick(price: float, bid: float = 0, ask: float = 0, asset: str = 'ETH', no_price: float = 0):
    """Called by polymarket_feed.py on every WS price update (~100-500ms).
    Throttled to max 5 broadcasts/sec to avoid overwhelming the frontend.
    Uses a thread-safe queue since this runs in the main asyncio thread
    while the dashboard's WebSockets live in uvicorn's thread."""
    global _last_price_tick
    now = time.time()
    if now - _last_price_tick < 0.05:  # 50ms throttle (20Hz updates)
        return
    _last_price_tick = now

    tick = {
        "type": "price_tick",
        "asset": asset,
        "price": float(price),
        "no_price": float(no_price),
        "bid": float(bid),
        "ask": float(ask),
        "ts": now,
    }

    price_history.append({
        "time": time.strftime("%H:%M:%S", time.localtime(now)),
        "price": float(price),
        "ts": now,
    })

    try:
        _price_tick_queue.put_nowait(tick)
    except _queue_mod.Full:
        pass  # Drop oldest if full


async def _drain_queues():
    """Unified drain loop running in uvicorn's event loop.
    Drains BOTH scan and price tick queues and broadcasts to clients.
    This is the ONLY place broadcasts happen — guaranteeing correct event loop."""
    while True:
        # Drain scan queue
        while not _scan_queue.empty():
            try:
                entry = _scan_queue.get_nowait()
                await _broadcast(entry)
            except _queue_mod.Empty:
                break

        # Drain price tick queue
        while not _price_tick_queue.empty():
            try:
                tick = _price_tick_queue.get_nowait()
                await _broadcast(tick)
            except _queue_mod.Empty:
                break

        await asyncio.sleep(0.01)  # Check queues every 10ms (100Hz)


@app.on_event("startup")
async def _start_queue_drainer():
    asyncio.create_task(_drain_queues())
    
    # Auto-open dashboard in browser (moved from start.sh for robustness)
    import webbrowser
    def open_browser():
        time.sleep(1)  # Small delay to ensure server is ready
        webbrowser.open("http://localhost:8081")
    
    # Run in thread to not block event loop
    import threading
    threading.Thread(target=open_browser, daemon=True).start()


async def _broadcast(data: dict):
    """Push data to all connected dashboard clients."""
    if not connected_clients:
        return
    msg = json.dumps(data, default=str)
    dead = []
    for ws in connected_clients:
        try:
            await ws.send_text(msg)
        except Exception as e:
            logger.error(f"WS Broadcast error: {e}")
            dead.append(ws)
    for ws in dead:
        if ws in connected_clients:
            connected_clients.remove(ws)


# ── WebSocket endpoint ────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    logger.info(f"Dashboard client connected ({len(connected_clients)} total)")

    # Send initial state
    try:
        # Get Perps Buffer Info (PerpsBuffer uses .states dict)
        buffer_info = {}
        for sym in getattr(buffer, 'states', {}):
            s = buffer.states[sym]
            buffer_info[f"{sym.lower()}_trades"] = len(s.trades)
            buffer_info[f"{sym.lower()}_binance_trades"] = len(s.binance_trades)
            buffer_info[f"{sym.lower()}_mark"] = s.mark_price
            buffer_info[f"{sym.lower()}_binance_mark"] = s.binance_mark
            buffer_info[f"{sym.lower()}_funding"] = s.funding_rate

        init = {
            "type": "init",
            "history": list(scan_history),
            "price_history": list(price_history)[-100:],
            "pnl_history": list(pnl_history)[-50:],
            "latest": dict(latest_scan) if latest_scan else {},
            "total_scans": total_scan_count,
            "buffer_info": buffer_info,
        }
        await ws.send_text(json.dumps(init, default=str))
    except Exception as e:
        logger.debug(f"Dashboard init send error: {e}")

    try:
        while True:
            # Keep connection alive, handle any incoming messages
            data = await ws.receive_text()
            # Client can request buffer stats
            if data == "ping":
                # Re-fetch per-asset buffer info for pong
                buffer_info = {}
                for sym in getattr(buffer, 'states', {}):
                    s = buffer.states[sym]
                    buffer_info[f"{sym.lower()}_trades"] = len(s.trades)
                    buffer_info[f"{sym.lower()}_binance_trades"] = len(s.binance_trades)
                    buffer_info[f"{sym.lower()}_mark"] = s.mark_price
                    buffer_info[f"{sym.lower()}_binance_mark"] = s.binance_mark
                    buffer_info[f"{sym.lower()}_funding"] = s.funding_rate
                
                await ws.send_text(json.dumps({
                    "type": "pong",
                    "buffer_info": buffer_info,
                }, default=str))
    except WebSocketDisconnect:
        connected_clients.remove(ws)
        logger.info(f"Dashboard client disconnected ({len(connected_clients)} total)")


# ── REST endpoints ────────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    buffer_info = {}
    for sym in getattr(buffer, 'states', {}):
        s = buffer.states[sym]
        buffer_info[f"{sym.lower()}_trades"] = len(s.trades)
        buffer_info[f"{sym.lower()}_mark"] = s.mark_price
    return {
        "latest_scan": dict(latest_scan) if latest_scan else {},
        "scan_count": total_scan_count,
        "connected_clients": len(connected_clients),
        "buffer": buffer_info,
    }


@app.get("/api/history")
async def get_history():
    return {"history": list(scan_history)}


@app.get("/api/prices")
async def get_prices():
    return {"prices": list(price_history)}


# ── Serve the frontend ────────────────────────────────────────
DASHBOARD_DIR = Path(__file__).parent / "static"


@app.get("/")
async def serve_dashboard():
    return FileResponse(DASHBOARD_DIR / "index.html")


@app.get("/style.css")
async def serve_css():
    return FileResponse(DASHBOARD_DIR / "style.css", media_type="text/css")


@app.get("/app.js")
async def serve_js():
    return FileResponse(DASHBOARD_DIR / "app.js", media_type="application/javascript")
