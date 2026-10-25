"""Local WebSocket Bridge Server for KUBET Auto-Bet Chrome Extension.
Author: cook45
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
import websockets

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [Bridge] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CONNECTED_CLIENTS = set()
AUTO_BET_STATE = {
    "enabled": True,
    "dry_run": True,
    "stake_amount": 50,
}


async def handler(websocket):
    """Handle incoming WebSocket connections from Chrome Extension."""
    CONNECTED_CLIENTS.add(websocket)
    logging.info(f"🟢 Chrome Extension connected! Total clients: {len(CONNECTED_CLIENTS)}")

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                mtype = data.get("type")

                if mtype == "CLIENT_READY":
                    logging.info(f"Client Ready on URL: {data.get('url')}")
                    # Send initial config to extension
                    await websocket.send(json.dumps({
                        "type": "CONFIG_UPDATE",
                        "auto_bet": AUTO_BET_STATE["enabled"],
                        "dry_run": AUTO_BET_STATE["dry_run"],
                        "amount": AUTO_BET_STATE["stake_amount"],
                    }))

                elif mtype == "BET_CONFIRMATION":
                    status = data.get("status")
                    match = data.get("match")
                    pick = data.get("pick")
                    amount = data.get("amount")
                    receipt = data.get("receipt_id")

                    log_msg = f"✅ บิลเดิมพันสำเร็จ [{status}]: {match} | {pick} | {amount} บ. (สลิป: {receipt})"
                    logging.info(log_msg)

                    # Telegram removed per user request

                elif mtype == "DOM_DUMP":
                    html_content = data.get("html", "")
                    out_path = ROOT_DIR / "data" / "kubet_dom_dump.html"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(html_content)
                    logging.info(f"💾 Saved full KUBET DOM dump ({len(html_content)} bytes) to {out_path}")

                elif mtype == "SCREENSHOT_DATA":
                    import base64
                    raw_b64 = data.get("data", "")
                    if "," in raw_b64:
                        raw_b64 = raw_b64.split(",", 1)[1]
                    img_bytes = base64.b64decode(raw_b64)
                    screen_path = ROOT_DIR / "data" / "kubet_live_screen.png"
                    screen_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(screen_path, "wb") as f:
                        f.write(img_bytes)
                    logging.info(f"📸 Live KUBET screenshot captured and saved ({len(img_bytes)} bytes) to {screen_path}")

            except Exception as e:
                logging.error(f"Error processing client message: {e}")

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        CONNECTED_CLIENTS.remove(websocket)
        logging.info(f"🔴 Chrome Extension disconnected. Remaining clients: {len(CONNECTED_CLIENTS)}")


async def broadcast_bet_signal(signal: dict):
    """Broadcast a bet signal to all connected Chrome Extension tabs."""
    if not CONNECTED_CLIENTS:
        logging.warning("⚠️ No Chrome Extension clients connected to receive bet signal.")
        return False

    payload = json.dumps({
        "type": "BET_SIGNAL",
        **signal
    })

    disconnected = set()
    for ws in list(CONNECTED_CLIENTS):
        try:
            await ws.send(payload)
        except Exception:
            disconnected.add(ws)

    for ws in disconnected:
        CONNECTED_CLIENTS.discard(ws)

    logging.info(f"📡 Broadcasted bet signal to {len(CONNECTED_CLIENTS)} clients: {signal.get('match')} | {signal.get('pick')}")
    return True


async def main():
    server = await websockets.serve(handler, "127.0.0.1", 8765)
    logging.info("⚡ Cook45 KUBET Quant Bridge WebSocket running on ws://127.0.0.1:8765")
    await asyncio.Future()  # run forever


if __name__ == "__main__":
    if "--test" in sys.argv:
        # Quick test broadcaster
        async def send_test():
            uri = "ws://127.0.0.1:8765"
            async with websockets.connect(uri) as ws:
                await ws.send(json.dumps({
                    "type": "BET_SIGNAL",
                    "match": "เซลติก vs ฮิเบอร์เนี่ยน",
                    "pick": "ต่อ เซลติก (-1.00)",
                    "team_target": "เซลติก",
                    "edge": 13.5,
                    "amount": 50,
                    "dry_run": True
                }))
                print("[Test] Signal sent to bridge!")
        asyncio.run(send_test())
    else:
        asyncio.run(main())
