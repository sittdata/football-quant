"""Cook45 KUBET Direct Remote Operator (CDP & WebSocket Bridge)
Allows the AI assistant to capture live screenshots and trigger real clicks.
"""

import asyncio
import json
import base64
import sys
from pathlib import Path
import urllib.request
import websockets

ROOT_DIR = Path(__file__).resolve().parent.parent

async def capture_screen_via_bridge():
    """Request live screenshot from Chrome Extension via WebSocket bridge."""
    uri = "ws://127.0.0.1:8765"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"type": "CAPTURE_SCREEN"}))
        print("[Operator] CAPTURE_SCREEN signal sent to Bridge...")
        
        # Wait for screenshot response or check data/kubet_live_screen.png
        for _ in range(10):
            await asyncio.sleep(0.5)
            img_path = ROOT_DIR / "data" / "kubet_live_screen.png"
            if img_path.exists():
                print(f"[Operator] ✅ Screenshot captured: {img_path} ({img_path.stat().st_size} bytes)")
                return str(img_path)
    return None

async def send_bet_signal_via_bridge(match: str, pick: str, team: str, amount: int = 10, dry_run: bool = True):
    """Send direct bet signal to the sports tab via WebSocket bridge."""
    uri = "ws://127.0.0.1:8765"
    payload = {
        "type": "BET_SIGNAL",
        "match": match,
        "pick": pick,
        "team_target": team,
        "amount": amount,
        "dry_run": dry_run,
        "edge": 15.0
    }
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps(payload))
        print(f"[Operator] 🚀 BET_SIGNAL broadcasted: {match} | {pick} | {amount} THB (DryRun={dry_run})")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if cmd == "screen":
        asyncio.run(capture_screen_via_bridge())
    elif cmd == "bet":
        match = sys.argv[2] if len(sys.argv) > 2 else "ฮิเบอร์เนี่ยน vs เซลติก"
        pick = sys.argv[3] if len(sys.argv) > 3 else "ต่อ เซลติก (-1.00)"
        team = sys.argv[4] if len(sys.argv) > 4 else "เซลติก"
        amount = int(sys.argv[5]) if len(sys.argv) > 5 else 10
        dry_run = "--real" not in sys.argv
        asyncio.run(send_bet_signal_via_bridge(match, pick, team, amount, dry_run))
