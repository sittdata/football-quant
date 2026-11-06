from http.server import BaseHTTPRequestHandler
import json
import math
import os
from pathlib import Path

# Load model data once at module startup
MODEL_PATH = Path(__file__).parent.parent / "public" / "model_data.json"
model_data = None

if MODEL_PATH.exists():
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        model_data = json.load(f)

def eval_node(node, feats):
    if "l" in node:
        return node["l"]
    val = feats[node["s"]]
    target = node["y"] if val < node["c"] else node["n"]
    if "ch" in node and str(target) in node["ch"]:
        return eval_node(node["ch"][str(target)], feats)
    return 0.0

def predict_probabilities(raw_features):
    if not model_data:
        return [0.333, 0.333, 0.334]

    min_v = model_data["scaler"]["data_min"]
    max_v = model_data["scaler"]["data_max"]

    scaled = []
    for i, val in enumerate(raw_features):
        denom = max_v[i] - min_v[i]
        scaled.append(0.0 if denom == 0 else (val - min_v[i]) / denom)

    margins = list(model_data["base_scores"])
    for idx, tree in enumerate(model_data["trees"]):
        cls = idx % 3
        margins[cls] += eval_node(tree, scaled)

    max_m = max(margins)
    exp_m = [math.exp(m - max_m) for m in margins]
    sum_exp = sum(exp_m)
    return [e / sum_exp for e in exp_m]

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")
        try:
            req = json.loads(body) if body else {}
        except Exception:
            req = {}

        home_xg = float(req.get("home_xg", 1.5))
        away_xg = float(req.get("away_xg", 1.2))
        home_xga = float(req.get("home_xga", 1.2))
        away_xga = float(req.get("away_xga", 1.5))
        handicap = float(req.get("handicap", 0.0))
        odds_fav = float(req.get("odds_fav", 1.90))

        probs = predict_probabilities([home_xg, away_xg, home_xga, away_xga, handicap, odds_fav])

        p_away = round(probs[0] * 100.0, 1)
        p_draw = round(probs[1] * 100.0, 1)
        p_home = round(probs[2] * 100.0, 1)

        winner = "HomeWin" if p_home >= p_away and p_home >= p_draw else ("AwayWin" if p_away >= p_home and p_away >= p_draw else "Draw")

        res_data = {
            "away_win_pct": p_away,
            "draw_pct": p_draw,
            "home_win_pct": p_home,
            "predicted_winner": winner,
            "input_features": {
                "home_xg": home_xg,
                "away_xg": away_xg,
                "home_xga": home_xga,
                "away_xga": away_xga,
                "handicap": handicap,
                "odds_fav": odds_fav
            }
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(res_data).encode("utf-8"))

    def do_GET(self):
        # Default test GET response
        probs = predict_probabilities([1.75, 1.17, 1.37, 1.77, -0.5, 1.95])
        res_data = {
            "status": "online",
            "model": "XGBoost Multi-Class Outcome Classifier",
            "features": ["Home_xG", "Away_xG", "Home_xGA", "Away_xGA", "Handicap", "Odds_Fav"],
            "sample_prediction": {
                "away_win_pct": round(probs[0] * 100.0, 1),
                "draw_pct": round(probs[1] * 100.0, 1),
                "home_win_pct": round(probs[2] * 100.0, 1)
            }
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(res_data).encode("utf-8"))
