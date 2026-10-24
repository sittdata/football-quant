import sqlite3
import sys
from pathlib import Path
import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]

# 1. Check sqlite
con = sqlite3.connect(ROOT_DIR / "data" / "football.db")
print("Tables:", pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)['name'].tolist())

df_recent = pd.read_sql("SELECT * FROM matches ORDER BY kickoff_at DESC LIMIT 10", con)
print("\nRecent 10 matches in DB:")
for _, r in df_recent.iterrows():
    print(r.get("kickoff_at"), r.get("league"), r.get("home_team"), "vs", r.get("away_team"), r.get("home_goals"), "-", r.get("away_goals"))

# 2. Check Thscore result or schedule API for yesterday
API_KEY = "B95794317BA548F2"
yesterday = "2026-10-23"

endpoints = [
    f"http://api2.thscore.info/football_th/result.aspx?date={yesterday}&api_key={API_KEY}",
    f"http://api2.thscore.info/football_th/schedule.aspx?date={yesterday}&api_key={API_KEY}",
]

for url in endpoints:
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            print(f"\nEndpoint: {url.split('?')[0]}")
            print(f"Total matches for {yesterday}: {len(data)}")
            if data:
                print("Sample match:", data[0].get("homeName"), "vs", data[0].get("awayName"), "Score:", data[0].get("homeScore"), "-", data[0].get("awayScore"))
    except Exception as e:
        print(f"Error {url}: {e}")
