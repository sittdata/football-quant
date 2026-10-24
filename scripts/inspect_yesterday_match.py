import requests
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

API_KEY = "B95794317BA548F2"
yesterday = "2026-10-23"
url = f"http://api2.thscore.info/football_th/schedule.aspx?date={yesterday}&api_key={API_KEY}"

resp = requests.get(url, timeout=10)
data = resp.json().get("data", [])
print(f"Total matches on {yesterday}: {len(data)}")

if data:
    # Print sample match fields
    sample = data[0]
    print("Keys in match dict:", list(sample.keys()))
    for k, v in sample.items():
        print(f"  {k}: {v}")
