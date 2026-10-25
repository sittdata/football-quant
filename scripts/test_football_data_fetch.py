import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

leagues = [
    ("D2", "Germany 2. Bundesliga"),
    ("N1", "Netherlands Eredivisie"),
    ("P1", "Portugal Primeira Liga"),
    ("B1", "Belgium Pro League"),
    ("SP2", "Spain Segunda"),
    ("I2", "Italy Serie B"),
]

print("Testing direct downloads from football-data.co.uk:")
for code, name in leagues:
    url = f"https://www.football-data.co.uk/mmz4281/2324/{code}.csv"
    try:
        df = pd.read_csv(url)
        has_ah = "AHh" in df.columns
        has_ou = "B365>2.5" in df.columns
        print(f"  ✅ {name:<26} ({code}): {len(df)} matches | Has AH: {has_ah} | Has OU: {has_ou}")
    except Exception as e:
        print(f"  ❌ {name:<26} ({code}): Failed - {e}")
