import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fbpredict.config import Config
from fbpredict.providers.five_dollar import FiveDollarFootballProvider

cfg = Config.from_env()
p = FiveDollarFootballProvider(cfg)

print("--- Checking 5DollarFootballAPI Leagues ---")
res = p.list_leagues(popular=False, per_page=100)
leagues = res.get("data", [])
print(f"Total leagues returned: {len(leagues)}")

for l in leagues:
    lid = l.get("id")
    name = l.get("name")
    country = l.get("country")
    print(f"{lid:>12} | {name} ({country})")

# Let's test if we can fetch fixtures for Champions League or other leagues on this plan
target_test_leagues = [
    ("UEFA Champions League", "UEFA"),
    ("Netherlands Eredivisie", "Netherlands"),
    ("Germany 2. Bundesliga", "Germany"),
    ("Portugal Primeira Liga", "Portugal"),
]

print("\n--- Testing Fixture Fetch Permissions for Extra Leagues ---")
for t_name, t_country in target_test_leagues:
    matched = [l for l in leagues if t_name.lower() in l.get("name", "").lower() or t_country.lower() in str(l.get("country", "")).lower()]
    if not matched:
        print(f"⚠️ League not found in list: {t_name}")
        continue
    target = matched[0]
    lid = target["id"]
    print(f"Testing {target['name']} (id={lid})...")
    try:
        fixtures = list(p.iter_league_fixtures(lid, max_pages=1))
        print(f"  ✅ SUCCESS: Fetched {len(fixtures)} fixtures for {target['name']}!")
    except Exception as e:
        print(f"  ❌ FAILED for {target['name']}: {e}")
