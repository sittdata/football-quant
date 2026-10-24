import sqlite3
import sys
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fbpredict import labels
from fbpredict.model import ModelBundle
from fbpredict.db import connect

con = connect(ROOT_DIR / "data" / "football.db")
query = """
SELECT SUBSTR(kickoff_at, 1, 10) as dt, count(*) as cnt, sum(CASE WHEN home_goals IS NOT NULL THEN 1 ELSE 0 END) as finished
FROM matches
GROUP BY dt
ORDER BY dt DESC
LIMIT 20
"""
df_dates = pd.read_sql_query(query, con)
print("Recent dates in database:")
print(df_dates)
