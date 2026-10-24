import sqlite3
import pandas as pd
con = sqlite3.connect('data/football.db')
query = """
SELECT kickoff_at, home_team, away_team, home_goals, away_goals 
FROM matches 
WHERE home_goals != 0 OR away_goals != 0 
ORDER BY kickoff_at DESC 
LIMIT 10
"""
df = pd.read_sql(query, con)
print(df)
