import requests, re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HEADERS = {'User-Agent': 'Mozilla/5.0'}
resp = requests.get('https://www.thscore.vip/gf/data/bf_th_nt.js', headers=HEADERS)
matches = re.findall(r'A\[\d+\]\s*=\s*\[([^\]]+)\];', resp.text)
print(f"Total raw matches: {len(matches)}")
for m in matches[:10]:
    items = [i.strip('\'" ') for i in m.split(',')]
    if len(items) > 10:
        print(f"ID: {items[0]} | Teams: {items[4]} vs {items[5]} | items[6]: '{items[6]}' | items[7]: '{items[7]}' | Status: {items[8]}")
