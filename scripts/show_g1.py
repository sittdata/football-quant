import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from scripts.print_tonight_quant import df_t
g1 = df_t[df_t['hour'].between(18, 20)]
print(f"Total Group 1 matches: {len(g1)}")
for i, r in g1.reset_index(drop=True).iterrows():
    print(f"{i+1:2d}. [{r['time']}] [{r['league']}] {r['home']} vs {r['away']} | {r['pick']} | Win {r['prob']*100:.1f}% | Edge +{r['edge']*100:.1f}% | น้ำ {r['odds']:.2f}")
