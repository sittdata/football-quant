import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fbpredict.quant_model import train_calibrated_models

df = pd.read_pickle(ROOT_DIR / "data" / "clean_quant_dataset.pkl")
ch = df[df["league"].str.contains("Championship", case=False, na=False)].reset_index(drop=True)
split_idx = int(len(ch) * 0.8)
train = ch.iloc[:split_idx].reset_index(drop=True)
test = ch.iloc[split_idx:].reset_index(drop=True)

print(f"Championship Total: {len(ch)} matches | Train: {len(train)} | Test: {len(test)}")

bundle, meta = train_calibrated_models(train, seed=42)

rows = []
for _, r in test.iterrows():
    p = bundle.predict_match(r.to_dict())
    if not p.get("safe"):
        continue
    
    rows.append({
        "kickoff": r["kickoff_at"][:10],
        "home": r["home_team"],
        "away": r["away_team"],
        "fav_side": r["fav_side"],
        "ah_open": r["ah_open"],
        "fair_fav": r["fair_fav_ah"],
        "fair_dog": r["fair_dog_ah"],
        "p_fav": p["p_fav"],
        "p_dog": p["p_dog"],
        "p_over": p["p_over"],
        "p_under": p["p_under"],
        "edge_dog": p["edge_dog"],
        "edge_fav": p["edge_fav"],
        "edge_over": p["edge_over"],
        "edge_under": p["edge_under"],
        "best_market": p["best_market"],
        "best_edge": p["best_edge"],
        "score": f"{r['home_goals']}-{r['away_goals']}",
        "ah_unit": r["ah_unit"],
        "ou_unit": r["ou_unit"],
        "h_gf": r["home_gf_avg"],
        "h_ga": r["home_ga_avg"],
        "a_gf": r["away_gf_avg"],
        "a_ga": r["away_ga_avg"],
        "draw_odds": r["draw_ft_odds"],
    })

res_df = pd.DataFrame(rows)

print("\n--- Summary of Best Markets Chosen in Test Set ---")
print(res_df["best_market"].value_counts())

print("\n--- AH_dog Predictions Breakdown (Top 15 Highest Edge) ---")
ah_dog_df = res_df[res_df["best_market"] == "AH_dog"].sort_values(by="best_edge", ascending=False)
for _, r in ah_dog_df.head(15).iterrows():
    won = "WON (Dog Covered)" if r["ah_unit"] < 0 else ("DRAW" if r["ah_unit"] == 0 else "LOST (Fav Covered)")
    print(f"[{r['kickoff']}] {r['home']} vs {r['away']} | Fav: {r['fav_side']} (Line: {r['ah_open']}) | Model p(Dog)={r['p_dog']:.3f} (Edge={r['edge_dog']:+.3f}) | Score: {r['score']} -> {won}")

print("\n--- Feature Importance / Model Coefficients Check ---")
clf_ah = bundle.ah_model
# Get underlying base estimator
base = clf_ah.calibrated_classifiers_[0].estimator
from sklearn.inspection import permutation_importance
X_train = train[meta["feature_columns"]]
y_train = train["ah_label"].dropna().astype(int)
X_train_clean = train.loc[y_train.index, meta["feature_columns"]]
perm = permutation_importance(base, X_train_clean, y_train, n_repeats=5, random_state=42)
sorted_importances_idx = perm.importances_mean.argsort()[::-1]
for idx in sorted_importances_idx[:10]:
    col = meta["feature_columns"][idx]
    print(f"{col:<20}: {perm.importances_mean[idx]:.4f} +/- {perm.importances_std[idx]:.4f}")
