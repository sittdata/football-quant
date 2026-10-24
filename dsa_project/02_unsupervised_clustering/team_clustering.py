"""Unsupervised Learning & Team Clustering (Week 14).

Implements:
1. Team Profiling (Aggregated stats: avg goals scored, conceded, match totals, cover rate)
2. K-Means Clustering (สไลด์หน้า 85)
3. DBSCAN Density-Based Clustering (สไลด์หน้า 87-88)
4. Gaussian Mixture Models - GMM (สไลด์หน้า 89)
5. PCA 2D Dimensionality Reduction & Visualization (สไลด์หน้า 90-91)
"""

from __future__ import annotations

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

SYS_PATH_PREP = Path(__file__).resolve().parents[1] / "01_data_prep_feature_selection"
if str(SYS_PATH_PREP) not in sys.path:
    sys.path.insert(0, str(SYS_PATH_PREP))

from prepare_data import load_and_clean_data

REPORTS_FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
REPORTS_FIG_DIR.mkdir(parents=True, exist_ok=True)


def build_team_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-team performance metrics to build clustering profiles."""
    home_stats = df.groupby("home_team").agg(
        matches=("match_id", "count"),
        gf=("home_goals", "mean"),
        ga=("away_goals", "mean"),
        total_goals=("target_total_goals", "mean"),
        btts_rate=("target_btts", "mean"),
        fav_rate=("fav_side", lambda s: (s == "home").mean()),
        ah_cover_rate=("target_ah_cover", "mean")
    )

    away_stats = df.groupby("away_team").agg(
        matches=("match_id", "count"),
        gf=("away_goals", "mean"),
        ga=("home_goals", "mean"),
        total_goals=("target_total_goals", "mean"),
        btts_rate=("target_btts", "mean"),
        fav_rate=("fav_side", lambda s: (s == "away").mean()),
        ah_cover_rate=("target_ah_cover", "mean")
    )

    # Combine home & away stats
    all_teams = sorted(list(set(home_stats.index).union(set(away_stats.index))))
    profiles = []
    for team in all_teams:
        h = home_stats.loc[team] if team in home_stats.index else None
        a = away_stats.loc[team] if team in away_stats.index else None
        
        total_matches = (h["matches"] if h is not None else 0) + (a["matches"] if a is not None else 0)
        if total_matches < 5:  # filter teams with too few matches
            continue
            
        gf_avg = ((h["gf"] * h["matches"] if h is not None else 0) + (a["gf"] * a["matches"] if a is not None else 0)) / total_matches
        ga_avg = ((h["ga"] * h["matches"] if h is not None else 0) + (a["ga"] * a["matches"] if a is not None else 0)) / total_matches
        tot_avg = ((h["total_goals"] * h["matches"] if h is not None else 0) + (a["total_goals"] * a["matches"] if a is not None else 0)) / total_matches
        btts_avg = ((h["btts_rate"] * h["matches"] if h is not None else 0) + (a["btts_rate"] * a["matches"] if a is not None else 0)) / total_matches
        fav_avg = ((h["fav_rate"] * h["matches"] if h is not None else 0) + (a["fav_rate"] * a["matches"] if a is not None else 0)) / total_matches
        ah_avg = ((h["ah_cover_rate"] * h["matches"] if h is not None else 0) + (a["ah_cover_rate"] * a["matches"] if a is not None else 0)) / total_matches

        profiles.append({
            "team": team,
            "matches": total_matches,
            "avg_goals_for": gf_avg,
            "avg_goals_against": ga_avg,
            "goal_diff_avg": gf_avg - ga_avg,
            "avg_total_goals": tot_avg,
            "btts_rate": btts_avg,
            "favourite_frequency": fav_avg,
            "ah_cover_rate": ah_avg
        })

    profile_df = pd.DataFrame(profiles).set_index("team")
    print(f"[Week 14] Built Team Profiles for {len(profile_df)} teams (minimum 5 matches).")
    return profile_df


def run_team_clustering(profile_df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Run K-Means, DBSCAN, and GMM, followed by PCA 2D visualization (สไลด์หน้า 85-91)."""
    features = [
        "avg_goals_for", "avg_goals_against", "goal_diff_avg",
        "avg_total_goals", "btts_rate", "favourite_frequency", "ah_cover_rate"
    ]
    X = profile_df[features].values
    X_scaled = StandardScaler().fit_transform(X)

    # 1. K-Means (สไลด์หน้า 85)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    profile_df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

    # 2. DBSCAN (สไลด์หน้า 87-88)
    dbscan = DBSCAN(eps=1.5, min_samples=3)
    profile_df["dbscan_cluster"] = dbscan.fit_predict(X_scaled)

    # 3. Gaussian Mixture Models - GMM (สไลด์หน้า 89)
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    profile_df["gmm_cluster"] = gmm.fit_predict(X_scaled)

    # 4. PCA Dimensionality Reduction to 2D (สไลด์หน้า 90-91)
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    profile_df["pca_1"] = pca_coords[:, 0]
    profile_df["pca_2"] = pca_coords[:, 1]
    
    evr = pca.explained_variance_ratio_
    print(f"  - PCA Explained Variance: PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}% (Total={sum(evr)*100:.1f}%)")

    # Interpret Clusters
    cluster_means = profile_df.groupby("kmeans_cluster")[features].mean()
    print("\n[K-Means Cluster Centroids]:")
    print(cluster_means.round(2).to_string())

    # Plot PCA Clusters (สไลด์หน้า 91-93)
    plt.figure(figsize=(11, 7))
    scatter = plt.scatter(
        profile_df["pca_1"],
        profile_df["pca_2"],
        c=profile_df["kmeans_cluster"],
        cmap="tab10",
        s=80,
        alpha=0.85,
        edgecolors="k"
    )

    # Annotate some notable teams
    notable = profile_df.index[:15]
    for team in notable:
        row = profile_df.loc[team]
        plt.annotate(
            team,
            (row["pca_1"], row["pca_2"]),
            fontsize=8,
            alpha=0.8,
            xytext=(4, 4),
            textcoords="offset points"
        )

    plt.title(f"Football Team Style Clustering via K-Means (k={n_clusters}) + PCA 2D Projection")
    plt.xlabel(f"Principal Component 1 ({evr[0]*100:.1f}% Var)")
    plt.ylabel(f"Principal Component 2 ({evr[1]*100:.1f}% Var)")
    plt.colorbar(scatter, label="K-Means Cluster")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = REPORTS_FIG_DIR / "team_pca_clusters.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"  - Saved PCA Clustering plot to: {plot_path}")

    return profile_df


if __name__ == "__main__":
    df = load_and_clean_data()
    profiles = build_team_profiles(df)
    run_team_clustering(profiles, n_clusters=4)
