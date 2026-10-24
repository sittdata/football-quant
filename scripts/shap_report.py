"""SHAP explanation plots for the frozen fbpredict model.

The pipeline trains a ``MultiOutputClassifier`` with two heads:

* head 0 -> ``p_fav_cover`` (the favourite covers the Asian handicap)
* head 1 -> ``p_over``      (the total goes over the line)

Both heads are linear, so SHAP values are exact and cheap: each head is a
``Pipeline(impute -> scale -> LogisticRegression)`` and the explanation is
computed on the *transformed* matrix that the classifier actually sees. The
plots therefore answer "which features pushed this prediction, and how",
rather than only "which feature is important on average".

Usage
-----
    py shap_report.py                      # explain the frozen model on its own training window
    py shap_report.py --split test         # explain the held-out window instead
    py shap_report.py --split all
    py shap_report.py --csv data/foo.csv   # read matches from a CSV instead of the database
    py shap_report.py --retrain            # ignore the saved bundle and refit now

Outputs (written to ``reports/``):
    shap_summary_AH.png   beeswarm for p_fav_cover
    shap_summary_OU.png   beeswarm for p_over
    shap_bar_AH.png       mean |SHAP| ranking for p_fav_cover
    shap_bar_OU.png       mean |SHAP| ranking for p_over
    shap_grid_AH.png      2x2 grid: bar + beeswarm for p_fav_cover
    shap_grid_OU.png      2x2 grid: bar + beeswarm for p_over
    shap_top_features.txt ranked mean |SHAP| per head (the numbers behind the bars)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this box has no display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from fbpredict import db, features, model
from fbpredict.config import load_config

#: default artefacts, matching the handoff document's chosen config
DEFAULT_FEATURE_SET = "filtered_novig"
DEFAULT_MODEL = Path("models") / "model_filtered_novig.pkl"

#: (head index, probability column, human label, output stem)
HEADS = (
    (0, "p_fav_cover", "AH - favourite covers the handicap", "AH"),
    (1, "p_over", "O/U - total goes over the line", "OU"),
)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_frame(args, cfg) -> pd.DataFrame:
    """Real matches with results (and prices where the CSV carries them)."""
    if args.csv:
        frame = pd.read_csv(args.csv)
        if "kickoff_at" in frame.columns:
            frame["kickoff_at"] = pd.to_datetime(
                frame["kickoff_at"], utc=True, format="ISO8601"
            )
        return frame

    con = db.init_db(cfg.db_path)
    try:
        frame = db.fetch_matches_df(
            con,
            include_synthetic=False,
            require_result=True,
        )
    finally:
        con.close()
    if frame.empty:
        raise SystemExit(
            "no real matches with results in the database yet - run a fetch first"
        )
    return frame


def restrict_to_priced(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only matches priced on every side a bet can be settled against."""
    needed = ("ah_open", "ah_fav_price", "ah_dog_price", "ou_over_price", "ou_under_price")
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        print(f"  ! cannot filter to priced matches: missing {missing}")
        return frame
    keep = pd.Series(True, index=frame.index)
    for column in needed:
        keep &= pd.to_numeric(frame[column], errors="coerce").notna()
    kept = frame[keep].reset_index(drop=True)
    print(f"  priced matches only     {len(kept)} of {len(frame)}")
    return kept


def window_slice(data: pd.DataFrame, split: str, bundle: model.ModelBundle) -> pd.DataFrame:
    """Pick the rows to explain: the model's own training window, or the rest."""
    if split == "all":
        return data

    trained_to = bundle.metadata.get("trained_to")
    if not trained_to:
        print(f"  ! bundle records no trained_to; falling back to --split all")
        return data

    cut = pd.Timestamp(trained_to)
    if cut.tzinfo is None:
        cut = cut.tz_localize("UTC")
    kickoff = pd.to_datetime(data["kickoff_at"], utc=True)

    if split == "train":
        return data[kickoff <= cut]
    return data[kickoff > cut]


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def transform_inputs(pipe, x: np.ndarray) -> tuple[np.ndarray, object]:
    """Apply every pipeline step except the final classifier.

    The classifier was fitted on the *transformed* matrix, so the explanation
    must be computed there too; explaining the raw matrix would silently ignore
    the imputation and the scaling.
    """
    steps = list(pipe.named_steps.items())
    transformed = x
    for _name, step in steps[:-1]:
        if hasattr(step, "transform"):
            transformed = step.transform(transformed)
    return np.asarray(transformed, dtype="float64"), steps[-1][1]


def explain_head(pipe, x: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, float]:
    """Exact SHAP values (log-odds space) and base value for one linear head."""
    transformed, clf = transform_inputs(pipe, x)
    # max_samples = every explained row, so the background mean is exact rather
    # than shap's default 100-row subsample.
    masker = shap.maskers.Independent(transformed, max_samples=transformed.shape[0])
    explainer = shap.LinearExplainer(clf, masker, feature_names=feature_names)
    values = explainer.shap_values(transformed)

    # Binary logistic heads return one matrix; be tolerant of the list form.
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:  # (n, features, classes)
        values = values[:, :, -1]

    base = float(np.asarray(explainer.expected_value).ravel()[-1])
    return values, base


def save_beeswarm(values: np.ndarray, x: pd.DataFrame, title: str, path: Path) -> None:
    plt.figure(figsize=(10, max(6.0, 0.30 * len(x.columns))))
    shap.summary_plot(values, x, feature_names=list(x.columns), show=False)
    axes = plt.gca()
    axes.set_title(title, fontsize=12)
    axes.set_xlabel(f"SHAP value (log-odds impact on {title.split(' - ')[0]})", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  wrote {path}")


def save_bar(values: np.ndarray, x: pd.DataFrame, title: str, path: Path) -> None:
    plt.figure(figsize=(10, max(6.0, 0.30 * len(x.columns))))
    shap.summary_plot(values, x, feature_names=list(x.columns), plot_type="bar", show=False)
    axes = plt.gca()
    axes.set_title(title, fontsize=12)
    axes.set_xlabel("mean |SHAP| (log-odds impact)", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  wrote {path}")


def save_grid(values: np.ndarray, x: pd.DataFrame, title: str, path: Path) -> None:
    """One sheet: importance ranking on the left, per-row detail on the right."""
    n_features = min(20, len(x.columns))
    fig, (ax_bar, ax_bee) = plt.subplots(
        1, 2, figsize=(21, 0.24 * n_features + 5), gridspec_kw={"wspace": 1.25}
    )

    plt.sca(ax_bar)
    shap.summary_plot(
        values,
        x,
        feature_names=list(x.columns),
        plot_type="bar",
        max_display=n_features,
        show=False,
    )
    ax_bar.set_title(f"{title}\nmean |SHAP| (top {n_features})", fontsize=12)
    ax_bar.set_xlabel("mean |SHAP| (log-odds impact)", fontsize=10)

    plt.sca(ax_bee)
    shap.summary_plot(
        values,
        x,
        feature_names=list(x.columns),
        max_display=n_features,
        show=False,
    )
    ax_bee.set_title(f"{title}\nper-match contribution (top {n_features})", fontsize=12)
    ax_bee.set_xlabel(f"SHAP value (log-odds impact on {title.split(' - ')[0]})", fontsize=10)

    fig.suptitle(
        "SHAP explanation of the frozen model (in-sample: rows it was fitted on)",
        fontsize=13,
    )
    fig.subplots_adjust(bottom=0.10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  wrote {path}")


def ranking(values: np.ndarray, x_frame: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| plus the contribution at low vs high feature values.

    A plain "mean signed SHAP" would be zero by construction (the background is
    the explained rows themselves), so the direction is read from the bottom and
    top of each feature's distribution instead.
    """
    rows = []
    for column_index, name in enumerate(x_frame.columns):
        feature = pd.to_numeric(x_frame[name], errors="coerce").to_numpy(dtype="float64")
        contributions = values[:, column_index]
        mean_abs = float(np.abs(contributions).mean())

        finite = np.isfinite(feature)
        low_shap = high_shap = float("nan")
        if finite.any():
            unique = np.unique(feature[finite])
            if unique.size <= 2:
                if unique.size == 2:
                    low_shap = float(contributions[finite & (feature == unique[0])].mean())
                    high_shap = float(contributions[finite & (feature == unique[1])].mean())
                else:
                    low_shap = high_shap = float(contributions[finite].mean())
            else:
                low_cut, high_cut = np.nanpercentile(feature[finite], [25, 75])
                low_mask = finite & (feature <= low_cut)
                high_mask = finite & (feature >= high_cut)
                low_shap = float(contributions[low_mask].mean()) if low_mask.any() else float("nan")
                high_shap = float(contributions[high_mask].mean()) if high_mask.any() else float("nan")

        rows.append(
            {
                "feature": name,
                "mean_abs_shap": mean_abs,
                "shap_when_low": low_shap,
                "shap_when_high": high_shap,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=None, help="read matches from this CSV instead of the DB")
    parser.add_argument(
        "--feature-set",
        default=DEFAULT_FEATURE_SET,
        choices=sorted(features.FEATURE_SETS),
        help=f"feature set to build (default: {DEFAULT_FEATURE_SET})",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help=f"saved ModelBundle to explain (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="ignore the saved bundle and fit a fresh model on the loaded rows",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test", "all"],
        help="rows to explain: the bundle's training window (default), the rest, or all",
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="output directory (default: cfg.reports_dir)",
    )
    parser.add_argument("--top", type=int, default=15, help="rows printed in the ranking")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config()
    reports_dir = Path(args.reports_dir) if args.reports_dir else cfg.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading matches ({args.csv or cfg.db_path})")
    frame = load_frame(args, cfg)
    frame = restrict_to_priced(frame)

    print(f"building features: {args.feature_set}")
    data = features.build_features(frame, feature_set=args.feature_set)

    ah_label = pd.to_numeric(data["ah_label"], errors="coerce")
    ou_label = pd.to_numeric(data["ou_label"], errors="coerce")
    usable = ah_label.notna() & ou_label.notna()
    data = data[usable].reset_index(drop=True)
    print(f"  rows with both labels   {len(data)}")

    if args.retrain:
        print("training a fresh model (--retrain)")
        bundle, report = model.train_model(data, feature_set=args.feature_set)
        print(f"  trained on              {report.n_rows_used} rows")
    else:
        bundle = model.ModelBundle.load(args.model)
        _, meta = model.ModelBundle.verify(args.model)
        print(f"loaded {args.model}")
        print(f"  hash matches sidecar    {bool(meta.get('sha256'))}")
        print(f"  kind / feature set      {bundle.kind} / {bundle.feature_set}")

    columns = list(bundle.feature_columns)
    sliced = window_slice(data, args.split, bundle).reset_index(drop=True)
    print(f"  explaining split        {args.split} ({len(sliced)} rows)")
    if sliced.empty:
        raise SystemExit(
            f"the {args.split!r} split has no rows; try --split all or --split train"
        )

    x = sliced[columns].to_numpy(dtype="float64")
    x_df = pd.DataFrame(x, columns=columns)

    ranked_lines: list[str] = [
        "SHAP feature ranking for the frozen fbpredict model",
        f"model        : {args.model} (feature set {bundle.feature_set}, kind {bundle.kind})",
        f"rows         : {len(sliced)} ({args.split} split)",
        "note         : in-sample explanation of the fitted model, not a performance claim",
        "",
    ]

    for head_index, proba_name, label, stem in HEADS:
        pipe = bundle.model.estimators_[head_index]
        print(f"\nhead {head_index} -> {proba_name} ({label})")
        values, base = explain_head(pipe, x, columns)
        base_probability = 1.0 / (1.0 + np.exp(-base))

        save_bar(values, x_df, f"{label} - feature importance", reports_dir / f"shap_bar_{stem}.png")
        save_beeswarm(values, x_df, f"{label} - SHAP summary", reports_dir / f"shap_summary_{stem}.png")
        save_grid(values, x_df, label, reports_dir / f"shap_grid_{stem}.png")

        table = ranking(values, x_df)
        ranked_lines.append(
            f"=== {proba_name} :: {label} ==="
        )
        ranked_lines.append(
            f"  base value: {base:+.4f} log-odds  (~{base_probability:.1%} average probability)"
        )
        for position, row in table.head(args.top).iterrows():
            ranked_lines.append(
                f"  {position + 1:>2}. {row['feature']:<28} "
                f"mean|SHAP|={row['mean_abs_shap']:>6.4f}  "
                f"low={row['shap_when_low']:+.4f}  high={row['shap_when_high']:+.4f}"
            )
        ranked_lines.append("")

        print(f"  base value              {base:+.4f} log-odds (~{base_probability:.1%})")
        print(f"  top features by mean |SHAP|:")
        for position, row in table.head(5).iterrows():
            print(
                f"    {position + 1:>2}. {row['feature']:<28} "
                f"mean|SHAP|={row['mean_abs_shap']:.4f}  "
                f"(low={row['shap_when_low']:+.3f}, high={row['shap_when_high']:+.3f})"
            )

    ranking_path = reports_dir / "shap_top_features.txt"
    ranking_path.write_text("\n".join(ranked_lines), encoding="utf-8")
    print(f"\n  wrote {ranking_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
