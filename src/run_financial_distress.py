from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from common import (
    FINANCIAL_FEATURES,
    build_financial_panel,
    evaluate,
    load_dataset,
    make_logistic,
    make_xgboost,
    model_importance,
    temporal_fit,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="下一年度亏损预警：逻辑回归与XGBoost")
    default_source = Path(__file__).resolve().parents[2] / "source_materials"
    parser.add_argument("--source-dir", type=Path, default=default_source)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "project1")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel, _, master = load_dataset(args.source_dir)
    df = build_financial_panel(panel, master)
    usable = df.loc[df["next_loss"].notna() & df["year"].between(2020, 2023)].copy()

    rows = []
    predictions = None
    importances = []
    fitted = {}
    for name, maker in [("LogisticRegression", make_logistic), ("XGBoost", make_xgboost)]:
        model, threshold, train, test, probability = temporal_fit(usable, FINANCIAL_FEATURES, maker)
        metrics = evaluate(test["next_loss"].astype(int), probability, threshold)
        rows.append({"model": name, **metrics, "train_rows": len(train), "test_rows": len(test)})
        importances.append(model_importance(model, FINANCIAL_FEATURES, name))
        fitted[name] = (model, threshold)
        if name == "XGBoost":
            predictions = test[["code", "year", "next_loss", "next_net_profit"]].copy()
            predictions["risk_probability"] = probability
            predictions["predicted_loss"] = (probability >= threshold).astype(int)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(args.output_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(importances).to_csv(args.output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    predictions.sort_values("risk_probability", ascending=False).to_csv(
        args.output_dir / "test_predictions_2024.csv", index=False, encoding="utf-8-sig"
    )

    best_model, threshold = fitted["XGBoost"]
    production_train = usable.loc[usable["year"] <= 2023]
    production_model = make_xgboost(FINANCIAL_FEATURES, production_train["next_loss"].astype(int))
    production_model.fit(production_train[FINANCIAL_FEATURES], production_train["next_loss"].astype(int))
    scoring = df.loc[df["year"] == 2024].copy()
    scoring["risk_probability"] = production_model.predict_proba(scoring[FINANCIAL_FEATURES])[:, 1]
    scoring["risk_level"] = pd.cut(
        scoring["risk_probability"], bins=[-0.01, 0.35, 0.65, 1.01], labels=["low", "medium", "high"]
    )
    scoring[["code", "year", "risk_probability", "risk_level"]].sort_values(
        "risk_probability", ascending=False
    ).to_csv(args.output_dir / "risk_watchlist_2025.csv", index=False, encoding="utf-8-sig")

    xgb_row = metrics_df.loc[metrics_df["model"] == "XGBoost"].iloc[0]
    write_json(
        args.output_dir / "summary.json",
        {
            "sample_rows": int(len(usable)),
            "sample_firms": int(usable["code"].nunique()),
            "test_rows_2023_to_2024": int((usable["year"] == 2023).sum()),
            "test_loss_rate": float(usable.loc[usable["year"] == 2023, "next_loss"].mean()),
            "xgboost_roc_auc": float(xgb_row["roc_auc"]),
            "xgboost_pr_auc": float(xgb_row["pr_auc"]),
            "xgboost_f1": float(xgb_row["f1"]),
            "xgboost_recall": float(xgb_row["recall"]),
            "features": FINANCIAL_FEATURES,
        },
    )

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, maker in [("Logistic", make_logistic), ("XGBoost", make_xgboost)]:
        model, t, _, test, prob = temporal_fit(usable, FINANCIAL_FEATURES, maker)
        RocCurveDisplay.from_predictions(test["next_loss"].astype(int), prob, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(test["next_loss"].astype(int), prob, name=name, ax=axes[1])
        if name == "XGBoost":
            ConfusionMatrixDisplay.from_predictions(
                test["next_loss"].astype(int), prob >= t, cmap="Blues", ax=axes[2], colorbar=False
            )
    axes[0].set_title("Out-of-time ROC: 2023 → 2024")
    axes[1].set_title("Out-of-time precision-recall")
    axes[2].set_title("XGBoost confusion matrix")
    fig.tight_layout()
    fig.savefig(args.output_dir / "model_performance.png", dpi=180)
    plt.close(fig)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()

