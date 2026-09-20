from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from common import (
    FINANCIAL_FEATURES,
    TEXT_FEATURES,
    add_text_features,
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
    parser = argparse.ArgumentParser(description="年报文本风险指标与下一年度亏损关联")
    default_source = Path(__file__).resolve().parents[2] / "source_materials"
    parser.add_argument("--source-dir", type=Path, default=default_source)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "project2")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel, text, master = load_dataset(args.source_dir)
    financial = build_financial_panel(panel, master)
    df = add_text_features(financial, text)
    usable = df.loc[df["next_loss"].notna() & df["year"].between(2020, 2023)].copy()

    configurations = [
        ("Text_Logistic", TEXT_FEATURES, make_logistic),
        ("Financial_XGBoost", FINANCIAL_FEATURES, make_xgboost),
        ("Financial_Text_XGBoost", FINANCIAL_FEATURES + TEXT_FEATURES, make_xgboost),
    ]
    results = []
    importance = []
    test_predictions = None
    plot_items = []
    production_spec = None
    for name, features, maker in configurations:
        model, threshold, train, test, probability = temporal_fit(usable, features, maker)
        metrics = evaluate(test["next_loss"].astype(int), probability, threshold)
        results.append({"model": name, **metrics, "train_rows": len(train), "test_rows": len(test)})
        importance.append(model_importance(model, features, name))
        plot_items.append((name, test["next_loss"].astype(int), probability))
        if name == "Financial_Text_XGBoost":
            test_predictions = test[["code", "year", "next_loss"]].copy()
            test_predictions["risk_probability"] = probability
            test_predictions["predicted_loss"] = (probability >= threshold).astype(int)
            production_spec = (features, threshold)

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(args.output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat(importance).to_csv(args.output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    test_predictions.sort_values("risk_probability", ascending=False).to_csv(
        args.output_dir / "test_predictions_2024.csv", index=False, encoding="utf-8-sig"
    )

    test_2023 = usable.loc[usable["year"] == 2023].copy()
    correlations = []
    for feature in TEXT_FEATURES:
        correlations.append(
            {
                "feature": feature,
                "spearman_with_next_loss": test_2023[[feature, "next_loss"]].corr(method="spearman").iloc[0, 1],
                "mean_profitable_next_year": test_2023.loc[test_2023["next_loss"] == 0, feature].mean(),
                "mean_loss_next_year": test_2023.loc[test_2023["next_loss"] == 1, feature].mean(),
            }
        )
    correlation_df = pd.DataFrame(correlations)
    correlation_df.to_csv(args.output_dir / "text_loss_relationship.csv", index=False, encoding="utf-8-sig")

    llm_path = args.source_dir / "llm_firmyear_metrics.csv"
    if llm_path.exists():
        llm = pd.read_csv(llm_path)
        llm["code"] = pd.to_numeric(llm["code"], errors="coerce").astype("Int64")
        pilot = usable[["code", "year", "next_loss"]].merge(llm, on=["code", "year"], how="inner")
        pilot_summary = pilot.groupby("next_loss", dropna=False).agg(
            firmyears=("code", "size"),
            llm_risk_tone=("LLM_Risk_Tone", "mean"),
            llm_risk_ratio=("LLM_Risk_Ratio", "mean"),
            llm_risk_quality=("LLM_Risk_Quality", "mean"),
            vagueness=("vagueness_avg", "mean"),
        ).reset_index()
        pilot_summary.to_csv(args.output_dir / "llm_pilot_summary.csv", index=False, encoding="utf-8-sig")
    else:
        pilot = pd.DataFrame()

    combined_row = metrics_df.loc[metrics_df["model"] == "Financial_Text_XGBoost"].iloc[0]
    financial_row = metrics_df.loc[metrics_df["model"] == "Financial_XGBoost"].iloc[0]
    write_json(
        args.output_dir / "summary.json",
        {
            "reports": int(len(usable)),
            "firms": int(usable["code"].nunique()),
            "test_reports": int((usable["year"] == 2023).sum()),
            "llm_pilot_firmyears": int(len(pilot)),
            "financial_roc_auc": float(financial_row["roc_auc"]),
            "combined_roc_auc": float(combined_row["roc_auc"]),
            "auc_change": float(combined_row["roc_auc"] - financial_row["roc_auc"]),
            "combined_f1": float(combined_row["f1"]),
            "combined_recall": float(combined_row["recall"]),
        },
    )

    features, threshold = production_spec
    production_train = usable.loc[usable["year"] <= 2023]
    production_model = make_xgboost(features, production_train["next_loss"].astype(int))
    production_model.fit(production_train[features], production_train["next_loss"].astype(int))
    scoring = df.loc[df["year"] == 2024].copy()
    scoring["risk_probability"] = production_model.predict_proba(scoring[features])[:, 1]
    scoring[["code", "year", "risk_probability"] + TEXT_FEATURES].sort_values(
        "risk_probability", ascending=False
    ).to_csv(args.output_dir / "text_enhanced_watchlist_2025.csv", index=False, encoding="utf-8-sig")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, y_true, probability in plot_items:
        RocCurveDisplay.from_predictions(y_true, probability, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_true, probability, name=name, ax=axes[1])
    axes[0].set_title("Text feature comparison: ROC")
    axes[1].set_title("Text feature comparison: PR")
    fig.tight_layout()
    fig.savefig(args.output_dir / "model_comparison.png", dpi=180)
    plt.close(fig)
    print(metrics_df.to_string(index=False))
    print(correlation_df.to_string(index=False))


if __name__ == "__main__":
    main()

