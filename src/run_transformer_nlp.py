from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    FINANCIAL_FEATURES,
    TEXT_FEATURES,
    add_text_features,
    best_f1_threshold,
    build_financial_panel,
    evaluate,
    load_dataset,
    make_xgboost,
)


def make_bge_model(components: int, c_value: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=components, whiten=True, random_state=42)),
        ("model", LogisticRegression(C=c_value, class_weight="balanced", max_iter=3000, random_state=42)),
    ])


def fit_probability(model, train, test, features):
    model.fit(train[features], train["next_loss"].astype(int))
    return model.predict_proba(test[features])[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="BGE年报语义向量与财务模型融合")
    default_source = Path(__file__).resolve().parents[2] / "source_materials"
    default_embedding = Path(__file__).resolve().parents[1] / "outputs" / "project2" / "bge_report_embeddings.csv"
    parser.add_argument("--source-dir", type=Path, default=default_source)
    parser.add_argument("--embeddings", type=Path, default=default_embedding)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "project2_transformer")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel, text, master = load_dataset(args.source_dir)
    df = add_text_features(build_financial_panel(panel, master), text)
    embedding = pd.read_csv(args.embeddings)
    embedding["code"] = pd.to_numeric(embedding["code"], errors="coerce").astype("Int64")
    df = df.merge(embedding, on=["code", "year"], how="inner", validate="one_to_one")
    df = df.loc[df["next_loss"].notna() & df["year"].between(2020, 2023)].copy()
    bge_features = [column for column in embedding.columns if column.startswith("bge_")]
    base_features = FINANCIAL_FEATURES + TEXT_FEATURES

    development = df.loc[df["year"] <= 2021]
    validation = df.loc[df["year"] == 2022]
    train = df.loc[df["year"] <= 2022]
    test = df.loc[df["year"] == 2023]

    tuning_rows = []
    for components in [16, 32, 64]:
        for c_value in [0.05, 0.2, 1.0]:
            model = make_bge_model(components, c_value)
            probability = fit_probability(model, development, validation, bge_features)
            tuning_rows.append({
                "components": components,
                "c": c_value,
                "validation_auc": roc_auc_score(validation["next_loss"], probability),
                "validation_pr_auc": average_precision_score(validation["next_loss"], probability),
            })
    tuning = pd.DataFrame(tuning_rows).sort_values(["validation_auc", "validation_pr_auc"], ascending=False)
    tuning.to_csv(args.output_dir / "bge_validation_tuning.csv", index=False, encoding="utf-8-sig")
    best = tuning.iloc[0]

    # 阈值、PCA维数和融合权重都只用验证集选择，测试集保持独立。
    bge_dev = make_bge_model(int(best["components"]), float(best["c"]))
    bge_val_probability = fit_probability(bge_dev, development, validation, bge_features)
    bge_threshold = best_f1_threshold(validation["next_loss"].astype(int), bge_val_probability)
    financial_dev = make_xgboost(base_features, development["next_loss"].astype(int))
    financial_val_probability = fit_probability(financial_dev, development, validation, base_features)
    weights = np.linspace(0, 0.5, 21)
    weight_scores = []
    for weight in weights:
        blended = (1 - weight) * financial_val_probability + weight * bge_val_probability
        weight_scores.append((weight, roc_auc_score(validation["next_loss"], blended)))
    bge_weight, _ = max(weight_scores, key=lambda item: item[1])
    blended_val = (1 - bge_weight) * financial_val_probability + bge_weight * bge_val_probability
    blended_threshold = best_f1_threshold(validation["next_loss"].astype(int), blended_val)

    bge_model = make_bge_model(int(best["components"]), float(best["c"]))
    bge_test_probability = fit_probability(bge_model, train, test, bge_features)
    financial_model = make_xgboost(base_features, train["next_loss"].astype(int))
    financial_test_probability = fit_probability(financial_model, train, test, base_features)
    blended_test_probability = (1 - bge_weight) * financial_test_probability + bge_weight * bge_test_probability

    results = [
        {"model": "BGE_Logistic", **evaluate(test["next_loss"].astype(int), bge_test_probability, bge_threshold)},
        {"model": "Financial_Lexicon_XGBoost", **evaluate(test["next_loss"].astype(int), financial_test_probability, blended_threshold)},
        {"model": "Financial_Lexicon_BGE_Ensemble", **evaluate(test["next_loss"].astype(int), blended_test_probability, blended_threshold)},
    ]
    metrics = pd.DataFrame(results)
    metrics["train_rows"] = len(train)
    metrics["test_rows"] = len(test)
    metrics.to_csv(args.output_dir / "transformer_model_comparison.csv", index=False, encoding="utf-8-sig")

    predictions = test[["code", "year", "next_loss"]].copy()
    predictions["financial_probability"] = financial_test_probability
    predictions["bge_probability"] = bge_test_probability
    predictions["ensemble_probability"] = blended_test_probability
    predictions.to_csv(args.output_dir / "transformer_test_predictions_2024.csv", index=False, encoding="utf-8-sig")

    financial_auc = float(metrics.loc[metrics["model"] == "Financial_Lexicon_XGBoost", "roc_auc"].iloc[0])
    ensemble_auc = float(metrics.loc[metrics["model"] == "Financial_Lexicon_BGE_Ensemble", "roc_auc"].iloc[0])
    summary = {
        "model": "BAAI/bge-small-zh-v1.5",
        "embedding_dimensions": len(bge_features),
        "selected_pca_components": int(best["components"]),
        "selected_c": float(best["c"]),
        "selected_bge_weight": float(bge_weight),
        "reports": int(len(df)),
        "firms": int(df["code"].nunique()),
        "test_reports": int(len(test)),
        "bge_only_auc": float(metrics.loc[metrics["model"] == "BGE_Logistic", "roc_auc"].iloc[0]),
        "financial_lexicon_auc": financial_auc,
        "ensemble_auc": ensemble_auc,
        "ensemble_auc_change": ensemble_auc - financial_auc,
        "ensemble_f1": float(metrics.loc[metrics["model"] == "Financial_Lexicon_BGE_Ensemble", "f1"].iloc[0]),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
