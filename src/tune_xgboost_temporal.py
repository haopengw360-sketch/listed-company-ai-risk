from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from common import (
    FINANCIAL_FEATURES,
    TEXT_FEATURES,
    add_text_features,
    best_f1_threshold,
    build_financial_panel,
    evaluate,
    load_dataset,
)


PARAMETER_SPACE = {
    "n_estimators": [250, 400, 550, 700],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.02, 0.04, 0.07],
    "min_child_weight": [3, 6, 10],
    "subsample": [0.75, 0.90],
    "colsample_bytree": [0.75, 0.90],
    "reg_alpha": [0.0, 0.1, 0.5],
    "reg_lambda": [1.5, 3.0, 6.0],
}


def make_model(params: dict, y: pd.Series, random_seed: int) -> Pipeline:
    positives = max(int(y.sum()), 1)
    negatives = max(int((1 - y).sum()), 1)
    estimator = XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=negatives / positives,
        random_state=random_seed,
        n_jobs=4,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def rolling_score(
    df: pd.DataFrame,
    features: list[str],
    params: dict,
    validation_years: list[int],
    random_seed: int,
) -> dict:
    fold_rows = []
    for validation_year in validation_years:
        train = df.loc[df["year"] < validation_year]
        validation = df.loc[df["year"] == validation_year]
        model = make_model(params, train["next_loss"].astype(int), random_seed)
        model.fit(train[features], train["next_loss"].astype(int))
        probability = model.predict_proba(validation[features])[:, 1]
        fold_rows.append({
            "validation_year": validation_year,
            "roc_auc": roc_auc_score(validation["next_loss"], probability),
            "pr_auc": average_precision_score(validation["next_loss"], probability),
        })
    result = {
        "mean_roc_auc": float(np.mean([row["roc_auc"] for row in fold_rows])),
        "mean_pr_auc": float(np.mean([row["pr_auc"] for row in fold_rows])),
    }
    for row in fold_rows:
        year = row["validation_year"]
        result[f"auc_{year}"] = row["roc_auc"]
        result[f"pr_auc_{year}"] = row["pr_auc"]
    return result


def tune_one(
    df: pd.DataFrame,
    features: list[str],
    candidates: int,
    validation_years: list[int],
    random_seed: int,
) -> tuple[dict, pd.DataFrame]:
    sampled = list(ParameterSampler(PARAMETER_SPACE, n_iter=candidates, random_state=random_seed))
    rows = []
    for index, params in enumerate(sampled, start=1):
        scores = rolling_score(df, features, params, validation_years, random_seed)
        rows.append({"candidate": index, **params, **scores})
        print(
            f"candidate={index}/{len(sampled)} "
            f"mean_auc={scores['mean_roc_auc']:.4f} mean_pr={scores['mean_pr_auc']:.4f}",
            flush=True,
        )
    leaderboard = pd.DataFrame(rows).sort_values(
        ["mean_roc_auc", "mean_pr_auc"], ascending=False
    ).reset_index(drop=True)
    best = {key: leaderboard.iloc[0][key] for key in PARAMETER_SPACE}
    for key in ["n_estimators", "max_depth", "min_child_weight"]:
        best[key] = int(best[key])
    for key in ["learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]:
        best[key] = float(best[key])
    return best, leaderboard


def train_locked_test(
    df: pd.DataFrame,
    features: list[str],
    params: dict,
    threshold_year: int,
    train_end_year: int,
    test_year: int,
    random_seed: int,
) -> tuple[dict, float]:
    development = df.loc[df["year"] < threshold_year]
    validation = df.loc[df["year"] == threshold_year]
    train = df.loc[df["year"] <= train_end_year]
    test = df.loc[df["year"] == test_year]

    threshold_model = make_model(params, development["next_loss"].astype(int), random_seed)
    threshold_model.fit(development[features], development["next_loss"].astype(int))
    validation_probability = threshold_model.predict_proba(validation[features])[:, 1]
    threshold = best_f1_threshold(validation["next_loss"].astype(int), validation_probability)

    model = make_model(params, train["next_loss"].astype(int), random_seed)
    model.fit(train[features], train["next_loss"].astype(int))
    test_probability = model.predict_proba(test[features])[:, 1]
    metrics = evaluate(test["next_loss"].astype(int), test_probability, threshold)
    metrics.update({"train_rows": len(train), "test_rows": len(test)})
    return metrics, threshold


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description="按配置执行滚动时间验证与锁定测试")
    default_source = Path(__file__).resolve().parents[2] / "source_materials"
    default_config = Path(__file__).resolve().parents[1] / "configs" / "experiment.json"
    parser.add_argument("--source-dir", type=Path, default=default_source)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "tuning")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--candidates", type=int, default=None, help="覆盖配置中的候选参数数量")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    candidates = args.candidates or int(config["parameter_candidates"])
    validation_years = [int(year) for year in config["rolling_validation_years"]]
    threshold_year = int(config["threshold_validation_year"])
    train_end_year = int(config["final_train_end_year"])
    test_year = int(config["locked_test_feature_year"])
    start_year = int(config["feature_year_start"])
    random_seed = int(config["random_seed"])
    industry_prefixes = tuple(config["industry_prefixes"])

    panel, text, master = load_dataset(args.source_dir)
    financial = build_financial_panel(panel, master, industry_prefixes)
    financial = financial.loc[
        financial["next_loss"].notna() & financial["year"].between(start_year, test_year)
    ].copy()
    combined = add_text_features(financial, text)

    tasks = [
        ("financial", financial, FINANCIAL_FEATURES),
        ("financial_text", combined, FINANCIAL_FEATURES + TEXT_FEATURES),
    ]
    all_metrics = []
    selected = {}
    for name, dataset, features in tasks:
        print(f"\nTUNING {name} rows={len(dataset)} features={len(features)}", flush=True)
        params, leaderboard = tune_one(
            dataset, features, candidates, validation_years, random_seed
        )
        leaderboard.to_csv(args.output_dir / f"{name}_leaderboard.csv", index=False, encoding="utf-8-sig")
        metrics, threshold = train_locked_test(
            dataset,
            features,
            params,
            threshold_year,
            train_end_year,
            test_year,
            random_seed,
        )
        all_metrics.append({"model": name, **metrics})
        selected[name] = {
            "params": params,
            "threshold": threshold,
            "rolling_validation_mean_auc": float(leaderboard.iloc[0]["mean_roc_auc"]),
            "rolling_validation_mean_pr_auc": float(leaderboard.iloc[0]["mean_pr_auc"]),
        }

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(args.output_dir / "tuned_test_metrics.csv", index=False, encoding="utf-8-sig")
    selected["experiment"] = {
        "name": config["experiment_name"],
        "config": config,
        "config_fingerprint": file_fingerprint(args.config),
        "data_fingerprints": {
            "financial_panel": file_fingerprint(args.source_dir / "panel_2020_2024_strict_effective.csv"),
            "text_metrics": file_fingerprint(args.source_dir / "text_metrics_fast_fulltext.csv"),
            "company_master": file_fingerprint(args.source_dir / "company_master.xlsx"),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "selected_parameters.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    registry_path = args.output_dir / "experiment_registry.csv"
    registry_row = pd.DataFrame([
        {
            "created_at_utc": selected["experiment"]["created_at_utc"],
            "experiment_name": config["experiment_name"],
            "config_fingerprint": selected["experiment"]["config_fingerprint"],
            "test_feature_year": test_year,
            "financial_roc_auc": metrics_df.loc[metrics_df["model"] == "financial", "roc_auc"].iloc[0],
            "financial_f1": metrics_df.loc[metrics_df["model"] == "financial", "f1"].iloc[0],
            "financial_text_roc_auc": metrics_df.loc[metrics_df["model"] == "financial_text", "roc_auc"].iloc[0],
            "financial_text_f1": metrics_df.loc[metrics_df["model"] == "financial_text", "f1"].iloc[0],
        }
    ])
    if registry_path.exists():
        registry_row.to_csv(registry_path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        registry_row.to_csv(registry_path, index=False, encoding="utf-8-sig")
    print("\nLOCKED TEST RESULTS")
    print(metrics_df.to_string(index=False))
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
