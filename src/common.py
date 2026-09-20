from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


FINANCIAL_FEATURES = [
    "size",
    "lev",
    "roa",
    "growth",
    "cashflow",
    "tangibility",
    "bm",
    "turnover",
    "ret",
    "sigma",
    "net_margin",
    "current_loss",
]

TEXT_FEATURES = [
    "log_char_count",
    "risk_freq",
    "negative_freq",
    "positive_freq",
    "uncertainty_freq",
    "vague_freq",
    "risk_tone",
]

FEATURE_NAMES_ZH = {
    "size": "企业规模",
    "lev": "资产负债率",
    "roa": "总资产收益率",
    "growth": "营业收入增长率",
    "cashflow": "经营现金流/总资产",
    "tangibility": "固定资产占比",
    "bm": "账面市值比",
    "turnover": "换手率",
    "ret": "年度股票收益率",
    "sigma": "收益波动率",
    "net_margin": "销售净利率",
    "current_loss": "当年亏损标记",
    "log_char_count": "年报文本长度（对数）",
    "risk_freq": "风险词频",
    "negative_freq": "负面词频",
    "positive_freq": "正面词频",
    "uncertainty_freq": "不确定性词频",
    "vague_freq": "模糊表达词频",
    "risk_tone": "风险语调",
}


def load_dataset(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(source_dir / "panel_2020_2024_strict_effective.csv")
    text = pd.read_csv(source_dir / "text_metrics_fast_fulltext.csv")
    master = pd.read_excel(source_dir / "company_master.xlsx")

    panel["code"] = pd.to_numeric(panel["code"], errors="coerce").astype("Int64")
    text["code"] = pd.to_numeric(text["code"], errors="coerce").astype("Int64")
    master["code"] = pd.to_numeric(master["Symbol"], errors="coerce").astype("Int64")
    master = master.loc[master["code"].notna(), ["code", "IndustryCode"]].drop_duplicates("code")
    return panel, text, master


def build_financial_panel(
    panel: pd.DataFrame,
    master: pd.DataFrame,
    industry_prefixes: tuple[str, ...] = ("I65", "C39"),
) -> pd.DataFrame:
    tech_codes = master.loc[
        master["IndustryCode"].astype(str).str.startswith(industry_prefixes), "code"
    ]
    df = panel.loc[panel["code"].isin(tech_codes)].copy()

    next_values = df[["code", "year", "net_profit", "revenue"]].copy()
    next_values["year"] -= 1
    next_values = next_values.rename(
        columns={"net_profit": "next_net_profit", "revenue": "next_revenue"}
    )
    df = df.merge(next_values, on=["code", "year"], how="left", validate="one_to_one")
    df["next_loss"] = np.where(
        df["next_net_profit"].notna(), (df["next_net_profit"] < 0).astype(int), np.nan
    )
    df["next_revenue_decline"] = np.where(
        df["next_revenue"].notna() & df["revenue"].notna(),
        (df["next_revenue"] < df["revenue"]).astype(int),
        np.nan,
    )
    df["net_margin"] = df["net_profit"] / df["revenue"].replace(0, np.nan)
    df["net_margin"] = df["net_margin"].clip(-2, 2)
    df["current_loss"] = (df["net_profit"] < 0).astype(float)
    return df


def add_text_features(financial: pd.DataFrame, text: pd.DataFrame) -> pd.DataFrame:
    keep = ["code", "year", "char_count", "risk_freq", "negative_freq", "positive_freq",
            "uncertainty_freq", "vague_freq", "risk_tone"]
    merged = financial.merge(text[keep], on=["code", "year"], how="inner", validate="one_to_one")
    merged["log_char_count"] = np.log1p(merged["char_count"].clip(lower=0))
    return merged


def make_logistic(features: list[str], _: pd.Series) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )


def make_xgboost(features: list[str], y: pd.Series) -> Pipeline:
    positives = max(int(y.sum()), 1)
    negatives = max(int((1 - y).sum()), 1)
    model = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=negatives / positives,
        random_state=42,
        n_jobs=4,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def best_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    thresholds = np.linspace(0.10, 0.90, 161)
    scores = [f1_score(y_true, probabilities >= t, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def evaluate(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "accuracy": accuracy_score(y_true, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "brier": brier_score_loss(y_true, probabilities),
        "threshold": threshold,
    }


def temporal_fit(
    df: pd.DataFrame,
    features: list[str],
    maker: Callable[[list[str], pd.Series], Pipeline],
) -> tuple[Pipeline, float, pd.DataFrame, pd.DataFrame, np.ndarray]:
    usable = df.loc[df["next_loss"].notna() & df["year"].between(2020, 2023)].copy()
    development = usable.loc[usable["year"] <= 2021]
    validation = usable.loc[usable["year"] == 2022]
    train = usable.loc[usable["year"] <= 2022]
    test = usable.loc[usable["year"] == 2023]

    threshold_model = maker(features, development["next_loss"].astype(int))
    threshold_model.fit(development[features], development["next_loss"].astype(int))
    validation_probability = threshold_model.predict_proba(validation[features])[:, 1]
    threshold = best_f1_threshold(validation["next_loss"].astype(int), validation_probability)

    final_model = maker(features, train["next_loss"].astype(int))
    final_model.fit(train[features], train["next_loss"].astype(int))
    test_probability = final_model.predict_proba(test[features])[:, 1]
    return final_model, threshold, train, test, test_probability


def model_importance(model: Pipeline, features: list[str], model_name: str) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    else:
        values = np.abs(estimator.coef_[0])
    out = pd.DataFrame({"feature": features, "importance": values})
    out["feature_zh"] = out["feature"].map(FEATURE_NAMES_ZH)
    out["model"] = model_name
    return out.sort_values("importance", ascending=False)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
