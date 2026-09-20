from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="用SQLite建立财务和年报文本特征库")
    default_source = Path(__file__).resolve().parents[2] / "source_materials"
    parser.add_argument("--source-dir", type=Path, default=default_source)
    parser.add_argument("--database", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "feature_store.sqlite")
    args = parser.parse_args()
    args.database.parent.mkdir(parents=True, exist_ok=True)

    financial = pd.read_csv(args.source_dir / "panel_2020_2024_strict_effective.csv")
    text = pd.read_csv(args.source_dir / "text_metrics_fast_fulltext.csv")
    with sqlite3.connect(args.database) as connection:
        financial.to_sql("financial_panel", connection, if_exists="replace", index=False)
        text.to_sql("text_metrics", connection, if_exists="replace", index=False)
        sql_path = Path(__file__).resolve().parents[1] / "sql" / "create_feature_store.sql"
        connection.executescript(sql_path.read_text(encoding="utf-8"))
        count = connection.execute("SELECT COUNT(*) FROM financial_text_features").fetchone()[0]
        firms = connection.execute("SELECT COUNT(DISTINCT code) FROM financial_text_features").fetchone()[0]
    print(f"joined_rows={count} firms={firms} database={args.database}")


if __name__ == "__main__":
    main()
