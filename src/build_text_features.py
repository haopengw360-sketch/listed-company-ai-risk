from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

RISK_WORDS = ["风险", "不确定", "压力", "挑战", "下滑", "下降", "波动", "冲击", "困难", "损失", "违约", "减值", "诉讼", "处罚", "亏损", "流动性", "信用风险", "市场风险", "经营风险", "财务风险", "合规风险", "政策风险", "汇率风险", "利率风险"]
NEGATIVE_WORDS = ["下降", "下滑", "减少", "降低", "恶化", "亏损", "损失", "不足", "困难", "压力", "不利", "失败", "滞后", "违规", "处罚", "诉讼", "减值", "违约", "受限"]
POSITIVE_WORDS = ["增长", "提升", "改善", "优化", "增强", "稳健", "良好", "积极", "持续", "突破", "创新", "完善", "提高", "扩大", "顺利", "有效", "加强"]
UNCERTAINTY_WORDS = ["可能", "预计", "或将", "不确定", "难以预测", "存在不确定性", "有待", "视情况", "取决于", "尚不明确", "波动"]
VAGUE_WORDS = ["积极推进", "持续加强", "不断完善", "进一步提升", "努力", "适时", "稳步推进", "多措并举", "持续优化", "加强管理", "有效控制", "总体可控", "保持稳定", "一定程度", "较大影响", "复杂多变"]


def parse_code_year(path: Path) -> tuple[str | None, int | None]:
    match = re.search(r"(?<!\d)(\d{6})[_-]?(20\d{2})(?!\d)", path.stem)
    return (match.group(1), int(match.group(2))) if match else (None, None)


def count(text: str, words: list[str]) -> int:
    return sum(text.count(word) for word in words)


def main() -> None:
    parser = argparse.ArgumentParser(description="从年报txt全文生成风险语义特征")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_dir.rglob("*.txt")):
        code, year = parse_code_year(path)
        if not code:
            continue
        text = re.sub(r"\s+", "", path.read_text(encoding="utf-8", errors="ignore"))
        if not text:
            continue
        risk, neg, pos = count(text, RISK_WORDS), count(text, NEGATIVE_WORDS), count(text, POSITIVE_WORDS)
        unc, vague = count(text, UNCERTAINTY_WORDS), count(text, VAGUE_WORDS)
        denom = neg + pos + len(text) / 1000
        rows.append({
            "code": code, "year": year, "file": str(path), "char_count": len(text),
            "risk_count": risk, "negative_count": neg, "positive_count": pos,
            "uncertainty_count": unc, "vague_count": vague,
            "risk_freq": risk / len(text), "negative_freq": neg / len(text),
            "positive_freq": pos / len(text), "uncertainty_freq": unc / len(text),
            "vague_freq": vague / len(text), "risk_tone": (neg - pos) / denom if denom else 0,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"reports={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()

