from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

RISK_TERMS = [
    "风险", "不确定", "压力", "挑战", "下滑", "下降", "波动", "冲击", "困难", "损失",
    "违约", "减值", "诉讼", "处罚", "亏损", "流动性", "信用", "经营风险", "财务风险",
    "合规", "政策", "汇率", "利率", "应收账款", "存货", "现金流", "偿债", "竞争加剧",
]


def select_risk_context(text: str, top_k: int = 18) -> tuple[str, int]:
    text = re.sub(r"[\t\r ]+", "", text)
    sentences = re.split(r"(?<=[。！？；])|\n+", text)
    candidates = []
    for index, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not 15 <= len(sentence) <= 500:
            continue
        hits = sum(sentence.count(term) for term in RISK_TERMS)
        if hits:
            number_bonus = 0.5 if re.search(r"\d+(?:\.\d+)?[%亿元万]", sentence) else 0.0
            candidates.append((hits + number_bonus, index, sentence))
    if not candidates:
        fallback = "".join(sentences[:30])[:4000]
        return fallback, 0
    chosen = sorted(candidates, key=lambda item: (-item[0], item[1]))[:top_k]
    chosen = sorted(chosen, key=lambda item: item[1])
    return "".join(item[2] for item in chosen), len(candidates)


def cls_embeddings(model, tokenizer, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
    chunks = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            vectors = model(**encoded).last_hidden_state[:, 0]
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        chunks.append(vectors.cpu().numpy())
        print(f"embedded={min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.vstack(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="用中文BGE编码年报高风险语句")
    parser.add_argument("--annual-report-dir", type=Path, required=True)
    parser.add_argument("--company-master", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    master = pd.read_excel(args.company_master)
    master["code"] = pd.to_numeric(master["Symbol"], errors="coerce").astype("Int64")
    tech_codes = set(
        master.loc[master["IndustryCode"].astype(str).str.startswith(("I65", "C39")), "code"]
        .dropna().astype(int).tolist()
    )

    records = []
    texts = []
    for path in sorted(args.annual_report_dir.rglob("*.txt")):
        match = re.search(r"(?<!\d)(\d{6})[_-]?(20\d{2})(?!\d)", path.stem)
        if not match:
            continue
        code, year = int(match.group(1)), int(match.group(2))
        if code not in tech_codes or not 2020 <= year <= 2024:
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        selected, candidate_count = select_risk_context(raw)
        if not selected:
            continue
        records.append({
            "code": code,
            "year": year,
            "risk_sentence_candidates": candidate_count,
            "selected_chars": len(selected),
        })
        texts.append(selected)

    print(f"reports_selected={len(records)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    embeddings = cls_embeddings(model, tokenizer, texts, args.batch_size, args.max_length)
    output = pd.DataFrame(records)
    embedding_columns = [f"bge_{i:03d}" for i in range(embeddings.shape[1])]
    output = pd.concat([output, pd.DataFrame(embeddings, columns=embedding_columns)], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"shape={output.shape} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
