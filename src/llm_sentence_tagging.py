from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

SYSTEM_PROMPT = """你是金融文本分析助手。对上市公司年报单句进行结构化分类，只输出JSON。
sentiment只能为positive、negative_risk、neutral；risk_degree、specificity、response、vagueness取0-3；quantification取0或1。
字段必须包括sentiment、risk_degree、specificity、quantification、response、vagueness。"""


def classify(base_url: str, api_key: str, model: str, sentence: str) -> dict:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": sentence[:1200]}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(base_url, data=payload, method="POST", headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        content = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"]
    return json.loads(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI兼容接口的年报风险句标注器")
    parser.add_argument("--input", type=Path, required=True, help="JSONL，每行至少含sentence字段")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("请通过环境变量 LLM_API_KEY 提供密钥，不要把密钥写入代码。")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/chat/completions")
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            item = json.loads(line)
            tagged = {**item, **classify(base_url, api_key, model, item["sentence"]), "model": model}
            target.write(json.dumps(tagged, ensure_ascii=False) + "\n")
            target.flush()
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()

