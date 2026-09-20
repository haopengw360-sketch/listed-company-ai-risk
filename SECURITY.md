# 安全与隐私

## API密钥

仓库不包含任何真实AI API Key。`src/llm_sentence_tagging.py`只从进程环境读取 `LLM_API_KEY`，默认调用DeepSeek的OpenAI兼容接口；通过 `LLM_BASE_URL` 和 `LLM_MODEL` 可以切换服务。

真实密钥只能保存在本机环境变量或被 `.gitignore` 排除的 `.env` 文件中。不要把密钥写入Python、JSON、Markdown、Notebook、命令历史截图或提交信息。

## 数据隐私

原始CSMAR数据、年报全文、逐公司测试预测、风险观察名单、SQLite数据库和Transformer向量均不进入公开仓库。公开内容仅包括代码、字段说明、聚合指标、模型图表和不含公司明细的实验结果。

## 提交前检查

每次提交前运行：

```bash
python src/check_secrets.py
git status --short
```

GitHub Actions也会在每次推送和拉取请求时执行同一项密钥扫描。若密钥曾经进入提交历史，应立即撤销该密钥、清理历史并重新生成密钥。
