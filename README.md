# 上市公司经营风险 AI 分析

项目包含财务困境预警和年报文本风险识别两个模块。建模样本覆盖A股信息技术相关行业（证监会行业代码 `I65`、`C39`），采用跨年度样本训练和严格的时间外测试评估模型。

已完成的滚动时间验证调参和锁定测试结果：财务困境 XGBoost 的 ROC-AUC 为 **0.869**、F1 为 **0.756**、召回率为 **83.2%**；财务与年报词典文本融合模型的 ROC-AUC 为 **0.873**、F1 为 **0.764**。完整口径见 `reports/project_report.md`。

## 项目一：财务困境预警模型

- 用公司第 t 年财务指标预测第 t+1 年是否亏损；
- 采用逻辑回归和 XGBoost；
- 2020—2022 年训练，2023 年指标预测 2024 年结果；
- 输出 ROC-AUC、PR-AUC、F1、召回率、特征重要性和风险名单；
- 另以 2024 年指标生成 2025 年待核验风险观察名单。

运行：

```bash
python src/run_financial_distress.py --source-dir ../source_materials
```

滚动时间验证调参：

```bash
python src/tune_xgboost_temporal.py --source-dir ../source_materials --candidates 36
```

详细训练逻辑见 `reports/training_guide.md`。

## 项目二：年报文本 NLP 风险识别

- 对年报全文提取风险、负面、正面、不确定性和模糊表达词频；
- 比较纯文本、纯财务和“财务＋文本”三类模型；
- 验证文本风险语义与下一年度亏损的相关性；
- 提供 OpenAI 兼容接口的 LLM 句级标签脚本；
- 已有 32 个公司年度的 LLM 抽样标注用于方法校验，全量结果使用透明可复现的词典指标。
- 高级版本使用 `BAAI/bge-small-zh-v1.5` 对每份年报的高风险语句编码为 512 维 Transformer 向量，经 PCA 压缩后与财务模型做验证集加权融合。

运行：

```bash
python src/run_annual_report_nlp.py --source-dir ../source_materials
```

生成 Transformer 特征并做时间外评估：

```bash
python src/build_transformer_embeddings.py --annual-report-dir D:/annual_reports --company-master ../source_materials/company_master.xlsx --output outputs/project2/bge_report_embeddings.csv
python src/run_transformer_nlp.py --source-dir ../source_materials
```

若要重新从年报 txt 提取词频：

```bash
python src/build_text_features.py --input-dir D:/annual_reports --output data/text_metrics.csv
```

LLM句级标注只通过环境变量读取密钥，代码和配置文件不保存真实密钥：

```bash
$env:LLM_API_KEY="在本机临时填写"
$env:LLM_BASE_URL="https://api.deepseek.com/chat/completions"
$env:LLM_MODEL="deepseek-chat"
python src/llm_sentence_tagging.py --input data/sentences.jsonl --output outputs/llm_tags.jsonl
```

当前仓库没有使用或保存任何真实AI API Key。默认接口是DeepSeek的OpenAI兼容地址，也可通过环境变量切换到其他兼容服务。安全规则见 `SECURITY.md`。

## 数据与防泄漏设计

- 原始 CSMAR 数据和年报文本受授权限制，不进入公开仓库；
- 标签来自下一年度净利润，任何下一年度字段都不进入特征；
- 验证集固定为 2022→2023，测试集固定为 2023→2024，避免随机拆分把未来信息泄漏到训练集；
- 分类阈值只在验证集选择，测试集只评估一次。

## 年度迭代

训练年份、滚动验证年份、测试年份、候选参数数量和随机种子集中保存在 `configs/experiment.json`。新增年度数据后：

1. 更新私有财务面板和年报文本指标；
2. 修改配置中的训练、验证和测试年份；
3. 运行SQLite特征库、基础模型和滚动调参；
4. 将新结果追加到实验登记表；
5. 只有锁定测试集指标达到门槛时才替换当前模型。

```bash
python src/build_feature_store.py --source-dir ../source_materials
python run_all.py
python src/tune_xgboost_temporal.py --source-dir ../source_materials --config configs/experiment.json
python src/check_secrets.py
```

## 目录

```text
src/                 数据处理、建模和LLM标签代码
configs/             可版本化的训练与评估配置
outputs/project1/    财务困境模型结果
outputs/project2/    年报NLP模型结果
data/README.md       私有数据字段和放置说明
reports/             项目报告、训练方法和迭代说明
```

结果以 `outputs` 中的可复现文件为准。原始授权数据、真实密钥、逐公司预测结果和大型向量文件不会进入公开仓库。
