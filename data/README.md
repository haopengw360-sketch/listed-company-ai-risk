# 数据说明

项目使用 CSMAR 财务数据和本地年报文本。原始数据受授权限制，不放入公开仓库。

运行时需提供三份文件：

1. `panel_2020_2024_strict_effective.csv`：公司年度财务面板；
2. `company_master.xlsx`：证券代码与证监会行业代码；
3. `text_metrics_fast_fulltext.csv`：由年报全文提取的词频特征。

可选文件 `llm_firmyear_metrics.csv` 是句级 LLM 标注聚合后的公司年度指标，只用于小样本方法校验。

默认脚本会从仓库同级的 `source_materials` 目录读取这些文件，也可通过命令行参数指定其他目录。

