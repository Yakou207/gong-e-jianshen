# 工e鉴审 · 反洗钱甄别质检工作台 + 受限取证 Agent

按 `../output/icbc_aml_spec_v0.4.1.md` 开发的合成数据研究原型：检查预警排除理由中的事实、对预警关注点的回应、材料与交易的支持关系，并在补证、纠错或规范迁移后保守增量重查、提示人工重新确认。不接入真实银行系统，不自动报送或处置账户，不判定“客户无洗钱风险”。

## 快速启动

需要 Python 3.11–3.13 与 [uv](https://docs.astral.sh/uv/)。

```sh
cd aml-qc
uv sync --frozen
cp .env.example .env          # 填写自己的 DEEPSEEK_API_KEY；离线演示可留空
.venv/bin/python -m uvicorn aml_qc.api:app --host 127.0.0.1 --port 8765
```

打开 http://127.0.0.1:8765 。首次启动只补入六个种子案例，不覆盖已修改的案件；数据保存在 `runtime/workbench.sqlite3`（请勿绑定公网，无生产认证）。可用深链接直接打开某个案件：`#workspace/seed-02`、`#delivery/seed-02`。

推荐演示：选 seed-02 → 离线固定流程 → 查看“仅支付一次”与三笔流水的矛盾 → 确认问题 → 修订理由 → 观察来源过期、增量重查与人工“需复核”记录。

## 目录

| 路径 | 内容 |
|---|---|
| `aml_qc/core.py` `ingest.py` `schema.py` | 案件校验、覆盖、F1/F2、四类事实核验、材料关系模板（整数分、半开区间） |
| `aml_qc/workflow.py` `llm.py` `contracts.py` `baseline.py` | LangGraph 编排、Fixed/Agent 流程、模型统一接口与输出契约、B0 直读基线 |
| `aml_qc/depgraph.py` `store.py` `annotations.py` `claim_edits.py` `leads.py` `migrations.py` `exports.py` | 依赖指纹与增量重查、不可变快照、人工作业与审计、规范迁移、导出 |
| `aml_qc/evaluation_budget.py` | 受累计预算约束的模型调用与账本 |
| `aml_qc/api.py` `web/` | FastAPI 接口与单页工作台 |
| `config/schema/S1.0.json` `config/prompts/` | 版本化标注规范与提示词 |
| `scripts/generate_heldout.py` `scripts/score_heldout.py` | 留出注入缺陷评测集生成与评分 |
| `tests/` | 回归测试（`.venv/bin/python -m pytest -q`） |

## 模型与费用

模型标识 `deepseek-flash`，通过 `.env` 配置。单任务最多 6 次模型调用；Agent 最多 3 轮追加、每轮 2 次只读工具；预算耗尽或输出校验失败保持“未完成”，不自动重试。工作台的付费运行入口默认禁用，真实评测通过受累计预算约束的运行器执行。离线固定流程不调用模型，也不算 Agent。

## 评测

- **留出集 heldout-v1**：`scripts/generate_heldout.py eval/heldout-v1` 以固定种子生成 32 个案件（8 类注入缺陷）；真值文件与任务包分离，F1/F2 与材料关系真值由独立实现复算并与核心代码逐项一致。评分：`scripts/score_heldout.py --output eval/heldout-v1/score-hv1.json`。
- **增量机制**：13 类关键变更的独立预期测试见 `tests/test_mutation_acceptance.py`。
- 真值为生成器注入，不是独立人工参考答案；结论仅限合成数据与本次运行。

## 已知限制

独立人工参考标注、从业人员访谈、正式冻结实验尚未完成；合成单账户；本地哈希链只验证链条一致性，不防止拥有写权限者重写整链；非生产权限系统。
