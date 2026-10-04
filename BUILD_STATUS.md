# 项目构建状态

当前交付范围：根据 Implementation Spec 建立真实数据管线和可运行的建模 v0，生成可复现 CSV split，并完成 Phase 1A、扩展正式认知预测基线套件及 Phase 1B item 编码核验与 numeric/semantic 公平对照。

| 部分 | 当前状态 | 证据/限制 |
|---|---|---|
| 目录核对 | 完成 | 三个真实目录逐文件索引，`reports/data_audit/mri_root_comparison.csv` |
| 原始量表审计 | 完成 | 15,453 visits / 4,316 subjects / 161 columns；逐列取值、缺失、co-occurrence、随访分布 |
| subject-wise split | 完成 | 3,021 / 431 / 864 subjects；原始 161 列保留，不插值、不填缺失 |
| MRI 配对 | 完成访视代码级匹配 | 9,436 visits；真实日期窗口/QC 尚不具备 |
| MRI 月份归类 | 完成 | 每条量表访视按同一受试者最近 `ses-Mxxx` 归类；提供全部记录、±6 月子集和 0/3/6/12 月敏感性统计 |
| 异步事件/未来索引 | 完成 | 独立 instrument 日期、明确 nominal 时间、严格 cutoff、未来目标仅来自实际观测 |
| 数据字典 | 完成官方核验 | 107/107 字段与官方表精确匹配；99 个字段通过编码/观测核验；8 个 MMSE 拼写字段保持关闭；68 个单调标量字段进入 Phase 1B |
| Numeric/masked learning v0 | 已实现并作短程检查 | CLI、train/val curves、checkpoint、per-item metrics、failure cases |
| Phase 1A 正式 numeric forecasting | 完成 | LOCF/LME/GRU-D/Transformer；3 seeds；逐终点/时间窗指标；subject bootstrap；测试集仅在协议冻结后评估 |
| 扩展正式 baseline suite | 完成 | 68 个已核验输入；LOCF/LME/GRU/LSTM/T-LSTM/GRU-D/Transformer；15 checkpoints；GRU-D 由验证集选中，测试 macro normalized MAE 0.3799 |
| Phase 1B semantic fair comparison | 完成 | numeric/name/task-semantics 三组使用相同输入、架构、训练流程及 269,325 个可训练参数；18 checkpoints；未来预测语义组略差、两项 cross-scale 的 CI 均跨 0，门控未通过 |
| Semantic branch | 已实现并正式比较固定 descriptor | 本地冻结 Bio_ClinicalBERT；结果不支持固定 item 任务描述的稳定增益 |
| MRI preprocessing / 3D ViT / grounding | 代码及真实图像前向检查完成 | 保留空间 tokens；header 检查不等于临床 QC；未训练 |
| Probabilistic forecasting | 代码及真实数据梯度检查完成 | 可变历史、Gaussian belief、时间转移、未来 item/total/FAQ targets；未做科研验证 |
| Multi-candidate future | 已实现 latent sampling API | 没有候选多样性/校准结论 |
| Structured report / SFT | schema、证据解析、本地 LoRA 和 epoch API | 无指定 LLM/SFT 数据，未训练 |
| RLVR / GRPO | 奖励分解、缺失归一化、证据/时间/domain 检查、critic、GRPO 目标与 gate API | 无真实 on-policy rollout，未训练；不声称完整 RL 实验已运行 |
| Leakage audit | 完成初筛 | 仅量表缺失模式预测 diagnosis，验证 balanced accuracy 0.420；应继续按 phase 控制协议混杂 |
| 科研 gates | Phase 1A 完成；Phase 1B 已判定未通过 | item 编码核验完成；semantic 对照结果为阴性；MRI 日期/QC、MRI 增益、校准和 SFT 稳定性仍需实证 |

工程核查日志和结果：

- `reports_tests.log`：自动化检查，包括真实 split 原值保留、无 subject 泄漏、严格未来、mask 隔离、因果 attention、奖励 availability、gate 拒绝冒烟结果。
- `outputs/stage1_smoke/`：真实数据上少量优化步，标记 `engineering_smoke_only=true`。这些数字只证明程序可运行。
- `reports/experiments/integration_smoke.json`：真实 T1 读取/resampling、空间 token、grounding 和 future backward 检查。
- `reports/experiments/locf_val/`：探索性 LOCF 基线及覆盖率；test 未用于选择模型。
- `reports/experiments/phase1_numeric_formal_protocol.md`：测试集评估前冻结的 Phase 1A 问题、输入、终点、模型、选择指标和统计方案。
- `outputs/phase1_numeric_formal_20261003/`：正式实验的 6 个神经 checkpoint、完整 val/test predictions、逐终点和 horizon 指标、LME 诊断、bootstrap 与报告。验证集选择 GRU-D；测试宏平均归一化 MAE 为 0.3889。
- `reports/experiments/phase1_baseline_suite_formal_protocol.md`：扩展基线套件在测试集加载前冻结的模型、训练预算、终点和统计方案。
- `outputs/phase1_baseline_suite_formal_20261004/`：15 个神经 checkpoint、完整预测、逐终点指标和 10 组 subject-cluster bootstrap。验证集选择 GRU-D；测试宏平均归一化 MAE 为 0.3799。
- `reports/data_audit/dictionary_verification.md`：ADNI 官方字段、编码、观测范围及关闭原因的逐项核验。
- `reports/experiments/phase1b_semantic_fair_protocol.md`：测试集启用前冻结的三臂公平性、任务、统计比较和停止规则。
- `outputs/phase1b_semantic_fair_20261003/`：18 个 checkpoint、完整 val/test predictions、逐任务指标、2,000 次 subject-cluster bootstrap、报告及门控决定。语义价值门控未通过。

后续必须按规范推进，不能仅填写 gate 的布尔值来替代 validation artifact。Phase 1B 的两项 cross-scale 成对实验和 semantic 公平对照已经完成；双向 interpolation 训练、冻结 representation 临床 probing、完整 calibration/robustness 实验矩阵、MRI grounding 和真实 SFT/GRPO 数据生成仍未完成。任何语义方案改动必须使用新协议和新 run ID，不能用本次测试集继续调参。

源 CSV 和外部 MRI 均未改动。本次仅在 MemReason 目录创建代码、派生表和报告；未创建 Git commit，因为该目录原先不是 Git 仓库。
