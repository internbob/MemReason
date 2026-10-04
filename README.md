# MemReason-RL

本项目依据 `MemReason_RL_ADNI_Implementation_Spec.md` 构建。**已完成真实数据 Phase 0、subject-wise split、可运行的认知建模 v0、Phase 1A、扩展正式基线套件，以及 Phase 1B item 编码核验与 numeric/semantic 公平对照。** Phase 1B 没有通过语义价值门控；MRI、SFT 和 GRPO 模块仍需后续科研验证。

## 当前应该使用的 MRI

主数据源：

```text
/public_bme2/bme-dgshen/WeilinZhou/AD_longitudinal/ADNI/ADNI_Linear
  /sub-ADNI002S0295/ses-M000/t1/affine.nii.gz
```

该目录有 1,981 人、10,230 次纵向 T1，与 `AD_longitudinal/ADNI.csv` 按 subject/session 对应。`surfGT/T1/ADNI/Step01_sMRI_data` 有 1,143 人各一次扫描；旧项目引用的 `Longitudinal/UniSurf/ADNI/Step01_sMRI_data` 有 100 人、523 次扫描。后者只有 2 个 image ID 能与当前 session 表精确对应，不沿用旧项目的 subject/year 推断映射。所有目录清单见 [MRI 比较报告](reports/data_audit/mri_root_comparison.csv)。

**实际 scan_date 和临床 QC 尚未获得。** 同访视匹配能用于数据整理，不代表已验证 90 天内同步。MRI 事件以 `nominal_session_month` 标记；正式 grounding 默认要求真实日期和 QC。

## 已生成的数据

| Split | 受试者 | 量表访视 | 其中有 MRI 的人 | MRI scans | 同月份匹配 | 最近 MRI ±6 月 |
|---|---:|---:|---:|---:|---:|---:|
| train | 3,021 | 10,764 | 1,385 | 7,153 | 6,594 | 6,739 |
| val | 431 | 1,516 | 186 | 973 | 893 | 919 |
| test | 864 | 3,173 | 410 | 2,104 | 1,949 | 2,008 |

seed=2026，按受试者 70/10/20 划分，不使用诊断分层。全部 4,316 人、15,453 次访视保留。MRI 较少的受试者不被删除。原始 161 列保留，CSV 空值维持空值。

- [`splits/train.csv`](splits/train.csv)、[`val.csv`](splits/val.csv)、[`test.csv`](splits/test.csv)：**建议使用的主表**，含全部原始量表与 `image_path`、`scan_id`、`mri_available`、匹配策略、时间来源、phase 等字段。未匹配 MRI 时路径为空。
- `splits/*_longitudinal_mri_subjects.csv`：至少两次认知访视、至少一次 MRI 的受试者的**全部认知访视**。适合纵向多模态分析，不要求每次访视都有 MRI。
- `splits/*_mri_matched.csv`：只有同访视 MRI 匹配的行，用于配对分析；不能取代主队列。
- `splits/*_mri.csv`：该 split 的全部主 MRI，包括没有同访视认知记录的 MRI。
- `splits/*_nearest_mri_month_all.csv`：将每条量表访视分配给该受试者最近的 `ses-Mxxx` MRI 月份；保留所有量表访视、原始月份、有符号月差、绝对月差和匹配等级。它是名义月份归类，不是实际扫描日期匹配。
- `splits/*_nearest_mri_month.csv`：上述表的保守 ±6 月子集。月份完全相同的严格子集仍使用 `*_mri_matched.csv`。
- `splits/*_events.csv`：异步 instrument/MRI 事件。`payload_json` 是观测证据，`state_payload_json` 仅为允许的认知输入候选。
- `splits/*_forecast.csv`：6/12/24/36 月真实未来目标索引；每条明确 cutoff 和 target event IDs。按 instrument 选 tolerance 内最近一次实际测量。任何 target 都严格晚于 cutoff。
- [`splits/subjects.csv`](splits/subjects.csv)：唯一 subject split 注册表。**不要再对任一派生 CSV 按行随机拆分。**

`processed/mri_metadata.csv` 保存主扫描及 alternate 元数据。`processed/events.csv` 保存全事件索引。时间原点只由该受试者最早名义访视确定，不使用未来随访日期估计基线。`train_normalization.json` 仅用训练受试者有效观测拟合。输入 provenance/hash 在 `reports/data_audit/provenance.json`。

## 运行

环境已在计算节点检查；使用现有 Python，不安装或修改全局环境：

```bash
cd /home_data/home/zhouwl2023/project/MemReason
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN -m memreason_rl.data.prepare'
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN -m pytest -q'
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN -m memreason_rl.train.stage1_cognition --smoke --exploratory'
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN scripts/verify_build.py'
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN -m memreason_rl.eval.baselines --split val --exploratory'
# 已冻结的 Phase 1A 正式实验；需要 GPU
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase1_numeric'
# 已冻结的 Phase 1B 正式公平对照；需要 GPU 和本地 Bio_ClinicalBERT
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase1b_semantic'
# 扩展正式基线：LOCF/LME/GRU/LSTM/T-LSTM/GRU-D/Transformer
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase1_baseline_suite'
```

wrapper 复用已有 Slurm allocation，CPU 检查隐藏 GPU。可用 `GPU_JOB_ID=...` 指定 allocation。普通环境可以直接执行 `python -m ...`；依赖定义于 `pyproject.toml`。

重建派生文件会覆盖当前 `processed/`、`splits/` 和数据审计，**不会修改原始输入**。item dictionary 已存在时不会覆盖人工核验。

## 模型与训练接口

- `data/dataset.py`：可变长事件历史、真实未来目标、item/instrument masking。填充位置带显式 mask，隐藏值不进入文本或 numeric branch。
- `models/cognitive_encoder.py`：numeric branch + 可选冻结本地 Hugging Face text encoder，通过门控融合。未指定 `--text-model` 时明确是 numeric baseline，不会伪装成预训练语义模型。
- `domain_aggregator.py`：Memory 为中心，其他认知域作条件。FAQ、诊断、总分不进入 state；FAQ/总分独立作预测目标。
- `temporal_dynamics.py`、`belief_state.py`：按真实/明确标记的名义间隔编码的因果 event Transformer、Gaussian belief、连续时间转移与 evidence update 接口。当前 v0 使用历史 Transformer posterior；独立逐事件 Gaussian correction API 还未做效果验证。
- `mri_encoder.py`、`cognitive_anatomy_grounder.py`：3D patch ViT 输出 spatial tokens，认知 query 对空间 token 做 cross-attention，保留 attention。随机初始化；尚未训练或证明医学解剖含义。
- `decoders.py`、`losses/`：binary BCE、有序累积阈值、未知顺序 categorical CE、训练统计归一化的 continuous Huber/NLL；不存在的 target 不生成 loss。
- `trajectory_generator.py`：从 belief 采样多候选 latent trajectory。
- `reasoning_llm.py`、`rl/`：JSON schema、历史证据验证、真实时间趋势、dictionary domain 验证、availability-aware reward、冻结主干的 clipped GRPO 目标；LLM 使用用户指定本地模型和 LoRA。
- `train/stage1_cognition.py` 和 `stage3_dynamics.py` 是 CLI；stage2、stage4 提供 gated epoch API，需要验证后的 matched batch/SFT/rollout loader。尚未选择 LLM 或生成真实 SFT/GRPO rollouts。

科学训练入口要求 `reports/experiments/stage*_entry_gate.json` 中对应 criteria 和真实 validation artifacts。`--smoke --exploratory` 只做少量梯度检查，状态明确记录 `engineering_smoke_only=true`，不推进任何 gate。主配置文件在 `configs/`；当前 CLI 超参数以 `--help` 为准，阶段 YAML 为实验参数记录模板。

## 第一个正式实验：Phase 1A numeric forecasting

预注册协议见 [`reports/experiments/phase1_numeric_formal_protocol.md`](reports/experiments/phase1_numeric_formal_protocol.md)，固定配置见 [`configs/phase1_numeric_formal.yaml`](configs/phase1_numeric_formal.yaml)。实验仅使用 64 个量表 item 数值、缺失 mask 和名义访视月份，不输入诊断、FAQ item、量表总分、MRI、文本语义或未来信息。预测 ADAS13、CDR-SB、FAQ、MMSE、MoCA 在 6/12/24/36 月的实际后续观测，并比较 LOCF、LME、GRU-D、time-aware Transformer；神经模型使用 2026/2027/2028 三个种子。

正式运行 `phase1_numeric_formal_20261003` 已完成：16,577 / 2,268 / 4,951 个 train/val/test forecast samples。验证集宏平均归一化 MAE 选择 **GRU-D**（0.4057 ± 0.0011；Transformer 0.4185 ± 0.0028）。冻结后一次性测试结果为：GRU-D ensemble 0.3889、Transformer ensemble 0.3950、LOCF 0.3983、LME 0.4193。GRU-D 相对 LME 的 pooled raw MAE 差为 -0.1481，subject-cluster bootstrap 95% CI [-0.2125, -0.0775]；相对 LOCF 为 -0.0682，95% CI [-0.1517, 0.0110]，区间跨 0，因此不能宣称总体显著优于 LOCF。完整逐终点/逐 horizon 指标、预测、诊断和 checkpoint 见 [`outputs/phase1_numeric_formal_20261003/report.md`](outputs/phase1_numeric_formal_20261003/report.md)。

该实验完成了纯数值强基线，不代表 semantic、MRI grounding、latent-state calibration 或 RL 已验证。LOCF 对部分 MoCA horizon 覆盖不足，且 MoCA 6 月仅有 6 个测试目标，不能单独泛化；LME 残差 Shapiro 检验均偏离正态，因此主比较使用 MAE 与受试者聚类 bootstrap，而不依赖正态理论 p 值。

### 扩展正式 baseline suite

在 item 编码核验后，`phase1_baseline_suite_formal_20261004` 使用同一组 68 个已核验单调标量 item，补齐 GRU、LSTM 和 T-LSTM，并统一重跑 GRU-D 与 time-aware Transformer。所有神经模型使用相同训练预算和 2026/2027/2028 三个种子；15 个 checkpoint 完成验证训练后才加载测试集。

| 方法 | 测试 macro normalized MAE | 平均覆盖率 |
|---|---:|---:|
| GRU-D ensemble | **0.3799** | 1.0000 |
| GRU ensemble | 0.3818 | 1.0000 |
| T-LSTM ensemble | 0.3857 | 1.0000 |
| Transformer ensemble | 0.3890 | 1.0000 |
| LSTM ensemble | 0.3903 | 1.0000 |
| LOCF | 0.3983 | 0.9622 |
| LME | 0.4193 | 1.0000 |

验证集选择 GRU-D。GRU-D 相对 LOCF 的 pooled normalized absolute-error 差为 -0.01960，2,000 次 subject-cluster bootstrap 95% CI [-0.03425, -0.00647]；相对 LME 为 -0.03585，95% CI [-0.04617, -0.02519]。完整协议与逐 endpoint/horizon 结果见 [基线协议](reports/experiments/phase1_baseline_suite_formal_protocol.md)和 [基线报告](outputs/phase1_baseline_suite_formal_20261004/report.md)。

## Phase 1B：item 编码核验与 numeric/semantic 公平对照

[`configs/cognitive_item_dictionary.yaml`](configs/cognitive_item_dictionary.yaml) 逐字段记录 domain、construct、取值类型、范围、用途、官方表名、说明、响应编码和核验状态。107 个 ADAS/CDR/FAQ/MMSE/MoCA 项目字段均与 ADNI 官方字段和表名精确匹配；其中 99 个通过观测值与编码核验，8 个 MMSE 拼写字段因官方在线字典缺少可审计响应编码而保持关闭。正式 Phase 1B 使用其中 68 个已核验、可标量比较且单调的字段。MoCA DELW1–5 已核验为 0=错误、1=无提示正确、2=类别提示正确、3=多选提示正确；由于它们是非单调提示类别，本次公平标量对照将其排除。RAVLT `Trial*Int`、原始 MoCA serial/letter 字段继续默认不进入 state。完整证据见 [编码核验报告](reports/data_audit/dictionary_verification.md)。

预注册协议见 [`reports/experiments/phase1b_semantic_fair_protocol.md`](reports/experiments/phase1b_semantic_fair_protocol.md)，固定配置见 [`configs/phase1b_semantic_fair.yaml`](configs/phase1b_semantic_fair.yaml)。三组只改变固定 item descriptor：`numeric_only` 使用全零描述符，`variable_name` 使用字段名 embedding，`task_semantics` 使用已核验任务描述 embedding；三组共享相同 68 个输入、mask、时间、样本、训练流程和 269,325 个可训练参数。描述符由本地冻结的 Bio_ClinicalBERT 一次性编码。每组运行 2026/2027/2028 三个种子，覆盖未来量表预测及两项并发 cross-scale 任务。

正式运行 `phase1b_semantic_fair_20261003` 已完成 18 个训练 checkpoint。测试集宏平均归一化 MAE 如下：

| 任务 | numeric only | variable name | task semantics |
|---|---:|---:|---:|
| ADAS+CDR → MoCA | 0.4261 | 0.4223 | 0.4226 |
| MMSE+CDR → ADAS13 | 0.3684 | 0.3674 | 0.3675 |
| 6/12/24/36 月未来预测 | **0.3766** | 0.3776 | 0.3779 |

相对 numeric-only，task semantics 在两项 cross-scale 上的差值为 -0.00351 和 -0.00098，但 2,000 次 subject-cluster bootstrap 的 95% CI 均跨 0；未来预测差值为 +0.00074，95% CI [0.00024, 0.00127]，即误差小幅上升。因此本协议下的**语义价值门控未通过**，不能声称固定任务描述带来稳定收益，也不据此推进 MRI 或 RL。完整结果见 [Phase 1B 报告](outputs/phase1b_semantic_fair_20261003/report.md)及 [门控决定](outputs/phase1b_semantic_fair_20261003/gate_decision.json)。

后续正式里程碑为：在新协议和新 run ID 下研究 response-aware 语义表示；获取 MRI 实际扫描日期、QC 与配准 provenance；再验证 grounding 增益、interpolation/calibration/robustness、SFT 和真实 rollout GRPO。当前负结果保留为正式实验结论，不使用测试集调参。

数据审计入口：[summary.md](reports/data_audit/summary.md)。完整工程状态：[BUILD_STATUS.md](BUILD_STATUS.md)。
