# MemReason-RL

本项目依据 `MemReason_RL_ADNI_Implementation_Spec.md` 构建。**已完成真实数据 Phase 0、subject-wise split、认知建模 Phase 1A–1F、cognitive-only irregular-time belief/calibration、MRI Phase 3A–3N、reasoning Phase 4A–4F、Phase 5B–5H measurement/reasoning audits、train-only exploratory Phase 6A–6D1 latent-state/MRI audits，以及独立的 Phase 7A native-T1 三分类/Memory 联合训练开发实验。** Phase 4F 已对冻结 GRU-D + Phase 4C 进行一次性官方 test 确认：GRU-D macro normalized MAE 为 0.37510，并显著优于 LOCF；grammar decoder 获得 100% JSON/evidence/forecast/uncertainty preservation 和零未来泄漏。完整 gate 因 nested temporal consistency=0.98990 略低于预注册 0.99000 而失败。Phase 5F/5G 将 held-out-instrument NMAE 降到约 0.42，并把 Memory cross-view CCC 提高到约 0.63，但未达到冻结的 0.70 门槛。Phase 6D1 证明 fold-aligned 128D MRI embedding 相对 cognition-only 和 subject-shuffled embedding 有微小、可复现的静态增益，但相对同一 encoder 的 AD 概率增益未通过预注册门槛，因此 Phase 6D2/6E 未开放。Phase 7A 中联合训练把 macro AUC 从 0.5958 提高到 0.6085、把未来 Memory MAE 从 0.08349 降到 0.08311，但 ACC/macro-F1 降低、MCI sensitivity 仅 0.061，且回归改善只出现在 3/5 folds，未通过冻结联合 gate。当前论文主方法仍是 **GRU-D forecasting + typed executable reasoning IR/compiler/verifier**，LLM 仅为可选渲染器。Phase 4D/4E GRPO 仍不支持增益，多模态 reasoning/GRPO 仍关闭。

GitHub 中保留的去标识化聚合结果见 [`reports/results/`](reports/results/README.md)；模型权重、MRI 数组、缓存、原始数据和受试者级预测均不进入 Git。

## 当前应该使用的 MRI

主数据源：

```text
/public_bme2/bme-dgshen/WeilinZhou/AD_longitudinal/ADNI/ADNI_Linear
  /sub-ADNI002S0295/ses-M000/t1/affine.nii.gz
```

该目录有 1,981 人、10,230 次纵向 T1，与 `AD_longitudinal/ADNI.csv` 按 subject/session 对应。`surfGT/T1/ADNI/Step01_sMRI_data` 有 1,143 人各一次扫描；旧项目引用的 `Longitudinal/UniSurf/ADNI/Step01_sMRI_data` 有 100 人、523 次扫描。后者只有 2 个 image ID 能与当前 session 表精确对应，不沿用旧项目的 subject/year 推断映射。所有目录清单见 [MRI 比较报告](reports/data_audit/mri_root_comparison.csv)。

IDA 导出 `idaSearch_10_12_2026 (2).csv` 已按 **Image ID + subject 双重一致**补入真实 `Study Date`：主目录 10,230 次 MRI 中 9,362 次有真实日期（91.52%），868 次仍缺日期。90 天主协议得到 9,446 条认知–MRI 匹配，其中 9,222 条通过真实日期窗口；详见 [IDA 日期连接审计](reports/data_audit/ida_scan_date_linkage.md)。Phase 3A 已对实际纳入的 6,465 个预处理 MRI 完成可读性、有限值、图像/掩膜维度与 affine、一致掩膜范围和非恒定前景检查，全部通过。该项目技术 QC 足以支持 global-anchor 开发实验，但不等同于 ADNI 临床 QC，也不支持 scanner/field-strength 或区域解剖结论。

## 已生成的数据

| Split | 受试者 | 量表访视 | 其中有 MRI 的人 | MRI scans | MRI 主匹配（90 天） | 最近 MRI ±6 月 |
|---|---:|---:|---:|---:|---:|---:|
| train | 3,021 | 10,764 | 1,385 | 7,153 | 6,600 | 6,739 |
| val | 431 | 1,516 | 186 | 973 | 896 | 919 |
| test | 864 | 3,173 | 410 | 2,104 | 1,950 | 2,008 |

seed=2026，按受试者 70/10/20 划分，不使用诊断分层。全部 4,316 人、15,453 次访视保留。MRI 较少的受试者不被删除。原始 161 列保留，CSV 空值维持空值。

- [`splits/train.csv`](splits/train.csv)、[`val.csv`](splits/val.csv)、[`test.csv`](splits/test.csv)：**建议使用的主表**，含全部原始量表与 `image_path`、`scan_id`、`mri_available`、匹配策略、时间来源、phase 等字段。未匹配 MRI 时路径为空。
- `splits/*_longitudinal_mri_subjects.csv`：至少两次认知访视、至少一次 MRI 的受试者的**全部认知访视**。适合纵向多模态分析，不要求每次访视都有 MRI。
- `splits/*_mri_matched.csv`：90 天主协议的 MRI 匹配行；优先同 visit-code，并为无同 visit-code 的访视补充真实日期近邻。使用 `date_window_verified` 区分真实日期窗口；不能取代主队列。
- `splits/*_mri.csv`：该 split 的全部主 MRI，包括没有同访视认知记录的 MRI。
- `splits/*_nearest_mri_month_all.csv`：将每条量表访视分配给该受试者最近的 `ses-Mxxx` MRI 月份；保留所有量表访视、原始月份、有符号月差、绝对月差和匹配等级。它是名义月份归类，不是实际扫描日期匹配。
- `splits/*_nearest_mri_month.csv`：上述表的保守 ±6 月子集。月份完全相同的严格子集仍使用 `*_mri_matched.csv`。
- `splits/*_events.csv`：异步 instrument/MRI 事件。`payload_json` 是观测证据，`state_payload_json` 仅为允许的认知输入候选。
- `splits/*_forecast.csv`：6/12/24/36 月真实未来目标索引；每条明确 cutoff 和 target event IDs。按 instrument 选 tolerance 内最近一次实际测量。任何 target 都严格晚于 cutoff。
- [`splits/subjects.csv`](splits/subjects.csv)：唯一 subject split 注册表。**不要再对任一派生 CSV 按行随机拆分。**

`processed/mri_metadata.csv` 保存主扫描、alternate 元数据、IDA Study Date 来源及序列描述。`processed/events.csv` 保存全事件索引；有 IDA 日期的 MRI 使用真实日期相对认知时间原点定位，剩余 868 次明确标为 `nominal_session_month`。时间原点不使用未来随访日期估计基线。`train_normalization.json` 仅用训练受试者有效观测拟合。输入 provenance/hash 在 `reports/data_audit/provenance.json`。

## 更新 GitHub

完成新的代码或实验后，在项目根目录执行：

```bash
bash scripts/git_update.sh "Describe the code or experiment update"
```

该脚本会拉取远端状态、导出去标识化的聚合结果、运行测试、按公开文件白名单暂存、检查大文件和常见凭据、创建 commit，并推送到 `origin`。它不会上传 `outputs/`、`processed/`、`splits/`、模型权重、NIfTI、IDA 下载文件或 ADNI 行级 CSV。公开结果位于 [`reports/results/`](reports/results/README.md)。可以临时设置 `SKIP_TESTS=1` 跳过测试，但正式结果更新建议保留默认测试。

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
# 联合 Stage I：masked item/instrument、cross-scale、future forecasting
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase1c_joint_stage1'
# 独立核验 9 个 checkpoint、train-only response vocab、预测和门控
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN scripts/verify_phase1c_joint_stage1.py'
# cognitive-only interpolation/extrapolation、probabilistic belief 与 calibration；需要 GPU
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase2_cognitive_belief'
# 独立核验 fit-only normalization、受试者隔离、ensemble 方差、temperature 与 gate
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN scripts/verify_phase2_cognitive_belief.py'
# validation-only MRI global anchor 与两类 shuffled negative controls；需要 GPU
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase3a_global_mri_anchor'
# 独立核验 MRI QC、样本配对、时间顺序、12 个 checkpoint 与 gate
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN scripts/verify_phase3a_global_mri_anchor.py'
# MRI 表示验证：demographics/global MRI/3D SFCN、subject shuffle 与未来变化量
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase3b_mri_representation'
# 独立核验 3D cache、9 个 checkpoint、ensemble、日期及双路径决定
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN scripts/verify_phase3b_mri_representation.py'
# 冻结 SFCN 的 concurrent target-to-spatial attention；需要 GPU
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase3c_spatial_grounding'
# 独立核验参数匹配、三类负对照、attention normalization/stability 与 gate
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN scripts/verify_phase3c_spatial_grounding.py'
# verified item/query + frozen MRI spatial tokens 的同期 masked reconstruction；需要 GPU
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase3d_multimodal_grounding'
# 独立核验 masked-value 隔离、9 个 checkpoint、ensemble、负对照与 gate
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN scripts/verify_phase3d_multimodal_grounding.py'
# 一次性冻结 Phase 3C heads，在 subject-disjoint test 上确认；无训练
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase3e_test_confirmation'
# 独立核验 test 隔离、冻结 checkpoint、ensemble、attention 与确认 gate
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN scripts/verify_phase3e_test_confirmation.py'
# 只用原 train subjects 的五折 diagnosis-heterogeneity cross-fit；需要 GPU
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase3f_diagnosis_crossfit'
# 独立核验 fold 隔离、30 个 checkpoint、OOF 预测、交互与 attention
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN scripts/verify_phase3f_diagnosis_crossfit.py'
# train-only frozen GRU-D + deterministic five-section structured reasoning；需要 GPU
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4a_structured_reasoning'
# 独立重算 normalization、checkpoint、16,577 份 JSON、证据和 forecast preservation
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN scripts/verify_phase4a_structured_reasoning.py'
# Phase 4B subject-balanced completion-only LoRA SFT；按 prepare/train/evaluate/summarize 顺序
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4b_reasoning_sft --stage prepare'
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4b_reasoning_sft --stage train'
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4b_reasoning_sft --stage evaluate-base'
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4b_reasoning_sft --stage evaluate-sft'
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN -m memreason_rl.experiments.phase4b_reasoning_sft --stage summarize'
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN scripts/verify_phase4b_reasoning_sft.py'
# Phase 4C：prepare 后依次在 dev 冻结 decoder，再打开新受试者 confirmation
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4c_grammar_decoding --stage prepare'
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4c_grammar_decoding --stage dev-raw'
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4c_grammar_decoding --stage dev-grammar'
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN -m memreason_rl.experiments.phase4c_grammar_decoding --stage select-dev'
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4c_grammar_decoding --stage confirm-raw'
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4c_grammar_decoding --stage confirm-selected'
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN -m memreason_rl.experiments.phase4c_grammar_decoding --stage summarize'
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN scripts/verify_phase4c_grammar_decoding.py'
# Phase 4D：真实 on-policy constrained GRPO；dev gate 失败时 finalize-dev，不加载 test
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4d_constrained_grpo --stage prepare-train'
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4d_constrained_grpo --stage train'
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4d_constrained_grpo --stage evaluate-dev'
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN -m memreason_rl.experiments.phase4d_constrained_grpo --stage finalize-dev'
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN scripts/verify_phase4d_constrained_grpo.py'
# Phase 4E：有界 trajectory action SFT + outcome-varying on-policy GRPO
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4e_trajectory_action_grpo --stage prepare'
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4e_trajectory_action_grpo --stage train-sft'
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4e_trajectory_action_grpo --stage train-grpo'
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4e_trajectory_action_grpo --stage evaluate-dev'
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN -m memreason_rl.experiments.phase4e_trajectory_action_grpo --stage finalize'
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN scripts/verify_phase4e_trajectory_action_grpo.py'
# Phase 4F：一次性冻结 GRU-D + Phase 4C 官方 test confirmation；无训练
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4f_frozen_test --stage prepare'
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4f_frozen_test --stage raw'
bash run-on-compute.sh '$PYTHON_BIN -m memreason_rl.experiments.phase4f_frozen_test --stage grammar'
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN -m memreason_rl.experiments.phase4f_frozen_test --stage summarize'
bash run-on-compute.sh 'CUDA_VISIBLE_DEVICES= $PYTHON_BIN scripts/verify_phase4f_frozen_test.py'
```

wrapper 复用已有 Slurm allocation，CPU 检查隐藏 GPU。可用 `GPU_JOB_ID=...` 指定 allocation。普通环境可以直接执行 `python -m ...`；依赖定义于 `pyproject.toml`。

重建派生文件会覆盖当前 `processed/`、`splits/` 和数据审计，**不会修改原始输入**。item dictionary 已存在时不会覆盖人工核验。

## 模型与训练接口

- `data/dataset.py`：可变长事件历史、真实未来目标、item/instrument masking。填充位置带显式 mask，隐藏值不进入文本或 numeric branch。
- `models/cognitive_encoder.py`：numeric branch + 可选冻结本地 Hugging Face text encoder，通过门控融合。未指定 `--text-model` 时明确是 numeric baseline，不会伪装成预训练语义模型。
- `domain_aggregator.py`：Memory 为中心，其他认知域作条件。FAQ、诊断、总分不进入 state；FAQ/总分独立作预测目标。
- `temporal_dynamics.py`、`belief_state.py`：按真实/明确标记的名义间隔编码的因果 event Transformer、Gaussian belief、连续时间转移与 evidence update 接口。Phase 2 另用冻结协议验证了逐事件 Gaussian predict-update、双向 interpolation product-of-experts 与外推；其显式 belief 门槛未通过。
- `mri_encoder.py`、`cognitive_anatomy_grounder.py`：提供 3D spatial token 与 cognitive-query cross-attention 接口。Phase 3C/3D 使用冻结 SFCN spatial tokens 完成了总分与 item-query 实验；attention 仅作 predictive grounding，不证明因果解剖含义。
- `decoders.py`、`losses/`：binary BCE、有序累积阈值、未知顺序 categorical CE、训练统计归一化的 continuous Huber/NLL；不存在的 target 不生成 loss。
- `trajectory_generator.py`：从 belief 采样多候选 latent trajectory。
- `reasoning_llm.py`、`rl/`：JSON schema、历史证据验证、真实时间趋势、dictionary domain 验证、availability-aware reward、冻结主干的 clipped GRPO 目标；LLM 使用用户指定本地模型和 LoRA。
- `train/stage1_cognition.py` 和 `stage3_dynamics.py` 是 CLI；stage2、stage4 提供 gated epoch API。Phase 4B 使用本地 DeepSeek-R1-Distill-Qwen-1.5B 训练 rank-16 LoRA，Phase 4C 冻结 token-trie decoder 后通过新受试者确认。Phase 4D/4E 均生成真实 on-policy rollouts并更新 LoRA，但各自 development gate 均未超过冻结基线。Phase 4F 不训练模型，在官方 test 上一次性确认冻结 GRU-D + Phase 4C。

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

## Phase 1C：response-aware 联合 Stage I

冻结协议见 [Phase 1C protocol](reports/experiments/phase1c_joint_stage1_protocol.md)，配置见 [`configs/phase1c_joint_stage1.yaml`](configs/phase1c_joint_stage1.yaml)。`numeric_only`、`task_semantics`、`response_aware` 三臂使用相同的 68 个输入、GRU-D、domain aggregation、联合任务采样和每臂 290,993 个可训练参数。response-aware descriptor 包含已核验任务、实际响应/分数、允许范围或类别标签以及分数方向；本地 Bio_ClinicalBERT 保持冻结。每臂使用 2026/2027/2028 三个种子，共生成 9 个 checkpoint。

三种子 ensemble 的测试 macro normalized MAE 为：

| 任务 | numeric only | task semantics | response aware |
|---|---:|---:|---:|
| ADAS+CDR → MoCA | **0.4213** | 0.4236 | 0.4236 |
| MMSE+CDR → ADAS13 | 0.3683 | 0.3663 | **0.3663** |
| 6/12/24/36 月未来预测 | **0.3804** | 0.3826 | 0.3828 |
| masked instrument | **0.3678** | 0.3746 | 0.3746 |
| masked item | **0.3423** | 0.3509 | 0.3507 |

response-aware 相对 numeric-only 只在 `MMSE+CDR → ADAS13` 上达到显著改善（差 -0.00194，95% CI [-0.00343, -0.00042]）。`ADAS+CDR → MoCA` 的区间跨 0；masked item 和 masked instrument 的误差显著增加；future forecasting 的差为 +0.00284，95% CI [-0.00188, 0.00736]，上界超过 +0.005 的预设非劣界值。因此 masked、双 cross-scale 和 future 三类条件均未满足，Stage I gate 为 **false**，不据此进入 MRI Stage II。完整结果见 [Phase 1C 报告](outputs/phase1c_joint_stage1_20261004/report.md)、[门控决定](outputs/phase1c_joint_stage1_20261004/gate_decision.json)和[独立产物核验](outputs/phase1c_joint_stage1_20261004/verification.json)。

该 Stage I 门控失败结论保持不变，不能用于支持 semantic 或 MRI reasoning。后续另行冻结的 cognitive-only 路线已完成 dynamics probing 和 Phase 4A deterministic structured reports，并只开放 Phase 4B cognitive-only SFT；它不改变 Stage I/MRI 的负结果，也不开放真实 rollout GRPO。

## Phase 1D–1F：validation-only 表示学习开发

Phase 1D–1F 均只读取 train/validation，未加载 test。依次实现了 typed response encoder、item-conditioned decoder、binary/ordinal/continuous 类型损失、per-item macro reconstruction loss、零初始化 semantic residual gate，以及按任务独立的 value/semantic gates。所有对照保持参数匹配。

| 开发实验 | Numeric validation macro | 最佳 semantic macro | 主要结果 |
|---|---:|---:|---|
| Phase 1D typed/item-conditioned | 0.39445 | 0.39264 | ADAS+CDR→MoCA 显著改善；联合门控失败 |
| Phase 1E gated + metric-aligned | 0.38826 | 0.38339 | masked item 显著改善；其余 CI 未满足 |
| Phase 1F task-specific gates | 0.38826 | **0.38334** | masked item 显著；masked instrument/future 点估计改善但 CI 未过界；cross-scale 未改善 |

Phase 1F 是当前 validation split 上预先限定的最后一次表示修订。其 future 差为 -0.00447，95% CI [-0.01457, 0.00623]；masked instrument 差为 -0.00284，95% CI [-0.01127, 0.00453]；两项 cross-scale 的区间均跨 0。因此不再在同一 validation 上继续调 semantic architecture，也不据此开放 MRI Stage II。完整报告见 [Phase 1D](outputs/phase1d_typed_representation_dev_20261004/report.md)、[Phase 1E](outputs/phase1e_gated_aligned_dev_20261009/report.md)、[Phase 1F](outputs/phase1f_task_gated_dev_20261009/report.md)和[终止决定](reports/experiments/phase1_representation_terminal_decision.md)。后续已按新协议完成 cognitive-only dynamics/calibration 实验，结果如下。

## Cognitive-only irregular-time belief dynamics 与 calibration

冻结协议见 [Phase 2 protocol](reports/experiments/phase2_cognitive_belief_dev_protocol.md)，固定配置见 [`configs/phase2_cognitive_belief_dev.yaml`](configs/phase2_cognitive_belief_dev.yaml)。实验不再宣称普适 semantic superiority，而将 verified typed item/response encoder 作为组件。官方 train 受试者进一步隔离为 fit/tune，validation 受试者隔离为 calibration/evaluation；normalization 只由 fit 受试者拟合，test 从未加载。三臂各运行 2026/2027/2028 三个种子，联合覆盖隐藏完整中间访视的 interpolation 与 6/12/24/36 月 extrapolation。

| 三种子 ensemble | normalized MAE | calibrated NLL | calibrated CRPS | 95% coverage | 多水平校准误差 |
|---|---:|---:|---:|---:|---:|
| GRU-D Gaussian | 0.3715 | 0.6163 | 0.2697 | 0.9468 | 0.0291 |
| typed direct Gaussian | **0.3609** | **0.6061** | **0.2659** | 0.9387 | 0.0350 |
| typed predict-update belief | 0.3813 | 0.6919 | 0.2805 | 0.9316 | **0.0286** |

显式 typed belief 达到预设绝对 calibration 门槛，但相对 GRU-D 的 overall normalized-MAE 差为 +0.00983（95% CI [-0.00074, 0.02187]），NLL 差为 +0.07564（95% CI [0.01566, 0.13653]），因此冻结 gate 为 **false**。typed direct Gaussian 的总体点估计最好，但 overall MAE 差的 95% CI 为 [-0.03355, 0.00577]，仍不能声称稳定总体优势；它在 interpolation 上的 MAE 改善为 -0.01437（95% CI [-0.02971, -0.00419]），保留为任务特异组件候选。GRU-D 继续作为确认后的认知主模型，MRI 仍不开放。完整结果与复核见 [实验报告](outputs/phase2_cognitive_belief_dev_20261010/report.md)、[门控决定](outputs/phase2_cognitive_belief_dev_20261010/development_decision.json)和[独立核验](outputs/phase2_cognitive_belief_dev_20261010/verification.json)。

## MRI Phase 3A：global anchor validation-only 开发实验

冻结协议见 [Phase 3A protocol](reports/experiments/phase3a_global_mri_anchor_dev_protocol.md)，配置见 [`configs/phase3a_global_mri_anchor_dev.yaml`](configs/phase3a_global_mri_anchor_dev.yaml)。实验只读取 train/validation，并要求 MRI 在真实日期 90 天窗口内匹配 cutoff visit、每个预测目标的实际量表日期严格晚于 MRI 日期。四臂 `cognition_only`、`observed_mri`、`shuffled_subject_mri` 和 `shuffled_time_mri` 使用相同样本、结构、402,958 个参数及三个随机种子。MRI 表示由全脑体积/形状及粗网格 mask/强度统计构成；这是全局信息价值测试，不是区域解剖 grounding。

正式运行纳入 13,775/1,877 个 train/validation forecast samples、1,349/185 名受试者和 6,465 个唯一 MRI。项目技术 QC 通过率 100%，validation 每臂有 7,580 个目标。三种子 ensemble macro normalized MAE 为：cognition-only **0.3795**、shuffled-subject 0.3867、observed MRI 0.3954、shuffled-time 0.3975。Observed MRI 相对 cognition-only 的配对 normalized absolute-error 差为 **+0.01679**，95% CI [0.00511, 0.02951]，即显著更差；12 月和 24 月差也均为正且区间下界大于 0。相对 shuffled-subject 和 shuffled-time 均未形成稳定优势。因此 global MRI-anchor gate 为 **false**，不在同一 validation 上继续调该融合结构；后续 Phase 3B 通过单独冻结的表示协议评估并开放了同期 grounding。完整结果见 [Phase 3A 报告](outputs/phase3a_global_mri_anchor_dev_20261013/report.md)和[独立核验脚本](scripts/verify_phase3a_global_mri_anchor.py)。

## MRI Phase 3B：representation validation

Phase 3B 将 MRI 表示能力与纵向融合拆开验证。冻结协议见 [Phase 3B protocol](reports/experiments/phase3b_mri_representation_dev_protocol.md)，配置见 [`configs/phase3b_mri_representation_dev.yaml`](configs/phase3b_mri_representation_dev.yaml)。正式实验使用 5,684/780 个 train/validation MRI anchors、1,349/185 名受试者；从原始对齐 MRI 构建 48×56×48 masked volume cache。`demographics_only`、`global_mri` 和 `sfcn_3d` 各运行三个种子，MRI 模型同时接受错受试者 MRI 的冻结 checkpoint 负对照。

同期认知 validation macro normalized MAE 为：SFCN **0.6031**、global MRI 0.6736、demographics 0.7466、shuffled global MRI 0.8396、shuffled SFCN 0.8521。SFCN 相对 demographics 的配对误差差为 **-0.16736**，95% CI [-0.23183, -0.11306]；相对 shuffled SFCN 为 **-0.26002**，95% CI [-0.31901, -0.20226]。ADAS13、CDR-SB、FAQ 和 MMSE 均显著改善，MoCA 区间跨零。Global MRI 也通过同期信号门槛。因此 `phase3b_concurrent_grounding_ready=true`，可以进入新的同期 spatial-grounding 协议。

未来变化量结果不同：current score + age + sex control 的 normalized MAE 为 **0.4648**，加入 SFCN 为 0.5187，加入 global MRI 为 0.5489。SFCN 相对 control 的 pooled 差为 +0.02882，95% CI [0.00026, 0.05972]；global MRI 为 +0.03740，95% CI [0.01745, 0.06002]。因此 `phase3b_longitudinal_fusion_ready=false`。当前允许研究同期 question-to-anatomy grounding，但不能声称 MRI 改善未来认知变化预测。完整结果和复核见 [Phase 3B 报告](outputs/phase3b_mri_representation_dev_20261014/report.md)与[verification](outputs/phase3b_mri_representation_dev_20261014/verification.json)。

## MRI Phase 3C：concurrent question-to-spatial grounding

冻结协议见 [Phase 3C protocol](reports/experiments/phase3c_concurrent_spatial_grounding_dev_protocol.md)，配置见 [`configs/phase3c_concurrent_spatial_grounding_dev.yaml`](configs/phase3c_concurrent_spatial_grounding_dev.yaml)。实验冻结 Phase 3B 三套 SFCN 的前四个卷积块，为每个 MRI 提取 3×4×3、64 通道空间 tokens。`global_pool` 与 `target_spatial_attention` head 均为 76,673 个可训练参数；attention 模型另外接受错受试者、特征—坐标空间置乱和量表 query 置乱测试。

三种子 ensemble normalized MAE 为：target spatial attention **0.5994**、global pooling 0.6281、spatial permutation 0.6626、query permutation 0.8054、shuffled-subject attention 0.8630。Attention 相对 global pooling 的配对差为 **-0.02219**，95% CI [-0.04346, -0.00140]；相对 spatial permutation 为 -0.06739，95% CI [-0.09171, -0.04247]；相对 query permutation 为 -0.24824，95% CI [-0.30940, -0.19596]。FAQ 与 MoCA 相对 global pooling 显著改善，达到两终点门槛。

跨 seed attention 的中位 Spearman 为 **0.3956**，top-20% token overlap 为 **0.50**，均通过冻结稳定性标准。因此 `concurrent_spatial_grounding_passed=true`，下一步可以冻结 multimodal concurrent grounding 协议。当前 attention center 只报告对齐模板毫米坐标；在没有经审计 atlas/segmentation 前不赋予具体脑区名称，也不外推为纵向 MRI 增益。完整结果见 [Phase 3C 报告](outputs/phase3c_concurrent_spatial_grounding_dev_20261014/report.md)和[独立核验](outputs/phase3c_concurrent_spatial_grounding_dev_20261014/verification.json)。

## MRI Phase 3D：multimodal concurrent item grounding

冻结协议见 [Phase 3D protocol](reports/experiments/phase3d_multimodal_item_grounding_dev_protocol.md)，配置见 [`configs/phase3d_multimodal_item_grounding_dev.yaml`](configs/phase3d_multimodal_item_grounding_dev.yaml)。实验沿用 5,684/780 个 train/validation MRI anchors、三套冻结 36×64 SFCN tokens 和 68 个已核验 item descriptor。它生成 22,728/3,120 个 masked-item/masked-instrument 样本，比较等参的 `cognition_only`、`global_mri` 与 `item_spatial_attention`（每臂 473,217 参数），每臂运行三个 seed。被 mask 的 response 在输入张量中显式置零，并由自动测试和独立 verifier 复核；test 未加载。

三种子 ensemble 的 masked-item normalized MAE 为：cognition-only **0.3787**、global MRI 0.3835、item spatial attention 0.3841；masked-instrument 分别为 **0.3696**、0.3741、0.3776。Spatial attention 相对 cognition-only 的 overall 配对差为 **+0.01106**，95% CI [0.00618, 0.01633]；相对 global MRI 为 **+0.00712**，95% CI [0.00291, 0.01231]，均显著更差。

正确 MRI 仍显著优于错受试者、空间置乱和 query 置乱对照，说明模型确实使用了 subject、spatial 和 query 信息；attention entropy 0.6619 也未塌缩。但跨 seed attention 中位 Spearman 仅 **0.0185**，top-20% overlap 为 **0.25**，未达到 0.25/0.35 的稳定性标准。故 `multimodal_item_grounding_passed=false`：保留 Phase 3C 的总分同期 grounding，当前不主张 68-item 级 multimodal grounding，也不据此开放纵向 MRI fusion、multimodal SFT 或 GRPO。完整结果见 [Phase 3D 报告](outputs/phase3d_multimodal_item_grounding_dev_20261014/report.md)和[独立核验](outputs/phase3d_multimodal_item_grounding_dev_20261014/verification.json)。

## MRI Phase 3E：frozen sealed-test confirmation

Phase 3E 在加载 test 之前冻结[确认协议](reports/experiments/phase3e_frozen_spatial_grounding_test_protocol.md)，只确认 Phase 3C；Phase 3D 因 validation gate 失败未进入 test。实验纳入 404 名 test subjects、1,905 个唯一 MRI 和 7,843 个同期量表目标，复用 Phase 3B train-only normalization、三套冻结 SFCN 和 Phase 3C 六个 head checkpoints，全程无训练、无 checkpoint 选择。

Test ensemble normalized MAE 为 spatial attention **0.6351**、global pooling 0.6454，点估计方向与 Phase 3C 一致。但配对差 **-0.01088** 的 95% CI 为 **[-0.02522, 0.00427]**，总体区间跨零；五个终点中只有 MMSE 显著改善，未达到至少两个终点的冻结标准。另一方面，spatial attention 仍显著优于错受试者（0.8913）、空间置乱（0.6840）和 query 置乱（0.8242），跨 seed attention Spearman **0.4018**、top-20% overlap **0.50**，说明 subject/spatial/query 依赖和稳定性得到复制。

因此 `phase3c_test_confirmed=false`。Phase 3C 现在只能作为有一致点估计和机制对照支持的 validation-only finding，不能作为已确认总体增益。探索性亚组显示 AD 显著受益、MCI 显著变差，提示明显诊断异质性；该发现不能使用当前 test 调参，后续需在新的 development/confirmation 划分或外部队列中预注册 diagnosis-conditioned 协议。完整结果见 [Phase 3E 报告](outputs/phase3e_frozen_spatial_grounding_test_20261014/report.md)和[独立核验](outputs/phase3e_frozen_spatial_grounding_test_20261014/verification.json)。

## MRI Phase 3F：train-only diagnosis-heterogeneity cross-fit

Phase 3F 在[冻结协议](reports/experiments/phase3f_diagnosis_heterogeneity_crossfit_protocol.md)下，只使用原 train split 的 1,349 名受试者、5,684 个 MRI 和 23,392 个同期目标。受试者被固定分入五个互斥 folds；每个 MRI 的预测均来自未见过该受试者的模型。实验复用 Phase 3C 的 36×64 SFCN tokens、架构和优化器，并固定各 arm/seed 的训练轮数，不做 fold 内 early stopping。共完成 2 arms × 3 seeds × 5 folds = 30 个 checkpoints。诊断仅用于预测后的统计，不进入模型输入。

Out-of-fold spatial attention 的 macro normalized MAE 为 **0.4145**，global pooling 为 0.4976。相对 global pooling，AD 的配对 normalized absolute-error 差为 **-0.2593**，95% CI [-0.2896, -0.2310]，并在 5/5 folds 改善；MCI 的差为 **-0.0210**，95% CI [-0.0282, -0.0143]，同样在 5/5 folds 改善，而非 Phase 3E 中的受损。AD-minus-MCI 交互为 **-0.2383**，95% CI [-0.2688, -0.2110]，说明 AD 获益更强，但不支持“MCI 受损”这一原始异质性模式。

三项机制对照均显著变差，cross-fold mean-map Spearman/top-20% overlap 为 0.7454/0.625；逐样本 cross-seed Spearman 为 0.3287，但 top-20% overlap 仅 **0.375**，低于 0.40 门槛。因此 `diagnosis_heterogeneity_replicated=false`。当前不应构建 diagnosis-conditioned 或 mixture-of-experts grounder；下一步优先转向外部队列确认、冻结认知模型的临床 robustness，或结束 MRI grounding 路线并整理论文负结果。完整结果见 [Phase 3F 报告](outputs/phase3f_diagnosis_heterogeneity_crossfit_20261014/report.md)和[独立核验脚本](scripts/verify_phase3f_diagnosis_crossfit.py)。

## MRI Phase 3G：MRI-inclusive concurrent diagnosis

Phase 3G 在[冻结协议](reports/experiments/phase3g_mri_diagnosis_crossfit_protocol.md)下，将真实日期相差不超过 90 天的预处理 T1 MRI 与 DXSUM CN/MCI/AD 标签连接。实验只使用原 train split 的 1,328 名受试者和 5,370 次 MRI-linked diagnostic visits；同一受试者的全部扫描固定在同一外层折。每折以其余受试者训练三个不同 seed 的 3D SFCN，在内部调优子集早停并校准，最终形成 15 个 checkpoints 和完整折外预测。对照包含 demographics、global MRI、SFCN MRI、截止扫描日之前的 cognition history、两种 cognition+MRI，以及 subject-shuffled MRI。该阶段的 SFCN 输入来自 `48×56×48` 三线性插值缓存，因此只作为低分辨率历史基线，不再作为后续主 MRI 结果。

受试者平衡的折外结果显示，SFCN MRI 的 macro-AUC 为 **0.6984**，高于 demographics 的 0.5618；2,000 次 subject-cluster bootstrap 差为 **+0.1365**，95% CI [0.1103, 0.1630]，支持 MRI 本身携带并发诊断信号。cognition-only 的 macro-AUC 为 **0.9739**，而 cognition+SFCN MRI 为 0.9298，配对差 **-0.0442**，95% CI [-0.0513, -0.0372]；balanced accuracy 也从 0.8924 降至 0.7879。正确 MRI 融合优于 shuffled MRI（macro-AUC 差 +0.0142 [0.0061, 0.0224]），说明模型使用了患者对应影像，但当前融合未在强认知基线上产生临床增量。

因此 `mri_concurrent_signal_supported=true`，`mri_increment_beyond_cognition_supported=false`。冻结协议规定只有完整增量 gate 通过才开放 MCI conversion，故本轮没有启动转化预测。这一结论属于 train-only 内部交叉验证，不能表述为外部临床验证。完整结果见 [Phase 3G 报告](outputs/phase3g_mri_diagnosis_crossfit_20261016/report.md)、[门槛决定](outputs/phase3g_mri_diagnosis_crossfit_20261016/development_decision.json)和[独立核验](outputs/phase3g_mri_diagnosis_crossfit_20261016/verification.json)。

## MRI Phase 3H：MCI-to-AD conversion classification

在用户明确要求开展两个分类实验后，Phase 3H 被单独标记为 exploratory follow-up；它不改写 Phase 3G 原冻结 gate 的失败结论。实验只使用 original-train subjects，并为每个时间窗选择每名受试者最早的、结局可判定的 MRI-linked MCI 基线。24 月队列包含 609 人和 128 名 converters，36 月队列包含 554 人和 183 名 converters。阳性为基线后在精确时间窗内首次诊断 AD；阴性必须有跨过完整时间窗的后续 DXSUM 诊断且窗内无 AD，从而排除无法判定的右删失基线。

每个时间窗独立完成五折 subject-wise cross-fitting、三个 SFCN seeds，共 30 个 checkpoints。七个模型臂与 Phase 3G 一致，并使用 0.5 阈值报告 ACC、conversion sensitivity、nonconversion specificity 和 positive-class F1；AUC 为 threshold-free ROC-AUC。主要折外结果如下：

| Horizon | Model | ACC | AUC | SEN | SEC | F1 |
|---|---|---:|---:|---:|---:|---:|
| 24 months | SFCN MRI | 0.7537 | 0.6031 | 0.1016 | 0.9272 | 0.1477 |
| 24 months | Cognitive history | **0.7668** | **0.8117** | **0.7031** | 0.7838 | **0.5590** |
| 24 months | Cognition + SFCN MRI | 0.7931 | 0.7158 | 0.1719 | **0.9584** | 0.2588 |
| 36 months | SFCN MRI | 0.6209 | 0.6057 | 0.2350 | 0.8113 | 0.2905 |
| 36 months | Cognitive history | **0.7690** | **0.8522** | **0.7377** | 0.7844 | **0.6784** |
| 36 months | Cognition + SFCN MRI | 0.6913 | 0.7133 | 0.3607 | **0.8544** | 0.4356 |

MRI-only 相对 demographics 的 AUC 差在 24 月为 +0.0913 [0.0207, 0.1649]，在 36 月为 +0.0554 [-0.0100, 0.1199]。正确 MRI 融合显著优于 shuffled-MRI fusion，但融合相对 cognition-only 在 24/36 月分别降低 AUC 0.0958 和 0.1390。因此 MRI 包含有限预后信号，但当前融合没有提供认知历史之外的增量。与 Phase 3G 相同，该阶段读取 `48×56×48` 缓存，结果保留为低分辨率探索性基线。完整结果见 [Phase 3H 报告](outputs/phase3h_mci_conversion_crossfit_20261017/report.md)、[联合指标表](outputs/phase3h_mci_conversion_crossfit_20261017/requested_classification_metrics.csv)、[论文图](outputs/phase3h_mci_conversion_crossfit_20261017/figures/fig_classification_metrics.pdf)和[独立核验](outputs/phase3h_mci_conversion_crossfit_20261017/verification.json)。

## MRI Phase 3I：native-resolution AD-vs-CN sanity benchmark

根据“不使用降采样 MRI”的更新要求，Phase 3I 使用完整 `193×229×193`、1 mm 各向同性 affine T1。对全部 6,464 个候选 MRI 的头信息审计确认形状、体素间距、mask 网格和 affine 完全一致，因此本阶段不做空间插值、裁剪或填充；float16 只用于缓存存储。冻结协议使用 original-train subjects、每人最早一张合格 MRI、五折 subject-wise cross-fitting，并报告 ACC/AUC/SEN/SEC/F1。896 名受试者（CN 503、AD 393）的折外结果中，native MRI 的 ACC/AUC/SEN/SEC/F1 为 0.7221/0.7986/0.7074/0.7336/0.6907；加入 demographics 后 AUC 为 0.7997，95% bootstrap CI [0.7701, 0.8289]。它比 demographics AUC 0.6339 明显更强，但比预设 0.8000 自动扩展门槛低 0.0003，因此严格 gate=false，不自动打开全分辨率三分类与 MCI conversion。详见 [Phase 3I protocol](reports/experiments/phase3i_native_resolution_adcn_protocol.md)与 [Phase 3I report](outputs/phase3i_native_resolution_adcn_20261017/report.md)。

## 后续 MRI 输入规范

Phase 3M 对固定中心裁剪进行了受试者、标签、五折、网络和预算完全匹配的敏感性试验。裁剪将每个体积的体素数减少约 42.4%，且 896 张审计扫描至少保留 98.44% 的脑掩膜；然而 AUC 非劣性区间下界未达到预设门槛。因此正式新实验暂时保留完整 `193×229×193` 输入，下一步验证由脑掩膜定义的 adaptive bounding box 后再决定是否缩小输入。标准固定裁剪加载器仍保留在 `memreason_rl/data/mri.py`，但不再作为正式默认方案。

## MRI Phase 3J：native longitudinal gated fusion

Phase 3J 已完成。冻结的 Phase 3I 五模型对 960 名受试者的 5,589 次 real-date、past-only MRI 生成结构负担轨迹；未来认知评价包含 185 名 MRI-linked subjects 和 8,718 个 target rows。longitudinal gated fusion 的 macro normalized MAE 为 **0.39572**，冻结 cognition 为 0.39691；subject-level 配对差为 **-0.00169**，95% CI [-0.00296, -0.00047]。该效应低于预设 0.005 最小改善，且相对 shuffled-MRI gated fusion 的差为 -0.00051，95% CI [-0.00117, 0.00011]。稀疏认知历史亚组区间也跨 0。

MCI-to-AD progression 中，24/36 月 cognition AUC 为 **0.8800/0.8669**，longitudinal gated fusion 为 0.8563/0.8589；相对 cognition 的差分别为 -0.02370 和 -0.00806。因而 `phase3j_gate_passed=false`、`open_multimodal_reasoning=false`。本阶段说明 MRI 具有独立结构信号，但尚未证明其在强认知历史基础上的稳定增量。完整结果见 [Phase 3J report](outputs/phase3j_native_longitudinal_gated_fusion_20261017/report.md)和[开发决策](outputs/phase3j_native_longitudinal_gated_fusion_20261017/development_decision.json)。

## MRI Phase 3K/3L：center-cropped T1 classification

该实验原先生成了 15 个三分类和 30 个 MCI 转化 checkpoints。Phase 3M 的逐体积一致性审计随后发现，旧裁剪缓存中 896 个匹配扫描只有 248 个非空，648 个为全零；旧审计只检查首尾缓存行，未发现中间损坏。因此 Phase 3K/3L 的 MRI-only、融合、shuffled-MRI 指标和 bootstrap 结论全部失效，不得用于论文。cognition-only 分支没有读取该缓存，数值本身不受影响。缓存构建器现已改为核验每一个体积；三分类四臂已由 Phase 3N 使用新缓存重算，MCI 转化仍待使用正确缓存重算。

## MRI Phase 3M：matched preprocessing sensitivity

[冻结协议](reports/experiments/phase3m_mri_preprocessing_sensitivity_protocol.md)复用 Phase 3I 的 896 名 original-train AD/CN 受试者、每人最早一次扫描、标签、五个 outer folds、inner tuning rule、SFCN、优化器、训练预算和种子，仅替换 MRI 输入。新建的 896-scan 裁剪缓存逐行核验全部非空，并抽样确认其与完整规范化 MRI 的中心子数组逐体素完全一致。官方 validation 和 test 均未加载。

完整网格、中心裁剪和低分辨率 MRI-only 的 ACC/AUC/SEN/SEC/F1 分别为 **0.7221/0.7986/0.7074/0.7336/0.6907**、**0.7243/0.7960/0.7023/0.7416/0.6909** 和 **0.7667/0.8135/0.7354/0.7913/0.7344**。中心裁剪相对完整网格的 AUC 差为 **-0.00257**，2,000 次配对受试者 bootstrap 95% CI **[-0.02872, 0.02327]**。点估计几乎相同，但区间下界低于预设 -0.02 非劣界值，因此严格 gate 未通过。下一步是 adaptive brain-mask bounding-box 敏感性实验，而不是直接把固定裁剪作为正式默认输入。完整结果见 [Phase 3M 报告](outputs/phase3m_mri_preprocessing_sensitivity_20261018/report.md)和[独立核验](outputs/phase3m_mri_preprocessing_sensitivity_20261018/independent_verification.json)。

## MRI Phase 3N：corrected center-cropped CN/MCI/AD classification

Phase 3N 从 `ADNI_Linear` 原始 `affine.nii.gz` 和 `mask.nii.gz` 路径重新读取 Phase 3G 的完整队列，将全部 5,370 张 `193×229×193`、1-mm T1 无插值中心裁剪为 `160×192×160`。缓存逐行复核 5,370/5,370 个体积均为有限且非空，并从 24 个分布位置重新读取原始 NIfTI，确认缓存与原始中心子数组逐元素一致。实验保持原来的 1,328 名 original-train subjects、诊断标签和 subject-wise 五折，每折训练 3 个 seed，共 15 个 SFCN checkpoints；官方 validation/test 未加载。

四臂折外 ACC/macro-AUC/macro-SEN/macro-SEC/macro-F1 分别为：MRI-only **0.4973/0.6854/0.4869/0.7318/0.4924**，cognition-only **0.8826/0.9739/0.8924/0.9409/0.8828**，cognition+MRI **0.8403/0.9497/0.8302/0.9134/0.8387**，cognition+shuffled-MRI **0.8226/0.9455/0.8029/0.9016/0.8188**。2,000 次受试者级 bootstrap 支持 MRI-only 优于 demographics，但融合相对 cognition-only 的 macro-AUC 差为 **-0.02412**，95% CI **[-0.02937, -0.01936]**；正确 MRI 相对 shuffled MRI 的 AUC 差为 +0.00428，区间跨 0。因而 MRI 含有并发诊断信号，但当前融合没有提供认知量表之外的增量。Phase 3N 取代旧 Phase 3K 的全部 MRI 三分类结论；旧 Phase 3L MCI 转化结果仍然无效。详见[四臂结果及置信区间](outputs/phase3n_corrected_cropped_mri_diagnosis_crossfit_20261018/requested_four_arm_report.md)、[完整报告](outputs/phase3n_corrected_cropped_mri_diagnosis_crossfit_20261018/report.md)与[独立复核](outputs/phase3n_corrected_cropped_mri_diagnosis_crossfit_20261018/independent_verification.json)。

## Phase 4A：frozen cognitive structured-reasoning baseline

[冻结协议](reports/experiments/phase4a_frozen_cognitive_reasoning_protocol.md)只使用原 train split 中具备未来预测样本的 1,671 名受试者，并新建 subject-disjoint build/dev/confirmation 划分：1,336/167/168 人，对应 13,238/1,582/1,757 份报告。三个 GRU-D checkpoint 使用 68 个已核验 cognitive items；normalization 只在 build subjects 拟合，dev 只用于 early stopping，confirmation 在冻结后评估。三种子均值写入 `forecast`，种子间标准差写入 `confidence`。

实验生成 **16,577 份**严格五段 JSON teacher reports。每份包含 `current_state`、真实 event/item/time/value evidence、可复算的 domain trend、6/12/24/36 月 forecast 和 uncertainty。未来真实目标只保存在独立 forecast evaluation 表中，不进入 prompt 或 completion；MRI 与诊断也不进入 reasoning input。

Confirmation 的 JSON validity、evidence precision/recall、numeric/time citation accuracy、domain、current-state、temporal、forecast 和 uncertainty preservation 均为 **1.000**，hallucinated evidence 与 future-evidence leakage 均为 **0**。Confirmation forecast macro normalized MAE 为 **0.3515**；ensemble uncertainty 与归一化绝对误差 Spearman 为 **0.1870**（p=5.30e-57，仅作描述）。独立 verifier 从原始事件、build normalization 和 ensemble 预测重算全部 16,577 份报告后通过。因此 `structured_reasoning_baseline_passed=true`、`phase4b_sft_ready=true`。这只开放 cognitive-only SFT，不开放 multimodal reasoning 或 GRPO。完整结果见 [Phase 4A 报告](outputs/phase4a_frozen_cognitive_reasoning_20261014/report.md)和[独立核验](outputs/phase4a_frozen_cognitive_reasoning_20261014/verification.json)。

## Phase 4B：cognitive-only structured-reasoning SFT

[冻结协议](reports/experiments/phase4b_cognitive_reasoning_sft_protocol.md)使用 Phase 4A 不变的 subject partitions，并从 16,577 份源报告中以稳定哈希选择 2,471/488/168 份 train/dev/confirmation 样本，覆盖 1,336/167/168 名受试者。未来真实结局、MRI 和诊断均不进入 prompt。全量序列最大 3,328 tokens，无 target 截断。正式模型为本地 `DeepSeek-R1-Distill-Qwen-1.5B`，只训练 4,358,144 个 rank-16 attention LoRA 参数；一轮训练后的 train/dev loss 为 0.00149/0.00113。

冻结 confirmation 上，base 模型 0/168 份输出通过 JSON schema；SFT 为 **166/168（98.81%）**，其中 **131/168（77.98%）**与 deterministic teacher 完全一致。全体 SFT 的 evidence precision/recall 为 0.9881/0.9851，current-state/temporal consistency 为 0.9802/0.9771，forecast/uncertainty preservation 均为 0.9881，未来证据泄漏为 0。两份输出在 1,280-token 冻结上限处截断；在 166 份合法 JSON 内，evidence precision、numeric/time/domain 和 forecast/uncertainty preservation 均为 1.000。由于严格协议要求 100% format 和至少 99% evidence/anchor preservation，`phase4b_sft_passed=false`、`cognitive_grpo_ready=false`。完整结果见 [Phase 4B 报告](outputs/phase4b_cognitive_reasoning_sft_20261014/report.md)和[独立核验](outputs/phase4b_cognitive_reasoning_sft_20261014/verification.json)。

## Phase 4C：dev-only grammar-constrained structured decoding

[冻结协议](reports/experiments/phase4c_grammar_constrained_decoding_protocol.md)不再更新模型参数。它只在 Phase 4A internal dev 的 167 名受试者上比较原始 SFT、token-trie `grammar_model` 和预声明阈值的 executable control。固定优先级下，`grammar_model` 满足全部 dev gate 并被锁定；原 Phase 4B 的 168 名 confirmation subjects 未参与选择或新确认。

锁定后，实验一次性纳入原始 ADNI validation split 中 234 名未参与 Phase 4A/B structured-reasoning 调参的受试者，每人稳定哈希选择一条 6/12/24/36 月样本。原始 SFT 生成 226/234（96.58%）份合法 JSON；`grammar_model` 为 **234/234（100%）**。约束臂的 evidence precision/recall、numeric/time/domain accuracy、forecast/uncertainty preservation 均为 1.000，hallucinated evidence 与 future leakage 为 0，current-state/temporal consistency 为 **0.9946/0.9940**。相对原始 SFT，JSON validity 改善 0.0342，subject-bootstrap 95% CI [0.0128, 0.0598]；其余主要改善区间也均高于 0。独立 verifier 重新评分并通过。

因此 `phase4c_structured_decoding_passed=true`、`cognitive_grpo_ready=true`。该结论只允许下一步冻结并开展 **cognitive-only constrained GRPO**；当前尚无 on-policy GRPO 结果，且 MRI/multimodal reasoning 仍关闭。完整结果见 [Phase 4C 报告](outputs/phase4c_grammar_constrained_decoding_20261015/report.md)和[独立核验](outputs/phase4c_grammar_constrained_decoding_20261015/verification.json)。

## Phase 4D：cognitive-only constrained GRPO

[冻结协议](reports/experiments/phase4d_cognitive_constrained_grpo_protocol.md)从 Phase 4B LoRA 和 Phase 4C enum trie 出发，只允许模型选择五个 state 和五个 trend 标签。64 名 Phase 4A build subjects 各提供一个 rollout history；每个 history 采样 4 个候选，共生成 **256 个真实 on-policy candidates**。训练完成 32 次 clipped GRPO updates，更新 4,358,144 个 LoRA 参数，并在 updates 8/16/24/32 保存 checkpoint。平均 categorical KL 为 0.01236，最大 update mean KL 为 0.05168；71.88% 的 rollout groups 有非零 reward variation，训练无安全终止。

所有 evidence、domain、format、forecast 和 uncertainty anchors 固定并逐候选核验；真实 future outcome 只作 availability-aware reward/audit，不进入 prompt。由于数字 forecast 在候选间不变，该分量不产生 group-relative gradient，实际可学习信号来自 state/trend actions。

在 167 名 internal-dev subjects 上，SFT direct action composite 为 **0.97784**。四个 GRPO checkpoints 均未超过它；最佳 update 32 为 **0.97246**，差 -0.00539。update 32 的 state/trend accuracy 为 0.97485/0.97006，锚定指标仍全部为 1.000，KL、reward variation 和安全条件也通过，但预注册的 improvement gate 失败。因此 `cognitive_rlvr_supported=false`，原始 test reasoning confirmation 未打开，Phase 4C 仍是当前支持策略。完整结果见 [Phase 4D 报告](outputs/phase4d_cognitive_constrained_grpo_20261015/report.md)和[独立核验](outputs/phase4d_cognitive_constrained_grpo_20261015/verification.json)。

## Phase 4E：trajectory-action SFT 与 outcome-varying GRPO

[冻结协议](reports/experiments/phase4e_trajectory_action_grpo_protocol.md)将 Phase 4A build 重新划分为 256/96/256 名互斥 action-SFT/rollout/development subjects，并排除 Phase 4D 的 64 名 rollout subjects。策略为五个认知终点分别选择 -0.5/-0.25/0/+0.25/+0.5 个训练标准差的 residual，以及 0.75/1.0/1.5 的 uncertainty multiplier。action-SFT 只学习零残差与原方差，不读取 future truth；GRPO 的 outcome、CRPS、future direction 和临床范围 reward 随候选动作变化。

训练完成 32 个 updates、256 个真实 on-policy candidates 和四个 checkpoints，无安全中止；31.25% 的 groups 有非零 reward variation。新 development 上冻结 GRU-D 的 normalized MAE/CRPS/temporal accuracy 为 **0.35108/0.31896/0.59382**，action-SFT 和全部 GRPO checkpoints 均为 **0.39970/0.35690/0.55247**。最佳 update 8 相对冻结 GRU-D 的 NMAE 差为 **+0.04862**，95% CI **[0.03620, 0.06064]**。动作诊断显示 action-SFT 和所有 GRPO checkpoint 对全部终点都贪心选择 `lower+wide`，GRPO 没有改变策略。完整 gate 失败，`trajectory_action_rlvr_supported=false`，原始 test 未加载，也不得在这 256 名 development subjects 上继续调参。完整结果见 [Phase 4E 报告](outputs/phase4e_trajectory_action_grpo_20261015/report.md)和[独立核验](outputs/phase4e_trajectory_action_grpo_20261015/verification.json)。

## Phase 4F：冻结 GRU-D + Phase 4C sealed-test confirmation

[冻结协议](reports/experiments/phase4f_frozen_grud_phase4c_test_protocol.md)在读取官方 test 前锁定 Phase 4A 三个 GRU-D checkpoints、Phase 4B 1.5B LoRA、Phase 4C `grammar_model`、prompt、schema、指标和阈值，并记录 SHA-256。实验无训练、无 checkpoint 选择，明确排除 Phase 4D/4E adapter。数值部分覆盖 495 名受试者、4,951 个 samples 和 20,007 个可用 targets；推理部分按稳定哈希每人一份，共 495 份报告。

冻结 GRU-D 的 test macro/pooled normalized MAE 为 **0.37510/0.36288**。在 LOCF 可覆盖的相同行上，GRU-D/LOCF 为 0.36200/0.37939，subject-mean 差 **-0.02144**，95% CI **[-0.03913, -0.00423]**。uncertainty-error Spearman 为 0.1584。数值预测三项 gate 全部通过。

Raw SFT 产生 488/495 份合法 JSON；grammar model 为 **495/495**。Grammar 的 evidence precision/recall、numeric/time/domain accuracy、forecast/uncertainty preservation 均为 1.000，幻觉和 future leakage 均为 0，nested state consistency 为 0.99515。Nested temporal consistency 为 **0.98990**，相对 raw SFT 改善 +0.01051，95% CI [0.00121, 0.02182]，但仍比预注册绝对门槛 0.99000 低约 0.00010。该分数平均每份报告五个认知域中的四项字段；grammar 保留全部 numeric/timepoint anchors，但 2,475 个域级 trend labels 中有 100 个不同，涉及 67/495 份报告。因此完整 `phase4f_frozen_system_confirmed=false`；不得在 test 上修改这些标签或降低阈值。论文可以报告已确认的数值性能和结构完整性改善，同时必须明确完整推理门槛窄幅失败。完整结果见 [Phase 4F 报告](outputs/phase4f_frozen_grud_phase4c_test_20261015/report.md)和[独立核验](outputs/phase4f_frozen_grud_phase4c_test_20261015/verification.json)。

## Phase 5B：冻结 structured-reasoning error decomposition

[分析协议](reports/experiments/phase5b_frozen_reasoning_error_decomposition_protocol.md)只读取 Phase 4F 已生成的 495 份 grammar reports，不训练模型、不重新生成报告，也不改变 ±0.5 state 和 ±0.25 trend 阈值。分析重建了 4,950 个 domain-field decisions，并以 2,000 次 subject-cluster bootstrap 分解 domain、horizon、expected label、history、missingness 和阈值距离。

Current-state 有 **36/2,475（1.45%，95% CI 0.97–1.98%）** 个 label mismatch；trend 有 **100/2,475（4.04%，95% CI 2.99–5.17%）**。Trend 错误在 Memory 最高（40/495，8.08%）；expected `improve` 的错误为 39/130（30.0%），最常见方向是 `improve → stable`（26 个）。Trend 阈值 0.10 z 内只有 8 个 decisions，其中 5 个错误；相对远阈值的差为 56.88 个百分点，但 95% CI 很宽 [13.99, 94.19]，只能作为小样本描述。Current-state 的 near-minus-far 差为 0.83 个百分点，95% CI [-0.17, 1.85]。

该分析确认偏差位于 categorical state/trend selection，而 Phase 4F 的数值、时间点、evidence、forecast 和 uncertainty anchors 均保持不变。它不改变 `phase4f_frozen_system_confirmed=false`，也不允许利用 495 名 test subjects 修复策略。下一项合规实验应是预先冻结的外部队列验证。完整结果见 [Phase 5B 报告](outputs/phase5b_frozen_reasoning_error_decomposition_20261016/report.md)和[独立核验](outputs/phase5b_frozen_reasoning_error_decomposition_20261016/verification.json)。

## Method reframing: interpretable cognitive memory state

论文主问题进一步聚焦为：能否从稀疏、异步、部分缺失的认知观测中学习一个具有可解释动态结构的长期 cognitive memory state，并用它预测和解释个体变化。新的 [paper/method reframing](reports/methods/memreason_state_paper_and_method_reframing.md) 将主方法定义为 typed item measurement、continuous-time predict-update state、progressive/transient decomposition 和 executable evidence reasoning；GRU-D 保留为强预测基线，Phase 4C 作为可靠 renderer，MRI/GRPO 作为次要或负结果。对应的 [Phase 5C train-only protocol](reports/experiments/phase5c_interpretable_memory_state_dev_protocol.md) 明确要求 nested subject-wise development 和后续外部确认，不重新使用已消费的 ADNI validation/test 进行选择。

## Phase 5C Stage 0/1：memory coding audit 与 measurement development

Stage 0 将 12 个新增 memory fields 与 ADNI 官方 ITEM dictionary 精确匹配。`AVLT_Trial1Int`–`Trial7Int` 是 intrusion counts，`AVLT_Delay_TotInt` 是 delayed intrusions，不能作为 recall scores；`AVLT_Delay_Rec` 才是 recognition score。Logical Memory immediate/delayed 按 0–25 分处理，99/999/9999 和负特殊码视为缺失。冻结 contract 使用原 68 个 verified items 加两项 Logical Memory，共 70 个 state items，其中 14 个属于 Memory；AVLT 只作 auxiliary validity。

Stage 1 只使用 original train，完成五折 nested subject-wise cross-fitting、15 个 neural checkpoints、1,079,180 条 OOF masked predictions 和 178,315 条 cross-view state pairs。无约束 autoencoder 的 held-out item/instrument macro normalized MAE 最好，为 **0.4446/0.4539**；概率 domain model 为 **0.6782/0.5507**。概率 domain model 相对无约束模型的 held-out-instrument subject-level error 差为 **+0.10739**，95% CI **[0.10099, 0.11331]**。其 cross-view Spearman 为 **-0.0776**，低于 fixed domain score 的 0.4684 和无约束模型的 0.9526。因此 `stage1_measurement_gate_passed=false`，不按当前结构进入 continuous-time dynamics。完整结果见 [Stage 1 报告](outputs/phase5c_memory_measurement_dev_20261016/report.md)和[独立核验](outputs/phase5c_memory_measurement_dev_20261016/verification.json)。

## Phase 5D：anchored bifactor MIRT 与 measurement invariance

[冻结协议](reports/experiments/phase5d_anchored_mirt_protocol.md)将状态识别为 positive-loading global impairment 与 Memory-specific impairment，分别使用 Bernoulli、beta 和 graded-response likelihood，并只在 original-train 内进行五折 nested subject cross-fitting。时间、诊断和 ADNI phase 的 residual DIF 只在 outer-fit 数据中审计；诊断不进入状态推断。实验复用 Phase 5C 的 fixed score 和 unconstrained autoencoder OOF 对照，共产生 15 个 MIRT checkpoints、1,348,975 条五臂预测和 29,247 条 cross-view records。

冻结规则没有在任何 fold 选择出具有实际效应的 time DIF，因此 partial-invariant 与 invariant arm 完全相同。其 held-out item/instrument macro normalized MAE 为 **0.5181/0.5273**，fixed score 为 0.4866/0.5212，unconstrained autoencoder 为 0.4446/0.4539。候选相对 autoencoder 的 held-out-instrument subject difference 为 **+0.07476**，95% CI **[0.07014, 0.07957]**。Memory cross-view CCC 为 **0.3713**，95% CI **[0.3547, 0.3873]**。8 个来自 ADAS/MMSE 的 Memory anchors 通过稳定性门槛，但预测非劣性、cross-view coherence 和 fold reproducibility 均失败。因此 `phase5d_measurement_gate_passed=false`、`continuous_time_dynamics_open=false`。完整结果见 [Phase 5D 报告](outputs/phase5d_anchored_mirt_dev_20261016/report.md)和[独立核验](outputs/phase5d_anchored_mirt_dev_20261016/verification.json)。

## Phase 5E：frozen reasoning faithfulness 与 renderer ablation

[冻结审计协议](reports/experiments/phase5e_reasoning_faithfulness_audit_protocol.md)在 Phase 4F test 已打开的前提下预先固定四种干预和解释边界，不选择 checkpoint、decoder 或阈值。实验对 495 份报告构造 1,013 个 leave-one-visit 候选，并用 Phase 4A 的三个冻结 GRU-D checkpoint 重新前向计算。在 278 名至少有两个可删除观测访视的受试者中，最高影响访视删除的 normalized forecast displacement 为 **0.2244**，匹配随机删除为 **0.0811**；配对差为 **0.1432**，95% CI **[0.1191, 0.1706]**。最高影响删除使 31.0% 的 endpoint forecast-direction 标签、17.1% 的 domain-state 标签和 52.9% 的 domain-trend 标签改变。

Memory score reflection 产生 1.5070 的平均预测位移，time-gap reversal 只产生 0.00268，说明当前冻结 GRU-D 对观测值明显敏感，而对仅改变间隔顺序的敏感度很低。所有反事实报告均由 typed executable IR 重新编译，verifier exact match/evidence precision 均为 1.000，删除证据残留和未来泄漏均为 0。

同一批 495 份报告上的 renderer 消融显示：`grammar_executable` 与 teacher 完全一致，exact report match 为 1.000；`grammar_model` 为 0.8141，raw SFT 为 0.8343。Executable 相对 grammar model 的 state/temporal consistency 分别提高 0.00485 和 0.01010。因此论文的核心方法应明确为 **frozen GRU-D + typed executable reasoning IR/compiler/verifier**；1.5B LLM 不再承担事实判断，只能作为可选语言渲染器。该结果是 post-test audit，不改变 `phase4f_frozen_system_confirmed=false`。完整结果见 [Phase 5E 报告](outputs/phase5e_reasoning_faithfulness_audit_20261016/report.md)和[独立核验](outputs/phase5e_reasoning_faithfulness_audit_20261016/independent_verification.json)。

## Phase 5F：hierarchical shared state、instrument residual 与 bridge graph

[冻结协议](reports/experiments/phase5f_hierarchical_bridge_protocol.md)将测量状态分为 global、五个 domain、三个预先固定的 Memory subconstruct 和只允许解释本量表题目的 instrument residual。Residual 在量表完全缺失时严格为零，并受到更强的 KL 与 loading shrinkage；bridge loss 只对真实日期/同期覆盖和 domain overlap 支持的量表边对齐 shared posterior，不对齐 instrument residual。实验只使用 original-train 的五折 nested subject cross-fitting，产生 15 个 checkpoints 和 1,348,975 条五臂 OOF masked predictions；原 validation/test、诊断、FAQ、MRI 和未来结果均未进入状态学习。

主候选 `hierarchical_bridge` 的 held-out item/instrument macro normalized MAE 为 **0.3989/0.4166**，优于 fixed score 的 0.4866/0.5212 和 unconstrained autoencoder 的 0.4446/0.4539。其相对 fixed score 的 held-out-instrument participant difference 为 **-0.09920**，95% CI **[-0.10317, -0.09516]**；相对 autoencoder 为 **-0.02525**，95% CI **[-0.02818, -0.02241]**。Bridge 将 Memory cross-view CCC 从 shared-only 的 0.4251 和 residual-only 的 0.4682 提高到 **0.6340**，95% participant-bootstrap CI **[0.6207, 0.6467]**，同时 Memory-state SD=0.2294、residual loading fraction=0.0854，均通过 anti-collapse/control 条件。代价是 bridge 相对 residual-only 的 held-out-instrument error 增加 **0.00861**，95% CI **[0.00665, 0.01038]**，说明一致性与预测存在真实 trade-off。LOGMEM 因无可用实际日期且构念重叠较弱，所有边均低于冻结支持阈值；当前 CCC 主要检验 ADAS/CDR/MMSE bridge，不能宣称 LOGMEM transport 已验证。CCC 下限仍未达到预注册 0.70，因此 `phase5f_measurement_gate_passed=false`、`continuous_time_dynamics_open=false`，不降低阈值、不打开 Phase 6。完整结果见 [Phase 5F 报告](outputs/phase5f_hierarchical_bridge_dev_20261028/report.md)和[独立核验](outputs/phase5f_hierarchical_bridge_dev_20261028/verification.json)。

## Phase 5G/5H：bridge Pareto 与 measurement identifiability

Phase 5G 在 original-train 五折中比较 11 个 bridge/gate variants（55 fits）。在预设的 +0.010 prediction tolerance 下，`ungated_w0p1` 为选中的 Pareto 点：held-out-instrument NMAE **0.4180**，Memory CCC **0.6296**，95% CI **[0.6163, 0.6423]**。Residual-adaptive gate 仅带来很小的预测改善，却在每个匹配系数上降低 CCC，因此没有支配 ungated frontier。Phase 5H 随后审计 4 个量表复合指标和 6 个同期量表对：衰减上限中位数为 **0.7711**，但仅 **4/6** 对支持 CCC≥0.70。该结果说明统一 0.70 门槛并非对所有量表对都可辨识，但这是描述性诊断，不能事后降低冻结门槛。完整结果见 [Phase 5G 报告](outputs/phase5g_bridge_pareto_gate_dev_20261029/report.md)和[Phase 5H 报告](outputs/phase5h_measurement_identifiability_audit_20261029/report.md)。

## Phase 6A：exploratory latent-state dynamics

在不重开 Phase 5F measurement gate 的前提下，Phase 6A 使用 Phase 5G OOF state 作为 noisy teacher，完成 original-train-only 五折 subject cross-fitting：7 个 arms、30 个 checkpoints、每臂 15,502 个 6/12/24/36 月预测样本。`reference_mri_filter` 的 latent MAE 为 **0.0905**，latent LOCF 为 **0.0949**；受试者配对差 **-0.00346**，95% CI **[-0.00436, -0.00255]**。真实 MRI 相对置乱 MRI 的 latent MAE 差为 **-0.00051**，但当前未对这项小差异作显著性声明。候选 95% coverage 仅 **0.8360**，未通过预设校准范围，因此只能报告点预测的探索性增益，不能声称已经确认连续时间潜在状态。完整结果见 [Phase 6A 报告](outputs/phase6a_latent_state_dynamics_dev_20261029/report.md)和[独立核验](outputs/phase6a_latent_state_dynamics_dev_20261029/verification.json)。

## Phase 6B：MRI incremental-value audit

Phase 6B 固定 Phase 6A folds，在 original-train-only 五折中训练 cognition-only、MRI metadata-only、real MRI、subject-shuffled MRI、within-subject time-shuffled MRI 和 MRI-only 六臂，共 30 个 checkpoints。主终点改为未来真实 observed Memory composite，而 Phase 5G latent teacher 仅作为辅助监督。Real MRI 的 composite MAE 为 **0.08450**，优于 cognition-only 的 0.08561、metadata-only 的 0.08547 和 subject-shuffled MRI 的 0.08557；对应受试者配对差分别为 **-0.001087**、**-0.000740** 和 **-0.000744**，且均在 5/5 folds、4/4 horizons 保持方向。临床区间 coverage 为 **0.9476**，进入冻结的 0.90--0.98 范围。

但是 real MRI 与 within-subject time-shuffled MRI 的受试者配对差仅为 **-0.000055**，95% participant-bootstrap CI **[-0.000193, +0.000076]**，只在 3/5 folds、2/4 horizons 占优。该结果支持患者特异的稳定结构表型，但不支持当前 burden/slope 表示利用了正确纵向顺序。因此 `mri_content_signal_supported=false`、`disagreement_filter_open=false`；下一步应先改进 MRI 纵向表征或时间对齐，而不是直接加入复杂分歧 gate。完整结果见 [Phase 6B 报告](outputs/phase6b_mri_incremental_value_dev_20261029/report.md)、[独立核验](outputs/phase6b_mri_incremental_value_dev_20261029/verification.json)和[独立图件](outputs/phase6b_mri_incremental_value_dev_20261029/figures/README.md)。

## Phase 6C：longitudinal MRI temporal-signal identifiability

Phase 6C 使用实际 MRI acquisition date，并将队列限制为截止日前至少 2 次 MRI、跨度至少 6 个月的 original-train 受试者，共 **732 人、7,669 个预测样本**。五个动态特征包括首末 burden 变化与实际时间斜率、最近两次变化与斜率、首末 disagreement 变化；所有负对照都严格保留同一受试者的最新 burden/disagreement、MRI availability、recency、scan count 和 observation span。实验完成 6 臂五折、30 个可审计 checkpoints；其中 sign-flip 是候选模型的推理期方向干预，因为全程符号翻转是可逆变换，重新训练不能构成有效负对照。

True longitudinal MRI 的 observed Memory composite MAE 为 **0.08756**，低于 cognition-only 的 0.09002，但没有优于 static MRI（0.08755）或 subject-shuffled dynamics（0.08753）。受试者配对的 true-minus-static 差为 **+0.000098**，95% CI **[-0.000084, +0.000290]**；true-minus-subject-shuffled 为 **+0.000157**，95% CI **[-0.000062, +0.000372]**。真实动态只在 2/5 folds、1/4 horizons 优于 static MRI。候选 coverage 为 **0.9485**，校准合格，但时间信号判据失败。因此 `longitudinal_mri_signal_supported=false`、`disagreement_filter_open=false`，当前 MRI 应冻结为静态辅助变量。完整结果见 [Phase 6C 报告](outputs/phase6c_longitudinal_mri_identifiability_dev_20261030/report.md)、[独立核验](outputs/phase6c_longitudinal_mri_identifiability_dev_20261030/verification.json)和[独立图件](outputs/phase6c_longitudinal_mri_identifiability_dev_20261030/figures/README.md)。

## Phase 6D0/6D1：fold-aligned 128D MRI embedding

Phase 6D0 在 native `193×229×193` T1 上重新训练五个 fold-specific SFCN encoder。对 fold `k`，encoder fit subjects 与该 fold 的 Phase 6 evaluation subjects 严格隔离；随后用每个 encoder 分别提取 original-train 的 **4,616** 个纵向扫描、每个 **128 维**的 embedding。五套独立 encoder 的坐标不作逐维平均，泄漏审计为零。完整结果见 [Phase 6D0 报告](outputs/phase6d0_native_embedding_extract_20261030/report.md)和[独立核验](outputs/phase6d0_native_embedding_extract_20261030/verification.json)。

Phase 6D1 对每个 outer fold 只使用对应 encoder，并在 outer-fit 内拟合标准化与 PCA-16；只有 acquisition date 不晚于 cognition cutoff 的扫描可进入模型。`embedding_static` 的 observed Memory composite MAE 为 **0.08434**，优于 cognition-only 的 0.08529（配对差 **-0.001048**，95% CI **[-0.001486, -0.000587]**）和 subject-shuffled embedding 的 0.08541（差 **-0.000836**，95% CI **[-0.001328, -0.000349]**）。但相对同一 encoder 的 AD probability（MAE 0.08455）差仅为 **-0.000240**，95% CI **[-0.000571, +0.000091]**，且只在 3/5 folds 占优。候选 coverage 为 **0.9505**，但严格 incremental gate 仍为 false。结论是 128D embedding 含有真实患者特异信息，却尚未证明其超出分类概率的额外维度能稳定改善预测；按冻结执行顺序，Phase 6D2 和 Phase 6E 均未开放。完整结果见 [Phase 6D1 报告](outputs/phase6d1_static_embedding_incremental_dev_20261030/report.md)和[独立核验](outputs/phase6d1_static_embedding_incremental_dev_20261030/verification.json)。

## Phase 7A：native-T1 三分类与 Memory 联合训练

Phase 7A 是不重开 Phase 6D1 gate 的独立开发分支。它在 original-train 的 **774 名受试者、4,263 次原生 `193×229×193` T1** 上做五折 subject cross-fitting，并比较 classification-only、regression-only 与 joint-multitask 三臂。每折 encoder 从 leakage-safe 的 Phase 6D0 对应 checkpoint 初始化后完全解冻；共享 128 维 MRI embedding 同时驱动 CN/MCI/AD 分类、当前 observed Memory 和 6/12/24/36 月未来 observed Memory Gaussian 回归。实验生成 15 个 checkpoint，并加入 joint 模型的 subject-shuffled MRI 推理负对照；原 validation/test 均未加载。

联合模型的三分类 ACC/AUC/macro-SEN/macro-SPE/macro-F1 为 **0.2691/0.6085/0.4320/0.6914/0.2763**；classification-only 为 **0.2988/0.5958/0.4192/0.6858/0.3049**。联合训练的 macro AUC 差为 **+0.01310**，95% CI **[-0.00725, +0.03235]**，满足非劣但不支持显著优越；同时 ACC 与 macro-F1 下降，MCI sensitivity 只有 **0.0610**。未来 Memory MAE 为 **0.08311**，低于 regression-only 的 0.08349；配对差 **-0.000325**，95% CI **[-0.000644, -0.000034]**，但只在 **3/5 folds** 改善，低于冻结的 4/5 要求。联合未来 95% coverage 为 **0.9471**。因此 `joint_multitask_gate_passed=false`：结果支持共享 embedding 中存在可被两任务利用的弱信号，但也暴露出明显的 MCI 类别不平衡/梯度冲突，下一步应优先改进类均衡采样、任务梯度协调和分阶段训练，不作确认性临床声明。完整结果见 [Phase 7A 报告](outputs/phase7a_joint_multitask_native_t1_dev_20261030/report.md)和[独立核验](outputs/phase7a_joint_multitask_native_t1_dev_20261030/verification.json)。一键运行入口为 `bash scripts/run_phase7a_joint_multitask_gpu.sh`。

数据审计入口：[summary.md](reports/data_audit/summary.md)。完整工程状态：[BUILD_STATUS.md](BUILD_STATUS.md)。
