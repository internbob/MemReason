# MemReason-RL 项目月度周报

**项目周期：四周汇总**
**项目名称：MemReason-RL——基于 ADNI 异步纵向认知与 MRI 的记忆状态建模**
**当前阶段：认知侧正式基线、表示学习开发、probabilistic belief/calibration、MRI Phase 3A–3D 开发实验及 Phase 3E 一次性冻结 test confirmation 已完成；Phase 3C 未通过完整 test confirmation，item-level multimodal 与纵向 MRI gate 也未通过；SFT 与 GRPO 尚未开展。**

## Brief Description

本项目面向 ADNI 异步纵向数据，目标是从不完整、不同步的认知量表与结构 MRI 中构建可更新的记忆状态，并进一步完成未来认知预测、跨量表推断、不确定性估计和有证据约束的纵向推理。当前工作首先完成数据审计、受试者级数据划分和 MRI 名义月份匹配，随后建立经典与深度纵向预测基线，并依次检验固定 semantic descriptor、response-aware semantic encoder、typed response representation、任务独立门控以及显式 probabilistic belief dynamics。

截至本月末，GRU-D 是得到正式测试支持的认知预测主模型；typed direct Gaussian 在隐藏中间访视插值上表现出任务特异优势。固定语义、response-aware semantic 和显式 predict-update belief 均未满足预先设定的完整门控，因此当前不宣称 semantic encoder 或显式 belief dynamics 具有普遍优势，也不据此开放正式 MRI Stage II。

## Rationale

- **先保证数据可信。** 纵向预测极易受到按 visit 随机划分、未来信息泄漏、错误量表编码和伪扫描日期影响，因此先建立 subject-wise split、严格 cutoff 和 item-level provenance。
- **先建立必须被超越的强基线。** 在引入 semantic、MRI 和 RL 前，统一比较 LOCF、LME、GRU、LSTM、T-LSTM、GRU-D 与 time-aware Transformer，避免把弱基线带来的表面增益归因于新方法。
- **用公平对照检验 semantic value。** numeric、variable-name、task-semantic 和 response-aware 分支使用相同输入、训练预算、任务和参数规模，使差异尽量只来自表示方式。
- **对部分观测任务进行联合验证。** masked item、masked instrument、cross-scale 与 future forecasting 共同检验 representation 是否能在真实缺失条件下形成稳定、可迁移的认知状态。
- **将负结果作为正式结论。** 当完整 Gate 1 未通过后，停止在相同 validation subjects 上继续调整 semantic 架构，防止验证集过拟合。
- **显式检查不确定性而不只看点预测。** belief dynamics 实验同时报告 MAE、Gaussian NLL、CRPS、区间覆盖率、校准误差及不确定性与误差/缺失程度的相关性。
- **严格执行阶段门控。** IDA 已为 91.52% 的主 MRI 补入真实 Study Date，但临床 QC、scanner/field strength 和完整配准 provenance 仍缺失；SFT/GRPO 也缺少正式训练数据，因此相关模块目前只具备工程接口，不作为已完成科研结果。

---

# Week 1：数据审计、纵向队列与 MRI 路径确认

## Description

第一周集中解决数据是否可用、时间是否可信以及不同数据源如何对齐的问题。工作覆盖原始量表审计、MRI 根目录比较、subject-wise split、异步事件流、未来目标索引和 MRI 名义月份匹配。

## Rationale

模型实验只有在受试者互斥、预测目标严格位于 cutoff 之后、量表缺失未被人工填补且 MRI 时间来源明确时才具有解释性。本周工作为后续所有认知、MRI 和纵向实验建立统一数据合同。

## To-do List

### Task 1：完成原始认知量表审计

- [x] 核对原始 CSV 的规模、字段、缺失情况和访视分布。
- [x] 确认共有 4,316 名受试者、15,453 次访视、161 个原始字段。
- [x] 保留原始空值，不对真实缺失进行预先插值或填补。
- [x] 标记候选异常值和特殊编码，避免静默修改原始数据。

### Task 2：确定正式 MRI 数据源

- [x] 比较 `ADNI_Linear`、`surfGT` 和旧 UniSurf longitudinal 目录。
- [x] 确定主 MRI 目录为 `/public_bme2/bme-dgshen/WeilinZhou/AD_longitudinal/ADNI/ADNI_Linear`。
- [x] 建立 10,230 次纵向 T1 的 subject/session 索引。
- [x] 使用 IDA Image ID+subject 精确连接，为 9,362/10,230 次主 MRI 补入真实 Study Date。
- [x] 记录剩余 868 次 MRI 缺少日期，以及 scanner 参数和临床 QC 仍未知的限制。

### Task 3：建立受试者级数据划分

- [x] 使用 seed=2026 完成 train/validation/test 的 70%/10%/20% subject-wise split。
- [x] 得到 3,021/431/864 名互斥受试者。
- [x] 将所有量表访视、MRI、事件和 future targets 继承受试者所属 split。
- [x] 禁止对派生 CSV 再按行随机拆分。

### Task 4：构建纵向事件和预测索引

- [x] 生成异步 instrument/MRI event stream。
- [x] 生成 6/12/24/36 月 future forecasting 索引。
- [x] 确保 target event 严格晚于 cutoff。
- [x] 建立同月份匹配、最近 MRI 月份匹配及保守 ±6 月子集。

## Weekly Summary

本周完成项目的数据地基。最终保留全部 4,316 名受试者及 15,453 次认知访视；主 MRI 数据源包含 1,981 名受试者和 10,230 次 T1。IDA 为其中 9,362 次提供真实 Study Date，90 天主协议得到 9,446 条配对，其中 9,222 条通过真实日期窗口。train/validation/test 完成受试者级隔离，且所有预测目标严格位于历史 cutoff 之后。临床 QC 仍未知，因此真实日期窗口通过不等同于正式 grounding eligibility。

---

# Week 2：认知预测基线、item 编码核验与 semantic 公平对照

## Description

第二周建立可靠的认知预测基线，并完成 item dictionary 的官方字段和响应编码核验。在此基础上，用完全一致的数据、任务、参数量和训练预算比较 numeric、variable-name 与 task-semantic representation。

## Rationale

semantic 或多模态方法的价值必须建立在强 numeric baseline 之上。若数据字段含义或响应方向不可靠，语言描述会放大编码错误，因此 item 编码核验必须先于 semantic value 实验。

## To-do List

### Task 1：完成 Phase 1A numeric forecasting

- [x] 使用量表 item、缺失 mask 和名义访视月份建立纯数值预测输入。
- [x] 预测 ADAS13、CDR-SB、FAQ、MMSE、MoCA 的 6/12/24/36 月结果。
- [x] 比较 LOCF、LME、GRU-D 和 time-aware Transformer。
- [x] 使用多随机种子、逐终点指标和 subject-cluster bootstrap。

### Task 2：完成扩展 baseline suite

- [x] 增加 GRU、LSTM 和 T-LSTM。
- [x] 统一重跑 GRU-D 和 Transformer，保持相同训练预算。
- [x] 完成 15 个神经模型 checkpoint。
- [x] 由 validation 选择模型后再读取 test。

### Task 3：完成 item dictionary 与编码核验

- [x] 对 107 个 ADAS/CDR/FAQ/MMSE/MoCA 项目字段进行官方字段匹配。
- [x] 99 个字段通过编码和观测值核验。
- [x] 8 个缺少可审计响应编码的 MMSE 拼写字段保持关闭。
- [x] 选出 68 个已核验、可作标量比较且方向明确的输入字段。

### Task 4：完成 Phase 1B numeric vs semantic 公平对照

- [x] 比较 `numeric_only`、`variable_name` 和 `task_semantics`。
- [x] 使用本地冻结 Bio_ClinicalBERT 编码 descriptor。
- [x] 保持三组输入、模型容量、训练流程和预测任务一致。
- [x] 完成未来预测与两项 cross-scale 任务的正式评估。

## Weekly Summary

扩展 baseline suite 中，GRU-D 获得最佳测试 macro normalized MAE **0.3799**，优于 GRU 0.3818、T-LSTM 0.3857、Transformer 0.3890、LSTM 0.3903、LOCF 0.3983 和 LME 0.4193。GRU-D 相对 LOCF 和 LME 的 pooled normalized absolute error 改善均获得 subject-cluster bootstrap 支持。

Phase 1B 中，task semantics 在两项 cross-scale 上只有很小且置信区间跨零的改善；future forecasting 相对 numeric-only 反而轻微变差。因此固定任务描述的 semantic value gate 未通过，不能声称 frozen semantic descriptor 带来稳定收益。

---

# Week 3：Response-aware 联合 Stage I 与表示学习迭代

## Description

第三周实现 response-aware semantic encoder，并将 masked item、masked instrument、cross-scale 和 future forecasting 纳入联合 Stage I。针对正式 Phase 1C 的负结果，随后依次开展 validation-only 的 typed response、metric-aligned loss 和 task-specific gate 开发实验。

## Rationale

固定 question descriptor 可能无法表达“问题—实际响应—评分方向”之间的关系，因此需要检验 response-aware representation。同时，完整 Stage I 必须在部分观测重建、跨量表迁移和未来预测上共同成立，不能依靠单一任务改善开放后续阶段。

## To-do List

### Task 1：实现 response-aware semantic encoder

- [x] 将任务描述、实际响应、允许范围/类别标签和评分方向编码为 descriptor。
- [x] 保持 Bio_ClinicalBERT 冻结。
- [x] 建立 numeric、task-semantic、response-aware 三臂公平对照。
- [x] 保持每臂 290,993 个可训练参数和相同联合任务采样。

### Task 2：完成联合 Stage I 正式实验

- [x] 完成 masked item reconstruction。
- [x] 完成 masked instrument reconstruction。
- [x] 完成 ADAS+CDR→MoCA 与 MMSE+CDR→ADAS13 cross-scale prediction。
- [x] 完成 6/12/24/36 月 future forecasting。
- [x] 运行 3 arms × 3 seeds，共 9 个 checkpoint，并独立核验结果。

### Task 3：完成 Phase 1D typed representation

- [x] 实现 continuous/binary/ordinal typed response encoder。
- [x] 实现 item-conditioned decoder 和类型对应的损失。
- [x] 比较 joint 与 two-stage 训练方式。
- [x] 全程只使用 train/validation，不加载 test。

### Task 4：完成 Phase 1E metric-aligned gated representation

- [x] 实现 per-item macro reconstruction loss。
- [x] 为 categorical target 加入 CE 与 expected-score Huber。
- [x] 加入零初始化 semantic residual gate。
- [x] 检验 semantic 增益是否从 masked item 扩展至其他任务。

### Task 5：完成 Phase 1F task-specific gates

- [x] 为五类任务分别设置 semantic/value gate。
- [x] 保持与 Phase 1E 相同的 metric-aligned loss。
- [x] 完成预先限定的最后一次当前 validation 表示修订。
- [x] 输出停止继续 semantic tuning 的 terminal decision。

## Weekly Summary

Phase 1C 中，response-aware 仅在 MMSE+CDR→ADAS13 上得到显著改善；masked item、masked instrument 明显变差，future forecasting 未满足预设非劣界值，因此完整 Stage I gate 为 false。

Phase 1D–1F 显示 typed/item-conditioned representation 对部分任务有价值。Phase 1F 的 semantic validation macro 为 **0.38334**，优于 numeric 的 0.38826；masked item 获得显著改善，masked instrument 和 future 的点估计也有改善，但其置信区间未满足门控，两项 cross-scale 亦未稳定改善。因此停止在相同 validation subjects 上继续调整 semantic architecture，保留“部分观测任务特异收益”，放弃“普遍 semantic superiority”的主张。

---

# Week 4：不规则时间 belief dynamics、校准与阶段决策

## Description

第四周将研究重点从 semantic superiority 转向 cognitive-only irregular-time dynamics。实验比较 GRU-D Gaussian、typed direct Gaussian 和 typed predict-update belief，联合覆盖隐藏完整中间访视的 interpolation 与 6/12/24/36 月 extrapolation，并对预测方差进行独立校准。

## Rationale

纵向医学预测不仅需要较低点预测误差，还需要在随访稀疏、历史长度不同和量表缺失时提供可信的不确定性。为避免 calibration information leakage，训练、调参、方差校准和最终 evaluation 必须按受试者隔离。

## To-do List

### Task 1：建立隔离的 dynamics/calibration 协议

- [x] 将官方 train subjects 进一步拆分为 fit 与 tune。
- [x] 将 validation subjects 拆分为 calibration 与 evaluation。
- [x] 仅使用 fit subjects 拟合 normalization。
- [x] 保证 calibration 与 evaluation subjects 完全互斥，且不加载 test。

### Task 2：完成三种动力学模型训练

- [x] 训练 GRU-D Gaussian 强基线。
- [x] 训练 typed direct Gaussian。
- [x] 训练显式 typed predict-update belief dynamics。
- [x] 每个模型运行 2026/2027/2028 三个种子，共生成 9 个 checkpoint。

### Task 3：完成 interpolation、forecast 与 calibration 评估

- [x] 评估 normalized MAE、Gaussian NLL 和 CRPS。
- [x] 评估 50%/80%/90%/95% coverage 与区间宽度。
- [x] 评估 multilevel calibration error。
- [x] 检查 uncertainty 与预测误差、缺失程度的 Spearman 相关性。
- [x] 使用 subject-cluster bootstrap 比较候选模型与 GRU-D。

### Task 4：完成独立核验和阶段门控

- [x] 核验 fit-only normalization。
- [x] 核验 calibration/evaluation subject isolation。
- [x] 核验 ensemble total variance 和 variance temperature。
- [x] 输出 cognitive dynamics gate 和是否进入 MRI 的决定。

### Task 5：整理工程与科研状态

- [x] 更新 README、BUILD_STATUS、实验协议和门控记录。
- [x] 明确已运行实验、仅有代码接口的模块和数据限制。
- [x] 保留所有负结果与停止规则。
- [x] 确定下一阶段为 frozen cognitive robustness/clinical probing 和 MRI 数据就绪工作。

## Weekly Summary

三种子 ensemble 中，typed direct Gaussian 得到最佳点估计：normalized MAE **0.3609**、校准后 NLL **0.6061**；GRU-D Gaussian 分别为 0.3715 和 0.6163。typed direct 的总体 MAE 差置信区间仍跨零，但 interpolation MAE 改善为 **-0.01437，95% CI [-0.02971, -0.00419]**，支持其作为插值组件候选。

显式 typed predict-update belief 的 normalized MAE 为 0.3813，校准后 NLL 为 0.6919。虽然其 95% coverage 和 calibration error 达到绝对门槛，但准确率与 NLL 没有优于 GRU-D，因此 cognitive dynamics gate 为 false。月末决定为：保留 GRU-D 作为确认后的认知主模型，保留 typed direct Gaussian 作为 interpolation 任务特异候选，不在当前 validation 上继续调 semantic/belief architecture。

---

# Monthly Summary

本月已完成从真实数据整理到认知纵向建模的完整实验链：数据审计与 split、强 numeric baselines、item 编码核验、numeric/semantic 公平对照、response-aware 联合 Stage I、三轮 validation-only 表示学习开发，以及 irregular-time probabilistic forecasting 与 calibration。项目已经具备可复现的数据、配置、checkpoint、预测、bootstrap、门控和独立 verifier。

当前最可靠的科研结论是：

1. GRU-D 是现阶段确认后的认知预测主模型。
2. typed/item-conditioned representation 对 masked item 有稳定的任务特异收益。
3. typed direct Gaussian 对隐藏中间访视 interpolation 有统计支持的改善。
4. 固定 task semantics、response-aware semantics 和显式 predict-update belief 均未显示跨全部任务的稳定优势。
5. 同期 MRI spatial attention 的 subject/spatial/query 依赖和 test 点估计可复现，但总体 test CI 跨零；item-level 与纵向 MRI gate 也未通过，因此不开放 SFT/GRPO。

下一工作周期不能继续使用已经评估过的 test 进行模型选择。应只用原 train subjects 建立新的 development/confirmation 划分，预注册 diagnosis-conditioned grounding，或使用外部队列复核 AD/MCI 异质性；只有新的确认 gate 通过后才进入 structured reasoning SFT 和 GRPO。

## Key Deliverables

- 数据与工程状态：[`BUILD_STATUS.md`](../BUILD_STATUS.md)
- 项目说明：[`README.md`](../README.md)
- 正式 baseline：[`phase1_baseline_suite_formal_20261004/report.md`](../outputs/phase1_baseline_suite_formal_20261004/report.md)
- 联合 Stage I：[`phase1c_joint_stage1_20261004/report.md`](../outputs/phase1c_joint_stage1_20261004/report.md)
- 表示学习终止决定：[`phase1_representation_terminal_decision.md`](experiments/phase1_representation_terminal_decision.md)
- Belief/calibration：[`phase2_cognitive_belief_dev_20261010/report.md`](../outputs/phase2_cognitive_belief_dev_20261010/report.md)

## 报告期后 MRI 更新（2026-10-13）

MRI Phase 3A 已完成 train/validation-only global-anchor 实验。实验要求 MRI 与 cutoff visit 处于真实日期 90 天窗口内，并要求预测目标的实际量表日期晚于 MRI 扫描日期；四臂分别为 cognition-only、observed MRI、shuffled-subject MRI 和 shuffled-time MRI，共运行 4×3 个 checkpoint。实际纳入 6,465 个唯一 MRI，全部通过项目级可读性、有限值、图像/掩膜一致性和掩膜范围检查；四臂共用 1,877 个 validation samples 和每臂 7,580 个观测目标。

Observed-MRI ensemble 的 macro normalized MAE 为 0.3954，cognition-only 为 0.3795。两者配对误差差为 +0.01679，95% CI [0.00511, 0.02951]，说明 observed MRI 在当前全局表示下显著更差；它也未稳定优于两项 shuffled controls，12/24 月条件均未满足。因此冻结的 global MRI-anchor gate 为 false，测试集保持封存。后续 Phase 3B 通过单独冻结的表示协议开放了同期 grounding。完整结果见 [Phase 3A 报告](../outputs/phase3a_global_mri_anchor_dev_20261013/report.md)。

## MRI 表示更新（2026-10-14）

Phase 3B 将 MRI 表示能力与纵向融合解耦。实验使用 5,684/780 个 train/validation MRI anchors，比较 demographics、global MRI 和直接从 3D volume 训练的 SFCN，并加入错受试者 MRI 对照。同期 macro normalized MAE 为：SFCN 0.6031、global MRI 0.6736、demographics 0.7466。SFCN 相对 demographics 改善 -0.16736（95% CI [-0.23183, -0.11306]），相对 shuffled SFCN 改善 -0.26002（95% CI [-0.31901, -0.20226]）；两种 MRI 表示均通过同期信号门槛。

在控制 current score、age 和 sex 后，MRI 没有改善 12/24 月变化量预测：control 为 0.4648，SFCN 为 0.5187，global MRI 为 0.5489。因此项目现在可以进入单独冻结的 concurrent question-to-anatomy grounding 实验，但 longitudinal MRI fusion 仍保持关闭。测试集未加载。完整结果见 [Phase 3B 报告](../outputs/phase3b_mri_representation_dev_20261014/report.md)。

## 空间 Grounding 更新（2026-10-14）

Phase 3C 冻结三套已验证的 SFCN backbone，并比较参数匹配的 global-pooling 与 target-conditioned spatial-attention heads。Spatial attention 的 normalized MAE 为 0.5994，global pooling 为 0.6281；配对差为 -0.02219（95% CI [-0.04346, -0.00140]）。空间坐标置乱后误差为 0.6626，query 置乱后为 0.8054，错受试者 MRI 为 0.8630。

跨 seed attention map 的中位 Spearman 为 0.3956，top-20% token overlap 为 0.50。Phase 3C 的全部冻结门槛通过，下一步开放 multimodal concurrent grounding。当前 attention center 只报告对齐模板毫米坐标，不作 atlas-region 或纵向 MRI 声明；测试集保持封存。完整结果见 [Phase 3C 报告](../outputs/phase3c_concurrent_spatial_grounding_dev_20261014/report.md)。

## 多模态 item grounding 更新（2026-10-14）

Phase 3D 将 68 个已核验 cognitive item query 与冻结的 36×64 SFCN spatial tokens 结合，生成 22,728/3,120 个 train/validation masked-item 和 masked-instrument 样本。实验训练了等参的 cognition-only、global-MRI 和 item-spatial-attention 三臂，每臂三个 seed；被 mask 的 response 在模型输入中显式置零，test 保持封存。

Masked-item normalized MAE 分别为 cognition-only 0.3787、global MRI 0.3835、spatial attention 0.3841；masked-instrument 分别为 0.3696、0.3741、0.3776。Spatial attention 相对 cognition-only 的 overall 差为 +0.01106（95% CI [0.00618, 0.01633]），相对 global MRI 为 +0.00712（95% CI [0.00291, 0.01231]），均显著更差。正确受试者、空间顺序和 item query 均包含显著信息，但跨 seed attention 的中位 Spearman 仅 0.0185，top-20% overlap 为 0.25。冻结的 item-level multimodal gate 因此未通过。当前保留 Phase 3C 的同期总分 spatial grounding 结论，不开放 item-level 或纵向 MRI fusion。完整结果见 [Phase 3D 报告](../outputs/phase3d_multimodal_item_grounding_dev_20261014/report.md)。

下一工作周期应优先开展冻结 Phase 3C 模型的 missingness/clinical robustness，或使用新 development/confirmation split 冻结新的 item-level 协议；不在当前 validation subjects 上继续调同一架构。Structured reasoning SFT 和 GRPO 仍是后续阶段。

## 冻结 Test Confirmation 更新（2026-10-14）

Phase 3E 在冻结 Phase 3C checkpoints、队列规则、对照和统计门槛后，一次性加载 sealed test。全程无训练和 checkpoint 选择，纳入 404 名 test subjects、1,905 个唯一 MRI 和 7,843 个同期量表目标。

Spatial attention normalized MAE 为 0.6351，global pooling 为 0.6454；配对差为 -0.01088，但 95% CI [-0.02522, 0.00427] 跨零。五个终点中只有 MMSE 显著改善，未达到至少两个终点的冻结要求。错受试者、空间置乱和 query 置乱均显著降低表现，跨 seed attention 稳定性也成功复制（Spearman 0.4018，top-20% overlap 0.50），但完整确认 gate 仍为 false。

冻结亚组分析显示明显异质性：AD 受试者显著改善 -0.08484（95% CI [-0.12142, -0.04863]），MCI 受试者显著变差 +0.03082（95% CI [0.01265, 0.05157]）。由于训练模型没有预设 diagnosis conditioning，这只能作为探索性发现。已经使用的 test 不得转化为调参集；后续 diagnosis-conditioned 修订必须使用新的 development/confirmation split 或外部队列。完整结果见 [Phase 3E 报告](../outputs/phase3e_frozen_spatial_grounding_test_20261014/report.md)。
