# 项目构建状态

当前交付范围：根据 Implementation Spec 建立真实数据管线和可运行的建模 v0，生成可复现 CSV split，并完成 Phase 1A–1F、cognitive-only irregular-time belief/calibration、MRI Phase 3A–3N、reasoning Phase 4A–4F、Phase 5B frozen error decomposition、Phase 5C–5H memory-state measurement development/audit、Phase 5E frozen reasoning-faithfulness/renderer-ablation audit，以及 train-only exploratory Phase 6A latent-state dynamics。

| 部分 | 当前状态 | 证据/限制 |
|---|---|---|
| 目录核对 | 完成 | 三个真实目录逐文件索引，`reports/data_audit/mri_root_comparison.csv` |
| 原始量表审计 | 完成 | 15,453 visits / 4,316 subjects / 161 columns；逐列取值、缺失、co-occurrence、随访分布 |
| subject-wise split | 完成 | 3,021 / 431 / 864 subjects；原始 161 列保留，不插值、不填缺失 |
| MRI 日期与配对 | 真实日期连接、窗口统计及项目技术 QC 完成 | IDA Image ID+subject 精确连接 9,362/10,230 scans；90 天主协议 9,446 visits，其中 9,222 条通过真实日期窗口；Phase 3A 纳入的 6,465 scans 全部通过可读性/有限值/image-mask 一致性/掩膜范围检查；不作临床 QC 声明 |
| MRI 月份归类 | 完成 | 每条量表访视按同一受试者最近 `ses-Mxxx` 归类；提供全部记录、±6 月子集和 0/3/6/12 月敏感性统计 |
| 异步事件/未来索引 | 完成 | 独立 instrument 日期、明确 nominal 时间、严格 cutoff、未来目标仅来自实际观测 |
| 数据字典 | 完成官方核验 | 107/107 字段与官方表精确匹配；99 个字段通过编码/观测核验；8 个 MMSE 拼写字段保持关闭；68 个单调标量字段进入 Phase 1B |
| Numeric/masked learning v0 | 已实现并作短程检查 | CLI、train/val curves、checkpoint、per-item metrics、failure cases |
| Phase 1A 正式 numeric forecasting | 完成 | LOCF/LME/GRU-D/Transformer；3 seeds；逐终点/时间窗指标；subject bootstrap；测试集仅在协议冻结后评估 |
| 扩展正式 baseline suite | 完成 | 68 个已核验输入；LOCF/LME/GRU/LSTM/T-LSTM/GRU-D/Transformer；15 checkpoints；GRU-D 由验证集选中，测试 macro normalized MAE 0.3799 |
| Phase 1B semantic fair comparison | 完成 | numeric/name/task-semantics 三组使用相同输入、架构、训练流程及 269,325 个可训练参数；18 checkpoints；未来预测语义组略差、两项 cross-scale 的 CI 均跨 0，门控未通过 |
| Phase 1C response-aware joint Stage I | 完成，门控未通过 | masked item、masked instrument、两项 cross-scale 与 future forecasting 联合训练；3 arms × 3 seeds；每臂 290,993 参数；9 checkpoints；response-aware 仅一项 cross-scale 改善，masked 两项显著变差，future 未满足非劣界值 |
| Phase 1D–1F representation development | 完成，terminal validation 决定未通过 | 全程未加载 test；typed/item-conditioned、metric-aligned loss、共享/任务独立 gate 依次验证；Phase 1F semantic macro 0.38334 vs numeric 0.38826，masked item 显著改善，但完整 Gate 1 仍未满足，停止在当前 validation 调参 |
| Cognitive-only belief dynamics/calibration | 完成，显式 belief gate 未通过 | train 内部 fit/tune 与 validation calibration/evaluation 均按受试者隔离；forecast 6/12/24/36 月 + 隐藏中间访视 interpolation；3 arms × 3 seeds、9 checkpoints。typed direct Gaussian 点估计最佳且 interpolation MAE 显著改善，但总体 CI 跨 0；显式 predict-update belief 的 accuracy/NLL 门槛失败 |
| Semantic branch | 已实现并正式比较固定及 response-aware descriptor | 本地冻结 Bio_ClinicalBERT；结果不支持当前 descriptor 融合方式的稳定增益 |
| MRI Phase 3A global anchor | 完成，冻结 gate 未通过 | train/val 13,775/1,877 samples；4 arms × 3 seeds、12 checkpoints；observed MRI normalized MAE 0.3954 vs cognition-only 0.3795，配对差 +0.01679 [0.00511, 0.02951]；test 未加载 |
| MRI Phase 3B representation validation | 完成；同期信号通过，纵向增量未通过 | 6,464 MRI；3 representations × 3 seeds、9 checkpoints；SFCN concurrent 0.6031 vs demographics 0.7466，并显著优于 subject shuffle；future-change 0.5187 vs control 0.4648 |
| MRI Phase 3C spatial grounding | 完成，冻结 gate 通过 | 3 frozen SFCN backbones；2 parameter-matched heads × 3 seeds、6 checkpoints；attention 0.5994 vs global 0.6281；空间/query 置乱显著变差；稳定性门槛通过 |
| MRI Phase 3D multimodal item grounding | 完成，冻结 gate 未通过 | 68 verified items；22,728/3,120 train/val masked samples；3 等参臂 × 3 seeds、9 checkpoints；spatial attention overall 显著差于 cognition-only/global MRI，且跨 seed attention 不稳定；test 未加载 |
| MRI Phase 3E frozen test confirmation | 完成，确认 gate 未通过 | 404 test subjects、1,905 MRI、7,843 targets；无训练；attention 0.6351 vs global 0.6454，差 -0.01088 [-0.02522, 0.00427]；仅 MMSE 显著，机制对照/稳定性通过 |
| MRI Phase 3F diagnosis cross-fit | 完成，异质性 gate 未通过 | 仅原 train 的 1,349 subjects、5,684 MRI、23,392 targets；五折 OOF、30 checkpoints；AD 在 5/5 folds 受益，MCI 也在 5/5 folds 受益，未复现 test 的 MCI harm；cross-seed top-token overlap 0.375 未达 0.40 |
| MRI Phase 3G concurrent diagnosis | 完成；低分辨率历史基线，MRI 信号通过、认知增量 gate 未通过 | SFCN 使用 `48×56×48` 三线性插值缓存；仅原 train 的 1,328 subjects、5,370 MRI-linked CN/MCI/AD visits；五折 subject-wise OOF、15 个 checkpoints。SFCN MRI macro-AUC 0.6984；该结果不再作为主 MRI 结论 |
| MRI Phase 3H MCI conversion | 完成；低分辨率 exploratory 历史基线，MRI 认知增量未通过 | SFCN 使用 `48×56×48` 缓存；24/36 月 cognition AUC 0.8117/0.8522，SFCN MRI 0.6031/0.6057；ACC/AUC/SEN/SEC/F1 和 bootstrap 均已完成，但后续必须以 native-resolution 复核 |
| MRI Phase 3I native-resolution AD-vs-CN | 完成；MRI 信号明确，严格 sanity gate 窄幅未通过 | 896 subjects（CN 503/AD 393）、完整 `193×229×193` 1-mm MRI、每人一张最早合格扫描、无插值/裁剪/填充、五折 OOF。native MRI ACC/AUC/SEN/SEC/F1=0.7221/0.7986/0.7074/0.7336/0.6907；demographics+MRI AUC=0.7997 [0.7701, 0.8289]，距预设 0.8000 门槛 0.0003，故不自动扩展三分类/MCI 转化 |
| MRI Phase 3J native longitudinal gated fusion | 完成，MRI 增量 gate 未通过 | 5,589 次完整 `193×229×193` MRI、960 subjects；185 名 MRI-linked evaluation subjects、8,718 个 forecast target rows。gated fusion NMAE 0.39572 vs frozen cognition 0.39691，paired difference -0.00169 [-0.00296, -0.00047]，未达到 0.005 最小效应且相对 shuffled MRI 的 CI 跨 0；24/36 月 MCI conversion AUC 均未超过 cognition。multimodal reasoning 保持关闭，test 未加载 |
| MRI Phase 3K/3L center-cropped classification | 原 MRI 结果失效；三分类已由 Phase 3N 重算，MCI 转化仍待重算 | 2026-10-18 的 Phase 3M 一致性审计发现旧裁剪缓存存在全零体积：896 个匹配扫描中仅 248 个非空、648 个全零。原 Phase 3K/3L 的 MRI-only、融合和 shuffled-MRI 结果均不得用于论文；cognition-only 不受影响 |
| MRI Phase 3M matched preprocessing sensitivity | 完成，严格非劣 gate 窄幅未通过 | 固定 Phase 3I 的 896 subjects、标签、五折、SFCN 和预算，仅比较完整 `193×229×193`、无插值 `160×192×160` crop 与 `48×56×48` 历史输入。重建 crop cache 后 896/896 非空且逐样本为完整网格的精确中心子数组。full/crop/low-res MRI AUC=0.7986/0.7960/0.8135；crop-full 差 -0.00257 [-0.02872, 0.02327]，下界未达到 -0.02；独立 verifier 通过，validation/test 未加载 |
| MRI Phase 3N corrected cropped three-class diagnosis | 完成；MRI 信号通过、认知增量 gate 未通过 | 从原始 T1/mask 路径重建全部 1,328 subjects、5,370 scans 的 `160×192×160` 无插值中心裁剪；5,370/5,370 行有限且非空，原始路径抽样逐元素一致；五折 × 3 seeds、15 checkpoints。MRI-only ACC/AUC/SEN/SEC/F1=0.4973/0.6854/0.4869/0.7318/0.4924；cognition-only=0.8826/0.9739/0.8924/0.9409/0.8828；融合 AUC 0.9497，低于 cognition-only；2,000 次受试者 bootstrap 与独立 verifier 均完成，validation/test 未加载 |
| 后续 MRI 预处理规范 | 待 adaptive crop 验证 | Phase 3N 证明完整队列固定裁剪可以被可靠读取和训练，但 Phase 3M 的严格 full-vs-crop 非劣 CI 未通过，故它仍不是正式默认预处理。下一项预处理实验应验证按脑掩膜得到的 adaptive bounding box；已损坏的 Phase 3K/3L MRI 结果继续禁止使用 |
| 3D / multimodal grounding | 保留 Phase 3C validation-only 证据 | Phase 3E 未确认总体增益；Phase 3D 未产生 item-level 增益；不开放 item-level/longitudinal grounding 或 multimodal SFT 声明 |
| Probabilistic forecasting | cognitive-only validation 开发实验完成 | Gaussian NLL/CRPS、50/80/90/95% coverage、PIT、history/missingness strata、subject bootstrap 和独立 verifier 均已生成；test 未加载，显式 belief 未通过冻结门槛 |
| Multi-candidate future | 已实现 latent sampling API | 没有候选多样性/校准结论 |
| Phase 4A structured reasoning | 完成，cognitive-only gate 通过 | 原 train 内 1,336/167/168 subjects 的 build/dev/confirmation；3 个冻结 GRU-D；16,577 份五段 JSON；confirmation 1,757 份，全部机械一致性指标 1.0、未来证据泄漏 0；独立 source-level verifier 通过 |
| Phase 4B structured-reasoning SFT | 完成，严格 gate 未通过 | 本地 DeepSeek-R1-Distill-Qwen-1.5B rank-16 LoRA；2,471/488/168 train/dev/confirmation；raw JSON validity 0%→98.81%，exact report 77.98%；2 条达到 token 上限而截断，未满足 100% format 与 99% evidence/anchor 门槛 |
| Phase 4C constrained decoding | 完成，冻结 gate 通过 | Phase 4A internal dev 167 subjects 选择 token-trie `grammar_model`；在 reasoning-new original-validation 234 subjects 上一次性确认：JSON/evidence/numeric/time/forecast/uncertainty 1.0，state/trend 0.9946/0.9940；原始 SFT JSON 0.9658；独立 verifier 通过 |
| Phase 4D constrained GRPO | 完成，internal-dev gate 未通过 | 64 rollout subjects、group=4、256 个真实 on-policy candidates、32 updates、4 个 checkpoints；best update 32 action composite 0.97246 vs SFT 0.97784；KL/reward variation/safety 通过，但改善门槛失败；test 未加载 |
| Phase 4E trajectory-action GRPO | 完成，新 development gate 未通过 | 256/96/256 名互斥 action-SFT/rollout/development subjects；32 updates、256 candidates；best NMAE 0.39970，冻结 GRU-D 为 0.35108，差 +0.04862 [0.03620, 0.06064]；test 未加载 |
| Phase 4F sealed-test confirmation | 完成，完整 gate 窄幅未通过 | 495 test subjects、4,951 samples、20,007 targets；GRU-D macro NMAE 0.37510，显著优于 LOCF；grammar JSON/evidence/anchors 全部通过，但 nested temporal consistency=0.98990，略低于 0.99000 |
| Phase 5B frozen error decomposition | 完成，post-test descriptive | 495 reports、4,950 domain-field decisions；state/trend mismatch 36/100；Memory trend 8.08%，expected improve 30.0%；输入哈希、完整分母、reference labels 与 Phase 4F 决定均独立复核；不允许 test 后调参 |
| Phase 5C interpretable memory-state redesign | Stage 0/1 完成，measurement gate 未通过 | AVLT/Logical Memory audit 与 70-item contract 完成；original-train 五折 nested OOF：unconstrained AE item/instrument 0.4446/0.4539，probabilistic domain 0.6782/0.5507；cross-view Spearman -0.0776，停止当前结构进入 dynamics |
| Phase 5D anchored bifactor MIRT | 完成，measurement gate 未通过 | train-only 五折 nested OOF；typed global+Memory IRT item/instrument 0.5181/0.5273，autoencoder 为 0.4446/0.4539；Memory cross-view CCC 0.3713 [0.3547, 0.3873]；8 个稳定 Memory anchors 来自 ADAS/MMSE，但预测与一致性门槛失败，dynamics 保持关闭 |
| Phase 5E reasoning faithfulness / renderer ablation | 完成，post-test frozen audit | 495 test reports；1,013 个 leave-one-visit 前向候选；278 名可配对受试者中最高影响/随机删除预测位移 0.2244/0.0811，差 0.1432 [0.1191, 0.1706]；全部反事实 IR 通过 verifier、零删除证据残留和零未来泄漏；executable renderer 与 teacher 完全一致，LLM renderer 无事实忠实度优势 |
| Phase 5F hierarchical bridge measurement | 完成，预测 gate 通过、measurement gate 未通过 | original-train 五折 nested OOF；hierarchical bridge held-out item/instrument NMAE 0.3989/0.4166，优于 fixed score 与 autoencoder；Memory CCC 0.6340 [0.6207, 0.6467]，但下限未达冻结 0.70 门槛 |
| Phase 5G bridge Pareto / residual gate | 完成，train-only development，measurement gate 仍关闭 | 11 variants、55 fits；在 +0.010 prediction tolerance 下选择 `ungated_w0p1`，held-out-instrument NMAE 0.4180，Memory CCC 0.6296 [0.6163, 0.6423]；residual-adaptive gate 未支配 ungated Pareto frontier |
| Phase 5H measurement identifiability audit | 完成，train-only audit，measurement gate 仍关闭 | 4 个量表复合指标、6 个同期量表对；Phase 5G state CCC 0.6296 [0.6157, 0.6433]；衰减上限中位数 0.7711，但仅 4/6 对支持 CCC≥0.70，故不能把统一 0.70 阈值视作所有量表对均可辨识，也不重开 Phase 5F gate |
| Phase 6A latent-state dynamics | 完成，train-only exploratory，校准 gate 未通过 | 五折 subject cross-fitting、7 臂、30 checkpoints、15,502 forecast samples/arm；reference+MRI filter latent MAE 0.0905 vs latent LOCF 0.0949，participant difference -0.00346 [-0.00436, -0.00255]；真实 MRI 比置乱 MRI 改善 0.00051，但 95% coverage=0.8360，未满足预设校准范围；不作 confirmatory continuous-time claim |
| RLVR / GRPO | 两项真实 cognitive-only 实验均完成，不支持增益声明 | Phase 4D 的 state/trend action 与 Phase 4E 的 forecast/uncertainty action 均未超过冻结基线；Phase 4C 仍为支持策略，multimodal GRPO 关闭 |
| Leakage audit | 完成初筛 | 仅量表缺失模式预测 diagnosis，验证 balanced accuracy 0.420；应继续按 phase 控制协议混杂 |
| 科研 gates | Phase 4A/4C 通过；Phase 4F 仅 temporal 差 1e-4 未通过；Phase 4B raw、Phase 4D/4E GRPO、MRI/dynamics 未通过 | test 后不得再调；论文可报告冻结 GRU-D 的确认结果与 Phase 4C 的结构完整性改善，但不能声称完整 confirmatory gate 通过 |

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
- `reports/data_audit/ida_scan_date_linkage.md`：IDA Study Date 与主 MRI 的 Image ID+subject 精确连接、缺失记录及 30/60/90/180 天窗口统计。
- `reports/experiments/phase1b_semantic_fair_protocol.md`：测试集启用前冻结的三臂公平性、任务、统计比较和停止规则。
- `outputs/phase1b_semantic_fair_20261003/`：18 个 checkpoint、完整 val/test predictions、逐任务指标、2,000 次 subject-cluster bootstrap、报告及门控决定。语义价值门控未通过。
- `reports/experiments/phase1c_joint_stage1_protocol.md`：测试任务、三臂公平性、联合优化、统计比较和 Stage I 停止规则的冻结协议。
- `outputs/phase1c_joint_stage1_20261004/`：9 个 checkpoint、完整 val/test predictions、masked item/instrument 类型指标、10 组 2,000 次 subject-cluster bootstrap、报告、门控及独立核验。Stage I gate 未通过。
- `outputs/phase1d_typed_representation_dev_20261004/`：typed response、item-conditioned decoder 和 two-stage 对照；4 个 validation-only checkpoint，test 未加载。
- `outputs/phase1e_gated_aligned_dev_20261009/`：per-item macro/metric-aligned reconstruction 与零初始化 semantic gate；3 个 validation-only checkpoint，masked item 显著改善。
- `outputs/phase1f_task_gated_dev_20261009/`：任务独立 value/semantic gates；3 个 validation-only checkpoint，记录当前 validation 上的 terminal representation decision。
- `reports/experiments/phase1_representation_terminal_decision.md`：停止在当前 validation 继续 semantic 调参的证据、解释和允许的后续路线。
- `reports/experiments/phase2_cognitive_belief_dev_protocol.md`：fit/tune、calibration/evaluation 受试者隔离，三臂动力学、插值/外推、校准和冻结门槛。
- `outputs/phase2_cognitive_belief_dev_20261010/`：9 个 checkpoint、5,936 个 evaluation targets/arm、三种子总预测方差、组内 variance temperature、完整 calibration/strata、6 组 subject bootstrap、报告和独立核验。test 未加载；显式 belief gate 为 false。
- `reports/experiments/phase3a_global_mri_anchor_dev_protocol.md`：真实日期 90 天窗口、target-after-scan 时间约束、项目技术 QC、四臂公平对照、shuffled negative controls 和冻结门槛。
- `outputs/phase3a_global_mri_anchor_dev_20261013/`：6,465 个 MRI 的 QC/全局特征、12 个 checkpoint、7,580 个 validation targets/arm、9 组 subject-cluster bootstrap、报告与 gate。test 未加载；global MRI anchor gate 为 false。
- `reports/experiments/phase3b_mri_representation_dev_protocol.md`：MRI 表示与纵向融合解耦、同期 cognition、subject-shuffle、12/24 月变化量和双路径决定协议。
- `outputs/phase3b_mri_representation_dev_20261014/`：6,464 个 3D MRI cache、9 个 checkpoint、3,237 个同期 validation targets、4,482 个 future-change targets、bootstrap、报告和独立核验。concurrent grounding ready=true，longitudinal fusion ready=false。
- `reports/experiments/phase3c_concurrent_spatial_grounding_dev_protocol.md`：冻结 SFCN、参数匹配 global/attention heads、subject/spatial/query controls、attention stability 和 gate。
- `outputs/phase3c_concurrent_spatial_grounding_dev_20261014/`：三套 36×64 空间特征、6 个 checkpoint、三 seed attention maps、24 组 bootstrap、模板坐标中心、报告和独立核验。concurrent spatial grounding gate=true。
- `reports/experiments/phase3d_multimodal_item_grounding_dev_protocol.md`：68 个 verified item query、masked item/instrument、等参 cognitive/global/spatial 三臂、三类破坏对照及稳定性门槛。
- `outputs/phase3d_multimodal_item_grounding_dev_20261014/`：9 个 checkpoint、413,140 行 seed/ensemble 预测、15 组 subject-cluster bootstrap、44,265 行跨 seed attention 稳定性、报告与独立核验。item-level multimodal gate=false，test 未加载。
- `reports/experiments/phase3e_frozen_spatial_grounding_test_protocol.md`：加载 test 前冻结的 cohort、六个 Phase 3C checkpoints、四项对照、统计门槛和亚组分析。
- `outputs/phase3e_frozen_spatial_grounding_test_20261014/`：1,905 test volumes、三套 spatial-token cache、188,232 行 seed/ensemble 预测、24 组 primary bootstrap、28,575 行 attention stability、诊断/时间/性别/年龄/availability 亚组结果和独立核验。test confirmation=false。
- `reports/experiments/phase3f_diagnosis_heterogeneity_crossfit_protocol.md`：只用原 train subjects、五折 subject cross-fitting、冻结 epoch、诊断交互、机制对照和稳定性 gate。
- `outputs/phase3f_diagnosis_heterogeneity_crossfit_20261014/`：30 个 fixed-epoch checkpoints、完整 OOF seed/ensemble 预测、诊断/交互 bootstrap、fold-wise effects、attention maps 和独立核验。diagnosis heterogeneity replicated=false。
- `reports/experiments/phase3g_mri_diagnosis_crossfit_protocol.md`：只用原 train subjects 的并发 CN/MCI/AD 诊断协议；真实日期 90 天窗口、五折 subject-wise cross-fitting、MRI shuffle 对照、校准、决策曲线和预设停止规则。
- `outputs/phase3g_mri_diagnosis_crossfit_20261016/`：1,328 人、5,370 个 MRI-linked visits、15 个 SFCN checkpoints、7 臂 OOF 预测、2,000 次 subject-cluster bootstrap、亚组/决策曲线、报告和独立核验。MRI 相对 demographics 有信号，但未增加 cognition-only 的诊断性能；MCI conversion 未开放。
- `reports/experiments/phase3h_mci_conversion_crossfit_protocol.md`：用户明确要求后单独冻结的 exploratory MCI-to-AD 预后协议；24/36 月精确窗口、完整随访阴性定义、五折 subject-wise cross-fitting 和七臂公平对照。
- `outputs/phase3h_mci_conversion_crossfit_20261017/`：24/36 月共 30 个二分类 SFCN checkpoints、完整 OOF seed/ensemble 预测、ACC/AUC/SEN/SEC/F1 与 2,000 次 bootstrap CI、配对 AUC 对照、联合论文图和独立核验。MRI-only 有有限预后信号，但未超过 cognition-only。
- `reports/experiments/phase3k_cropped_mri_classification_protocol.md`：冻结中心裁剪、全样本保留、原受试者/标签/五折复用、四臂对照及统计口径；对应旧缓存现已确认损坏。
- `outputs/phase3k_cropped_mri_diagnosis_crossfit_20261017/` 与 `outputs/phase3l_cropped_mci_conversion_crossfit_20261017/`：保留为失败 provenance；MRI 相关结果因缓存中大量全零体积而失效，不得引用，等待全队列重建与重算。
- `reports/experiments/phase3m_mri_preprocessing_sensitivity_protocol.md` 与 `outputs/phase3m_mri_preprocessing_sensitivity_20261018/`：896 名受试者的 full/crop/low-res 匹配五折实验、10 个新 checkpoints、全行缓存审计、ACC/AUC/SEN/SEC/F1、2,000 次配对 bootstrap、掩膜保留分层、资源统计、拒绝尝试 provenance 和独立 verifier。固定 crop 非劣 gate=false。
- `reports/experiments/phase4a_frozen_cognitive_reasoning_protocol.md`：原 train 内部 subject split、build-only normalization、冻结 GRU-D ensemble、五段 JSON、未来真值隔离和 Phase 4B 进入 gate。
- `outputs/phase4a_frozen_cognitive_reasoning_20261014/`：3 个 GRU-D checkpoints、完整 seed/ensemble forecasts、16,577 份 structured JSONL、逐报告核验、confirmation preview、报告、gate 和独立 source-level verification。Phase 4B SFT ready=true。
- `reports/experiments/phase4b_cognitive_reasoning_sft_protocol.md`：本地 causal LLM、subject-balanced 选择、completion-only LoRA、raw generation 对照与严格 SFT gate。
- `outputs/phase4b_cognitive_reasoning_sft_20261014/`：47MB LoRA adapter、base/SFT raw generations、168-subject metrics、2,000 次 subject bootstrap、failure summary、gate 报告及独立 verification。Phase 4B SFT gate=false。
- `reports/experiments/phase4c_grammar_constrained_decoding_protocol.md`：dev-only token-trie decoder 选择、固定 arm priority、新受试者一次性确认和严格结构化 gate。
- `outputs/phase4c_grammar_constrained_decoding_20261015/`：167-subject dev、234-subject reasoning-new confirmation、raw/constrained generations、2,000 次 subject bootstrap、报告、gate 和独立 verification。Phase 4C gate=true，cognitive GRPO protocol ready=true。
- `reports/experiments/phase4d_cognitive_constrained_grpo_protocol.md`：subject-isolated rollout/dev/confirmation 顺序、availability-aware reward、clip/KL、停止规则和 test-opening gate。
- `outputs/phase4d_cognitive_constrained_grpo_20261015/`：256 个真实 on-policy rollouts、32-update curves、4 个 LoRA checkpoints、835 条 dev generations/metrics、停止决定、报告及独立 verification。dev improvement gate=false，原 test 未加载。
- `reports/experiments/phase4e_trajectory_action_grpo_protocol.md`：新的 subject partition、有界 residual/uncertainty action、真实 future reward、KL/clip/停止规则及 original-test opening gate。
- `outputs/phase4e_trajectory_action_grpo_20261015/`：action-SFT adapter、256 个真实 on-policy candidates、32-update curves、4 个 checkpoints、1,536 条 development action 记录、subject bootstrap、动作分布诊断、报告及独立 verification。development gate=false，原 test 未加载。
- `reports/experiments/phase4f_frozen_grud_phase4c_test_protocol.md`：在 test 加载前冻结的 checkpoint hashes、reasoning cohort、raw/grammar controls、forecast/reasoning metrics 与一次性确认门槛。
- `outputs/phase4f_frozen_grud_phase4c_test_20261015/`：三 seed test forecasts、495 份 subject-balanced raw/grammar/executable/teacher reports、forecast 与 reasoning bootstrap、robustness strata、pretest lock、报告和独立 verification。只有 temporal consistency 门槛失败，test 后不再调参。
- `reports/experiments/phase5b_frozen_reasoning_error_decomposition_protocol.md`：冻结 Phase 4F 输出、decision-level 分母、阈值距离、分层因素、subject bootstrap 和 post-test interpretation 规则。
- `outputs/phase5b_frozen_reasoning_error_decomposition_20261016/`：4,950 个重建 decisions、完整 confusion/rate/near-far 表、2,000 次 subject-cluster bootstrap、PDF/PNG 论文图、报告和独立 verification；无训练、无重新生成、Phase 4F 决定不变。
- `reports/experiments/phase5e_reasoning_faithfulness_audit_protocol.md`：冻结 leave-one-visit、随机匹配、time-gap reversal、Memory score reflection、renderer ablation 和 post-test 解释边界。
- `outputs/phase5e_reasoning_faithfulness_audit_20261016/`：1,013 个冻结 GRU-D 候选前向、四种反事实、2,000 次 subject bootstrap、四臂 renderer 消融、typed reasoning IR、报告及独立 verification。该审计不重开 Phase 4F gate。
- `reports/methods/memreason_state_paper_and_method_reframing.md`：以可解释 memory state 为核心的论文问题、数学结构、reasoning 定义、实验矩阵、可保留证据和不可声称结论。
- `reports/experiments/phase5c_interpretable_memory_state_dev_protocol.md`：train-only nested development、state validity、faithfulness 和 external-validation gate；Stage 0/1 已执行并记录 negative measurement decision，当前架构未开放 Stage 2。
- `reports/data_audit/phase5c_memory_field_audit.md` 与 `configs/phase5c_memory_state_item_contract.yaml`：12 个 AVLT/Logical Memory 字段的官方语义、特殊码和角色核验；冻结 70-item state contract。
- `outputs/phase5c_memory_measurement_dev_20261016/`：五折 nested train-only measurement experiment，15 个 neural checkpoints、完整 OOF masked predictions、cross-view states、subject bootstrap、negative gate、报告和独立 verification；validation/test 未加载。
- `reports/experiments/phase5d_anchored_mirt_protocol.md`：global+Memory bifactor、typed likelihood、time/diagnosis/ADNI-phase DIF、anchor、nested folds、participant bootstrap 和 dynamics-opening gate。
- `outputs/phase5d_anchored_mirt_dev_20261016/`：15 个 MIRT checkpoints、1,348,975 条五臂 OOF predictions、29,247 条 cross-view records、4,300 个 DIF audit decisions、8 个稳定 Memory anchors、negative gate 和独立 verification；validation/test 未加载。

后续必须按规范推进，不能仅填写 gate 的布尔值来替代 artifact。Phase 4F 已一次性打开官方 test，冻结 GRU-D 的数值预测获得确认：macro NMAE 0.37510，且相对 LOCF 的 subject-mean 差为 -0.02144，95% CI [-0.03913, -0.00423]。Phase 4C grammar 在 test 上得到 100% JSON/evidence/forecast/uncertainty preservation、零幻觉和零未来泄漏，并显著改善 raw SFT；但 nested temporal consistency 为 0.98990，低于预注册 0.99000，因此完整 Phase 4F gate=false。Phase 5E 进一步显示最高影响访视删除产生的预测位移显著高于匹配随机删除，且 executable compiler 在所有反事实下机械通过 verifier；同时 deterministic executable renderer 与 teacher 完全一致，带 LLM 的 grammar renderer 没有额外事实忠实度。论文当前主方法应写成“冻结 GRU-D forecasting + typed executable reasoning IR/compiler/verifier”，LLM 只作可选 renderer。Phase 5B/5E 均为 post-test descriptive evidence，不重开 Phase 4F gate。Phase 5C 的 neural domain state 和 Phase 5D 的 anchored bifactor MIRT 均未同时获得跨量表一致性和预测非劣性；不得继续扩展成 continuous-time dynamics 或声称已学到有效 memory state。Phase 4D/4E 的 GRPO 负结果和 MRI longitudinal/multimodal gate 继续保持不变。

源 CSV 和外部 MRI 均未改动。本次仅在 MemReason 目录创建代码、派生表和报告；公开仓库只导出不含受试者标识的聚合结果。
