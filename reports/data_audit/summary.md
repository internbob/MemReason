# MemReason Phase 0 数据审计

原始数据：15453 visits，4316 subjects，161 columns。原始 CSV 未修改；缺失值未填补。

## MRI 数据源

主目录：`/public_bme2/bme-dgshen/WeilinZhou/AD_longitudinal/ADNI/ADNI_Linear`。全目录 10230 scans；认知队列内 10230 scans；同 subject + visit 匹配 9436 visits。

详见 `mri_root_comparison.csv`。保留全部认知访视；MRI 不作为纳入认知队列的条件。

## Split

按 subject 随机划分，seed=2026，70/10/20。所有 visits、events、MRI、future targets 随 subject 归属。主文件 `splits/train.csv`、`val.csv`、`test.csv` 保留原始 161 列及真实 MRI 路径。`*_mri_matched.csv` 是同步匹配子集，不能取代全部认知队列。

split  subjects  visits  subjects_with_mri  longitudinal_subjects  primary_mri  matched_visits  nearest_mri_month_visits  forecast_samples
train      3021   10764               1385                   1783         7153            6594                      6739             15533
  val       431    1516                186                    253          973             893                       919              2125
 test       864    3173                410                    532         2104            1949                      2008              4647

## 必须保留的限制

- 配套 MRI CSV 提供 image_id/session，没有实际 scan_date、scanner、field_strength 或 QC；均保持未知。不能将同访视匹配说成已验证 ≤90 天。30/60/90/180 天敏感性暂不可计算。
- MRI 名义时间只用于明确标记的探索性事件流。正式 grounding 还需真实采集日期和 QC。MRI header 检查不等于 QC。
- ADAS/CDR/FAQ/MMSE/MoCA 的 107 个项目字段已与 ADNI 官方字段和表名精确匹配；99 个通过编码与观测值核验，8 个 MMSE 拼写字段因官方在线字典缺少可审计响应编码而保持关闭。MoCA DELW 已核验为非单调提示类别，不进入标量公平对照。
- RAVLT Trial*Int 不按学习词数解释；原值保留在 CSV 和事件证据中，默认不进入 state。999/9999 及候选范围异常单独记录在 `invalid_observations.csv`，原始表不改写。
- FAQ、诊断、总分不进入 memory state encoder；CDR function items 只作 outcome。源文件 presence flags 保留，availability 另按实际 payload 统计。
- `*_userdate` 不作为采集日期；缺失真实日期时，time_source 标为 nominal。原始 ITEM 的可用测量日期按原合并规则补回，未创造日期。
- 原始 study phase 按 RID+instrument date 回溯，匹配不到仍未知。缺失模式预测 diagnosis 的检查见后续 leakage audit。

## Gate 状态

Phase 0 数据与 split 构建完成；20206 个观测被标记为候选范围异常/特殊码。Item 字典核验与 Phase 1B numeric/semantic 公平对照已经完成；固定 task-semantics descriptor 未通过语义价值门控。MRI、SFT 与 RL 仍须依规范通过各自实证 gate。
