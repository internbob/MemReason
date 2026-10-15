# MemReason Phase 0 数据审计

原始数据：15453 visits，4316 subjects，161 columns。原始 CSV 未修改；缺失值未填补。

## MRI 数据源

主目录：`/public_bme2/bme-dgshen/WeilinZhou/AD_longitudinal/ADNI/ADNI_Linear`。全目录 10230 scans；其中 9362 scans 通过 Image ID 与 IDA Study Date 精确连接，868 scans 因缺少或未命中 Image ID 仍无真实日期；认知队列内 10230 scans；90 天主协议匹配 9446 visits。

详见 `mri_root_comparison.csv`。保留全部认知访视；MRI 不作为纳入认知队列的条件。

## Split

按 subject 随机划分，seed=2026，70/10/20。所有 visits、events、MRI、future targets 随 subject 归属。主文件 `splits/train.csv`、`val.csv`、`test.csv` 保留原始 161 列及真实 MRI 路径。`*_mri_matched.csv` 是同步匹配子集，不能取代全部认知队列。

split  subjects  visits  subjects_with_mri  longitudinal_subjects  primary_mri  matched_visits  nearest_mri_month_visits  forecast_samples
train      3021   10764               1385                   1783         7153            6600                      6739             15533
  val       431    1516                186                    253          973             896                       919              2125
 test       864    3173                410                    532         2104            1950                      2008              4647

## 必须保留的限制

- IDA 导出 `/home_data/home/zhouwl2023/project/MemReason/idaSearch_10_12_2026 (2).csv` 提供 Image ID 级 `Study Date`；仅在 Image ID 和 subject 同时一致时写入 `scan_date`。真实日期覆盖和 30/60/90/180 天匹配敏感性见 `mri_date_coverage.csv` 与 `mri_cognition_matching_stats.csv`。
- 未获得日期的 MRI 继续使用明确标记的 nominal session month。scanner、field strength 和临床 QC 仍未知，因此 `main_grounding_eligible` 继续为 false；MRI header 检查不等于 QC。
- ADAS/CDR/FAQ/MMSE/MoCA 的官方编码核验由独立脚本维护；重建 Phase 0 数据不会覆盖已核验字典。当前核验状态见 `reports/data_audit/dictionary_verification.md`；`verified_from_dictionary=false` 的字段不能进入正式语义实验。
- RAVLT Trial*Int 不按学习词数解释；原值保留在 CSV 和事件证据中，默认不进入 state。999/9999 及候选范围异常单独记录在 `invalid_observations.csv`，原始表不改写。
- FAQ、诊断、总分不进入 memory state encoder；CDR function items 只作 outcome。源文件 presence flags 保留，availability 另按实际 payload 统计。
- `*_userdate` 不作为采集日期；缺失真实日期时，time_source 标为 nominal。原始 ITEM 的可用测量日期按原合并规则补回，未创造日期。
- 原始 study phase 按 RID+instrument date 回溯，匹配不到仍未知。缺失模式预测 diagnosis 的检查见后续 leakage audit。

## Gate 状态

Phase 0 数据与 split 构建完成；278 个观测被标记为候选范围异常/特殊码。Item 字典核验与 Phase 1B 的状态由独立报告记录；MRI、SFT 与 RL 仍须依规范通过各自实证 gate。
