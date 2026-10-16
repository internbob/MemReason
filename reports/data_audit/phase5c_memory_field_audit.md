# Phase 5C memory-field coding audit

The audit matched all 12 requested fields exactly to the official ADNI online dictionary and the ITEM table. `AVLT_Trial1Int` through `AVLT_Trial7Int` are intrusion counts, `AVLT_Delay_TotInt` is delayed total intrusions, and `AVLT_Delay_Rec` is the recognition score. The intrusion fields are not recall totals and are excluded from the primary state definition.

Logical Memory IA immediate and IIA delayed scores are bounded to 0–25 using the ADNI protocol definition. Values 99/999/9999 and negative special codes are missing. The two scores are added as primary Memory items. The frozen contract contains 70 primary items, including 14 Memory items; AVLT fields remain auxiliary validity variables.

Only `splits/train.csv` was profiled. Original validation and test rows were not loaded. The existing global cognitive dictionary was not changed, preserving Phase 1–4 reproducibility.

Artifacts: `phase5c_memory_field_audit.csv` and `configs/phase5c_memory_state_item_contract.yaml`.
