# IDA MRI Study Date linkage audit

## Source and linkage rule

The source file is `/home_data/home/zhouwl2023/project/MemReason/idaSearch_10_12_2026 (2).csv` (SHA-256 `f7c72a016de40ba9ae5f0b04554d19d5509e6d926f783c68ad8beaa7edd7db53`). It contains 245,025 unique Image IDs with `Subject ID`, `Study Date`, sequence description, and image type.

A date is attached to an MRI only when both conditions hold:

1. the numeric IDA `Image ID` exactly matches the MRI session metadata `image_id`; and
2. the normalized IDA `Subject ID` exactly matches the MRI subject.

All linked `Study Date` values parse with the explicit `%m/%d/%Y` format. No date is inferred from session month, folder year, file modification time, or NIfTI header.

## Coverage

| MRI source | scans | scans with Image ID | scans with Study Date | date coverage |
|---|---:|---:|---:|---:|
| ADNI Linear primary | 10,230 | 9,363 | 9,362 | 91.52% |
| surfGT alternate | 1,143 | 1,143 | 1,143 | 100% |
| UniSurf longitudinal alternate | 523 | 523 | 523 | 100% |

All 9,362 primary links have consistent subjects. Of the 868 undated primary scans, 867 have no Image ID in the session table. The remaining record is `130_S_0886/ses-M036`, Image ID `436705`, which is absent from this IDA export.

## Cognitive–MRI matching after date linkage

The primary matching protocol uses a 90-day window, prioritizes the same visit code, and permits a nearest true-date match when no same-visit scan exists. Same-visit matches outside the date window remain available for data audit but have `date_window_verified=false`.

| window | matched visits | true-date evaluable | within date window | median evaluable gap |
|---:|---:|---:|---:|---:|
| 30 days | 9,442 | 9,368 | 8,695 | 2 days |
| 60 days | 9,445 | 9,371 | 9,122 | 2 days |
| 90 days | 9,446 | 9,372 | 9,222 | 2 days |
| 180 days | 9,512 | 9,438 | 9,389 | 2 days |

Under the 90-day protocol, 9,436 matches use the same visit code and 10 additional matches use nearest true dates. Split-level matched visits are 6,600 train, 896 validation, and 1,950 test.

## Remaining limitation

IDA `Study Date` resolves temporal alignment but does not provide a clinical QC decision for the processed `affine.nii.gz` files. Scanner, field strength, and complete preprocessing/registration provenance also remain incomplete. Consequently, `main_grounding_eligible` remains false until a separate QC/provenance audit is completed.
