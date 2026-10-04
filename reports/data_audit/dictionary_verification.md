# ADNI item encoding verification

The project cognitive fields were checked against the official ADNI online Data Dictionary Search on 2026-10-03. The audit requires an exact field name and source-table match; definitions returned for every available ADNI phase are retained with URL and response SHA-256.

## Result

- Project ADAS/CDR/FAQ/MMSE/MoCA fields queried: 107
- Exact official field + table matches: 107
- Official coding consistent with all observed non-missing values: 99
- Fully verified after construct review: 99
- Still closed: 8
- Verified state candidates: 73
- Scalar, monotonic inputs eligible for the Phase 1B fair comparison: 68

The eight closed fields are the seven raw MMSE spelling-letter fields and `mmse_worldscore`, whose online dictionary rows do not provide an auditable code specification. The raw MoCA serial and letter-vigilance fields have official descriptions but remain excluded because their numbers are responses/counts rather than comparable item scores.

MoCA `DELW1`–`DELW5` are now verified as four non-monotonic response categories: incorrect, correct without cue, correct with category cue, and correct with multiple-choice cue. They are stored as categorical variables and are not treated as binary or continuous. They are excluded from the scalar Phase 1B comparison so every arm receives the same defensible numeric response representation.

Primary evidence:

- Official search rows: `adni_official_dictionary_rows.csv`
- Observed-code validation: `adni_official_dictionary_validation.csv`
- Unmatched rows: `adni_official_dictionary_unmatched.csv` (header only; zero unmatched fields)
- Reproducible acquisition: `scripts/audit_adni_official_dictionary.py`
- Conservative application policy: `scripts/apply_adni_dictionary_verification.py`

Authoritative sources:

- https://adni.loni.usc.edu/data-samples/data-dictionary-search/
- https://adni.loni.usc.edu/quick-start-guide-asset/datadic.html
- https://adni.loni.usc.edu/wp-content/uploads/2008/07/ADNI_GO_Procedures_Manual_06102011.pdf

Verification applies to field meaning and encoding. It does not resolve clinical QC, visit timing, or MRI acquisition dates.
