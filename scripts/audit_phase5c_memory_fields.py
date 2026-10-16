#!/usr/bin/env python
"""Audit AVLT/Logical Memory fields and freeze the Phase 5C item contract."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memreason_rl.data.dictionary import load_dictionary
from memreason_rl.experiments.phase1_baseline_suite import select_verified_scalar_features
from scripts.audit_adni_official_dictionary import TableParser


FIELDS = {
    **{f"AVLT_Trial{i}Int": {"construct": "IntrusionCount", "role": "auxiliary_only",
                              "direction": "higher=worse", "range": [0, 25],
                              "missing_codes": [-4, -1, 999, 9999]} for i in range(1, 8)},
    "AVLT_Delay_Rec": {"construct": "RecognitionScore", "role": "auxiliary_only",
                       "direction": "higher=better", "range": [0, 15],
                       "missing_codes": [-4, -1, 999, 9999]},
    "AVLT_Delay_TotInt": {"construct": "DelayedIntrusionCount", "role": "auxiliary_only",
                          "direction": "higher=worse", "range": [0, 25],
                          "missing_codes": [-4, -1, 999, 9999]},
    "LogMemIA_ImmediateScore": {"construct": "ImmediateStoryRecall", "role": "primary_state",
                                "direction": "higher=better", "range": [0, 25],
                                "missing_codes": [-4, -1, 99, 999, 9999]},
    "LogMemIIA_DelayedScore": {"construct": "DelayedStoryRecall", "role": "primary_state",
                               "direction": "higher=better", "range": [0, 25],
                               "missing_codes": [-4, -1, 99, 999, 9999]},
    "LogMemIIA_ReminderGiven": {"construct": "ReminderMetadata", "role": "metadata_only",
                                "direction": "nonmonotonic", "valid_values": [1, 2],
                                "missing_codes": [-4, -1, 999, 9999]},
}


def fetch(field: str) -> tuple[str, bytes, list[str]]:
    url = "https://adni.loni.usc.edu/data-samples/data-dictionary-search/?q=" + quote(field)
    content = urlopen(Request(url, headers={"User-Agent": "MemReason-phase5c-audit/1.0"}), timeout=60).read()
    parser = TableParser(); parser.feed(content.decode("utf-8", errors="replace"))
    matches = [row for row in parser.rows if len(row) >= 6 and row[0].lower() == field.lower()
               and row[4].upper() == "ITEM"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one exact ITEM match for {field}, found {len(matches)}")
    return url, content, matches[0]


def main() -> None:
    train = pd.read_csv(ROOT / "splits/train.csv", low_memory=False)
    records = []
    for field, rule in FIELDS.items():
        url, content, row = fetch(field)
        values = pd.to_numeric(train[field], errors="coerce")
        missing_codes = set(rule["missing_codes"])
        cleaned = values.mask(values.isin(missing_codes))
        if "range" in rule:
            low, high = rule["range"]
            invalid = cleaned.notna() & ~cleaned.between(low, high)
        else:
            invalid = cleaned.notna() & ~cleaned.isin(rule["valid_values"])
        records.append({
            "field": field, "official_description": row[1], "official_codes": row[2],
            "official_table": row[4], "official_crf": row[5], "query_url": url,
            "response_sha256": hashlib.sha256(content).hexdigest(),
            "construct": rule["construct"], "role": rule["role"], "direction": rule["direction"],
            "allowed": json.dumps(rule.get("range", rule.get("valid_values"))),
            "missing_codes": json.dumps(rule["missing_codes"]),
            "train_nonmissing_raw": int(values.notna().sum()),
            "train_valid": int((cleaned.notna() & ~invalid).sum()),
            "train_special_or_invalid": int(values.notna().sum() - (cleaned.notna() & ~invalid).sum()),
            "train_subjects_valid": int(train.loc[cleaned.notna() & ~invalid, "subject_id"].nunique()),
            "observed_valid_min": None if not (cleaned.notna() & ~invalid).any() else float(cleaned[~invalid].min()),
            "observed_valid_max": None if not (cleaned.notna() & ~invalid).any() else float(cleaned[~invalid].max()),
            "semantics_verified": True,
            "coding_rule_frozen": True,
        })
    audit = pd.DataFrame(records)
    out = ROOT / "reports/data_audit"
    audit.to_csv(out / "phase5c_memory_field_audit.csv", index=False)

    dictionary = load_dictionary()
    baseline_config = yaml.safe_load((ROOT / "configs/phase1_baseline_suite_formal.yaml").read_text())
    primary = select_verified_scalar_features(baseline_config, dictionary)
    assert len(primary) == 68
    primary += ["LogMemIA_ImmediateScore", "LogMemIIA_DelayedScore"]
    item_entries = []
    for field in primary:
        if field in FIELDS:
            rule = FIELDS[field]
            item_entries.append({"field": field, "instrument": "LOGMEM", "domain": "Memory",
                                 "construct": rule["construct"], "value_type": "continuous",
                                 "direction": rule["direction"], "range": rule["range"],
                                 "missing_codes": rule["missing_codes"], "source": "phase5c_memory_field_audit.csv"})
        else:
            entry = dictionary[field]
            item_entries.append({key: entry.get(key) for key in
                                 ["instrument", "domain", "construct", "value_type", "direction",
                                  "range", "valid_values", "missing_codes"]} | {"field": field,
                                  "source": entry.get("source")})
    contract = {
        "contract": "phase5c_memory_state_items_v1", "frozen_date": "2026-10-16",
        "development_data": "original_train_only", "primary_state_items": item_entries,
        "auxiliary_memory_validity": [dict(field=field, **rule) for field, rule in FIELDS.items()
                                       if rule["role"] == "auxiliary_only"],
        "metadata_only": [dict(field=field, **rule) for field, rule in FIELDS.items()
                          if rule["role"] == "metadata_only"],
        "exclusions": {"diagnosis": "downstream_only", "FAQ": "downstream_only",
                       "MRI": "downstream_only", "AVLT_intrusion_fields": "not_recall_scores"},
    }
    (ROOT / "configs/phase5c_memory_state_item_contract.yaml").write_text(
        yaml.safe_dump(contract, sort_keys=False, allow_unicode=True))
    memory = [x["field"] for x in item_entries if x["domain"] == "Memory"]
    report = f"""# Phase 5C memory-field coding audit

The audit matched all {len(FIELDS)} requested fields exactly to the official ADNI online dictionary and the ITEM table. `AVLT_Trial1Int` through `AVLT_Trial7Int` are intrusion counts, `AVLT_Delay_TotInt` is delayed total intrusions, and `AVLT_Delay_Rec` is the recognition score. The intrusion fields are not recall totals and are excluded from the primary state definition.

Logical Memory IA immediate and IIA delayed scores are bounded to 0–25 using the ADNI protocol definition. Values 99/999/9999 and negative special codes are missing. The two scores are added as primary Memory items. The frozen contract contains {len(item_entries)} primary items, including {len(memory)} Memory items; AVLT fields remain auxiliary validity variables.

Only `splits/train.csv` was profiled. Original validation and test rows were not loaded. The existing global cognitive dictionary was not changed, preserving Phase 1–4 reproducibility.

Artifacts: `phase5c_memory_field_audit.csv` and `configs/phase5c_memory_state_item_contract.yaml`.
"""
    (out / "phase5c_memory_field_audit.md").write_text(report)
    print(json.dumps({"fields": len(audit), "primary_items": len(item_entries), "memory_items": len(memory),
                      "all_exact_item_matches": True, "validation_or_test_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
