"""Apply conservative official-ADNI item verification to the modeling dictionary."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memreason_rl.data.common import ROOT


MISSING_CODES = {-4.0, -1.0, 999.0, 9999.0}
RANGE_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*\.\.\s*(-?\d+(?:\.\d+)?)")
CODE_PATTERN = re.compile(r"(?:^|;)\s*(-?\d+(?:\.\d+)?)\s*=\s*([^;]+)")
CURATED_CONSTRUCTS = {
    "adas_q5score_cue": ("Language", "NamingWithCue"),
    "adas_q10score": ("Language", "SpokenLanguageComprehension"),
    "adas_q11score": ("Language", "WordFindingDifficulty"),
    "adas_q12score": ("Language", "LanguageAbility"),
}


def numeric_key(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def extract_codes(texts: list[str]) -> tuple[dict[float, str], list[tuple[float, float]]]:
    labels: dict[float, str] = {}
    ranges: list[tuple[float, float]] = []
    for text in texts:
        for low, high in RANGE_PATTERN.findall(text):
            ranges.append((float(low), float(high)))
        for value, label in CODE_PATTERN.findall(text):
            labels.setdefault(float(value), " ".join(label.split()))
    return labels, ranges


def direction(column: str) -> str:
    if column.startswith(("adas_", "cdr_", "faq_")):
        return "higher=worse"
    if column.startswith(("mmse_", "moca_")):
        return "higher=better"
    return "verify"


def main() -> None:
    audit_path = ROOT / "reports/data_audit/adni_official_dictionary_rows.csv"
    dictionary_path = ROOT / "configs/cognitive_item_dictionary.yaml"
    source_path = ROOT / "adni_all_longitudinal_scales_visit_with_ravlt_logmem.csv"
    rows = pd.read_csv(audit_path)
    data = pd.read_csv(source_path, low_memory=False)
    document = yaml.safe_load(dictionary_path.read_text())
    dictionary = document["items"]
    validation = []

    for column, group in rows.groupby("project_column", sort=True):
        labels, ranges = extract_codes(group.official_codes.fillna("").tolist())
        observed = set(pd.to_numeric(data[column], errors="coerce").dropna().astype(float).unique()) - MISSING_CODES
        has_codes = bool(labels or ranges)
        unsupported = sorted(
            value
            for value in observed
            if value not in labels and not any(low <= value <= high for low, high in ranges)
        )
        existing = dictionary.get(column)
        if existing and column in CURATED_CONSTRUCTS:
            domain, construct = CURATED_CONSTRUCTS[column]
            existing.update(
                domain=domain,
                construct=construct,
                use_for_state=True,
                use_for_aux=True,
                core_memory=False,
                default_drop=False,
            )
        semantics_curated = bool(existing and existing.get("domain") != "Unverified" and existing.get("construct") != "Unverified")
        coding_verified = has_codes and not unsupported
        verified = bool(coding_verified and semantics_curated)
        descriptions = [str(value) for value in group.official_description.dropna() if str(value).strip()]
        description = max(descriptions, key=len) if descriptions else ""

        validation.append(
            {
                "project_column": column,
                "matched_official_rows": len(group),
                "official_tables": "|".join(sorted(group.official_table.unique())),
                "official_phases": "|".join(sorted(group.official_phase.unique())),
                "official_description": description,
                "observed_nonmissing_values": "|".join(map(str, sorted(observed))),
                "official_enumerated_values": "|".join(map(str, sorted(labels))),
                "official_ranges": "|".join(f"{low}..{high}" for low, high in ranges),
                "unsupported_observed_values": "|".join(map(str, unsupported)),
                "coding_verified": coding_verified,
                "semantics_curated": semantics_curated,
                "verified_from_dictionary": verified,
            }
        )
        if not existing:
            continue
        existing.update(
            official_field=str(group.official_field.iloc[0]),
            official_table=str(group.official_table.iloc[0]),
            official_phases=sorted(group.official_phase.unique().tolist()),
            official_description=description,
            source=str(group.query_url.iloc[0]),
            verification_note=(
                "Exact field and ADNI source-table match; observed non-missing codes checked against all returned phase definitions."
                if coding_verified
                else "Official field/table matched, but coding could not be verified from returned codes or observed values."
            ),
            verified_from_dictionary=verified,
        )
        if description:
            existing["task_description"] = description
        if coding_verified:
            existing["direction"] = direction(column)
            if labels:
                existing["valid_values"] = [numeric_key(value) for value in sorted(labels)]
                existing["response_labels"] = {
                    str(numeric_key(value)): label for value, label in sorted(labels.items())
                }
                if set(labels) == {0.0, 1.0}:
                    existing["value_type"] = "binary"
                elif column.startswith("moca_delw"):
                    existing["value_type"] = "categorical"
                    existing["direction"] = "nonmonotonic_cue_category"
                else:
                    existing["value_type"] = "ordinal"
            if ranges:
                existing["range"] = [numeric_key(min(x[0] for x in ranges)), numeric_key(max(x[1] for x in ranges))]

    # Raw-answer fields are officially described, but remain excluded because
    # their numeric values are responses rather than comparable item scores.
    for key in ["moca_letters", *[f"moca_serial{i}" for i in range(1, 6)]]:
        if key in dictionary:
            dictionary[key]["default_drop"] = True
            dictionary[key]["use_for_state"] = False
            dictionary[key]["verification_note"] += " Raw response retained in source data and excluded from scored-state inputs."

    document["verification"] = {
        "authority": "ADNI official online Data Dictionary Search",
        "audit_rows": "reports/data_audit/adni_official_dictionary_rows.csv",
        "validation": "reports/data_audit/adni_official_dictionary_validation.csv",
        "policy": "Exact field+table match, official code specification, observed-code check, and curated construct are all required.",
    }
    dictionary_path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True))
    validation_frame = pd.DataFrame(validation)
    validation_frame.to_csv(ROOT / "reports/data_audit/adni_official_dictionary_validation.csv", index=False)
    print(
        {
            "officially_matched": int(validation_frame.project_column.nunique()),
            "coding_verified": int(validation_frame.coding_verified.sum()),
            "fully_verified": int(validation_frame.verified_from_dictionary.sum()),
            "still_excluded": int((~validation_frame.verified_from_dictionary).sum()),
        }
    )


if __name__ == "__main__":
    main()
