"""Assign cognitive visits to the nearest available nominal MRI session month.

This is a nominal-month grouping operation, not a scan-date match.  It is useful
when the MRI directory exposes sessions such as ``ses-M000`` and ``ses-M006``
but no acquisition date.  Every source visit is retained in the all-visits
output; distant assignments remain explicitly ineligible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


ASSIGNMENT_COLUMNS = [
    "nearest_mri_scan_id",
    "nearest_mri_image_id",
    "nearest_mri_session_id",
    "nearest_mri_visit_month",
    "nearest_mri_visit_code",
    "nearest_mri_image_path",
    "nearest_mri_mask_path",
    "cognition_minus_mri_months",
    "absolute_mri_month_gap",
    "nearest_mri_tie_count",
    "mri_month_match_class",
    "mri_month_match_eligible",
    "mri_month_match_basis",
]


def _match_class(gap: float | None) -> str:
    if gap is None or not np.isfinite(gap):
        return "no_mri_for_subject"
    if gap == 0:
        return "exact_month"
    if gap <= 3:
        return "within_3_months"
    if gap <= 6:
        return "within_6_months"
    if gap <= 12:
        return "within_12_months"
    return "outside_12_months"


def align_visits_to_nearest_mri_month(
    visits: pd.DataFrame,
    mri: pd.DataFrame,
    max_gap_months: float = 6,
) -> pd.DataFrame:
    """Return visits plus their nearest same-subject primary MRI session.

    Ties prefer an MRI at or before the cognitive visit, then the earlier MRI.
    This prevents an arbitrary preference for future structural evidence.  The
    signed gap is ``cognitive_visit_month - MRI_session_month``.
    """
    required_visits = {"subject_id", "visit_month"}
    required_mri = {
        "subject_id",
        "scan_id",
        "image_id",
        "session_id",
        "visit_month",
        "visit_code",
        "image_path",
        "mask_path",
    }
    if missing := required_visits.difference(visits.columns):
        raise ValueError(f"Missing visit columns: {sorted(missing)}")
    if missing := required_mri.difference(mri.columns):
        raise ValueError(f"Missing MRI columns: {sorted(missing)}")
    if max_gap_months < 0:
        raise ValueError("max_gap_months must be nonnegative")
    if visits[["subject_id", "visit_month"]].isna().any().any():
        raise ValueError("Visit subject/month cannot be missing")

    scans = mri[mri["primary"].eq(True)].copy() if "primary" in mri else mri.copy()
    scans["visit_month"] = pd.to_numeric(scans["visit_month"], errors="coerce")
    scans = scans.dropna(subset=["subject_id", "visit_month"])
    if scans.duplicated(["subject_id", "scan_id"]).any():
        raise ValueError("Duplicate primary scan identity")

    by_subject = {
        subject: group.sort_values(["visit_month", "scan_id"], kind="stable").to_dict("records")
        for subject, group in scans.groupby("subject_id", sort=False)
    }
    assignments: list[dict] = []
    for row in visits[["subject_id", "visit_month"]].itertuples(index=False):
        candidates = by_subject.get(row.subject_id, [])
        if not candidates:
            assignments.append(
                {
                    **{column: None for column in ASSIGNMENT_COLUMNS},
                    "mri_month_match_class": "no_mri_for_subject",
                    "mri_month_match_eligible": False,
                    "mri_month_match_basis": "nominal_session_month",
                }
            )
            continue

        ranked = []
        for scan in candidates:
            signed = float(row.visit_month) - float(scan["visit_month"])
            # Prefer past/present MRI (signed >= 0) only when absolute gaps tie.
            rank = (abs(signed), signed < 0, float(scan["visit_month"]), str(scan["scan_id"]))
            ranked.append((rank, scan, signed))
        ranked.sort(key=lambda value: value[0])
        _, selected, signed = ranked[0]
        gap = abs(signed)
        tie_count = sum(np.isclose(item[0][0], gap) for item in ranked)
        assignments.append(
            {
                "nearest_mri_scan_id": selected["scan_id"],
                "nearest_mri_image_id": selected["image_id"],
                "nearest_mri_session_id": selected["session_id"],
                "nearest_mri_visit_month": float(selected["visit_month"]),
                "nearest_mri_visit_code": selected["visit_code"],
                "nearest_mri_image_path": selected["image_path"],
                "nearest_mri_mask_path": selected["mask_path"],
                "cognition_minus_mri_months": signed,
                "absolute_mri_month_gap": gap,
                "nearest_mri_tie_count": tie_count,
                "mri_month_match_class": _match_class(gap),
                "mri_month_match_eligible": bool(gap <= max_gap_months),
                "mri_month_match_basis": "nominal_session_month",
            }
        )

    aligned = pd.concat(
        [visits.reset_index(drop=True).copy(), pd.DataFrame(assignments)], axis=1
    )
    if len(aligned) != len(visits):
        raise AssertionError("Nearest-month alignment lost visits")
    if aligned.loc[aligned["mri_month_match_eligible"], "nearest_mri_image_path"].isna().any():
        raise AssertionError("Eligible assignment has no MRI path")

    # Describe many-to-one MRI-centered bins without collapsing scale visits.
    eligible = aligned["mri_month_match_eligible"]
    counts = aligned.loc[eligible].groupby("nearest_mri_scan_id").size()
    aligned["mri_month_group_visit_count"] = (
        aligned["nearest_mri_scan_id"].map(counts).fillna(0).astype(int)
    )
    aligned["mri_month_group_rank"] = pd.Series(pd.NA, index=aligned.index, dtype="Int64")
    if eligible.any():
        order = (
            aligned.loc[eligible]
            .sort_values(
                ["nearest_mri_scan_id", "absolute_mri_month_gap", "visit_month"],
                kind="stable",
            )
            .groupby("nearest_mri_scan_id")
            .cumcount()
            .add(1)
        )
        aligned.loc[order.index, "mri_month_group_rank"] = order.astype("Int64")
    return aligned


def summarize_nearest_month_alignment(aligned: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity table for exact/3/6/12-month nominal windows."""
    gap = pd.to_numeric(aligned["absolute_mri_month_gap"], errors="coerce")
    records = []
    for window in [0, 3, 6, 12]:
        selected = gap.notna() & gap.le(window)
        records.append(
            {
                "window_months": window,
                "matched_visits": int(selected.sum()),
                "matched_subjects": int(aligned.loc[selected, "subject_id"].nunique()),
                "unique_mri_scans": int(
                    aligned.loc[selected, "nearest_mri_scan_id"].nunique()
                ),
                "match_basis": "nominal_session_month",
            }
        )
    records.append(
        {
            "window_months": "no_limit",
            "matched_visits": int(gap.notna().sum()),
            "matched_subjects": int(aligned.loc[gap.notna(), "subject_id"].nunique()),
            "unique_mri_scans": int(aligned.loc[gap.notna(), "nearest_mri_scan_id"].nunique()),
            "match_basis": "nominal_session_month",
        }
    )
    return pd.DataFrame(records)
