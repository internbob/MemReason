"""Build a reproducible FreeSurfer MRI inventory without replacing primary MRI data.

Run with::

    python -m memreason_rl.data.build_freesurfer_manifest

The inventory is linked to the existing UniSurf metadata by exact ADNI subject
and Image ID.  FreeSurfer ``brainmask.mgz`` is an intensity image; downstream
code should obtain its binary brain mask with ``volume > 0``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from .common import DAYS_PER_MONTH, ROOT, save_csv, save_json
from .build_mri_metadata import normalize_image_id


DEFAULT_FREESURFER_ROOT = Path(
    "/public_bme/data/lianzf2024/UniSurf/Longitudinal/FreeSurfer/ADNI"
)

DIRECTORY_PATTERN = re.compile(r"(\d{3}_S_\d{4})-(\d{4})-I(\d+)")

FREESURFER_FILES = {
    "brainmask_path": "mri/brainmask.mgz",
    "brainmask_auto_path": "mri/brainmask.auto.mgz",
    "t1_path": "mri/T1.mgz",
    "brain_path": "mri/brain.mgz",
    "brain_finalsurfs_path": "mri/brain.finalsurfs.mgz",
    "rawavg_path": "mri/rawavg.mgz",
    "orig_path": "mri/orig.mgz",
    "norm_path": "mri/norm.mgz",
    "aseg_path": "mri/aseg.mgz",
    "aparc_aseg_path": "mri/aparc+aseg.mgz",
    "aparc_dkt_aseg_path": "mri/aparc.DKTatlas+aseg.mgz",
    "aparc_a2009s_aseg_path": "mri/aparc.a2009s+aseg.mgz",
    "wmparc_path": "mri/wmparc.mgz",
    "aseg_stats_path": "stats/aseg.stats",
    "lh_aparc_stats_path": "stats/lh.aparc.stats",
    "rh_aparc_stats_path": "stats/rh.aparc.stats",
    "lh_thickness_path": "surf/lh.thickness",
    "rh_thickness_path": "surf/rh.thickness",
}


def inventory_freesurfer(root: Path) -> pd.DataFrame:
    records = []
    for directory in sorted(root.iterdir(), key=lambda path: path.name):
        match = DIRECTORY_PATTERN.fullmatch(directory.name)
        if match is None or not directory.is_dir():
            continue
        subject_id, year, image_id = match.groups()
        record = {
            "subject_id": subject_id,
            "image_id_key": normalize_image_id(image_id),
            "freesurfer_subject_dir": str(directory),
            "freesurfer_directory_name": directory.name,
            "freesurfer_directory_year": int(year),
        }
        for column, relative in FREESURFER_FILES.items():
            path = directory / relative
            record[column] = str(path) if path.is_file() else None
            record[column.replace("_path", "_exists")] = path.is_file()
        records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError(f"No ADNI FreeSurfer directories found under {root}")
    if frame.duplicated(["subject_id", "image_id_key"]).any():
        duplicates = frame.loc[
            frame.duplicated(["subject_id", "image_id_key"], keep=False),
            ["subject_id", "image_id_key", "freesurfer_directory_name"],
        ]
        raise ValueError(f"Duplicate FreeSurfer identity:\n{duplicates}")
    return frame


def link_metadata(
    freesurfer: pd.DataFrame, metadata_path: Path, primary_path: Path
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path, low_memory=False, dtype={"image_id": str})
    source = metadata[metadata.source.eq("unisurf_longitudinal")].copy()
    source["image_id_key"] = source.image_id.map(normalize_image_id)
    if source.duplicated(["subject_id", "image_id_key"]).any():
        raise ValueError("UniSurf metadata is not unique by subject and Image ID")

    linked = freesurfer.merge(
        source,
        on=["subject_id", "image_id_key"],
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_unisurf"),
    )
    missing = linked[linked._merge.ne("both")]
    if not missing.empty:
        examples = missing[
            ["subject_id", "image_id_key", "freesurfer_directory_name"]
        ].head(10)
        raise ValueError(f"FreeSurfer scans missing UniSurf metadata:\n{examples}")
    linked = linked.drop(columns="_merge")

    linked = linked.rename(
        columns={
            "scan_id": "source_scan_id",
            "image_path": "unisurf_brain_path",
            "mask_path": "unisurf_mask_path",
            "path_exists": "unisurf_path_exists",
            "file_size_bytes": "unisurf_file_size_bytes",
            "source": "source_metadata_collection",
            "primary": "source_primary",
            "preprocessing": "source_preprocessing",
        }
    )
    linked["scan_id"] = "freesurfer:" + linked.freesurfer_directory_name
    linked["image_id"] = linked.image_id_key
    linked["image_path"] = linked.brainmask_path
    # brainmask.mgz is an intensity image; the mask contract is nonzero voxels.
    linked["mask_path"] = linked.brainmask_path
    linked["mask_rule"] = "nonzero"
    linked["source"] = "unisurf_freesurfer"
    linked["sequence_name"] = "T1_FreeSurfer_brainmask"
    linked["primary"] = False
    linked["path_exists"] = linked.brainmask_exists
    linked["file_size_bytes"] = linked.brainmask_path.map(
        lambda value: Path(value).stat().st_size if isinstance(value, str) else None
    )
    linked["qc_pass"] = pd.NA
    linked["qc_status"] = "not_requested"
    linked["metadata_link"] = "exact_subject_image_id_to_unisurf_longitudinal"
    linked["preprocessing"] = (
        "FreeSurfer conformed 1mm brainmask.mgz; binary mask is nonzero voxels"
    )
    linked["image_eligible"] = linked.brainmask_exists
    linked["roi_eligible"] = (
        linked.aseg_exists
        & linked.aparc_aseg_exists
        & linked.aseg_stats_exists
        & linked.lh_aparc_stats_exists
        & linked.rh_aparc_stats_exists
    )
    linked["surface_eligible"] = (
        linked.lh_thickness_exists & linked.rh_thickness_exists
    )

    primary = pd.read_csv(primary_path, low_memory=False, dtype={"image_id": str})
    primary["scan_date"] = pd.to_datetime(primary.scan_date, errors="coerce")
    primary = primary[primary.scan_date.notna()].copy()
    nearest_records = []
    for row in linked.itertuples(index=False):
        scan_date = pd.to_datetime(row.scan_date, errors="coerce")
        candidates = primary[primary.subject_id.eq(row.subject_id)].copy()
        if pd.isna(scan_date) or candidates.empty:
            nearest_records.append({"scan_id": row.scan_id})
            continue
        candidates["date_gap_days"] = (candidates.scan_date - scan_date).dt.days.abs()
        nearest = candidates.sort_values(["date_gap_days", "scan_id"]).iloc[0]
        nearest_records.append(
            {
                "scan_id": row.scan_id,
                "nearest_primary_scan_id": nearest.scan_id,
                "nearest_primary_image_id": normalize_image_id(nearest.image_id),
                "nearest_primary_session_id": nearest.session_id,
                "nearest_primary_visit_month": nearest.visit_month,
                "nearest_primary_visit_code": nearest.visit_code,
                "nearest_primary_scan_date": nearest.scan_date.date().isoformat(),
                "nearest_primary_date_gap_days": int(nearest.date_gap_days),
                "same_day_primary_match": bool(nearest.date_gap_days == 0),
            }
        )
    nearest = pd.DataFrame(nearest_records)
    linked = linked.merge(nearest, on="scan_id", how="left", validate="one_to_one")
    linked["scan_date"] = pd.to_datetime(linked.scan_date, errors="coerce").dt.date.astype("string")
    return linked


def order_columns(frame: pd.DataFrame, primary_path: Path) -> pd.DataFrame:
    standard = list(pd.read_csv(primary_path, nrows=0).columns)
    missing = [column for column in standard if column not in frame.columns]
    if missing:
        raise ValueError(f"FreeSurfer inventory lacks primary-schema columns: {missing}")
    extras = [column for column in frame.columns if column not in standard]
    return frame[standard + extras].sort_values(
        ["split", "subject_id", "scan_date", "scan_id"], na_position="last"
    ).reset_index(drop=True)


def build_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame[frame.image_eligible & frame.scan_date.notna()].copy()
    eligible["scan_date"] = pd.to_datetime(eligible.scan_date, errors="raise")
    eligible = eligible.sort_values(["subject_id", "scan_date", "scan_id"])
    pair_columns = [
        "scan_id", "image_id", "scan_date", "image_path", "mask_path",
        "aseg_path", "aparc_aseg_path", "roi_eligible",
    ]
    records = []
    for subject_id, group in eligible.groupby("subject_id", sort=True):
        rows = list(group.itertuples(index=False))
        for first, second in zip(rows[:-1], rows[1:]):
            delta_days = int((second.scan_date - first.scan_date).days)
            record = {
                "pair_id": f"{first.scan_id}__to__{second.scan_id}",
                "subject_id": subject_id,
                "split": first.split,
                "delta_days": delta_days,
                "delta_months": delta_days / DAYS_PER_MONTH,
                "interval_6_36_months": bool(
                    6 <= delta_days / DAYS_PER_MONTH <= 36
                ),
            }
            first_values = first._asdict()
            second_values = second._asdict()
            for column in pair_columns:
                record[f"{column}_1"] = first_values[column]
                record[f"{column}_2"] = second_values[column]
            records.append(record)
    return pd.DataFrame(records)


def build_summary(frame: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    records = []
    for split in ["all", "train", "val", "test"]:
        scans = frame if split == "all" else frame[frame.split.eq(split)]
        pair_block = pairs if split == "all" else pairs[pairs.split.eq(split)]
        usable = scans[scans.image_eligible]
        counts = usable.groupby("subject_id").size()
        records.append(
            {
                "split": split,
                "directories": len(scans),
                "subjects": scans.subject_id.nunique(),
                "image_eligible_scans": int(scans.image_eligible.sum()),
                "image_eligible_subjects": usable.subject_id.nunique(),
                "roi_eligible_scans": int(scans.roi_eligible.sum()),
                "surface_eligible_scans": int(scans.surface_eligible.sum()),
                "subjects_with_at_least_2_images": int((counts >= 2).sum()),
                "adjacent_pairs": len(pair_block),
                "adjacent_pairs_6_36_months": int(
                    pair_block.interval_6_36_months.sum()
                ) if not pair_block.empty else 0,
                "subjects_with_6_36_month_pairs": pair_block.loc[
                    pair_block.interval_6_36_months, "subject_id"
                ].nunique() if not pair_block.empty else 0,
                "same_day_primary_matches": int(scans.same_day_primary_match.fillna(False).sum()),
            }
        )
    return pd.DataFrame(records)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small audit table without pandas' optional tabulate dependency."""
    if frame.empty:
        return "None."
    columns = [str(column) for column in frame.columns]
    rows = []
    for values in frame.itertuples(index=False, name=None):
        rows.append([
            "" if pd.isna(value) else str(value).replace("|", "\\|")
            for value in values
        ])
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def write_report(
    frame: pd.DataFrame, pairs: pd.DataFrame, summary: pd.DataFrame, path: Path
) -> None:
    coverage = []
    for column in FREESURFER_FILES:
        exists_column = column.replace("_path", "_exists")
        coverage.append(
            {
                "file": FREESURFER_FILES[column],
                "available_scans": int(frame[exists_column].sum()),
                "total_scans": len(frame),
                "coverage_fraction": float(frame[exists_column].mean()),
            }
        )
    coverage_frame = pd.DataFrame(coverage)
    save_csv(coverage_frame, path.with_name("mri_freesurfer_file_coverage.csv"))
    missing_masks = frame.loc[
        ~frame.brainmask_exists,
        ["subject_id", "image_id", "freesurfer_directory_name"],
    ]
    report = f"""# FreeSurfer MRI inventory audit

This inventory is an additive data source. It does not replace or modify
`processed/mri_primary.csv`.

## Coverage

{markdown_table(summary)}

## File coverage

{markdown_table(coverage_frame)}

`brainmask.mgz` is a FreeSurfer-conformed intensity image, not a binary label.
The mask contract for downstream loading is `brainmask.mgz > 0`.

## Missing brainmask files

{markdown_table(missing_masks)}

## Pair rule

`processed/mri_freesurfer_pairs.csv` contains consecutive, real-date scans
within each subject. `interval_6_36_months` marks the current longitudinal
analysis window. No registration, scanner-protocol, motion-artifact, or
clinical-QC exclusion is applied.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freesurfer-root", type=Path, default=DEFAULT_FREESURFER_ROOT)
    parser.add_argument("--metadata", type=Path, default=ROOT / "processed/mri_metadata.csv")
    parser.add_argument("--primary", type=Path, default=ROOT / "processed/mri_primary.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "processed/mri_freesurfer.csv")
    parser.add_argument("--pairs-output", type=Path, default=ROOT / "processed/mri_freesurfer_pairs.csv")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT / "reports/data_audit/mri_freesurfer_summary.csv",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "reports/data_audit/mri_freesurfer_audit.md",
    )
    args = parser.parse_args()

    inventory = inventory_freesurfer(args.freesurfer_root)
    linked = link_metadata(inventory, args.metadata, args.primary)
    linked = order_columns(linked, args.primary)
    pairs = build_pairs(linked)
    summary = build_summary(linked, pairs)

    save_csv(linked, args.output)
    save_csv(pairs, args.pairs_output)
    save_csv(summary, args.summary_output)
    for split in ["train", "val", "test"]:
        save_csv(
            linked[linked.split.eq(split)],
            ROOT / f"splits/{split}_mri_freesurfer.csv",
        )
    write_report(linked, pairs, summary, args.report_output)
    save_json(
        {
            "passed": True,
            "freesurfer_root": str(args.freesurfer_root),
            "metadata_source": str(args.metadata),
            "primary_reference": str(args.primary),
            "directories": len(linked),
            "subjects": int(linked.subject_id.nunique()),
            "image_eligible_scans": int(linked.image_eligible.sum()),
            "roi_eligible_scans": int(linked.roi_eligible.sum()),
            "adjacent_pairs": len(pairs),
            "adjacent_pairs_6_36_months": int(pairs.interval_6_36_months.sum()),
            "standard_columns_preserved": list(pd.read_csv(args.primary, nrows=0).columns),
        },
        ROOT / "reports/data_audit/mri_freesurfer_validation.json",
    )
    print(json.dumps(summary.iloc[0].to_dict(), default=str), flush=True)


if __name__ == "__main__":
    main()
