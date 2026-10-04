"""Audit project cognitive columns against the official ADNI online dictionary.

This script is deliberately fail-closed: an item is matched only when both the
official field name and source table agree. It writes evidence tables and never
marks the modeling dictionary verified by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memreason_rl.data.common import ROOT


BASE_URL = "https://adni.loni.usc.edu/data-samples/data-dictionary-search/?q={}"
PREFIX_TABLE = {
    "adas": "ADAS",
    "cdr": "CDR",
    "faq": "FAQ",
    "mmse": "MMSE",
    "moca": "MOCA",
}
METADATA_SUFFIXES = {"visit_date", "userdate", "present"}


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_cell = False
        self.cell_parts: list[str] = []
        self.row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "td":
            self.in_cell = True
            self.cell_parts = []
        elif tag == "tr":
            self.row = []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self.in_cell:
            self.row.append(" ".join(" ".join(self.cell_parts).split()))
            self.in_cell = False
        elif tag == "tr" and self.row:
            self.rows.append(self.row)


def source_columns(csv_path: Path) -> list[tuple[str, str, str]]:
    columns = pd.read_csv(csv_path, nrows=0).columns
    result = []
    for column in columns:
        if "_" not in column:
            continue
        prefix, suffix = column.split("_", 1)
        if prefix not in PREFIX_TABLE or suffix in METADATA_SUFFIXES:
            continue
        result.append((column, suffix.upper(), PREFIX_TABLE[prefix]))
    return result


def fetch(field: str, cache_dir: Path | None = None) -> tuple[str, bytes, list[list[str]]]:
    url = BASE_URL.format(quote(field))
    cache_path = cache_dir / f"{field}.html" if cache_dir else None
    if cache_path and cache_path.exists():
        content = cache_path.read_bytes()
    else:
        request = Request(url, headers={"User-Agent": "MemReason-ADNI-dictionary-audit/1.0"})
        with urlopen(request, timeout=60) as response:
            content = response.read()
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(content)
    parser = TableParser()
    parser.feed(content.decode("utf-8", errors="replace"))
    return url, content, parser.rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "adni_all_longitudinal_scales_visit_with_ravlt_logmem.csv",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    requested = source_columns(args.csv)
    unique_fields = sorted({field for _, field, _ in requested})
    fetched = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(fetch, field, args.cache_dir): field for field in unique_fields}
        for future in as_completed(jobs):
            field = jobs[future]
            fetched[field] = future.result()

    records = []
    unmatched = []
    for project_column, field, expected_table in requested:
        url, content, rows = fetched[field]
        matches = [row for row in rows if len(row) >= 6 and row[0].upper() == field and row[4].upper() == expected_table]
        if not matches:
            unmatched.append(
                {
                    "project_column": project_column,
                    "official_field": field,
                    "expected_table": expected_table,
                    "query_url": url,
                    "response_sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        for row in matches:
            records.append(
                {
                    "project_column": project_column,
                    "official_field": row[0],
                    "official_description": row[1],
                    "official_codes": row[2],
                    "official_phase": row[3],
                    "official_table": row[4],
                    "official_crf": row[5],
                    "query_url": url,
                    "response_sha256": hashlib.sha256(content).hexdigest(),
                }
            )

    output = ROOT / "reports" / "data_audit"
    output.mkdir(parents=True, exist_ok=True)
    matched = pd.DataFrame(records).sort_values(["project_column", "official_phase"])
    missing = pd.DataFrame(
        unmatched,
        columns=["project_column", "official_field", "expected_table", "query_url", "response_sha256"],
    ).sort_values("project_column")
    matched.to_csv(output / "adni_official_dictionary_rows.csv", index=False)
    missing.to_csv(output / "adni_official_dictionary_unmatched.csv", index=False)
    print(
        {
            "project_columns": len(requested),
            "matched_columns": int(matched.project_column.nunique()),
            "unmatched_columns": int(len(missing)),
            "official_rows": int(len(matched)),
        }
    )


if __name__ == "__main__":
    main()
