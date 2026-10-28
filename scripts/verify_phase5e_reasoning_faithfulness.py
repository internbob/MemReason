#!/usr/bin/env python3
"""Independently verify the Phase 5E frozen audit artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/phase5e_reasoning_faithfulness_audit_20261016"


def main() -> None:
    verification = json.loads((OUTPUT / "verification.json").read_text())
    decision = json.loads((OUTPUT / "analysis_decision.json").read_text())
    per_report = pd.read_csv(OUTPUT / "counterfactual_per_report_metrics.csv")
    selection = pd.read_csv(OUTPUT / "visit_deletion_selection.csv")
    contrast = json.loads((OUTPUT / "primary_deletion_contrast.json").read_text())
    ablation = pd.read_csv(OUTPUT / "renderer_ablation_metrics.csv").set_index("arm")
    required_conditions = {
        "highest_influence_visit_deletion", "matched_random_visit_deletion",
        "reversed_time_gaps", "memory_score_reflection",
    }
    checks = {
        "four_interventions_present": set(per_report.condition) == required_conditions,
        "paired_deletion_subjects_match": (
            selection.groupby("condition").subject_id.nunique().eq(278).all()
            and contrast["subjects"] == 278
        ),
        "primary_influence_contrast_positive": contrast["ci_low"] > 0,
        "all_ir_reports_exactly_verified": per_report.verifier_exact_report_match.eq(1).all(),
        "all_evidence_precise": per_report.verifier_evidence_precision.eq(1).all(),
        "no_future_leakage": per_report.verifier_future_evidence_leakage.eq(0).all(),
        "no_deleted_event_retained": per_report.deleted_visit_stale_citations.eq(0).all(),
        "executable_exactly_matches_teacher": (
            ablation.loc["grammar_executable", "exact_report_match"] == 1
            and ablation.loc["teacher", "exact_report_match"] == 1
        ),
        "llm_advantage_not_claimed": not verification["renderer_ablation"][
            "llm_renderer_factual_advantage_supported"
        ],
        "audit_remains_post_test": (
            verification["analysis_is_post_test_descriptive"]
            and not verification["phase4f_gate_reopened"]
        ),
        "paper_method_matches_results": (
            decision["paper_core_method"]
            == "frozen_grud_plus_typed_executable_reasoning_ir_compiler_verifier"
            and decision["llm_role"] == "optional_renderer_only"
            and not decision["latent_clinical_memory_state_supported"]
        ),
    }
    result = {"passed": bool(all(checks.values())), "checks": {k: bool(v) for k, v in checks.items()}}
    (OUTPUT / "independent_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
