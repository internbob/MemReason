#!/usr/bin/env python3
"""Export aggregate, de-identified experiment artifacts for the public repository.

Detailed outputs remain under the git-ignored ``outputs/`` directory.  This
script copies only an explicit allowlist of reports and aggregate tables.  It
also rejects files containing ADNI-style participant identifiers.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "reports" / "results"
SUBJECT_ID = re.compile(rb"\b\d{3}_S_\d{4,5}\b")
MAX_BYTES = 2 * 1024 * 1024

EXPORTS = {
    "outputs/phase1_baseline_suite_formal_20261004/report.md":
        "phase1_baseline_suite_report.md",
    "outputs/phase1b_semantic_fair_20261003/report.md":
        "phase1b_semantic_comparison_report.md",
    "outputs/phase1c_joint_stage1_20261004/report.md":
        "phase1c_joint_stage1_report.md",
    "outputs/phase2_cognitive_belief_dev_20261010/report.md":
        "phase2_cognitive_belief_report.md",
    "outputs/phase3e_frozen_spatial_grounding_test_20261014/report.md":
        "phase3e_spatial_grounding_test_report.md",
    "outputs/phase3i_native_resolution_adcn_20261017/report.md":
        "phase3i_native_resolution_adcn_report.md",
    "outputs/phase3i_native_resolution_adcn_20261017/classification_metrics.csv":
        "phase3i_native_resolution_adcn_metrics.csv",
    "outputs/phase3i_native_resolution_adcn_20261017/development_decision.json":
        "phase3i_native_resolution_adcn_decision.json",
    "outputs/phase3i_native_resolution_adcn_20261017/verification.json":
        "phase3i_native_resolution_adcn_verification.json",
    "outputs/phase3m_mri_preprocessing_sensitivity_20261018/report.md":
        "phase3m_mri_preprocessing_sensitivity_report.md",
    "outputs/phase3m_mri_preprocessing_sensitivity_20261018/final_decision.json":
        "phase3m_mri_preprocessing_sensitivity_decision.json",
    "outputs/phase3m_mri_preprocessing_sensitivity_20261018/mask_retention_strata_metrics.csv":
        "phase3m_mri_preprocessing_mask_retention.csv",
    "outputs/phase3m_mri_preprocessing_sensitivity_20261018/independent_verification.json":
        "phase3m_mri_preprocessing_verification.json",
    "outputs/phase3n_corrected_cropped_mri_diagnosis_crossfit_20261018/report.md":
        "phase3n_corrected_cropped_diagnosis_report.md",
    "outputs/phase3n_corrected_cropped_mri_diagnosis_crossfit_20261018/requested_four_arm_report.md":
        "phase3n_corrected_cropped_four_arm_report.md",
    "outputs/phase3n_corrected_cropped_mri_diagnosis_crossfit_20261018/requested_four_arm_metrics_with_ci.csv":
        "phase3n_corrected_cropped_four_arm_metrics.csv",
    "outputs/phase3n_corrected_cropped_mri_diagnosis_crossfit_20261018/development_decision.json":
        "phase3n_corrected_cropped_diagnosis_decision.json",
    "outputs/phase3n_corrected_cropped_mri_diagnosis_crossfit_20261018/independent_verification.json":
        "phase3n_corrected_cropped_diagnosis_verification.json",
    "outputs/phase4a_frozen_cognitive_reasoning_20261014/report.md":
        "phase4a_structured_reasoning_report.md",
    "outputs/phase4b_cognitive_reasoning_sft_20261014/report.md":
        "phase4b_reasoning_sft_report.md",
    "outputs/phase4c_grammar_constrained_decoding_20261015/report.md":
        "phase4c_grammar_decoding_report.md",
    "outputs/phase4d_cognitive_constrained_grpo_20261015/report.md":
        "phase4d_constrained_grpo_report.md",
    "outputs/phase4e_trajectory_action_grpo_20261015/report.md":
        "phase4e_trajectory_grpo_report.md",
    "outputs/phase4f_frozen_grud_phase4c_test_20261015/report.md":
        "phase4f_frozen_test_report.md",
    "outputs/phase4f_frozen_grud_phase4c_test_20261015/final_decision.json":
        "phase4f_final_decision.json",
    "outputs/phase4f_frozen_grud_phase4c_test_20261015/verification.json":
        "phase4f_verification.json",
    "outputs/phase4f_frozen_grud_phase4c_test_20261015/forecast_metrics_by_target_horizon.csv":
        "phase4f_forecast_metrics_by_target_horizon.csv",
    "outputs/phase4f_frozen_grud_phase4c_test_20261015/forecast_subject_bootstrap.csv":
        "phase4f_forecast_subject_bootstrap.csv",
    "outputs/phase4f_frozen_grud_phase4c_test_20261015/reasoning_aggregate_metrics.csv":
        "phase4f_reasoning_aggregate_metrics.csv",
    "outputs/phase4f_frozen_grud_phase4c_test_20261015/reasoning_subject_bootstrap.csv":
        "phase4f_reasoning_subject_bootstrap.csv",
    "outputs/phase5b_frozen_reasoning_error_decomposition_20261016/report.md":
        "phase5b_reasoning_error_decomposition_report.md",
    "outputs/phase5b_frozen_reasoning_error_decomposition_20261016/mismatch_rate_summary.csv":
        "phase5b_mismatch_rate_summary.csv",
    "outputs/phase5b_frozen_reasoning_error_decomposition_20261016/confusion_matrix.csv":
        "phase5b_confusion_matrix.csv",
    "outputs/phase5b_frozen_reasoning_error_decomposition_20261016/near_far_boundary_contrast.csv":
        "phase5b_near_far_boundary_contrast.csv",
    "outputs/phase5b_frozen_reasoning_error_decomposition_20261016/analysis_decision.json":
        "phase5b_analysis_decision.json",
    "outputs/phase5b_frozen_reasoning_error_decomposition_20261016/verification.json":
        "phase5b_verification.json",
    "reports/data_audit/phase5c_memory_field_audit.md":
        "phase5c_memory_field_audit.md",
    "reports/data_audit/phase5c_memory_field_audit.csv":
        "phase5c_memory_field_audit.csv",
    "outputs/phase5c_memory_measurement_dev_20261016/report.md":
        "phase5c_memory_measurement_report.md",
    "outputs/phase5c_memory_measurement_dev_20261016/aggregate_metrics.csv":
        "phase5c_memory_measurement_aggregate_metrics.csv",
    "outputs/phase5c_memory_measurement_dev_20261016/cross_view_summary.csv":
        "phase5c_memory_measurement_cross_view_summary.csv",
    "outputs/phase5c_memory_measurement_dev_20261016/subject_bootstrap.csv":
        "phase5c_memory_measurement_subject_bootstrap.csv",
    "outputs/phase5c_memory_measurement_dev_20261016/development_decision.json":
        "phase5c_memory_measurement_decision.json",
    "outputs/phase5c_memory_measurement_dev_20261016/verification.json":
        "phase5c_memory_measurement_verification.json",
    "outputs/phase5d_anchored_mirt_dev_20261016/report.md":
        "phase5d_anchored_mirt_report.md",
    "outputs/phase5d_anchored_mirt_dev_20261016/aggregate_metrics.csv":
        "phase5d_anchored_mirt_aggregate_metrics.csv",
    "outputs/phase5d_anchored_mirt_dev_20261016/cross_view_summary.csv":
        "phase5d_anchored_mirt_cross_view_summary.csv",
    "outputs/phase5d_anchored_mirt_dev_20261016/subject_bootstrap.csv":
        "phase5d_anchored_mirt_subject_bootstrap.csv",
    "outputs/phase5d_anchored_mirt_dev_20261016/anchor_stability.csv":
        "phase5d_anchored_mirt_anchor_stability.csv",
    "outputs/phase5d_anchored_mirt_dev_20261016/instrument_connectivity.csv":
        "phase5d_anchored_mirt_instrument_connectivity.csv",
    "outputs/phase5d_anchored_mirt_dev_20261016/development_decision.json":
        "phase5d_anchored_mirt_decision.json",
    "outputs/phase5d_anchored_mirt_dev_20261016/verification.json":
        "phase5d_anchored_mirt_verification.json",
    "outputs/phase5e_reasoning_faithfulness_audit_20261016/report.md":
        "phase5e_reasoning_faithfulness_report.md",
    "outputs/phase5e_reasoning_faithfulness_audit_20261016/primary_deletion_contrast.json":
        "phase5e_primary_deletion_contrast.json",
    "outputs/phase5e_reasoning_faithfulness_audit_20261016/counterfactual_summary_bootstrap.csv":
        "phase5e_counterfactual_summary_bootstrap.csv",
    "outputs/phase5e_reasoning_faithfulness_audit_20261016/renderer_ablation_metrics.csv":
        "phase5e_renderer_ablation_metrics.csv",
    "outputs/phase5e_reasoning_faithfulness_audit_20261016/analysis_decision.json":
        "phase5e_analysis_decision.json",
    "outputs/phase5e_reasoning_faithfulness_audit_20261016/independent_verification.json":
        "phase5e_independent_verification.json",
    "outputs/phase5f_hierarchical_bridge_dev_20261028/report.md":
        "phase5f_hierarchical_bridge_report.md",
    "outputs/phase5f_hierarchical_bridge_dev_20261028/aggregate_metrics.csv":
        "phase5f_hierarchical_bridge_aggregate_metrics.csv",
    "outputs/phase5f_hierarchical_bridge_dev_20261028/cross_view_summary.csv":
        "phase5f_hierarchical_bridge_cross_view_summary.csv",
    "outputs/phase5f_hierarchical_bridge_dev_20261028/subject_bootstrap.csv":
        "phase5f_hierarchical_bridge_subject_bootstrap.csv",
    "outputs/phase5f_hierarchical_bridge_dev_20261028/connection_graph.csv":
        "phase5f_hierarchical_bridge_connection_graph.csv",
    "outputs/phase5f_hierarchical_bridge_dev_20261028/development_decision.json":
        "phase5f_hierarchical_bridge_decision.json",
    "outputs/phase5f_hierarchical_bridge_dev_20261028/verification.json":
        "phase5f_hierarchical_bridge_verification.json",
    "outputs/phase5g_bridge_pareto_gate_dev_20261029/report.md":
        "phase5g_bridge_pareto_report.md",
    "outputs/phase5g_bridge_pareto_gate_dev_20261029/aggregate_metrics.csv":
        "phase5g_bridge_pareto_aggregate_metrics.csv",
    "outputs/phase5g_bridge_pareto_gate_dev_20261029/pareto_frontier.csv":
        "phase5g_bridge_pareto_frontier.csv",
    "outputs/phase5g_bridge_pareto_gate_dev_20261029/development_decision.json":
        "phase5g_bridge_pareto_decision.json",
    "outputs/phase5g_bridge_pareto_gate_dev_20261029/verification.json":
        "phase5g_bridge_pareto_verification.json",
    "outputs/phase5h_measurement_identifiability_audit_20261029/report.md":
        "phase5h_measurement_identifiability_report.md",
    "outputs/phase5h_measurement_identifiability_audit_20261029/instrument_reliability.csv":
        "phase5h_measurement_identifiability_reliability.csv",
    "outputs/phase5h_measurement_identifiability_audit_20261029/cross_instrument_pairs.csv":
        "phase5h_measurement_identifiability_pairs.csv",
    "outputs/phase5h_measurement_identifiability_audit_20261029/decision.json":
        "phase5h_measurement_identifiability_decision.json",
    "outputs/phase5h_measurement_identifiability_audit_20261029/verification.json":
        "phase5h_measurement_identifiability_verification.json",
    "outputs/phase6a_latent_state_dynamics_dev_20261029/report.md":
        "phase6a_latent_state_dynamics_report.md",
    "outputs/phase6a_latent_state_dynamics_dev_20261029/aggregate_metrics.csv":
        "phase6a_latent_state_dynamics_aggregate_metrics.csv",
    "outputs/phase6a_latent_state_dynamics_dev_20261029/metrics_by_horizon.csv":
        "phase6a_latent_state_dynamics_horizon_metrics.csv",
    "outputs/phase6a_latent_state_dynamics_dev_20261029/subject_bootstrap.csv":
        "phase6a_latent_state_dynamics_bootstrap.csv",
    "outputs/phase6a_latent_state_dynamics_dev_20261029/development_decision.json":
        "phase6a_latent_state_dynamics_decision.json",
    "outputs/phase6a_latent_state_dynamics_dev_20261029/verification.json":
        "phase6a_latent_state_dynamics_verification.json",
}


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    missing = []
    exported = []
    for source_name, destination_name in EXPORTS.items():
        source = ROOT / source_name
        if not source.is_file():
            missing.append(source_name)
            continue
        content = source.read_bytes()
        if len(content) > MAX_BYTES:
            raise RuntimeError(f"Refusing aggregate artifact larger than 2 MiB: {source}")
        if SUBJECT_ID.search(content):
            raise RuntimeError(f"Refusing artifact containing an ADNI subject identifier: {source}")
        destination = DESTINATION / destination_name
        shutil.copyfile(source, destination)
        exported.append(destination.relative_to(ROOT).as_posix())
    if missing:
        raise FileNotFoundError("Missing required public-result artifacts:\n" + "\n".join(missing))
    print(f"Exported {len(exported)} aggregate artifacts to {DESTINATION.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
