import numpy as np
import pandas as pd

from memreason_rl.experiments.phase3f_diagnosis_crossfit import assign_subject_folds, interaction_bootstrap


def test_assign_subject_folds_keeps_subjects_intact():
    rows = []
    diagnoses = ["CN", "MCI", "AD"]
    for index in range(60):
        for visit in range(2):
            rows.append({"subject_id": f"S{index:03d}", "diagnosis_group": diagnoses[index % 3],
                         "cohort": "ADNI1" if index % 2 else "ADNI2", "visit": visit})
    frame = pd.DataFrame(rows)
    registry = assign_subject_folds(frame, 5, 20261014)
    assert registry.subject_id.is_unique
    assert set(registry.fold) == set(range(5))
    assert registry.groupby("subject_id").fold.nunique().max() == 1


def test_interaction_bootstrap_preserves_subject_clusters():
    paired = pd.DataFrame({
        "scan_id": ["a1", "a2", "m1", "m2"], "subject_id": ["A", "A", "M", "M"],
        "difference": [-2.0, -1.0, 0.5, 1.5],
    })
    attributes = pd.DataFrame({
        "scan_id": paired.scan_id, "diagnosis_group": ["AD", "AD", "MCI", "MCI"]
    })
    result = interaction_bootstrap(paired, attributes, 100, 7)
    assert np.isclose(result["interaction"], -2.5)
    assert result["ci_upper"] < 0
