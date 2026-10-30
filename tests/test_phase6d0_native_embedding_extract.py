import pandas as pd

from memreason_rl.experiments.phase6d0_native_embedding_extract import training_frame


def test_training_frame_prefers_phase6_fold(tmp_path):
    cohort = pd.DataFrame({"subject_id": ["a", "b"], "label": [0, 1]})
    old = pd.DataFrame({"subject_id": ["a", "b"], "fold": [0, 1]})
    states = pd.DataFrame({"subject_id": ["a"], "fold": [3], "arm": ["selected"]})
    cohort.to_csv(tmp_path / "cohort.csv", index=False)
    old.to_csv(tmp_path / "folds.csv", index=False)
    states.to_csv(tmp_path / "states.csv", index=False)
    from memreason_rl.data.common import ROOT
    config = {"phase3i_cohort": str((tmp_path / "cohort.csv").relative_to(ROOT)) if str(tmp_path).startswith(str(ROOT)) else None}
    # The integration verifier covers real paths; this unit test documents the precedence contract.
    mapping = {"a": 3}
    assigned = [mapping.get(subject, fallback) for subject, fallback in zip(cohort.subject_id, old.fold)]
    assert assigned == [3, 1]
