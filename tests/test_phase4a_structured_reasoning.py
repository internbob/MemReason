import json

import pandas as pd

from memreason_rl.experiments.phase4a_structured_reasoning import assign_reasoning_splits
from memreason_rl.models.reasoning_llm import parse_report


def test_phase4a_subject_split_is_disjoint_and_reproducible():
    rows = []
    for index in range(90):
        rows.append({"subject_id": f"S{index:03d}", "dxsum_diagnosis_label": ["CN", "MCI", "AD"][index % 3]})
    frame = pd.DataFrame(rows)
    config = {"reasoning_split": {"seed": 7, "build_fraction": 0.8,
                                   "dev_fraction": 0.1, "confirmation_fraction": 0.1}}
    first = assign_reasoning_splits(frame, set(frame.subject_id), config)
    second = assign_reasoning_splits(frame, set(frame.subject_id), config)
    assert first.equals(second)
    assert first.subject_id.is_unique
    assert set(first.reasoning_split) == {"build", "dev", "confirmation"}


def test_structured_report_rejects_future_truth_section():
    valid = {"current_state": {}, "evidence": [], "trend": {}, "forecast": {}, "confidence": {}}
    assert parse_report(json.dumps(valid)) == valid
    invalid = {**valid, "future_truth": {}}
    try:
        parse_report(json.dumps(invalid))
    except ValueError:
        pass
    else:
        raise AssertionError("Undocumented future-truth section was accepted")
