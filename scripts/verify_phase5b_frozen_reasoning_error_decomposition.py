#!/usr/bin/env python
"""Independent verification of Phase 5B source provenance and decomposition."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memreason_rl.experiments.phase5b_reasoning_error_decomposition import reconstruct_decisions


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/phase5b_frozen_reasoning_error_decomposition.yaml").read_text())
    source = ROOT / "outputs" / config["source_run_id"]
    output = ROOT / "outputs" / config["run_id"]
    lock = json.loads((output / "source_lock.json").read_text())
    assert lock["analysis_status"] == "post_test_descriptive"
    assert not lock["training_performed"] and not lock["generation_performed"]
    for path, expected_hash in lock["artifacts"].items():
        assert digest(Path(path)) == expected_hash
    examples = read_jsonl(source / "test_reasoning_examples.jsonl")
    generations = read_jsonl(source / "test_generations_grammar_model.jsonl")
    assert len(examples) == len(generations) == 495
    assert len({row["subject_id"] for row in examples}) == 495
    fresh = reconstruct_decisions(examples, generations)
    stored = pd.read_csv(output / "decision_level_decomposition.csv")
    keys = ["sample_id", "field", "domain"]
    fresh = fresh.sort_values(keys).reset_index(drop=True)
    stored = stored.sort_values(keys).reset_index(drop=True)
    assert len(fresh) == len(stored) == 4950
    for column in ["subject_id", "expected", "predicted", "margin_band", "boundary_group"]:
        assert fresh[column].astype(str).equals(stored[column].astype(str))
    for column in ["mismatch", "decision_value", "threshold_margin", "timepoints", "items"]:
        assert np.allclose(fresh[column].astype(float), stored[column].astype(float), equal_nan=True)
    state = stored[stored.field.eq("current_state")]
    trend = stored[stored.field.eq("trend")]
    assert len(state) == len(trend) == 2475
    assert state.mismatch.sum() == 36 and state.loc[state.mismatch.eq(1), "sample_id"].nunique() == 32
    assert trend.mismatch.sum() == 100 and trend.loc[trend.mismatch.eq(1), "sample_id"].nunique() == 67
    summary = pd.read_csv(output / "mismatch_rate_summary.csv")
    overall = summary[summary.dimension.eq("overall")].set_index("field")
    assert np.isclose(overall.loc["current_state", "mismatch_rate"], 36 / 2475)
    assert np.isclose(overall.loc["trend", "mismatch_rate"], 100 / 2475)
    confusion = pd.read_csv(output / "confusion_matrix.csv")
    assert confusion.groupby("field")["count"].sum().to_dict() == {"current_state": 2475, "trend": 2475}
    mismatch_counts = confusion.loc[confusion.expected.ne(confusion.predicted)].groupby("field")["count"].sum()
    assert int(mismatch_counts["current_state"]) == 36
    assert int(mismatch_counts["trend"]) == 100
    contrast = pd.read_csv(output / "near_far_boundary_contrast.csv")
    assert set(contrast.field) == {"current_state", "trend"}
    decision = json.loads((output / "analysis_decision.json").read_text())
    source_decision = json.loads((source / "final_decision.json").read_text())
    assert decision["analysis_only"] and decision["post_test_descriptive"]
    assert not decision["training_performed"] and not decision["generation_performed"]
    assert not decision["phase4f_decision_changed"] and not decision["further_test_tuning_permitted"]
    assert decision["phase4f_frozen_system_confirmed"] == source_decision["phase4f_frozen_system_confirmed"]
    assert decision["phase4f_frozen_system_confirmed"] is False
    assert not (output / "checkpoints").exists()
    assert (output / "figures/phase5b_error_decomposition.pdf").stat().st_size > 1000
    assert (output / "figures/phase5b_error_decomposition.png").stat().st_size > 1000
    result = {
        "status": "passed", "reports": 495, "subjects": 495, "decisions": 4950,
        "current_state_mismatches": 36, "trend_mismatches": 100,
        "source_hashes_verified": True, "reference_labels_recomputed": True,
        "denominators_verified": True, "phase4f_decision_unchanged": True,
        "training_performed": False, "generation_performed": False,
        "further_test_tuning_permitted": False,
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
