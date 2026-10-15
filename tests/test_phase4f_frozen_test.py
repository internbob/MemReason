import json

from memreason_rl.experiments.phase4f_frozen_test import frozen_lock, load_config


def test_phase4f_uses_only_frozen_phase4a_to_phase4c_components():
    config = load_config()
    assert config["decoding"]["frozen_selected_arm"] == "grammar_model"
    assert "phase4d_run_id" not in config and "phase4e_run_id" not in config
    lock = frozen_lock(config)
    assert lock["selected_decoder"] == "grammar_model"
    assert lock["grpo_adapters_excluded"] is True
    assert lock["training_performed"] is False
    assert set(lock["artifacts"]) >= {"grud_2026", "grud_2027", "grud_2028", "phase4b_adapter"}
    assert all(len(value["sha256"]) == 64 for value in lock["artifacts"].values())


def test_phase4f_gate_is_frozen_before_test_construction():
    config = load_config()
    assert config["forecast_gate"]["maximum_macro_normalized_mae"] == 0.42
    assert config["reasoning_gate"]["minimum_format_validity"] == 1.0
    assert config["reasoning_gate"]["maximum_future_evidence_leakage"] == 0.0
    protocol = (__import__("pathlib").Path(__file__).parents[1] /
                "reports/experiments/phase4f_frozen_grud_phase4c_test_protocol.md")
    assert protocol.exists()
    assert "further adjustment on test subjects" in protocol.read_text()
