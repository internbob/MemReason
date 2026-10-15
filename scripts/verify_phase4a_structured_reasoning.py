"""Independent source-level verification of Phase 4A artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memreason_rl.data.dictionary import load_dictionary
from memreason_rl.experiments.phase1_numeric import clean_frame, fit_normalization
from memreason_rl.experiments.phase4a_structured_reasoning import severity
from memreason_rl.models.reasoning_llm import parse_report


CONFIG = ROOT / "configs/phase4a_frozen_cognitive_reasoning.yaml"


def expected_state_and_trend(history, dictionary, normalization, config):
    latest = {}
    for row in history:
        latest[row["item"]] = row
    state, trend = {}, {}
    state_threshold = float(config["reasoning"]["state_z_threshold"])
    trend_threshold = float(config["reasoning"]["trend_z_threshold"])
    for domain in ["Memory", "Executive", "Attention", "Language", "Orientation"]:
        current_rows = [row for row in latest.values() if row["domain"] == domain]
        domain_rows = [row for row in history if row["domain"] == domain]
        if not current_rows:
            state[domain] = {"status": "unavailable", "severity_z": None, "items": 0}
            trend[domain] = {"label": "insufficient_history", "start_severity_z": None,
                             "end_severity_z": None, "timepoints": 0}
            continue
        current = float(np.mean([severity(row["value"], row["item"], dictionary, normalization)
                                 for row in current_rows]))
        status = "elevated_impairment" if current > state_threshold else (
            "lower_impairment" if current < -state_threshold else "within_training_reference"
        )
        state[domain] = {"status": status, "severity_z": round(current, 6), "items": len(current_rows)}
        visits = []
        for visit in sorted({float(row["visit_month"]) for row in domain_rows}):
            values = [severity(row["value"], row["item"], dictionary, normalization)
                      for row in domain_rows if float(row["visit_month"]) == visit]
            visits.append((visit, float(np.mean(values))))
        if len(visits) < 2:
            label = "insufficient_history"
        else:
            delta = visits[-1][1] - visits[0][1]
            label = "decline" if delta > trend_threshold else ("improve" if delta < -trend_threshold else "stable")
        trend[domain] = {"label": label, "start_severity_z": round(visits[0][1], 6),
                         "end_severity_z": round(visits[-1][1], 6), "timepoints": len(visits)}
    return state, trend


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    output = ROOT / "outputs" / config["run_id"]
    registry = pd.read_csv(output / "subject_registry.csv")
    manifest = json.loads((output / "manifest.json").read_text())
    decision = json.loads((output / "development_decision.json").read_text())
    normalization = json.loads((output / "normalization.json").read_text())
    dictionary = load_dictionary()
    features = manifest["features"]

    train_subjects = set(pd.read_csv(ROOT / "splits/train.csv", usecols=["subject_id"]).subject_id.astype(str))
    val_subjects = set(pd.read_csv(ROOT / "splits/val.csv", usecols=["subject_id"]).subject_id.astype(str))
    test_subjects = set(pd.read_csv(ROOT / "splits/test.csv", usecols=["subject_id"]).subject_id.astype(str))
    used = set(registry.subject_id.astype(str))
    assert registry.subject_id.is_unique and used <= train_subjects
    assert not used & (val_subjects | test_subjects)
    assert set(registry.reasoning_split) == {"build", "dev", "confirmation"}

    source = pd.read_csv(ROOT / "splits/train.csv", low_memory=False)
    build_subjects = set(registry.loc[registry.reasoning_split.eq("build"), "subject_id"].astype(str))
    clean = clean_frame(source[source.subject_id.astype(str).isin(build_subjects)], features,
                        list(config["targets"]), dictionary)
    recomputed = fit_normalization(clean, features, list(config["targets"]))
    for key in features + list(config["targets"]):
        assert recomputed[key]["n"] == normalization[key]["n"]
        assert np.isclose(recomputed[key]["mean"], normalization[key]["mean"])
        assert np.isclose(recomputed[key]["std"], normalization[key]["std"])

    checkpoints = sorted((output / "checkpoints/gru_d").glob("*/best.pt"))
    assert len(checkpoints) == len(config["neural_seeds"]) == 3
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert checkpoint["architecture"] == "GRU-D"
        assert checkpoint["features"] == features and checkpoint["targets"] == list(config["targets"])
        curves = pd.read_csv(checkpoint_path.parent / "curves.csv")
        assert np.isclose(checkpoint["validation_macro_normalized_mae"],
                          curves.validation_macro_normalized_mae.min())

    predictions = pd.read_csv(output / "ensemble_forecast_predictions.csv")
    assert predictions.seeds.eq(3).all()
    prediction_lookup = predictions.set_index(["sample_id", "target"])
    events = pd.read_csv(ROOT / "splits/train_events.csv", low_memory=False)
    event_store = {}
    for row in events.itertuples(index=False):
        payload = json.loads(row.payload_json) if isinstance(row.payload_json, str) else {}
        event_store[str(row.event_id)] = {
            "subject_id": str(row.subject_id), "visit_month": float(row.visit_month),
            "time_month": float(row.time_month), "payload": payload,
        }

    count = 0
    split_counts = {"build": 0, "dev": 0, "confirmation": 0}
    confirmation_count = 0
    forbidden_keys = {"truth", "target_month", "future_targets", "diagnosis", "diagnosis_group"}
    with (output / "structured_reasoning_examples.jsonl").open() as stream:
        for line in stream:
            example = json.loads(line)
            count += 1
            split_counts[example["reasoning_split"]] += 1
            report = parse_report(example["completion"])
            prompt = example["prompt"]
            assert not forbidden_keys & set(prompt)
            assert not forbidden_keys & set(report)
            assert prompt["constraints"] == {
                "use_only_verified_history": True,
                "do_not_infer_diagnosis": True,
                "MRI_available_for_reasoning": False,
            }
            history = prompt["verified_history"]
            latest = {}
            for observation in history:
                event = event_store[observation["event_id"]]
                assert event["subject_id"] == example["subject_id"]
                assert event["visit_month"] <= float(prompt["cutoff_visit_month"]) + 1e-6
                assert event["time_month"] <= float(prompt["evidence_cutoff_time_month"]) + 1e-6
                assert observation["item"] in event["payload"]
                assert np.isclose(float(observation["value"]), float(event["payload"][observation["item"]]))
                assert observation["domain"] == dictionary[observation["item"]]["domain"]
                latest[observation["item"]] = observation
            expected_evidence = [{key: row[key] for key in ["event_id", "item", "time_month", "value", "domain"]}
                                 for row in sorted(latest.values(), key=lambda x: (x["domain"], x["item"]))]
            assert report["evidence"] == expected_evidence
            state, trend = expected_state_and_trend(history, dictionary, normalization, config)
            assert report["current_state"] == state
            assert report["trend"] == trend
            for target in config["targets"]:
                row = prediction_lookup.loc[(example["sample_id"], target)]
                forecast = report["forecast"][target]
                confidence = report["confidence"][target]
                assert forecast["horizon_months"] == int(prompt["forecast_horizon_months"])
                assert np.isclose(float(forecast["mean"]), round(float(row.prediction), 6), atol=5e-7)
                assert np.isclose(float(confidence["ensemble_std"]), round(float(row.ensemble_std), 6), atol=5e-7)
                expected_norm = round(float(row.ensemble_std) / float(normalization[target]["std"]), 6)
                assert np.isclose(float(confidence["normalized_std"]), expected_norm, atol=5e-7)
            if example["reasoning_split"] == "confirmation":
                confirmation_count += 1
    assert count == manifest["structured_examples"] == sum(manifest["sample_counts"].values())
    assert split_counts == manifest["sample_counts"]
    assert confirmation_count == decision["confirmation_reports"]
    assert decision["structured_reasoning_baseline_passed"] and decision["phase4b_sft_ready"]
    result = {"status": "passed", "subjects": len(used), "examples": count,
              "confirmation_examples": confirmation_count, "checkpoints": len(checkpoints),
              "build_only_normalization": True, "future_truth_absent_from_reports": True,
              "phase4b_sft_ready": True}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
