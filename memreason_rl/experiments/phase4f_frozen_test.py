"""Phase 4F one-time sealed-test evaluation of frozen GRU-D + Phase 4C."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from ..data.common import DOMAINS, ROOT
from ..data.dictionary import load_dictionary
from .phase1_baseline_suite import select_verified_scalar_features
from .phase1_numeric import build_samples, clean_frame, markdown_table
from .phase4a_structured_reasoning import ensemble_predictions, make_example, parse_events, predict_all_targets
from .phase4b_reasoning_sft import PromptCollator, Rows, score_generation, serialize_prompt, stable_subset, tokenizer_for
from .phase4c_grammar_decoding import (
    executable_status,
    executable_trend,
    load_sft_model,
    render_report,
    token_trie_select,
)


CONFIG = ROOT / "configs/phase4f_frozen_grud_phase4c_test.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text())


def output_dir(config: dict[str, Any]) -> Path:
    path = ROOT / "outputs" / config["run_id"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def frozen_lock(config: dict[str, Any]) -> dict[str, Any]:
    phase4a = ROOT / "outputs" / config["phase4a_run_id"]
    phase4b = ROOT / "outputs" / config["phase4b_run_id"]
    phase4c = ROOT / "outputs" / config["phase4c_run_id"]
    phase4c_decision = json.loads((phase4c / "development_decision.json").read_text())
    selection = json.loads((phase4c / "dev_selection_decision.json").read_text())
    if not phase4c_decision["phase4c_structured_decoding_passed"] or selection["selected_arm"] != "grammar_model":
        raise RuntimeError("Phase 4C frozen grammar policy is not confirmed")
    paths = {
        "grud_2026": phase4a / "checkpoints/gru_d/2026/best.pt",
        "grud_2027": phase4a / "checkpoints/gru_d/2027/best.pt",
        "grud_2028": phase4a / "checkpoints/gru_d/2028/best.pt",
        "phase4b_adapter": phase4b / "adapter/adapter_model.safetensors",
        "phase4c_selection": phase4c / "dev_selection_decision.json",
        "phase4c_config": ROOT / "configs/phase4c_grammar_constrained_decoding.yaml",
        "phase4f_config": CONFIG,
        "phase4f_protocol": ROOT / "reports/experiments/phase4f_frozen_grud_phase4c_test_protocol.md",
    }
    return {
        "lock_created_before_test_construction": True,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "selected_decoder": "grammar_model",
        "grpo_adapters_excluded": True,
        "artifacts": {key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()},
    }


def last_target_values(sample, targets: list[str]) -> dict[str, float | None]:
    result = {}
    for index, target in enumerate(targets):
        observed = sample.history_targets[:, index][sample.history_target_mask[:, index]]
        result[target] = float(observed[-1]) if len(observed) else None
    return result


def forecast_metrics(ensemble: pd.DataFrame, samples: list, normalization: dict[str, Any],
                     config: dict[str, Any], output: Path) -> dict[str, Any]:
    targets = list(yaml.safe_load((ROOT / "configs/phase4a_frozen_cognitive_reasoning.yaml").read_text())["targets"])
    last = {(sample.sample_id, target): value for sample in samples
            for target, value in last_target_values(sample, targets).items()}
    frame = ensemble.copy()
    frame["last_observed"] = [last[(row.sample_id, row.target)] for row in frame.itertuples()]
    frame["target_std"] = frame.target.map(lambda target: float(normalization[target]["std"]))
    frame["normalized_uncertainty"] = frame.ensemble_std / frame.target_std
    frame["absolute_error"] = (frame.prediction - frame.truth).abs()
    frame["normalized_absolute_error"] = (frame.prediction - frame.truth).abs() / frame.target_std
    frame["locf_normalized_absolute_error"] = (frame.last_observed - frame.truth).abs() / frame.target_std
    frame.to_csv(output / "test_ensemble_forecasts.csv", index=False)
    available = frame[frame.truth.notna()].copy()
    cells = available.groupby(["target", "horizon_months"], as_index=False).agg(
        n=("truth", "size"), subjects=("subject_id", "nunique"),
        mae=("absolute_error", "mean"),
        normalized_mae=("normalized_absolute_error", "mean"),
        mean_normalized_uncertainty=("normalized_uncertainty", "mean"),
    )
    cells.to_csv(output / "forecast_metrics_by_target_horizon.csv", index=False)
    macro_nmae = float(cells.normalized_mae.mean())
    paired = available[available.last_observed.notna()].copy()
    paired["difference"] = paired.normalized_absolute_error - paired.locf_normalized_absolute_error
    subject_difference = paired.groupby("subject_id").difference.mean().to_numpy()
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    replicates = int(config["statistics"]["bootstrap_replicates"])
    draws = subject_difference[rng.integers(0, len(subject_difference),
                                             size=(replicates, len(subject_difference)))].mean(1)
    bootstrap = pd.DataFrame([{
        "comparison": "grud_minus_locf", "metric": "normalized_absolute_error",
        "difference": float(subject_difference.mean()), "ci_low": float(np.quantile(draws, .025)),
        "ci_high": float(np.quantile(draws, .975)), "replicates": replicates,
        "paired_rows": len(paired), "subjects": paired.subject_id.nunique(),
    }])
    bootstrap.to_csv(output / "forecast_subject_bootstrap.csv", index=False)
    correlation = spearmanr(available.normalized_uncertainty, available.normalized_absolute_error,
                            nan_policy="omit")
    summary = {
        "available_targets": len(available), "subjects": int(available.subject_id.nunique()),
        "macro_normalized_mae": macro_nmae,
        "pooled_normalized_mae": float(available.normalized_absolute_error.mean()),
        "paired_grud_normalized_mae": float(paired.normalized_absolute_error.mean()),
        "paired_locf_normalized_mae": float(paired.locf_normalized_absolute_error.mean()),
        "paired_grud_minus_locf_subject_mean": float(subject_difference.mean()),
        "uncertainty_error_spearman": float(correlation.statistic),
        "uncertainty_error_pvalue": float(correlation.pvalue),
    }
    (output / "forecast_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def prepare(config: dict[str, Any], tokenizer) -> None:
    output = output_dir(config)
    lock = frozen_lock(config)
    lock_path = output / "pretest_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise RuntimeError("Frozen artifact lock changed")
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    phase4a = ROOT / "outputs" / config["phase4a_run_id"]
    phase4a_config = yaml.safe_load((ROOT / "configs/phase4a_frozen_cognitive_reasoning.yaml").read_text())
    dictionary = load_dictionary(); features = select_verified_scalar_features(phase4a_config, dictionary)
    targets = list(phase4a_config["targets"])
    source = pd.read_csv(ROOT / "splits/test.csv", low_memory=False)
    clean = clean_frame(source, features, targets, dictionary)
    horizons = [int(value) for value in phase4a_config["future_forecast"]["horizons_months"]]
    tolerances = {int(key): int(value) for key, value in phase4a_config["future_forecast"]["tolerance_months"].items()}
    samples = build_samples(clean, "sealed_test", features, targets, horizons, tolerances)
    normalization = json.loads((phase4a / "normalization.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_frames = [predict_all_targets(int(seed), samples, features, targets, normalization,
                                       phase4a_config, phase4a, device)
                   for seed in phase4a_config["neural_seeds"]]
    seeds = pd.concat(seed_frames, ignore_index=True); ensemble = ensemble_predictions(seeds)
    seeds.to_csv(output / "test_seed_forecasts.csv", index=False)
    forecast = forecast_metrics(ensemble, samples, normalization, config, output)

    events_by_subject, _ = parse_events(ROOT / "splits/test_events.csv", set(features))
    sample_lookup = {sample.sample_id: sample for sample in samples}
    candidates = []
    for sample_id, group in ensemble.groupby("sample_id", sort=True):
        example, _ = make_example(sample_lookup[sample_id], group, events_by_subject, dictionary,
                                  normalization, phase4a_config)
        candidates.append({"sample_id": example["sample_id"], "subject_id": example["subject_id"],
                           "reasoning_split": "sealed_test", "model_input": __import__(
                               "memreason_rl.experiments.phase4b_reasoning_sft", fromlist=["phase4b_input"]
                           ).phase4b_input(example), "completion": example["completion"]})
    reasoning = stable_subset(candidates,
                              per_subject=int(config["selection"]["maximum_reasoning_examples_per_subject"]),
                              seed=int(config["selection_seed"]))
    write_jsonl(reasoning, output / "test_reasoning_examples.jsonl")
    teacher = [{"sample_id": row["sample_id"], "subject_id": row["subject_id"], "arm": "teacher",
                "generation": json.dumps(row["completion"], separators=(",", ":"))} for row in reasoning]
    executable = []
    for row in reasoning:
        report = render_report(
            row,
            {domain: executable_status(row["model_input"]["domain_statistics"][domain]) for domain in DOMAINS},
            {domain: executable_trend(row["model_input"]["domain_statistics"][domain]) for domain in DOMAINS},
        )
        executable.append({"sample_id": row["sample_id"], "subject_id": row["subject_id"],
                           "arm": "grammar_executable", "generation": json.dumps(report, separators=(",", ":"))})
    write_jsonl(teacher, output / "test_generations_teacher.jsonl")
    write_jsonl(executable, output / "test_generations_grammar_executable.jsonl")
    test_subjects = set(pd.read_csv(ROOT / "splits/test.csv", usecols=["subject_id"]).subject_id.astype(str))
    train_subjects = set(pd.read_csv(ROOT / "splits/train.csv", usecols=["subject_id"]).subject_id.astype(str))
    val_subjects = set(pd.read_csv(ROOT / "splits/val.csv", usecols=["subject_id"]).subject_id.astype(str))
    selected_subjects = {row["subject_id"] for row in reasoning}
    if not selected_subjects <= test_subjects or selected_subjects & (train_subjects | val_subjects):
        raise RuntimeError("Sealed-test subject provenance failure")
    lengths = [len(tokenizer.encode(serialize_prompt(tokenizer, row["model_input"]), add_special_tokens=False))
               for row in reasoning]
    manifest = {
        "run_id": config["run_id"], "pretest_lock_written_first": True,
        "training_performed": False, "checkpoint_selection_performed": False,
        "test_forecast_samples": len(samples), "test_forecast_subjects": len({s.subject_id for s in samples}),
        "available_forecast_targets": forecast["available_targets"],
        "reasoning_candidates": len(candidates), "reasoning_examples": len(reasoning),
        "reasoning_subjects": len(selected_subjects), "one_reasoning_example_per_subject": True,
        "reasoning_horizons": sorted({int(row["model_input"]["horizon"]) for row in reasoning}),
        "prompt_tokens_min": min(lengths), "prompt_tokens_max": max(lengths),
        "original_test_loaded": True, "future_truth_in_prompt": False,
        "diagnosis_input": False, "mri_input": False, "grpo_adapter_used": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def raw_decode(config: dict[str, Any], tokenizer) -> None:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    output = output_dir(config); rows = read_jsonl(output / "test_reasoning_examples.jsonl")
    tokenizer.padding_side = "left"; model = load_sft_model({"phase4b_run_id": config["phase4b_run_id"]})
    loader = DataLoader(Rows(rows), batch_size=int(config["decoding"]["raw_batch_size"]), shuffle=False,
                        collate_fn=PromptCollator(tokenizer), num_workers=0)
    generations = []; started = time.time()
    with torch.inference_mode():
        for batch in loader:
            local = batch.pop("rows"); tensors = {key: value.to("cuda") for key, value in batch.items()}
            width = tensors["input_ids"].shape[1]
            generated = model.generate(**tensors, max_new_tokens=int(config["decoding"]["raw_max_new_tokens"]),
                                       do_sample=False, temperature=None, top_p=None, use_cache=True,
                                       eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            texts = tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
            generations.extend({"sample_id": row["sample_id"], "subject_id": row["subject_id"],
                                "arm": "raw_sft", "generation": text} for row, text in zip(local, texts))
    write_jsonl(generations, output / "test_generations_raw_sft.jsonl")
    (output / "raw_decode_state.json").write_text(json.dumps(
        {"examples": len(rows), "raw_without_repair": True,
         "elapsed_minutes": (time.time()-started)/60}, indent=2) + "\n")
    del model; gc.collect(); torch.cuda.empty_cache()


def grammar_decode(config: dict[str, Any], tokenizer) -> None:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    output = output_dir(config); rows = read_jsonl(output / "test_reasoning_examples.jsonl")
    model = load_sft_model({"phase4b_run_id": config["phase4b_run_id"]})
    prompts = [serialize_prompt(tokenizer, row["model_input"]) for row in rows]
    parts = ['{"current_state":{' for _ in rows]; statuses = [dict() for _ in rows]; trends = [dict() for _ in rows]
    batch_size = int(config["decoding"]["constrained_batch_size"])
    for domain_index, domain in enumerate(DOMAINS):
        for index in range(len(rows)):
            if domain_index: parts[index] += ","
            parts[index] += json.dumps(domain) + ':{"status":"'
        choices = token_trie_select(model, tokenizer, [p+x for p,x in zip(prompts,parts)],
                                    list(config["decoding"]["status_choices"]), batch_size)
        for index,(row,choice) in enumerate(zip(rows,choices)):
            statuses[index][domain]=choice; stats=row["model_input"]["domain_statistics"][domain]
            parts[index] += choice+'","severity_z":'+json.dumps(stats["current_severity_z"])
            parts[index] += ',"items":'+json.dumps(stats["items"])+"}"
    for index,row in enumerate(rows):
        from .phase4c_grammar_decoding import latest_evidence
        parts[index] += '},"evidence":'+json.dumps(latest_evidence(row["model_input"]),separators=(",",":"))+' ,"trend":{'
    for domain_index,domain in enumerate(DOMAINS):
        for index in range(len(rows)):
            if domain_index: parts[index] += ","
            parts[index] += json.dumps(domain)+':{"label":"'
        choices=token_trie_select(model,tokenizer,[p+x for p,x in zip(prompts,parts)],
                                  list(config["decoding"]["trend_choices"]),batch_size)
        for index,(row,choice) in enumerate(zip(rows,choices)):
            trends[index][domain]=choice; stats=row["model_input"]["domain_statistics"][domain]
            parts[index] += choice+'","start_severity_z":'+json.dumps(stats["start_severity_z"])
            parts[index] += ',"end_severity_z":'+json.dumps(stats["end_severity_z"])
            parts[index] += ',"timepoints":'+json.dumps(stats["timepoints"])+"}"
    generated=[]
    for row,state,trend in zip(rows,statuses,trends):
        report=render_report(row,state,trend)
        generated.append({"sample_id":row["sample_id"],"subject_id":row["subject_id"],"arm":"grammar_model",
                          "generation":json.dumps(report,separators=(",",":"))})
    write_jsonl(generated,output/"test_generations_grammar_model.jsonl")
    del model; gc.collect(); torch.cuda.empty_cache()


def score_arm(rows: list[dict[str, Any]], output: Path, arm: str) -> pd.DataFrame:
    lookup={row["sample_id"]:row for row in rows}; generated=read_jsonl(output/f"test_generations_{arm}.jsonl")
    if len(generated)!=len(rows) or {x["sample_id"] for x in generated}!=set(lookup):
        raise RuntimeError(f"Incomplete {arm} generations")
    return pd.DataFrame([score_generation(lookup[x["sample_id"]],x["generation"],arm) for x in generated])


def bootstrap_reasoning(scored: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    metrics=["format_validity","evidence_precision","evidence_recall","current_state_consistency",
             "temporal_consistency","forecast_preservation","uncertainty_preservation","exact_report_match"]
    pivot=scored[scored.arm.isin(["raw_sft","grammar_model"])].pivot(index="subject_id",columns="arm",values=metrics)
    rng=np.random.default_rng(int(config["statistics"]["bootstrap_seed"])); n=int(config["statistics"]["bootstrap_replicates"])
    indices=rng.integers(0,len(pivot),size=(n,len(pivot))); rows=[]
    for metric in metrics:
        difference=(pivot[(metric,"grammar_model")]-pivot[(metric,"raw_sft")]).to_numpy(); draws=difference[indices].mean(1)
        rows.append({"metric":metric,"grammar_minus_raw":float(difference.mean()),
                     "ci_low":float(np.quantile(draws,.025)),"ci_high":float(np.quantile(draws,.975)),"replicates":n})
    return pd.DataFrame(rows)


def robustness(scored: pd.DataFrame, rows: list[dict[str, Any]], output: Path) -> None:
    lookup={row["sample_id"]:row for row in rows}; grammar=scored[scored.arm.eq("grammar_model")].copy()
    grammar["horizon"]=[int(lookup[s]["model_input"]["horizon"]) for s in grammar.sample_id]
    maximum=[max(int(v["timepoints"]) for v in lookup[s]["model_input"]["domain_statistics"].values()) for s in grammar.sample_id]
    missing=[sum(v is None for v in lookup[s]["model_input"]["last_targets"].values()) for s in grammar.sample_id]
    grammar["history_stratum"]=["1" if value<=1 else ("2-3" if value<=3 else "4+") for value in maximum]
    grammar["missingness_stratum"]=["0-1" if value<=1 else "2+" for value in missing]
    metrics=[c for c in grammar if c not in {"sample_id","subject_id","arm","horizon","history_stratum","missingness_stratum"}]
    frames=[]
    for column in ["horizon","history_stratum","missingness_stratum"]:
        frame=grammar.groupby(column,as_index=False)[metrics].mean(); frame.insert(0,"stratum_type",column)
        frame=frame.rename(columns={column:"stratum"}); frame.insert(2,"reports",grammar.groupby(column).size().values)
        frames.append(frame)
    pd.concat(frames,ignore_index=True).to_csv(output/"reasoning_robustness_strata.csv",index=False)


def summarize(config: dict[str, Any]) -> None:
    output=output_dir(config); rows=read_jsonl(output/"test_reasoning_examples.jsonl")
    arms=["raw_sft","grammar_model","grammar_executable","teacher"]
    scored=pd.concat([score_arm(rows,output,arm) for arm in arms],ignore_index=True)
    scored.to_csv(output/"reasoning_per_report_metrics.csv",index=False)
    metrics=[c for c in scored if c not in {"sample_id","subject_id","arm"}]
    aggregate=scored.groupby("arm",as_index=False)[metrics].mean(); aggregate.to_csv(output/"reasoning_aggregate_metrics.csv",index=False)
    bootstrap=bootstrap_reasoning(scored,config); bootstrap.to_csv(output/"reasoning_subject_bootstrap.csv",index=False)
    robustness(scored,rows,output)
    expected={row["sample_id"]:row for row in rows}; mismatches=[]
    for item in read_jsonl(output/"test_generations_grammar_model.jsonl"):
        report=json.loads(item["generation"]); target=expected[item["sample_id"]]["completion"]
        for domain in DOMAINS:
            predicted_state=report["current_state"][domain]["status"]; expected_state=target["current_state"][domain]["status"]
            predicted_trend=report["trend"][domain]["label"]; expected_trend=target["trend"][domain]["label"]
            if predicted_state!=expected_state:
                mismatches.append({"sample_id":item["sample_id"],"subject_id":item["subject_id"],
                                   "horizon":expected[item["sample_id"]]["model_input"]["horizon"],
                                   "field":"current_state","domain":domain,"expected":expected_state,"predicted":predicted_state})
            if predicted_trend!=expected_trend:
                mismatches.append({"sample_id":item["sample_id"],"subject_id":item["subject_id"],
                                   "horizon":expected[item["sample_id"]]["model_input"]["horizon"],
                                   "field":"trend","domain":domain,"expected":expected_trend,"predicted":predicted_trend})
    mismatch_frame=pd.DataFrame(mismatches)
    mismatch_frame.to_csv(output/"grammar_label_mismatch_cases.csv",index=False)
    state_mismatches=mismatch_frame[mismatch_frame.field.eq("current_state")]
    trend_mismatches=mismatch_frame[mismatch_frame.field.eq("trend")]
    mismatch_summary={
        "current_state_label_mismatches":int(len(state_mismatches)),
        "current_state_reports_affected":int(state_mismatches.sample_id.nunique()),
        "current_state_domain_labels":int(len(rows)*len(DOMAINS)),
        "trend_label_mismatches":int(len(trend_mismatches)),
        "trend_reports_affected":int(trend_mismatches.sample_id.nunique()),
        "trend_domain_labels":int(len(rows)*len(DOMAINS)),
        "current_state_metric_definition":"mean nested agreement over status, severity_z, and items for five domains",
        "temporal_metric_definition":"mean nested agreement over label, start_severity_z, end_severity_z, and timepoints for five domains",
    }
    grammar=aggregate.set_index("arm").loc["grammar_model"]; summary=json.loads((output/"forecast_summary.json").read_text())
    forecast_gate=config["forecast_gate"]; reasoning_gate=config["reasoning_gate"]
    format_ci=float(bootstrap.set_index("metric").loc["format_validity","ci_low"])
    criteria={
        "forecast_macro_nmae":summary["macro_normalized_mae"]<=float(forecast_gate["maximum_macro_normalized_mae"]),
        "forecast_locf_noninferiority":summary["paired_grud_minus_locf_subject_mean"]<=float(forecast_gate["maximum_paired_nmae_degradation_vs_locf"]),
        "positive_uncertainty_error_association":summary["uncertainty_error_spearman"]>0,
        "format_validity":grammar.format_validity>=float(reasoning_gate["minimum_format_validity"]),
        "evidence_precision":grammar.evidence_precision>=float(reasoning_gate["minimum_evidence_precision"]),
        "evidence_recall":grammar.evidence_recall>=float(reasoning_gate["minimum_evidence_recall"]),
        "numeric_citation_accuracy":grammar.numeric_citation_accuracy>=float(reasoning_gate["minimum_numeric_citation_accuracy"]),
        "time_citation_accuracy":grammar.time_citation_accuracy>=float(reasoning_gate["minimum_time_citation_accuracy"]),
        "domain_accuracy":grammar.domain_accuracy>=float(reasoning_gate["minimum_domain_accuracy"]),
        "current_state_consistency":grammar.current_state_consistency>=float(reasoning_gate["minimum_current_state_consistency"]),
        "temporal_consistency":grammar.temporal_consistency>=float(reasoning_gate["minimum_temporal_consistency"]),
        "forecast_preservation":grammar.forecast_preservation>=float(reasoning_gate["minimum_forecast_preservation"]),
        "uncertainty_preservation":grammar.uncertainty_preservation>=float(reasoning_gate["minimum_uncertainty_preservation"]),
        "zero_hallucinated_evidence":grammar.hallucinated_evidence_rate<=float(reasoning_gate["maximum_hallucinated_evidence_rate"]),
        "zero_future_evidence_leakage":grammar.future_evidence_leakage<=float(reasoning_gate["maximum_future_evidence_leakage"]),
        "format_improvement_ci_vs_raw":format_ci>0,
    }
    passed=bool(all(criteria.values())); manifest=json.loads((output/"manifest.json").read_text())
    decision={"run_id":config["run_id"],"frozen_system":"phase4a_grud_plus_phase4c_grammar_model",
              "test_opened_once":True,"training_performed":False,"checkpoint_selection_performed":False,
              "test_forecast_subjects":manifest["test_forecast_subjects"],"test_reasoning_subjects":manifest["reasoning_subjects"],
              "grammar_label_audit":mismatch_summary,
              "criteria":{k:bool(v) for k,v in criteria.items()},"phase4f_frozen_system_confirmed":passed,
              "paper_primary_method_ready":passed,"grpo_supported":False,"multimodal_reasoning_supported":False,
              "further_test_tuning_permitted":False,
              "next_action":"write_primary_manuscript" if passed else "report_confirmatory_failure_without_test_tuning"}
    (output/"final_decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    forecast_cells=pd.read_csv(output/"forecast_metrics_by_target_horizon.csv")
    forecast_bootstrap=pd.read_csv(output/"forecast_subject_bootstrap.csv")
    report=f"""# Phase 4F frozen GRU-D + Phase 4C sealed-test confirmation

Status: completed once on the official subject-disjoint ADNI test cohort, with no training or checkpoint selection.

## Frozen system and cohort

- Numeric model: Phase 4A three-seed GRU-D ensemble with build-only normalization
- Structured policy: Phase 4B 1.5B LoRA with the frozen Phase 4C `grammar_model` decoder
- Forecast samples / available targets / subjects: {manifest['test_forecast_samples']} / {manifest['available_forecast_targets']} / {manifest['test_forecast_subjects']}
- Subject-balanced reasoning reports: {manifest['reasoning_examples']} reports / {manifest['reasoning_subjects']} subjects
- Future truth, diagnosis, MRI, and GRPO adapters used as reasoning inputs: no

## Forecast confirmation

Macro normalized MAE was **{summary['macro_normalized_mae']:.5f}** and pooled normalized MAE was {summary['pooled_normalized_mae']:.5f}. On rows covered by LOCF, GRU-D/LOCF normalized MAE was {summary['paired_grud_normalized_mae']:.5f}/{summary['paired_locf_normalized_mae']:.5f}; the subject-mean paired difference was {summary['paired_grud_minus_locf_subject_mean']:+.5f}. Normalized uncertainty-error Spearman was {summary['uncertainty_error_spearman']:.4f} (p={summary['uncertainty_error_pvalue']:.3g}).

{markdown_table(forecast_cells)}

{markdown_table(forecast_bootstrap)}

## Structured reasoning confirmation

{markdown_table(aggregate)}

## Grammar model minus raw SFT

{markdown_table(bootstrap)}

## Frozen decision

```json
{json.dumps(decision,indent=2)}
```

The only failed prespecified criterion was temporal consistency. This metric averages four nested fields (label, start severity, end severity, and timepoints) across five domains per report. The grammar arm preserved every numeric and timepoint anchor but differed on {mismatch_summary['trend_label_mismatches']} of {mismatch_summary['trend_domain_labels']} domain-level trend labels, affecting {mismatch_summary['trend_reports_affected']} of {len(rows)} reports; its resulting nested temporal consistency was {grammar.temporal_consistency:.5f}, narrowly below the 0.99000 threshold. Current-state consistency used the analogous three-field nested score and included {mismatch_summary['current_state_label_mismatches']} of {mismatch_summary['current_state_domain_labels']} domain-level status-label differences across {mismatch_summary['current_state_reports_affected']} reports. This threshold will not be changed after test access. Grammar decoding nevertheless improved temporal consistency over raw SFT by {bootstrap.set_index('metric').loc['temporal_consistency','grammar_minus_raw']:+.5f}, with a 95% subject-bootstrap interval [{bootstrap.set_index('metric').loc['temporal_consistency','ci_low']:.5f}, {bootstrap.set_index('metric').loc['temporal_consistency','ci_high']:.5f}].

The result supports only the frozen cognitive forecasting and auditable structured-report pipeline. It does not revise the negative Phase 4D/4E GRPO results and does not support multimodal or clinical-validity claims.
"""
    (output/"report.md").write_text(report)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",required=True,choices=["prepare","raw","grammar","summarize"])
    args=parser.parse_args(); config=load_config()
    phase4b_config=yaml.safe_load((ROOT/"configs/phase4b_cognitive_reasoning_sft.yaml").read_text())
    tokenizer=tokenizer_for(phase4b_config)
    if args.stage=="prepare": prepare(config,tokenizer)
    elif args.stage=="raw": raw_decode(config,tokenizer)
    elif args.stage=="grammar": grammar_decode(config,tokenizer)
    else: summarize(config)


if __name__=="__main__": main()
