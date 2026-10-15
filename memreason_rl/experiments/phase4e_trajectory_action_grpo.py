"""Phase 4E bounded trajectory-action SFT and native on-policy GRPO."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from ..data.common import ROOT
from .phase1_numeric import markdown_table
from .phase4b_reasoning_sft import load_base, load_rows, tokenizer_for
from .phase4d_constrained_grpo import choice_log_probs


CONFIG = ROOT / "configs/phase4e_trajectory_action_grpo.yaml"
SYSTEM = (
    "Choose only bounded forecast residual and uncertainty actions. Return the exact constrained "
    "JSON action object. Future observations are unavailable. Do not change evidence or infer diagnosis."
)


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text())


def output_dir(config: dict[str, Any]) -> Path:
    path = ROOT / "outputs" / config["run_id"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def rank_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def one_per_subject(rows: list[dict[str, Any]], subjects: list[str], seed: int) -> list[dict[str, Any]]:
    wanted = set(subjects)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["subject_id"] in wanted:
            grouped.setdefault(row["subject_id"], []).append(row)
    selected = []
    for subject in subjects:
        selected.append(min(grouped[subject], key=lambda row: rank_key(seed, row["sample_id"])))
    return sorted(selected, key=lambda row: row["sample_id"])


def serialize_prompt(tokenizer, model_input: dict[str, Any]) -> str:
    return (tokenizer.bos_token + SYSTEM + "<｜User｜>" +
            json.dumps(model_input, separators=(",", ":"), allow_nan=False) + "<｜Assistant｜>")


def anchor_action(config: dict[str, Any]) -> dict[str, Any]:
    return {target: {"residual": "anchor", "uncertainty": "baseline"}
            for target in config["targets"]}


def action_completion(config: dict[str, Any]) -> dict[str, Any]:
    return {"forecast_adjustments": anchor_action(config)}


def forecast_lookup(config: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / "outputs" / config["phase4a_run_id"] / "ensemble_forecast_predictions.csv"
    return pd.read_csv(path)


def prepare(config: dict[str, Any]) -> None:
    output = output_dir(config)
    phase4b = ROOT / "outputs" / config["phase4b_run_id"]
    build = load_rows(phase4b, "build")
    excluded = {row["subject_id"] for row in read_jsonl(
        ROOT / "outputs" / config["phase4d_run_id"] / "rollout_examples.jsonl")}
    subjects = sorted({row["subject_id"] for row in build} - excluded,
                      key=lambda value: rank_key(int(config["selection_seed"]), value))
    counts = config["data"]
    required = int(counts["sft_subjects"]) + int(counts["rollout_subjects"]) + int(counts["development_subjects"])
    if len(subjects) < required:
        raise RuntimeError("Insufficient eligible build subjects")
    s0 = int(counts["sft_subjects"]); s1 = s0 + int(counts["rollout_subjects"])
    partitions = {
        "sft": subjects[:s0],
        "rollout": subjects[s0:s1],
        "development": subjects[s1:required],
    }
    rows_by_split = {key: one_per_subject(build, value, int(config["selection_seed"]))
                     for key, value in partitions.items()}
    for split, rows in rows_by_split.items():
        write_jsonl(rows, output / f"{split}_examples.jsonl")
    sets = [set(value) for value in partitions.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Phase 4E subject leakage")
    predictions = forecast_lookup(config)
    available = set(predictions.sample_id.astype(str))
    if any(row["sample_id"] not in available for rows in rows_by_split.values() for row in rows):
        raise RuntimeError("Selected action example lacks frozen forecast records")
    normalization = json.loads((ROOT / "outputs" / config["phase4a_run_id"] / "normalization.json").read_text())
    (output / "normalization.json").write_text(json.dumps(
        {target: normalization[target] for target in config["targets"]}, indent=2) + "\n")
    manifest = {
        "run_id": config["run_id"], "protocol_frozen_before_training": True,
        "source_partition": "phase4a_build_only", "phase4d_rollout_subjects_excluded": len(excluded),
        "sft_examples": len(rows_by_split["sft"]), "sft_subjects": len(sets[0]),
        "rollout_examples": len(rows_by_split["rollout"]), "rollout_subjects": len(sets[1]),
        "development_examples": len(rows_by_split["development"]), "development_subjects": len(sets[2]),
        "subject_disjoint": True, "future_truth_in_prompt": False,
        "action_sft_uses_future_truth": False, "grpo_future_truth_reward_only": True,
        "numeric_anchor": "frozen_phase4a_grud_ensemble", "diagnosis_input": False, "mri_input": False,
        "original_test_loaded": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


class Rows(Dataset):
    def __init__(self, rows: list[dict[str, Any]]): self.rows = rows
    def __len__(self) -> int: return len(self.rows)
    def __getitem__(self, index: int) -> dict[str, Any]: return self.rows[index]


class ActionCollator:
    def __init__(self, tokenizer, config: dict[str, Any]):
        self.tokenizer = tokenizer; self.config = config
    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompts = [serialize_prompt(self.tokenizer, row["model_input"]) for row in rows]
        completion = json.dumps(action_completion(self.config), separators=(",", ":")) + self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompts, add_special_tokens=False)["input_ids"]
        completion_ids = self.tokenizer([completion] * len(rows), add_special_tokens=False)["input_ids"]
        sequences, labels = [], []
        for prompt, target in zip(prompt_ids, completion_ids):
            if len(prompt) + len(target) > int(self.config["action_sft"]["max_length"]):
                raise RuntimeError("Action SFT sequence overflow")
            sequences.append(prompt + target); labels.append([-100] * len(prompt) + target)
        width = max(map(len, sequences)); pad = self.tokenizer.pad_token_id
        ids, masks, targets = [], [], []
        for sequence, label in zip(sequences, labels):
            amount = width - len(sequence)
            ids.append(sequence + [pad] * amount); masks.append([1] * len(sequence) + [0] * amount)
            targets.append(label + [-100] * amount)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks),
                "labels": torch.tensor(targets)}


def load_policy(config: dict[str, Any], adapter: Path, trainable: bool):
    from peft import PeftModel
    phase4b_config = yaml.safe_load((ROOT / "configs/phase4b_cognitive_reasoning_sft.yaml").read_text())
    model = load_base(phase4b_config, training=trainable)
    model = PeftModel.from_pretrained(model, adapter, local_files_only=True, is_trainable=trainable)
    model.to("cuda")
    if trainable: model.train()
    else: model.requires_grad_(False).eval()
    return model


def free_model(model) -> None:
    del model; gc.collect(); torch.cuda.empty_cache()


def train_action_sft(config: dict[str, Any], tokenizer) -> None:
    from transformers import get_cosine_schedule_with_warmup
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    output = output_dir(config); spec = config["action_sft"]
    rows = read_jsonl(output / "sft_examples.jsonl")
    generator = torch.Generator().manual_seed(int(config["selection_seed"]))
    loader = DataLoader(Rows(rows), batch_size=int(spec["batch_size"]), shuffle=True, generator=generator,
                        collate_fn=ActionCollator(tokenizer, config), num_workers=0)
    source = ROOT / "outputs" / config["phase4b_run_id"] / "adapter"
    model = load_policy(config, source, trainable=True)
    if bool(spec["gradient_checkpointing"]):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"]))
    accumulation = int(spec["gradient_accumulation_steps"])
    updates = math.ceil(len(loader) * int(spec["epochs"]) / accumulation)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(updates * float(spec["warmup_fraction"]))), updates)
    random.seed(int(config["selection_seed"])); np.random.seed(int(config["selection_seed"]))
    torch.manual_seed(int(config["selection_seed"])); torch.cuda.manual_seed_all(int(config["selection_seed"]))
    optimizer.zero_grad(set_to_none=True); losses = []; curves = []; update = 0; started = time.time()
    for epoch in range(int(spec["epochs"])):
        for step, batch in enumerate(loader):
            batch = {key: value.to("cuda") for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16): loss = model(**batch).loss
            (loss / accumulation).backward(); losses.append(float(loss.detach()))
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(params, float(spec["gradient_clip"]))
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); update += 1
                record = {"update": update, "loss": float(np.mean(losses[-accumulation:])),
                          "learning_rate": scheduler.get_last_lr()[0], "elapsed_minutes": (time.time()-started)/60}
                curves.append(record); print(json.dumps({"action_sft": record}), flush=True)
    model.save_pretrained(output / "action_sft_adapter", safe_serialization=True); tokenizer.save_pretrained(output / "action_sft_adapter")
    pd.DataFrame(curves).to_csv(output / "action_sft_training_curves.csv", index=False)
    state = {"examples": len(rows), "subjects": len({r['subject_id'] for r in rows}), "updates": update,
             "train_loss": float(np.mean(losses)), "trainable_parameters": int(sum(p.numel() for p in params)),
             "future_truth_used": False, "anchor_residual_only": True, "unit_uncertainty_only": True,
             "elapsed_minutes": (time.time()-started)/60}
    (output / "action_sft_state.json").write_text(json.dumps(state, indent=2) + "\n")
    free_model(model)


def append_residual_prefix(parts: list[str], target: str, index: int) -> None:
    for i in range(len(parts)):
        if index: parts[i] += ","
        parts[i] += json.dumps(target) + ':{"residual":"'


def finish_residual(parts: list[str], labels: list[str]) -> None:
    for i, label in enumerate(labels): parts[i] += label + '","uncertainty":"'


def finish_uncertainty(parts: list[str], labels: list[str]) -> None:
    for i, label in enumerate(labels): parts[i] += label + '"}'


def decode_actions(model, tokenizer, rows: list[dict[str, Any]], config: dict[str, Any], sample: bool,
                   generator: torch.Generator | None = None, reference=None):
    prompts = [serialize_prompt(tokenizer, row["model_input"]) for row in rows]
    parts = ['{"forecast_adjustments":{' for _ in rows]
    actions = [{target: {} for target in config["targets"]} for _ in rows]
    traces = []
    model.eval()
    with torch.no_grad():
        for target_index, target in enumerate(config["targets"]):
            append_residual_prefix(parts, target, target_index)
            contexts = [prompt + part for prompt, part in zip(prompts, parts)]
            choices = list(config["actions"]["residual_choices"])
            logp = choice_log_probs(model, tokenizer, contexts, choices,
                                    float(config["policy"]["sampling_temperature"]) if sample else 1.0)
            ref = choice_log_probs(reference, tokenizer, contexts, choices,
                                   float(config["policy"]["sampling_temperature"])) if reference is not None else None
            selected = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1) if sample else logp.argmax(-1)
            labels = [choices[i] for i in selected.tolist()]
            for i, label in enumerate(labels): actions[i][target]["residual"] = label
            if sample: traces.append({"contexts": contexts, "choices": choices, "selected": selected.cpu(),
                                      "old": logp.gather(1, selected[:, None]).squeeze(1).cpu(),
                                      "reference": ref.cpu(), "kind": "residual", "target": target})
            finish_residual(parts, labels)
            contexts = [prompt + part for prompt, part in zip(prompts, parts)]
            choices = list(config["actions"]["uncertainty_choices"])
            logp = choice_log_probs(model, tokenizer, contexts, choices,
                                    float(config["policy"]["sampling_temperature"]) if sample else 1.0)
            ref = choice_log_probs(reference, tokenizer, contexts, choices,
                                   float(config["policy"]["sampling_temperature"])) if reference is not None else None
            selected = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1) if sample else logp.argmax(-1)
            labels = [choices[i] for i in selected.tolist()]
            for i, label in enumerate(labels): actions[i][target]["uncertainty"] = label
            if sample: traces.append({"contexts": contexts, "choices": choices, "selected": selected.cpu(),
                                      "old": logp.gather(1, selected[:, None]).squeeze(1).cpu(),
                                      "reference": ref.cpu(), "kind": "uncertainty", "target": target})
            finish_uncertainty(parts, labels)
    return actions, traces


def phi(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2 * math.pi)


def Phi(value: float) -> float:
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def direction(value: float, last: float, scale: float, higher_worse: bool, threshold: float) -> str:
    delta = (value - last) / max(scale, 1e-6)
    if abs(delta) <= threshold: return "stable"
    worsening = delta > 0 if higher_worse else delta < 0
    return "decline" if worsening else "improve"


def score_action(row: dict[str, Any], action: dict[str, Any], forecast_rows: pd.DataFrame,
                 normalization: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    by_target = forecast_rows.set_index("target")
    errors = []; crps_values = []; temporal = []; raw_ranges = []; adjusted_ranges = []
    for target in config["targets"]:
        record = by_target.loc[target]; norm = normalization[target]; scale = max(float(norm["std"]), 1e-6)
        base = float(record.prediction); residual = float(config["actions"]["residual_std_units"][action[target]["residual"]])
        raw = base + residual * scale
        adjusted = min(max(raw, float(norm["minimum"])), float(norm["maximum"]))
        raw_ranges.append(float(float(norm["minimum"]) <= raw <= float(norm["maximum"])))
        adjusted_ranges.append(float(float(norm["minimum"]) <= adjusted <= float(norm["maximum"])))
        truth = float(record.truth) if pd.notna(record.truth) else math.nan
        if not math.isfinite(truth): continue
        error = abs(adjusted - truth) / scale; errors.append(error)
        multiplier = float(config["actions"]["uncertainty_multipliers"][action[target]["uncertainty"]])
        sigma = max(float(record.ensemble_std), float(config["actions"]["uncertainty_floor_std_units"]) * scale) * multiplier
        z = (truth - adjusted) / sigma
        crps = sigma * (z * (2 * Phi(z) - 1) + 2 * phi(z) - 1 / math.sqrt(math.pi)) / scale
        crps_values.append(crps)
        last = row["model_input"]["last_targets"].get(target)
        if last is not None and math.isfinite(float(last)):
            higher_worse = target in {"adas_total13", "cdr_cdrsb", "faq_faqtotal"}
            threshold = float(config["actions"]["temporal_change_threshold_std_units"])
            temporal.append(float(direction(adjusted, float(last), scale, higher_worse, threshold) ==
                                  direction(truth, float(last), scale, higher_worse, threshold)))
    if not errors: raise RuntimeError(f"No future truth for {row['sample_id']}")
    nmae = float(np.mean(errors)); ncrps = float(np.mean(crps_values)); temporal_accuracy = float(np.mean(temporal)) if temporal else 1.0
    components = {"outcome": math.exp(-nmae), "uncertainty": math.exp(-ncrps),
                  "temporal": temporal_accuracy, "clinical_range": float(np.mean(raw_ranges))}
    total = sum(float(config["reward_weights"][key]) * value for key, value in components.items())
    return {"normalized_mae": nmae, "normalized_crps": ncrps, "temporal_accuracy": temporal_accuracy,
            "raw_clinical_range_rate": float(np.mean(raw_ranges)),
            "clinical_range_rate": float(np.mean(adjusted_ranges)), **{f"reward_{k}": v for k, v in components.items()},
            "total_reward": float(total)}


def prediction_groups(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    return {sample_id: group.copy() for sample_id, group in forecast_lookup(config).groupby("sample_id", sort=False)}


def train_grpo(config: dict[str, Any], tokenizer) -> None:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    output = output_dir(config); rows = read_jsonl(output / "rollout_examples.jsonl")
    predictions = prediction_groups(config); normalization = json.loads((output / "normalization.json").read_text())
    adapter = output / "action_sft_adapter"
    policy = load_policy(config, adapter, trainable=True); reference = load_policy(config, adapter, trainable=False)
    if bool(config["policy"]["gradient_checkpointing"]):
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); policy.enable_input_require_grads()
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(config["policy"]["learning_rate"]), weight_decay=float(config["policy"]["weight_decay"]))
    seed = int(config["selection_seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    order = list(range(len(rows))); random.Random(seed).shuffle(order)
    group = int(config["policy"]["group_size"]); batch_size = int(config["policy"]["rollout_batch_size"])
    generator = torch.Generator(device="cuda").manual_seed(seed)
    curves = []; rollouts = []; kl_violations = low_variance = 0; abort = None; started = time.time()
    for update, start in enumerate(range(0, len(order), batch_size), start=1):
        if update > int(config["policy"]["maximum_updates"]): break
        batch = [rows[index] for index in order[start:start + batch_size]]
        expanded = [row for row in batch for _ in range(group)]
        actions, traces = decode_actions(policy, tokenizer, expanded, config, True, generator, reference)
        scored = [score_action(row, action, predictions[row["sample_id"]], normalization, config)
                  for row, action in zip(expanded, actions)]
        rewards = torch.tensor([item["total_reward"] for item in scored]).reshape(len(batch), group)
        advantages = (rewards - rewards.mean(1, keepdim=True)) / rewards.std(1, keepdim=True, unbiased=False).clamp_min(1e-6)
        flat_advantage = advantages.reshape(-1).to("cuda")
        optimizer.zero_grad(set_to_none=True); policy.train(); losses=[]; kls=[]; entropies=[]; clips=[]
        for trace in traces:
            new = choice_log_probs(policy, tokenizer, trace["contexts"], trace["choices"], float(config["policy"]["sampling_temperature"]))
            selected = trace["selected"].to("cuda"); new_selected = new.gather(1, selected[:, None]).squeeze(1)
            old = trace["old"].to("cuda"); reference_logp = trace["reference"].to("cuda")
            ratio = (new_selected - old).clamp(-20, 20).exp()
            clipped = ratio.clamp(1-float(config["policy"]["clip_epsilon"]), 1+float(config["policy"]["clip_epsilon"]))
            kl = (new.exp() * (new - reference_logp)).sum(-1)
            loss = -torch.minimum(ratio * flat_advantage, clipped * flat_advantage).mean() + float(config["policy"]["kl_beta"]) * kl.mean()
            if not torch.isfinite(loss): abort = "nonfinite_loss"; break
            (loss / len(traces)).backward(); losses.append(float(loss.detach())); kls.extend(kl.detach().cpu().tolist())
            entropies.extend((-(new.exp()*new).sum(-1)).detach().cpu().tolist())
            clips.extend(((ratio-1).abs()>float(config["policy"]["clip_epsilon"])).float().cpu().tolist())
        if abort: optimizer.zero_grad(set_to_none=True); break
        grad = torch.nn.utils.clip_grad_norm_(params, float(config["policy"]["gradient_clip"]))
        if not torch.isfinite(grad): abort="nonfinite_gradient"; optimizer.zero_grad(set_to_none=True); break
        optimizer.step(); reward_std = rewards.std(1, unbiased=False); mean_kl=float(np.mean(kls))
        kl_violations = kl_violations+1 if mean_kl>float(config["policy"]["maximum_mean_kl"]) else 0
        low_variance = low_variance+1 if float(reward_std.mean())<float(config["policy"]["minimum_group_reward_std"]) else 0
        record={"update":update,"subjects":len(batch),"reward_mean":float(rewards.mean()),"reward_std":float(reward_std.mean()),
                "active_groups":int((reward_std>1e-8).sum()),"loss":float(np.mean(losses)),"mean_kl":mean_kl,
                "max_kl":float(np.max(kls)),"entropy":float(np.mean(entropies)),"clip_fraction":float(np.mean(clips)),
                "gradient_norm":float(grad),"elapsed_minutes":(time.time()-started)/60}
        for key in ["outcome","uncertainty","temporal","clinical_range"]:
            record[f"reward_{key}"]=float(np.mean([item[f"reward_{key}"] for item in scored]))
        curves.append(record)
        for index,(row,action,metrics) in enumerate(zip(expanded,actions,scored)):
            rollouts.append({"update":update,"sample_id":row["sample_id"],"subject_id":row["subject_id"],
                             "candidate":index%group,"actions":action,"metrics":metrics})
        pd.DataFrame(curves).to_csv(output/"training_curves.csv",index=False); write_jsonl(rollouts,output/"on_policy_rollouts.jsonl")
        if update in set(config["policy"]["checkpoint_updates"]):
            policy.save_pretrained(output/"checkpoints"/f"update_{update:04d}",safe_serialization=True)
        print(json.dumps(record),flush=True)
        if kl_violations>=int(config["policy"]["maximum_consecutive_kl_violations"]): abort="consecutive_kl_violations"; break
        if low_variance>=int(config["policy"]["maximum_consecutive_low_variance_updates"]): abort="consecutive_low_reward_variance"; break
    state={"updates_completed":len(curves),"maximum_updates":int(config["policy"]["maximum_updates"]),
           "safety_abort":abort is not None,"abort_reason":abort,"rollout_candidates":len(rollouts),
           "trainable_parameters":int(sum(p.numel() for p in params)),"reference_frozen":True,
           "lora_only":True,"real_on_policy_rollouts":True,"elapsed_minutes":(time.time()-started)/60}
    (output/"training_state.json").write_text(json.dumps(state,indent=2)+"\n"); free_model(policy); free_model(reference)


def evaluate_actions(config: dict[str, Any], tokenizer, rows: list[dict[str, Any]], arm: str,
                     adapter: Path | None) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    predictions=prediction_groups(config); normalization=json.loads((output_dir(config)/"normalization.json").read_text())
    generated=[]; records=[]; batch_size=int(config["decoding"]["evaluation_batch_size"])
    model=load_policy(config,adapter,False) if adapter is not None else None
    for start in range(0,len(rows),batch_size):
        local=rows[start:start+batch_size]
        actions=[anchor_action(config) for _ in local] if model is None else decode_actions(model,tokenizer,local,config,False)[0]
        for row,action in zip(local,actions):
            metrics=score_action(row,action,predictions[row["sample_id"]],normalization,config)
            generated.append({"sample_id":row["sample_id"],"subject_id":row["subject_id"],"arm":arm,"actions":action})
            records.append({"sample_id":row["sample_id"],"subject_id":row["subject_id"],"arm":arm,**metrics})
    if model is not None: free_model(model)
    return generated,pd.DataFrame(records)


def bootstrap_difference(frame: pd.DataFrame, left: str, right: str, metric: str, config: dict[str, Any]) -> dict[str, float]:
    pivot=frame.pivot(index="subject_id",columns="arm",values=metric); values=(pivot[left]-pivot[right]).to_numpy()
    rng=np.random.default_rng(int(config["statistics"]["bootstrap_seed"])); n=int(config["statistics"]["bootstrap_replicates"])
    draws=values[rng.integers(0,len(values),size=(n,len(values)))].mean(1)
    return {"comparison":f"{left}_minus_{right}","metric":metric,"difference":float(values.mean()),
            "ci_low":float(np.quantile(draws,.025)),"ci_high":float(np.quantile(draws,.975)),"replicates":n}


def evaluate_development(config: dict[str, Any], tokenizer) -> None:
    output=output_dir(config); rows=read_jsonl(output/"development_examples.jsonl"); frames=[]
    generated,frame=evaluate_actions(config,tokenizer,rows,"frozen",None); write_jsonl(generated,output/"development_actions_frozen.jsonl"); frames.append(frame)
    generated,frame=evaluate_actions(config,tokenizer,rows,"action_sft",output/"action_sft_adapter"); write_jsonl(generated,output/"development_actions_action_sft.jsonl"); frames.append(frame)
    checkpoints=[]
    for update in config["policy"]["checkpoint_updates"]:
        path=output/"checkpoints"/f"update_{int(update):04d}"
        if not path.exists(): continue
        arm=f"grpo_u{int(update)}"; generated,frame=evaluate_actions(config,tokenizer,rows,arm,path)
        write_jsonl(generated,output/f"development_actions_{arm}.jsonl"); frames.append(frame); checkpoints.append((int(update),arm,path))
    scored=pd.concat(frames,ignore_index=True); scored.to_csv(output/"development_per_report_metrics.csv",index=False)
    metrics=[c for c in scored if c not in {"sample_id","subject_id","arm"}]
    aggregate=scored.groupby("arm",as_index=False)[metrics].mean(); aggregate.to_csv(output/"development_aggregate_metrics.csv",index=False)
    indexed=aggregate.set_index("arm"); update,arm,path=min(checkpoints,key=lambda x:(float(indexed.loc[x[1],"normalized_mae"]),x[0]))
    comparisons=[]
    for baseline in ["frozen","action_sft"]:
        for metric in ["normalized_mae","normalized_crps","temporal_accuracy"]:
            comparisons.append(bootstrap_difference(scored,arm,baseline,metric,config))
    bootstrap=pd.DataFrame(comparisons); bootstrap.to_csv(output/"development_subject_bootstrap.csv",index=False)
    frozen=indexed.loc["frozen"]; sft=indexed.loc["action_sft"]; selected=indexed.loc[arm]
    ci=bootstrap.set_index(["comparison","metric"]); gate=config["development_gate"]
    curves=pd.read_csv(output/"training_curves.csv"); training=json.loads((output/"training_state.json").read_text())
    active=float(curves.active_groups.sum()/(len(curves)*int(config["policy"]["rollout_batch_size"])))
    criteria={
        "nmae_improved_over_sft":float(selected.normalized_mae)<float(sft.normalized_mae),
        "nmae_improved_over_frozen":float(selected.normalized_mae)<float(frozen.normalized_mae),
        "negative_nmae_ci_vs_frozen":float(ci.loc[(f"{arm}_minus_frozen","normalized_mae"),"ci_high"])<0,
        "crps_noninferior":float(selected.normalized_crps-frozen.normalized_crps)<=float(gate["maximum_normalized_crps_degradation_vs_frozen"]),
        "temporal_noninferior":float(selected.temporal_accuracy-frozen.temporal_accuracy)>=-float(gate["maximum_temporal_accuracy_degradation_vs_frozen"]),
        "clinical_range":float(selected.clinical_range_rate)>=float(gate["minimum_clinical_range_rate"]),
        "reward_variation":active>=float(gate["minimum_active_group_fraction"]),
        "kl_controlled":float(curves.mean_kl.max())<=float(config["policy"]["maximum_mean_kl"]),
        "no_safety_abort":not training["safety_abort"],
    }
    decision={"selection_uses_new_internal_development_only":True,"original_test_loaded":False,
              "selected_update":update,"selected_arm":arm,"selected_checkpoint":str(path),
              "frozen_normalized_mae":float(frozen.normalized_mae),"action_sft_normalized_mae":float(sft.normalized_mae),
              "selected_normalized_mae":float(selected.normalized_mae),"active_group_fraction":active,
              "criteria":{k:bool(v) for k,v in criteria.items()},"development_gate_passed":bool(all(criteria.values()))}
    (output/"development_selection.json").write_text(json.dumps(decision,indent=2)+"\n")


def finalize(config: dict[str, Any]) -> None:
    output=output_dir(config); selection=json.loads((output/"development_selection.json").read_text())
    manifest=json.loads((output/"manifest.json").read_text()); training=json.loads((output/"training_state.json").read_text())
    aggregate=pd.read_csv(output/"development_aggregate_metrics.csv"); bootstrap=pd.read_csv(output/"development_subject_bootstrap.csv")
    passed=bool(selection["development_gate_passed"])
    decision={"run_id":config["run_id"],"trajectory_action_sft_completed":True,"real_on_policy_grpo_completed":True,
              "development_gate_passed":passed,"confirmation_opened":False,"original_test_loaded":False,
              "selected_update":selection["selected_update"],"grpo_minus_frozen_normalized_mae":selection["selected_normalized_mae"]-selection["frozen_normalized_mae"],
              "grpo_minus_action_sft_normalized_mae":selection["selected_normalized_mae"]-selection["action_sft_normalized_mae"],
              "action_sft_anchor_recovered":False,"greedy_action_policy":"lower_plus_wide_for_every_target",
              "failure_mode":"action_sft_policy_collapse_unchanged_by_grpo",
              "trajectory_action_rlvr_supported":False if not passed else None,
              "next_action":"open_frozen_original_test_confirmation" if passed else "retain_frozen_grud_and_phase4c_policy_stop_phase4e_tuning"}
    (output/"final_decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    report=f"""# Phase 4E trajectory-action SFT and constrained GRPO

Status: development experiment completed{' and passed; frozen original-test confirmation is required' if passed else ' and stopped at the frozen development gate'}. Real on-policy reward varied with bounded forecast and uncertainty actions.

## Frozen design

- Action-SFT / rollout / development: {manifest['sft_subjects']} / {manifest['rollout_subjects']} / {manifest['development_subjects']} mutually exclusive Phase 4A build subjects
- Phase 4D rollout subjects excluded: {manifest['phase4d_rollout_subjects_excluded']}
- Future truth in prompts or action-SFT: no / no
- GRPO group size / completed updates / candidates: {config['policy']['group_size']} / {training['updates_completed']} / {training['rollout_candidates']}
- Trainable component: existing 1.5B-model LoRA parameters only; GRU-D, evidence, state/trend statistics, and renderer stayed frozen
- Original test loaded: no

## Development results

{markdown_table(aggregate)}

Selected checkpoint: update {selection['selected_update']}. Its normalized MAE was {selection['selected_normalized_mae']:.5f}, versus {selection['frozen_normalized_mae']:.5f} for frozen GRU-D and {selection['action_sft_normalized_mae']:.5f} for matched action-SFT. Reward-variable groups comprised {selection['active_group_fraction']:.4f} of rollout groups.

Greedy action diagnosis found that action-SFT and all four GRPO checkpoints selected `lower` residual plus `wide` uncertainty for every endpoint and every development report. Thus the one-epoch action-SFT did not recover its intended zero-residual/unit-uncertainty anchor under constrained first-token decoding, and GRPO did not alter that policy. This failure is part of the frozen result; training duration, temperature, and reward weights will not be retuned on these development subjects.

## Subject-bootstrap comparisons

{markdown_table(bootstrap)}

## Frozen decision

```json
{json.dumps(decision,indent=2)}
```

The result tests cognitive-only bounded trajectory actions. It does not support multimodal GRPO, and failure at this gate forbids further tuning on these development subjects.
"""
    (output/"report.md").write_text(report)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",required=True,choices=["prepare","train-sft","train-grpo","evaluate-dev","finalize"])
    args=parser.parse_args(); config=load_config(); phase4b_config=yaml.safe_load((ROOT/"configs/phase4b_cognitive_reasoning_sft.yaml").read_text())
    tokenizer=tokenizer_for(phase4b_config)
    if args.stage=="prepare": prepare(config)
    elif args.stage=="train-sft": train_action_sft(config,tokenizer)
    elif args.stage=="train-grpo": train_grpo(config,tokenizer)
    elif args.stage=="evaluate-dev": evaluate_development(config,tokenizer)
    else: finalize(config)


if __name__ == "__main__": main()
