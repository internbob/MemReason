"""Original-train-only measurement-identifiability and reliability audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

from ..data.common import ROOT
from .phase5c_memory_measurement import clean_matrix, fit_scaling, to_severity
from .phase5d_anchored_mirt import concordance_correlation


CONFIG = ROOT / "configs/phase5h_measurement_identifiability_audit.yaml"


def safe_correlation(function: Callable, x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) < 1e-8 or np.std(y[valid]) < 1e-8:
        return float("nan")
    result = function(x[valid], y[valid])
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def spearman_brown(correlation: float) -> float:
    if not np.isfinite(correlation) or correlation <= -1:
        return float("nan")
    return float(2 * correlation / (1 + correlation))


def cluster_bootstrap(
    frame: pd.DataFrame,
    value_a: str,
    value_b: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    local = frame[["subject_id", value_a, value_b]].dropna()
    subjects = local.subject_id.astype(str).unique()
    x = local[value_a].to_numpy(float); y = local[value_b].to_numpy(float)
    point = concordance_correlation(x, y)
    if len(subjects) < 2:
        return {"ccc": point, "ci_low": float("nan"), "ci_high": float("nan")}
    # CCC depends only on first and second moments. Pre-aggregating these by
    # participant preserves cluster resampling while avoiding repeated frame
    # concatenation for every bootstrap draw.
    sufficient = []
    for _, group in local.groupby(local.subject_id.astype(str), sort=False):
        values = group[[value_a, value_b]].to_numpy(float); gx, gy = values[:, 0], values[:, 1]
        sufficient.append([len(values), gx.sum(), gy.sum(), np.square(gx).sum(),
                           np.square(gy).sum(), (gx * gy).sum()])
    sufficient = np.asarray(sufficient, float)
    rng = np.random.default_rng(seed); draws = []
    chunk_size = 256
    for start in range(0, replicates, chunk_size):
        count = min(chunk_size, replicates - start)
        indices = rng.integers(0, len(sufficient), size=(count, len(sufficient)))
        total = sufficient[indices].sum(1)
        n, sx, sy, sxx, syy, sxy = total.T
        mean_x, mean_y = sx / n, sy / n
        var_x = sxx / n - mean_x ** 2; var_y = syy / n - mean_y ** 2
        covariance = sxy / n - mean_x * mean_y
        denominator = var_x + var_y + (mean_x - mean_y) ** 2
        draws.extend(np.divide(2 * covariance, denominator,
                               out=np.full_like(covariance, np.nan), where=denominator > 0))
    finite = np.asarray(draws, float); finite = finite[np.isfinite(finite)]
    return {
        "ccc": point,
        "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else float("nan"),
        "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else float("nan"),
    }


def instrument_scores(
    frame: pd.DataFrame,
    items: list[dict],
) -> tuple[pd.DataFrame, dict[str, list[int]], np.ndarray]:
    raw = clean_matrix(frame, items); mask = np.isfinite(raw)
    scaling = fit_scaling(raw, np.ones(len(raw), bool), items)
    severity = to_severity(raw, scaling)
    memory = np.asarray([item["domain"] == "Memory" for item in items])
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        if memory[index]:
            groups.setdefault(item["instrument"], []).append(index)
    result = frame[["subject_id", "visit_month", "visit_code"]].copy()
    for instrument, indices in groups.items():
        local = mask[:, indices]; count = local.sum(1)
        score = np.where(count > 0, np.nansum(severity[:, indices], axis=1) / np.maximum(count, 1), np.nan)
        result[instrument] = score
    return result, groups, severity


def run(config_path: Path = CONFIG) -> Path:
    config = yaml.safe_load(config_path.read_text())
    contract_path = ROOT / config["item_contract"]
    items = yaml.safe_load(contract_path.read_text())["primary_state_items"]
    frame = pd.read_csv(ROOT / "splits/train.csv", low_memory=False)
    scores, groups, severity = instrument_scores(frame, items)
    output = ROOT / "outputs" / config["run_id"]; output.mkdir(parents=True, exist_ok=True)
    repetitions = int(config["statistics"]["bootstrap_replicates"])
    base_seed = int(config["statistics"]["bootstrap_seed"])
    minimum = int(config["statistics"]["minimum_pairs"])

    reliability_rows = []
    field_names = [item["field"] for item in items]
    raw = clean_matrix(frame, items); mask = np.isfinite(raw)
    scaling = fit_scaling(raw, np.ones(len(raw), bool), items); z = to_severity(raw, scaling)
    for instrument, indices in sorted(groups.items()):
        ordered = sorted(indices, key=lambda index: hashlib.sha256(field_names[index].encode()).hexdigest())
        left, right = ordered[::2], ordered[1::2]
        internal = float("nan"); corrected = float("nan"); internal_pairs = 0
        if left and right:
            left_count, right_count = mask[:, left].sum(1), mask[:, right].sum(1)
            valid = (left_count > 0) & (right_count > 0)
            left_score = np.nansum(z[:, left], 1) / np.maximum(left_count, 1)
            right_score = np.nansum(z[:, right], 1) / np.maximum(right_count, 1)
            internal_pairs = int(valid.sum())
            internal = safe_correlation(pearsonr, left_score[valid], right_score[valid])
            corrected = spearman_brown(internal)
        temporal_pairs = []
        local = scores[["subject_id", "visit_month", instrument]].dropna().sort_values(
            ["subject_id", "visit_month"], kind="stable")
        low = float(config["temporal_reliability"]["minimum_gap_months"])
        high = float(config["temporal_reliability"]["maximum_gap_months"])
        for subject, group in local.groupby("subject_id"):
            values = group[instrument].to_numpy(float); months = group.visit_month.to_numpy(float)
            for index in range(len(group) - 1):
                gap = months[index + 1] - months[index]
                if low <= gap <= high:
                    temporal_pairs.append((str(subject), values[index], values[index + 1], gap))
        temporal = pd.DataFrame(temporal_pairs, columns=["subject_id", "first", "second", "gap"])
        temporal_r = safe_correlation(pearsonr, temporal.get("first", pd.Series(dtype=float)).to_numpy(),
                                      temporal.get("second", pd.Series(dtype=float)).to_numpy())
        reliability_rows.append({
            "instrument": instrument,
            "memory_items": len(indices),
            "split_half_pairs": internal_pairs,
            "split_half_pearson": internal,
            "spearman_brown": corrected,
            "temporal_pairs": len(temporal),
            "temporal_pearson": temporal_r,
            "preferred_reliability": corrected if np.isfinite(corrected) and corrected > 0 else temporal_r,
        })
    reliability = pd.DataFrame(reliability_rows); reliability.to_csv(output / "instrument_reliability.csv", index=False)

    pair_rows = []; instruments = sorted(groups)
    reliability_map = reliability.set_index("instrument").preferred_reliability.to_dict()
    for pair_index, left in enumerate(instruments):
        for right in instruments[pair_index + 1:]:
            local = scores[["subject_id", left, right]].dropna()
            if len(local) < minimum:
                continue
            x, y = local[left].to_numpy(float), local[right].to_numpy(float)
            bootstrap = cluster_bootstrap(local, left, right, repetitions, base_seed + pair_index * 101)
            reliability_left, reliability_right = reliability_map.get(left), reliability_map.get(right)
            ceiling = (math.sqrt(max(reliability_left, 0) * max(reliability_right, 0))
                       if np.isfinite(reliability_left) and np.isfinite(reliability_right) else float("nan"))
            pair_rows.append({
                "instrument_a": left, "instrument_b": right,
                "rows": len(local), "subjects": local.subject_id.nunique(),
                "pearson": safe_correlation(pearsonr, x, y),
                "spearman": safe_correlation(spearmanr, x, y),
                **bootstrap,
                "attenuation_ceiling": ceiling,
                "ccc_fraction_of_ceiling": bootstrap["ccc"] / ceiling if np.isfinite(ceiling) and ceiling > 0 else float("nan"),
            })
    pairs = pd.DataFrame(pair_rows); pairs.to_csv(output / "cross_instrument_pairs.csv", index=False)

    state_path = ROOT / config["phase5g_states"]
    states = pd.read_csv(state_path)
    states = states[states.arm.eq(config["phase5g_selected_arm"])].copy()
    state_reference = cluster_bootstrap(states.rename(columns={"view_a": "a", "view_b": "b"}),
                                        "a", "b", repetitions, base_seed + 9001)
    threshold = float(config["reference_threshold"])
    finite_ceilings = pairs.attenuation_ceiling.dropna() if len(pairs) else pd.Series(dtype=float)
    decision = {
        "run_id": config["run_id"],
        "descriptive_only": True,
        "original_train_only": True,
        "validation_or_test_loaded": False,
        "selected_phase5g_state_ccc": state_reference,
        "reference_threshold": threshold,
        "instrument_pairs": len(pairs),
        "pairs_with_estimable_ceiling": int(len(finite_ceilings)),
        "pairs_with_ceiling_at_least_threshold": int((finite_ceilings >= threshold).sum()),
        "threshold_empirically_universal": bool(len(finite_ceilings) > 0 and (finite_ceilings >= threshold).all()),
        "phase5f_gate_reopened": False,
    }
    (output / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    median_ceiling = float(finite_ceilings.median()) if len(finite_ceilings) else float("nan")
    report = f"""# Phase 5H: Measurement-identifiability audit

This original-train-only descriptive audit evaluated {len(reliability)} instrument-specific Memory composites and {len(pairs)} concurrent instrument pairs. Original validation and test subjects were not loaded.

The selected Phase 5G state had graph-supported CCC **{state_reference['ccc']:.4f}**, participant-bootstrap 95% CI **[{state_reference['ci_low']:.4f}, {state_reference['ci_high']:.4f}]**. Among {len(finite_ceilings)} pairs with an estimable reliability ceiling, the median attenuation ceiling was **{median_ceiling:.4f}** and {(finite_ceilings >= threshold).sum()} reached the reference {threshold:.2f} threshold.

Split-half coefficients quantify internal consistency of the available item subsets; 3--18 month coefficients mix reliability with true longitudinal change. The attenuation ceilings are therefore diagnostics rather than revised gates. Phase 5F is not reopened, and any threshold revision requires a separately frozen confirmation protocol.
"""
    (output / "report.md").write_text(report)
    manifest = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "phase5g_states_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "train_subjects": int(frame.subject_id.nunique()),
        "train_rows": len(frame),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(); print(run(args.config))


if __name__ == "__main__":
    main()
