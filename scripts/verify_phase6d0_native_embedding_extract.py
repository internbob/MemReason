#!/usr/bin/env python3
"""Independent structural verification for Phase 6D0."""

import json
import numpy as np
import pandas as pd
import yaml

from memreason_rl.data.common import ROOT


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/phase6d0_native_embedding_extract.yaml").read_text())
    output = ROOT / "outputs" / config["run_id"]
    index = pd.read_csv(output / "scan_index.csv")
    audit = pd.read_csv(output / "leakage_and_embedding_audit.csv")
    decision = json.loads((output / "development_decision.json").read_text())
    assert len(audit) == int(config["folds"])
    assert (audit.leaked_subjects == 0).all()
    assert audit.embedding_valid.all() and audit.probability_valid.all()
    for fold in range(int(config["folds"])):
        embedding = np.load(output / f"embeddings_fold{fold}_float16.npy", mmap_mode="r")
        probability = np.load(output / f"probability_fold{fold}_float32.npy", mmap_mode="r")
        assert embedding.shape == (len(index), int(config["encoding"]["embedding_dimension"]))
        assert probability.shape == (len(index),)
    assert decision["fold_aligned_embeddings_ready"]
    result = {"status": "passed", "folds": int(config["folds"]), "scans": len(index),
              "embedding_dimension": int(config["encoding"]["embedding_dimension"]),
              "leaked_subjects": 0, "validation_or_test_loaded": False}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(output / "verification.json")


if __name__ == "__main__": main()
