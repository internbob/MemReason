#!/usr/bin/env python3
"""Independent structural checks for the Phase 5H audit."""

import json
from pathlib import Path

import pandas as pd

from memreason_rl.data.common import ROOT


def main() -> None:
    output = ROOT / "outputs/phase5h_measurement_identifiability_audit_20261029"
    reliability = pd.read_csv(output / "instrument_reliability.csv")
    pairs = pd.read_csv(output / "cross_instrument_pairs.csv")
    decision = json.loads((output / "decision.json").read_text())
    assert len(reliability) == 4
    assert set(reliability.instrument) == {"ADAS", "CDR", "LOGMEM", "MMSE"}
    assert len(pairs) == 6 and (pairs.rows > 0).all()
    assert pairs.ccc.between(-1, 1).all()
    assert decision["original_train_only"] and not decision["validation_or_test_loaded"]
    assert not decision["phase5f_gate_reopened"]
    verification = {"status": "passed", "instruments": len(reliability), "instrument_pairs": len(pairs),
                    "phase5f_gate_reopened": False, "validation_or_test_loaded": False}
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(output / "verification.json")


if __name__ == "__main__":
    main()
