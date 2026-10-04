"""Snapshot build provenance after successful checks; no changes to input data."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memreason_rl.data.common import ROOT,read_config,save_json,sha256

cfg=read_config()
inputs=[ROOT/cfg['cognition_csv'],Path(cfg['mri']['session_metadata']),Path(cfg['item_provenance_csv']),ROOT/'configs/data.yaml',ROOT/'configs/cognitive_item_dictionary.yaml']
inputs.extend(sorted(Path(cfg['clinical_source_dir']).glob('*_*.csv')))
provenance=json.loads((ROOT/'reports/data_audit/provenance.json').read_text())
provenance['inputs']={str(p):sha256(p) for p in inputs if p.is_file()}
save_json(provenance,ROOT/'reports/data_audit/provenance.json')
test_log=(ROOT/'reports_tests.log').read_text()
if 'failed' in test_log or 'passed' not in test_log:
    raise RuntimeError('Test run did not pass')
status_paths=[ROOT/'outputs/stage1_smoke/status.json',ROOT/'reports/experiments/integration_smoke.json',ROOT/'reports/experiments/locf_val/status.json',ROOT/'reports/data_audit/protocol_leakage.json']
results={str(p.relative_to(ROOT)):json.loads(p.read_text()) for p in status_paths}
source_files=sorted((ROOT/'memreason_rl').rglob('*.py'))+sorted((ROOT/'configs').glob('*.yaml'))+sorted((ROOT/'tests').glob('*.py'))
save_json(dict(test_summary=test_log.strip().splitlines()[-1],input_source_sha256=sha256(ROOT/cfg['cognition_csv']),
    code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in source_files},checks=results,
    data_construction_complete=True,scientific_training_complete=False),ROOT/'reports/build_manifest.json')
print(test_log.strip().splitlines()[-1])
