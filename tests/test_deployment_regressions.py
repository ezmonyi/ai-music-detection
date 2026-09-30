import json
from pathlib import Path

import pytest

from music_detector.calibration import predict_deployment
from music_detector.scoring import FAMILIES

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('receipt', ['all_artifacts_upload_result.json', 'all_artifacts_final_upload_result.json'])
def test_real_browser_all_artifacts_result_excludes_bc_from_probability(receipt):
    result = json.loads((ROOT / 'validation/ui' / receipt).read_text())
    bundle = json.loads((ROOT / 'src/music_detector/assets/deployment_models.json').read_text())
    families = [name for name in FAMILIES if name != 'BC']
    measurements = {name: result['feature_values'][name] for family in families for name in FAMILIES[family]}
    replay = predict_deployment(bundle, measurements, families)
    assert replay['probability'] == result['prediction']['probability']
    assert 'BC' not in result['prediction']['families']
    bc = next(row for row in result['features'] if row['id'] == 'BC')
    assert bc['contribution'] is None
    assert len(result['feature_values']) == 55


def test_observed_human_false_positive_is_not_hidden():
    result = json.loads((ROOT / 'validation/ui/human_upload_result.json').read_text())
    assert result['audio']['name'] == 'human_maestro_0124.flac'
    assert result['prediction']['decision'] == 'likely_ai'
    assert result['prediction']['probability'] == 0.7162271317341915
