"""Parent-reviewed, exact-contract CPU launch; no feature or classifier fitting."""
from pathlib import Path
import json
import materialize_native30_prior2174_v1 as m

ROOT = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
EXPECTED = '0d8d404eb16747b6abb8b72f823e50bc7c61e7491079298c238849ecc4ba9dc7'

if __name__ == '__main__':
    contract, dsp = m.prepare(
        ROOT / 'audit/native30_origin_plan_v2.json',
        ROOT / 'audit/native30_origin_plan_v2_parent_review.json',
        ROOT / 'audit/equal30_candidates_screen_v1.json',
        '/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/native30_prior2174_v1')
    m.require(m.base.value_hash(contract) == EXPECTED, 'parent-frozen contract differs')
    print(json.dumps({'event': 'parent_contract_verified', 'contract_sha256': EXPECTED,
                      'workers': 4, 'classifier_fits': 0}), flush=True)
    result = m.run_contract(contract, dsp, workers=4)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    raise SystemExit(1 if result['status'] == 'partial_no_COMMIT' else 0)
