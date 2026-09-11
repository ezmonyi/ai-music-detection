"""Publish exact parent-reviewed development-only execution authority."""
from pathlib import Path
import hashlib
import importlib

ROOT = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
CODE_SHA = '2c607b1175694b6966b3ce5200c6e306f20c40dd00de8f0229e6236387cc013e'
path = ROOT / 'code/run_bc_guitarset_codec_development_v1.py'
assert hashlib.sha256(path.read_bytes()).hexdigest() == CODE_SHA
m = importlib.import_module('run_bc_guitarset_codec_development_v1')
assert Path(m.__file__).resolve() == path
draft_dir = ROOT / 'preregistration/bc_guitarset_codec_development_v1_r2'
draft_sha = '2f53c482e7934d23c120ab20343f3e66f377efd3626a8466608944e4f4ad30ed'
document, commit = m.verify_draft(draft_dir, draft_sha)
assert m.binding(draft_dir / 'draft.json')['sha256'] == '24d0544582d0992441b856bc00594f52082a97fb6dfce2780e6825bca29aa8f7'
assert document['bindings']['runner']['sha256'] == CODE_SHA
assert document['bindings']['tests']['sha256'] == '2933e683107e8a364bf59fd28841dcf837ad3b305541e916d51af5c3df61cf97'
assert document['bindings']['protocol']['sha256'] == '91b3610be9b7e6f5653447488dcfed3efe91e3ab796535d5a222154fcde7fedc'
assert len(document['rows']) == 270
assert {row['split_role'] for row in document['rows']} == {'development'}
assert len({row['item_id'] for row in document['rows']}) == 90
assert not Path(document['output_root']).exists()
freeze = {'version': m.FREEZE_VERSION,
          'status': 'authorized_development_codec_measurement_not_admission',
          'draft_COMMIT': commit, 'draft': m.binding(draft_dir / 'draft.json'),
          'output_root': document['output_root'], **m.SCOPE}
target = ROOT / 'preregistration/bc_guitarset_codec_development_parent_freeze_v1.json'
m.write_json(target, freeze)
m.verify_freeze(draft_dir, draft_sha, target, m.digest(target))
print(m.canonical(m.binding(target)).decode())
