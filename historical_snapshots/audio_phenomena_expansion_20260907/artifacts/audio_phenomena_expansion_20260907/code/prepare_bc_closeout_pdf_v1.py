"""Create a presentation-only TeX derivative; preserve committed report bytes."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
source_root = root / 'results/bc_reserved_verified_report_v2'
commit = json.loads((source_root / 'COMMIT.json').read_text())
for name, binding in commit['products'].items():
    data = (source_root / name).read_bytes()
    assert len(data) == binding['bytes']
    assert hashlib.sha256(data).hexdigest() == binding['sha256']
source = (source_root / 'REPORT.tex').read_text()
start = r'\subsection*{Corrected-audit provenance}'
end = r'\subsection*{Interpretation}'
assert source.count(start) == source.count(end) == 1
text = source.replace(start, '\\begingroup\\raggedright\n' + start, 1)
text = text.replace(end, '\\par\\endgroup\n' + end, 1)
target = root / 'latex/bc_reserved_verified_closeout_en.tex'
with target.open('x') as out:
    out.write(text)
print(json.dumps({'verified_products': len(commit['products']),
                  'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                  'presentation': str(target),
                  'change': 'raggedright grouping for provenance only; no result changes'}))
