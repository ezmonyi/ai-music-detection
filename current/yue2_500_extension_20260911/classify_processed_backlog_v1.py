"""Partition pending objects without conflating job dispatch and acceptance."""
from collections import Counter
import hashlib
import json
from pathlib import Path

R=Path('/Users/yi/Documents/report/music_ai_detection_20260912/datasets')
OUT=R/'processed_backlog_v1'


def read(path,pin):
    raw=path.read_bytes(); assert hashlib.sha256(raw).hexdigest()==pin
    return json.loads(raw)


def main():
    assert not OUT.exists()
    rows=read(R/'processed_delivery_reconciliation_v1/records.json',
              '7c6687764402f8539f77f55ece6996f86622246871256b9fcd89b48dd7fa06aa')
    fma=read(R/'fma_test_view_rights_v2/records.json',
             'e308b3bf3e2a4b82f61e3734b23b1db7c8538129841bb90a1f9d502aa6c694fd')
    fma={r['sha256']:r['status'] for r in fma}
    aime={'AudioLDM 2 Large','AudioLDM 2 Music','MusicGen Large','MusicGen Medium',
          'MusicGen Small','Mustango','Riffusion','Stable Audio v1','Stable Audio v2','Udio'}
    results=[]; counts=Counter(); sizes=Counter()
    for r in rows:
        if r['status']!='not_yet_accepted': continue
        sources=set(r['sources']); assert len(sources)==1
        s=next(iter(sources))
        if s in aime: state='dispatched_aime_not_accepted'
        elif s in {'ACE-Step','HeartMuLa'}: state='dispatched_open_models_not_accepted'
        elif s=='Suno': state='prepared_suno_not_dispatched'
        elif s=='FMA':
            state={'derivative_license_candidate_requires_notice':'dispatched_fma_not_accepted',
                   'hold_no_derivatives':'hold_fma_nd',
                   'hold_unresolved_or_mixed_license':'hold_fma_unresolved'}[fma[r['sha256']]]
        else: state='other_source_rights_or_provenance_unresolved'
        counts[state]+=1; sizes[state]+=r['bytes']
        results.append(dict(**r,next_action_category=state))
    assert len(results)==12000
    assert counts['dispatched_aime_not_accepted']==5500
    assert counts['dispatched_open_models_not_accepted']==800
    assert counts['dispatched_fma_not_accepted']==316
    assert counts['prepared_suno_not_dispatched']==900
    OUT.mkdir()
    raw=(json.dumps(results,indent=2)+'\n').encode()
    (OUT/'records.json').write_bytes(raw)
    summary=dict(objects=12000,counts=dict(counts),bytes_by_category=dict(sizes),
        records_sha256=hashlib.sha256(raw).hexdigest(),
        note='Dispatch means a publisher was started, not live transfer or completed publication; scope excludes unenumerated stems',
        whole_project_complete=False)
    (OUT/'COMMIT.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
