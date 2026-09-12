"""Prepare source-specific notices for the two audited historical Suno collections."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main(audit,out):
    raw=(audit/'records.json').read_bytes()
    commit=json.loads((audit/'COMMIT.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==commit['records_sha256']
    rows=json.loads(raw)
    assert len(rows)==len({r['id'] for r in rows})==500
    assert Counter(r['source_collection'] for r in rows)=={'humair_suno':250,'suno_unknown':250}
    sources={
        'humair_suno':dict(dataset='humair025/suno-audio',license='MIT',
            metadata_revision='344c67dd2992063779b8f40504ff112f6083e7f2',
            attribution='humair025 / Humair332 (dataset card citation); nyuuzyou/suno (upstream); recorded track creator'),
        'suno_unknown':dict(dataset='Kukedlc/suno-ai-music-dataset',license='CC-BY-4.0',
            metadata_revision='bdff424e70c10ef62dca13ba43659ad9e7e1fcdb',
            attribution='Kukedlc, Suno AI Music Dataset publisher')}
    manifest=[]
    for r in rows:
        assert r['status']=='verified' and r['actual_original_sha256']==r['expected_original_sha256']
        s=sources[r['source_collection']]
        manifest.append(dict(id=r['id'],source_collection=r['source_collection'],
            path='audio/historical_suno_originals_v1/'+r['source_collection']+'/'+r['id']+'.mp3',
            source_relative_path=r['source_relative_path'],bytes=r['original_bytes'],sha256=r['actual_original_sha256'],
            source_dataset=s['dataset'],source_card_revision=s['metadata_revision'],
            publisher_declared_license=s['license'],attribution=s['attribution'],
            creator_recorded=r['creator'],model_name_historical=r['model_name'],
            historical_role=r['historical_role'],group_id=r['historical_group_id'],
            source_duration_recorded_s=r['source_duration_recorded_s'],
            historical_crop_start_s=r['historical_crop_start_s'],
            modification='None; recovered original MP3 bytes, not a cropped analysis view'))
    out.mkdir(exist_ok=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'README.md').write_text('''# Historical Suno originals: two separate collections

500 original MP3 recordings, 250 from each source below. These are previously
used experimental identities, not a new untouched test set. Original bytes are
unchanged; duration and historical crop offsets are preserved. No standardized
views, separated stems, prompt text, lyrics or artwork are included here.

## humair_suno — 250 recordings

Source: https://huggingface.co/datasets/humair025/suno-audio
Card revision: 344c67dd2992063779b8f40504ff112f6083e7f2.
The card declares MIT and credits Humair332; repository owner is humair025.
Upstream credited by the card: https://huggingface.co/datasets/nyuuzyou/suno .
Recorded individual creator handles are retained in the manifest.
MIT reference: https://opensource.org/license/mit . Preserve upstream notices.
No independent third-party recording-rights clearance is asserted by this plan.

## suno_unknown — 250 recordings

Source: https://huggingface.co/datasets/Kukedlc/suno-ai-music-dataset
Revision: bdff424e70c10ef62dca13ba43659ad9e7e1fcdb. Attribution: Kukedlc.
Publisher declares CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ .
All 250 original hashes matched the pinned upstream audio. Historical model
labels remain unknown; a separate post-experiment provenance supplement records
six row-level model labels. Do not substitute the card's general V5.5 prose for
those individual labels or silently rewrite the frozen experimental grouping.

## Publication boundary

Each source's declared terms are separate, not a blanket license for the whole
project. No endorsement, exclusive ownership, or independent clearance of the
generation service and all possible third-party rights is claimed. This is a
verified-byte publication plan, not proof of successful public upload. Keep the
source cards and any supplied notices alongside the eventual release. Upload
requires a terminal receipt and independent remote verification before claiming
completion. Current card revisions document consulted provenance; they are not
asserted to be the original acquisition revision for the Humair audio.
''')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'COMMIT.json').write_text(json.dumps(dict(status='500_originals_publication_plan_not_uploaded',
        files=500,bytes=sum(r['bytes'] for r in manifest),source_audit_commit_sha256=sha(audit/'COMMIT.json'),
        manifest_sha256=sha(out/'manifest.json'),whole_project_complete=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.audit,a.out)
