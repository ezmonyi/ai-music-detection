"""Recover selected attribution from the public viewer, without audio or prompts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


def get(url):
    with urllib.request.urlopen(url,timeout=30) as response:return json.load(response)


def main(audit,out):
    rows=json.loads(audit.read_text())
    selected={Path(r['relative_source_path']).stem:r for r in rows if r['memberships'][0]['source']=='humair_suno'}
    assert len(selected)==100
    api='https://huggingface.co/api/datasets/humair025/suno-audio'
    revision=get(api)['sha']
    def page(offset):
        q=urllib.parse.urlencode(dict(dataset='humair025/suno-audio',config='default',split='train',offset=offset,length=100))
        return get('https://datasets-server.huggingface.co/rows?'+q)
    first=page(0);total=first['num_rows_total']
    assert 0<total<=50000
    with ThreadPoolExecutor(max_workers=3) as pool:pages=[first]+list(pool.map(page,range(100,total,100)))
    assert get(api)['sha']==revision,'Source changed while scanning'
    visible=[r['row'] for p in pages for r in p['rows']]
    assert all(p['num_rows_total']==total for p in pages)
    assert len(visible)==total and len({r['id'] for r in visible})==total
    fields=['id','title','display_name','handle','model_name','duration','created_at']
    found=[{k:r.get(k) for k in fields} for r in visible if r['id'] in selected]
    missing=sorted(set(selected)-{r['id'] for r in found})
    out.mkdir(exist_ok=False)
    raw=json.dumps(found,ensure_ascii=False,indent=2).encode();(out/'selected_metadata.json').write_bytes(raw)
    receipt=dict(status='viewer_metadata_scan_not_source_audio_verification',source_revision=revision,
        viewer_rows=total,selected_ids=100,recovered_ids=len(found),missing_ids=missing,
        metadata_sha256=hashlib.sha256(raw).hexdigest(),audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
        viewer_revision_pinning_available=False,audio_downloaded=False,audio_uploaded=False)
    (out/'COMMIT.json').write_text(json.dumps(receipt,indent=2))
    print('Viewer rows:',total,'Recovered:',len(found),'Unresolved:',len(missing))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();main(args.audit,args.out)
