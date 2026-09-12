"""Read primary Internet Archive metadata for the 44 existing SALAMI inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse,unquote,quote
import urllib.request

PIN='e16d269194b5aec1958db9267c7e5c5647f9036fd0f39c4a4adc1812e3055058'


def fetch(identity):
    url='https://archive.org/metadata/'+quote(identity,safe='')
    try:
        with urllib.request.urlopen(url,timeout=25) as response:raw=response.read()
        body=json.loads(raw);metadata=body.get('metadata',{})
        assert metadata.get('identifier')==identity
        fields=['identifier','creator','title','date','licenseurl','rights','usage','collection']
        return dict(status='retrieved',url=url,response_sha256=hashlib.sha256(raw).hexdigest(),
            metadata={k:metadata.get(k) for k in fields},files=body.get('files',[]))
    except Exception as exc:
        return dict(status='fetch_failed',url=url,error_type=type(exc).__name__)


def main(audit,out):
    assert hashlib.sha256(audit.read_bytes()).hexdigest()==PIN
    rows=[r for r in json.loads(audit.read_text()) if r['dataset']=='SALAMI'];assert len(rows)==44
    mapping={}
    for row in rows:
        url=urlparse(row['source_url']);parts=url.path.split('/')
        assert url.hostname=='archive.org' and parts[1]=='download'
        mapping[row['id']]=(unquote(parts[2]),unquote('/'.join(parts[3:])))
    fetched={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks={pool.submit(fetch,item):item for item in sorted({v[0] for v in mapping.values()})}
        for future in as_completed(tasks):
            item=tasks[future];fetched[item]=future.result()
            print(f'Reviewed item {len(fetched)}/{len(tasks)}: {fetched[item]["status"]}',flush=True)
    records=[]
    for row in rows:
        item,filename=mapping[row['id']];result=fetched[item]
        record=dict(id=row['id'],original_sha256=row['actual_sha256'],source_url=row['source_url'],
            archive_identifier=item,filename=filename,status=result['status'],audio_publication_authorized=False)
        if result['status']=='retrieved':
            matches=[f for f in result['files'] if f.get('name')==filename]
            record.update(metadata=result['metadata'],metadata_response_sha256=result['response_sha256'],
                exact_filename_matches=len(matches),source_file_metadata=[{k:f.get(k) for k in ['name','size','md5','sha1','source','original']} for f in matches],
                explicit_license_url_present=bool(result['metadata'].get('licenseurl')))
        else:record['error_type']=result['error_type']
        records.append(record)
    out.mkdir(exist_ok=False)
    raw=(json.dumps(records,ensure_ascii=False,indent=2)+'\n').encode();(out/'records.json').write_bytes(raw)
    summary=dict(records=44,distinct_archive_items=len(fetched),retrieved=sum(r['status']=='retrieved' for r in records),
        explicit_license_url_records=sum(r.get('explicit_license_url_present',False) for r in records),
        exact_file_matched_records=sum(r.get('exact_filename_matches')==1 for r in records),
        source_audit_sha256=PIN,records_sha256=hashlib.sha256(raw).hexdigest(),
        audio_uploaded=False,whole_project_complete=False)
    (out/'COMMIT.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.audit,a.out)
