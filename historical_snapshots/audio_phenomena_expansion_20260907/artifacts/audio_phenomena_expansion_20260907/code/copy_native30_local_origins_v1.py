"""Copy only the397 bound local stereo origins; preserve existing destination files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

REPORT_SHA='a3dcf87f703b7709b01d963e1a1c2e96f9a223d159180d1b793021337a3b160a'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def check(ok,message):
    if not ok:raise ValueError(message)


def run(report,local_root,remote_root,host,remote_python,output):
    report,local_root,output=Path(report),Path(local_root),Path(output)
    check(not output.exists() and output.parent.is_dir(),'exclusive receipt path required')
    check(sha(report)==REPORT_SHA,'source report changed')
    check(remote_root.startswith('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/') and '..' not in Path(remote_root).parts,'scoped destination required')
    code_sha=sha(__file__);rows=[]
    for r in json.loads(report.read_text())['candidates']:
        n=r['native_evidence'];p=Path(n['path'])
        if n['channels']!=2 or not str(p).startswith('/Users/'):continue
        check(r['source_group'] in {'FMA','Suno'},'unexpected local source')
        rel=p.relative_to(local_root)
        check('..' not in rel.parts and p.is_file() and not p.is_symlink(),'unsafe or nonregular native origin')
        check(sha(p)==n['sha256'],'local native hash changed:'+str(p))
        rows.append({'id':r['id'],'source_group':r['source_group'],'local_path':str(p),
                     'relative_path':str(rel),'remote_path':str(Path(remote_root)/rel),
                     'bytes':p.stat().st_size,'sha256':n['sha256']})
    check(len(rows)==397 and len({r['relative_path'] for r in rows})==397,'expected397distinctlocalorigins')
    command=['rsync','-a','--ignore-existing','--from0','--files-from=-',str(local_root)+'/',host+':'+remote_root+'/']
    subprocess.run(['ssh',host,'mkdir -p '+shlex.quote(remote_root)],check=True,timeout=30)
    print(json.dumps({'stage':'copying_bound_native_origins','files':397,'bytes':sum(r['bytes'] for r in rows)}),flush=True)
    subprocess.run(command,input=b'\0'.join(r['relative_path'].encode() for r in rows)+b'\0',check=True,timeout=1800)
    remote="""import json,hashlib,os,sys
rows=json.load(sys.stdin)
for r in rows:
 p=r['remote_path']
 if not os.path.isfile(p) or os.path.islink(p) or os.path.getsize(p)!=r['bytes']:raise ValueError('remote file properties: '+p)
 with open(p,'rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
 if d!=r['sha256']:raise ValueError('remote copy hash: '+p)
print(json.dumps({'verified':len(rows),'bytes':sum(r['bytes'] for r in rows)}))
"""
    result=subprocess.run(['ssh',host,shlex.quote(remote_python)+' -B -c '+shlex.quote(remote)],
                          input=json.dumps(rows),text=True,capture_output=True,check=True,timeout=300)
    checked=json.loads(result.stdout);check(checked['verified']==397,'remote copy count')
    check(all(sha(r['local_path'])==r['sha256'] for r in rows),'local source changed while copying')
    check(sha(report)==REPORT_SHA and sha(__file__)==code_sha,'copy inputs/code changed')
    receipt={'status':'397_native_origin_copies_byte_verified','report_sha256':REPORT_SHA,
             'code_sha256':code_sha,'host':host,'remote_python':remote_python,'remote_root':remote_root,
             'rsync_command':command,'rsync_stdin_encoding':'NUL-separated relative_path values in rows order, trailing NUL',
             'local_hashes_before_and_after_verified':True,'remote_full_hashes_verified':True,
             'remote_verification':checked,'rows':rows,'source_audio_modified':False,'classifier_fits':0}
    temp=output.with_name('.'+output.name+'.tmp')
    with temp.open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    try:os.link(temp,output)
    finally:temp.unlink()
    return {'status':receipt['status'],'receipt_sha256':sha(output),**checked}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['report','local-root','remote-root','host','remote-python','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(run(a.report,a.local_root,a.remote_root,a.host,a.remote_python,a.output)))
