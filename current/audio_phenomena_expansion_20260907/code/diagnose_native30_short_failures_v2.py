"""Recheck terminal short-input failures without producer imports or padding.

Primary duration is measured under the pinned libsndfile decoder. FFmpeg is a
separate diagnostic, not a source-specific replacement decoder.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid
import numpy as np
import soundfile as sf

CONTRACT='7cf9b3d2436958c8e7d0d4e5121ed317b04a1126abaafe6ba97022e88dc82e4a'
RUN='9746294937094572a41241fe33d48ae7'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def signature(path):
    s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def vh(value):
    return hashlib.sha256((json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()).hexdigest()


def sequential(path,rate,channels,block):
    count=0;h=hashlib.sha256()
    with sf.SoundFile(path) as f:
        if f.samplerate!=rate or f.channels!=channels:raise ValueError('native format disagreement')
        header=int(f.frames)
        while True:
            x=f.read(block,dtype='float64',always_2d=True)
            if not len(x):break
            if not np.isfinite(x).all():raise ValueError('nonfinite native audio')
            count+=len(x);h.update(np.ascontiguousarray(x,dtype='<f8').tobytes())
    return {'frames':count,'header_frames':header,'pcm_sha256':h.hexdigest(),'empty_eof_observed':True}


def ffmpeg_observation(path,rate,channels):
    ffmpeg,ffprobe=shutil.which('ffmpeg'),shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:return {'status':'unavailable'}
    p=subprocess.run([ffprobe,'-v','error','-select_streams','a:0','-show_entries','stream=sample_rate,channels','-of','json',str(path)],capture_output=True,check=True,timeout=60)
    streams=json.loads(p.stdout)['streams']
    if len(streams)!=1 or int(streams[0]['sample_rate'])!=rate or streams[0]['channels']!=channels:raise ValueError('ffprobe format disagreement')
    p=subprocess.run([ffmpeg,'-v','error','-nostdin','-i',str(path),'-map','0:a:0','-f','f64le','-acodec','pcm_f64le','-'],capture_output=True,check=True,timeout=120)
    if len(p.stdout)%(8*channels):raise ValueError('partial FFmpeg PCM frame')
    values=np.frombuffer(p.stdout,dtype='<f8')
    if not np.isfinite(values).all():raise ValueError('FFmpeg nonfinite audio')
    return {'status':'decoded_without_resampling_or_channel_conversion','frames':len(p.stdout)//(8*channels),
            'pcm_sha256':hashlib.sha256(p.stdout).hexdigest(),'stderr':p.stderr.decode(errors='replace'),
            'executable':ffmpeg,'executable_sha256':sha(ffmpeg)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    contract_path=a.root/'contract.json';run_path=a.root/'runs'/(RUN+'.json')
    assert sha(contract_path)==CONTRACT
    c=json.loads(contract_path.read_text());run=json.loads(run_path.read_text())
    assert run['status']=='partial_no_COMMIT' and run['expected']==1695 and run['completed']==1656 and run['failed']==39
    assert run['contract_sha256']==CONTRACT and not (a.root/'COMMIT.json').exists()
    assert sf.__version__==c['runtime']['soundfile'] and sf.__libsndfile_version__==c['runtime']['libsndfile'] and np.__version__==c['runtime']['numpy']
    rows={r['id']:r for r in c['rows']};records=[]
    bindings={str(contract_path):sha(contract_path),str(run_path):sha(run_path),str(Path(__file__).resolve()):sha(__file__)}
    for value in [c['runtime']['libsndfile_binary'],*c['runtime']['module_files'].values()]:
        assert sha(value['path'])==value['sha256'];bindings[value['path']]=value['sha256']
    for failure in run['failures']:
        r=rows[failure['id']];native=r['native_evidence'];path=Path(r['execution_native_path'])
        fp=a.root/'failures'/(r['id']+'.'+RUN+'.json')
        assert json.loads(fp.read_text())==failure and failure['row_sha256']==vh(r)
        assert failure['exception']=='ValueError' and failure['message']=='native region shorter than30s: padding forbidden'
        assert not (a.root/'items'/(r['id']+'.json')).exists() and not (a.root/'audio'/(r['id']+'.wav')).exists()
        before=signature(path);assert sha(path)==native['sha256']
        rate=native['sample_rate_hz'];channels=native['channels'];assert channels==2
        first=sequential(path,rate,channels,32771);second=sequential(path,rate,channels,65536)
        assert first['frames']==second['frames'] and first['header_frames']==second['header_frames'] and 0<=first['frames']<30*rate
        other=ffmpeg_observation(path,rate,channels)
        assert sha(path)==native['sha256'] and signature(path)==before
        record={k:r[k] for k in ['id','source_group','label','group_id','component_id','role']}
        record.update(native_path=str(path),native_sha256=native['sha256'],native_rate_hz=rate,native_channels=channels,
                      actual_frames=first['frames'],primary_second_pass_frames=second['frames'],primary_pcm_sha256=first['pcm_sha256'],
                      primary_second_pcm_sha256=second['pcm_sha256'],different_block_pcm_bit_exact=first['pcm_sha256']==second['pcm_sha256'],
                      header_frames=first['header_frames'],required_frames=30*rate,deficit_frames=30*rate-first['frames'],
                      deficit_seconds=(30*rate-first['frames'])/rate,empty_eof_observed=True,
                      diagnosis_method='two_fresh_libsndfile_sequential_passes_different_blocks_to_empty_EOF',
                      producer_failure={'path':str(fp),'sha256':sha(fp)},source_binding={'path':str(path),'sha256':native['sha256']},
                      source_byte_count=before[2],source_stat_signature=before,ffmpeg_diagnostic=other)
        records.append(record);bindings[str(fp)]=sha(fp)
        print(json.dumps({'id':r['id'],'deficit_frames':record['deficit_frames'],'ffmpeg_frames':other.get('frames')}),flush=True)
    assert len(records)==len({r['id'] for r in records})==39
    for path,digest in bindings.items():assert sha(path)==digest
    for r in records:assert sha(r['native_path'])==r['native_sha256'] and signature(r['native_path'])==r['source_stat_signature']
    out={'status':'confirmed_39_native_short_inputs_source_blind_policy','contract_sha256':CONTRACT,'run_summary_sha256':sha(run_path),
         'policy':{'eligibility':'sequential_actual_frames >= 30*native_rate_hz','source_or_label_used':False,'padding':False,'replacement':False,'relabel':False},
         'decoder_scope':'pinned_libsndfile_sequential_EOF_not_universal_audio_duration','runtime':{'soundfile':sf.__version__,'libsndfile':sf.__libsndfile_version__,'numpy':np.__version__},
         'records':records,'input_bindings':bindings,'classifier_fits':0,'cohort_admitted':False,'feature_extraction_authorized':False}
    temp=a.output.with_name('.'+a.output.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('xb') as f:f.write((json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());f.flush();os.fsync(f.fileno())
        os.link(temp,a.output)
        fd=os.open(a.output.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temp.unlink(missing_ok=True)
    print(json.dumps({'status':out['status'],'records':39,'output_sha256':sha(a.output),'min_deficit_seconds':min(r['deficit_seconds'] for r in records),'max_deficit_seconds':max(r['deficit_seconds'] for r in records),'ffmpeg_agrees_primary_frames':sum(r['ffmpeg_diagnostic'].get('frames')==r['actual_frames'] for r in records)}),flush=True)


if __name__=='__main__':main()
