#!/usr/bin/env python3
"""Read-only bounded decoder diagnosis of the sole Mureka500 frame discrepancy."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import signal
import subprocess
import tempfile

import numpy as np
import soundfile as sf

SOURCE = Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/music8k_mureka500_v1')
EXPECTED_SHA = 'a0094d936053b54df7e73c07b1ab5bcd07facfc963123f63dada262c8d41c050'
REMUX_SHA = '29cb2baed4b3bcde20162bea6128f4a36a57a8e10897886f1b0b1a7eb629cbfc'
WINDOWS_SHA = 'a320afd280c9b2dbfbb2dfd1035b9cacc65c19a3680b345ae7d715af7fc80ed5'
CROP = 2646000
BLOCK = 65536


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()


def pcm_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a,dtype='<f4').tobytes()).hexdigest()


def stats(a):
    nonzero = np.flatnonzero(np.any(a != 0,axis=1)) if len(a) else np.array([],dtype=int)
    return dict(frames=len(a), finite=bool(np.isfinite(a).all()), sha256=pcm_sha(a),
                nonzero_frames=len(nonzero), first_nonzero_frame=int(nonzero[0]) if len(nonzero) else None,
                last_nonzero_frame=int(nonzero[-1]) if len(nonzero) else None,
                peak=float(np.max(np.abs(a))) if len(a) else None,
                rms=float(np.sqrt(np.mean(a.astype(np.float64)**2))) if len(a) else None)


def compare(a,b):
    count=min(len(a),len(b))
    changed=np.any(a[:count]!=b[:count],axis=1)
    indices=np.flatnonzero(changed)
    return dict(a_frames=len(a),b_frames=len(b),exact_equal=bool(np.array_equal(a,b)),
                compared_frames=count,changed_frames=int(changed.sum()),
                first_changed_frame=int(indices[0]) if len(indices) else None,
                max_abs_error=float(np.max(np.abs(a[:count].astype(np.float64)-b[:count]))) if count else None)


def sf_decode(path, use_blocks=False):
    chunks, blocks, total = [], [], 0
    with sf.SoundFile(path) as f:
        if use_blocks:
            iterator=f.blocks(blocksize=BLOCK,dtype='float32',always_2d=True)
        else:
            def reads():
                while True:
                    data=f.read(BLOCK,dtype='float32',always_2d=True)
                    if not len(data):break
                    yield data
            iterator=reads()
        for data in iterator:
            if not len(data):
                break
            # Copy: soundfile block buffers may be reused by the generator.
            chunks.append(data.copy())
            blocks.append(dict(start=total,frames=len(data),sha256=pcm_sha(data),
                               nonzero_samples=int(np.count_nonzero(data)),tell=int(f.tell())))
            total+=len(data)
            if total>20000000:
                raise RuntimeError('Bounded one-file frame limit exceeded')
    return np.concatenate(chunks),blocks


def ffmpeg_decode(binary,path):
    cmd=[str(binary),'-v','error','-xerror','-i',str(path),'-map','0:a:0','-c:a','pcm_f32le','-f','f32le','-']
    chunks=[]
    with tempfile.TemporaryFile() as errors:
        with subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=errors) as proc:
            try:
                total=0
                while True:
                    chunk=proc.stdout.read(1<<20)
                    if not chunk:break
                    chunks.append(chunk); total+=len(chunk)
                    if total>20000000*8:
                        raise RuntimeError('FFmpeg frame limit exceeded')
                rc=proc.wait(timeout=120)
            except BaseException:
                proc.kill();proc.wait();raise
        errors.seek(0);stderr=errors.read().decode(errors='replace')
    data=b''.join(chunks)
    if len(data)%8:raise RuntimeError('Partial stereo float frame')
    return np.frombuffer(data,dtype='<f4').reshape(-1,2),dict(command=cmd,exit_code=rc,stderr=stderr,
        executable_sha256=sha(binary),version=subprocess.check_output([str(binary),'-version'],text=True).splitlines()[0])


def window_comparison(sequential, path, acquisition_frames):
    assert sha(path)==WINDOWS_SHA
    rows=[]
    with np.load(path,allow_pickle=False) as windows:
        assert set(windows.files)=={'first','center','tail_sf','tail_ff'}
        for name,start in [('first',0),('center',(acquisition_frames-CROP)//2),('tail_sf',len(sequential)-8192)]:
            reference=windows[name]
            assert reference.shape==(8192,2) and np.isfinite(reference).all()
            same=sequential[start:start+len(reference)]
            error=same.astype(np.float64)-reference
            # Search only the supplied window's immediate +/-2304-frame neighborhood.
            low=max(0,start-2304); high=min(len(sequential)-len(reference),start+2304)
            region=sequential[low:high+len(reference)].astype(np.float64)
            ref=reference.astype(np.float64)
            cross=sum(np.correlate(region[:,ch],ref[:,ch],mode='valid') for ch in range(2))
            squared=np.concatenate(([0.0],np.cumsum(np.sum(region**2,axis=1))))
            sse=squared[len(ref):]-squared[:-len(ref)]+np.sum(ref**2)-2*cross
            best=low+int(np.argmin(sse))
            rows.append(dict(name=name,start=start,reference_sha256=pcm_sha(reference),
                same_position=compare(same,reference),rms_error=float(np.sqrt(np.mean(error**2))),
                samples_error_above_1e_6=int(np.count_nonzero(np.abs(error)>1e-6)),
                samples_error_above_1e_5=int(np.count_nonzero(np.abs(error)>1e-5)),
                searched_start_frame_min=low,searched_start_frame_max=high,best_start_frame=best,
                best_offset_frames=best-start,best_alignment=compare(sequential[best:best+len(ref)],reference)))
        tail=windows['tail_ff']
        assert tail.shape==(1152,2)
        tail_stats=stats(tail)
    assert sha(path)==WINDOWS_SHA
    return dict(path=str(path),sha256=WINDOWS_SHA,origin='parent_local_ffmpeg_full_decode',
        full_decode_sha256='bc4831bf24ff69396e292fcb736f05c55b255a6b9e0a80c2f0a0b4c69b62f926',
        windows=rows,ffmpeg_final1152=tail_stats)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ffmpeg',type=Path)
    p.add_argument('--diagnostic-remux',type=Path)
    p.add_argument('--ffmpeg-windows',type=Path)
    args=p.parse_args()
    signal.signal(signal.SIGALRM,lambda *unused:(_ for _ in ()).throw(TimeoutError('180s diagnosis bound')))
    signal.alarm(180)
    original=SOURCE/'raw/mureka_v9/97136.mp3'; receipt_path=SOURCE/'items/97136.json'
    receipt=json.loads(receipt_path.read_text())
    receipt_sha=sha(receipt_path)
    assert sha(original)==EXPECTED_SHA==receipt['sha256']
    path=args.diagnostic_remux or original
    decoded_sha=REMUX_SHA if args.diagnostic_remux else EXPECTED_SHA
    assert sha(path)==decoded_sha and path.stat().st_size==3352246
    info=sf.info(path)
    sequential,blocks=sf_decode(path)
    generator,generator_blocks=sf_decode(path,True)
    native_frames=receipt['decoded_frames']
    block_counts=Counter(b['sha256'] for b in generator_blocks if b['frames']==BLOCK)
    repeats=[dict(sha256=digest,starts=[b['start'] for b in generator_blocks if b['sha256']==digest])
             for digest,count in block_counts.items() if count>1]
    result=dict(status='read_only_diagnosis',source_path=str(path),source_sha256=decoded_sha,
        original_path=str(original),original_sha256=EXPECTED_SHA,diagnostic_remux_only=bool(args.diagnostic_remux),
        source_bytes=path.stat().st_size,receipt_sha256=receipt_sha,code_sha256=sha(__file__),
        python=platform.python_version(),numpy=np.__version__,soundfile=sf.__version__,libsndfile=sf.__libsndfile_version__,
        acquisition_frames=native_frames,acquisition_pcm_sha256=receipt['decoded_native_float32_sha256'],
        sf_header=dict(frames=info.frames,samplerate=info.samplerate,channels=info.channels,format=info.format,subtype=info.subtype),
        sf_sequential=stats(sequential),sf_blocks=stats(generator),
        sf_read_vs_blocks=compare(sequential,generator),sf_sequential_blocks=blocks,
        sf_generator_blocks=generator_blocks,
        repeated_full_block_hashes={k:v for k,v in block_counts.items() if v>1},
        repeated_generator_blocks=repeats,
        sf_prefix_at_acquisition_length=stats(sequential[:native_frames]),
        sf_tail_after_acquisition=stats(sequential[native_frames:]),
        generator_tail_after_sequential_eof=stats(generator[len(sequential):]),
        generator_tail_after_acquisition=stats(generator[native_frames:]),seek_tests=[],changed_admission=False)
    for name,start in [('acquisition_center',(native_frames-CROP)//2),('sf_header_center',(info.frames-CROP)//2),
                       ('acquisition_end_minus_1s',native_frames-44100),('after_acquisition_end',native_frames)]:
        expected=sequential[start:start+CROP]
        for prime in (False,True):
            with sf.SoundFile(path) as f:
                if prime:
                    while len(f.read(BLOCK,dtype='float32',always_2d=True)):pass
                seek_return=f.seek(start)
                actual=f.read(CROP,dtype='float32',always_2d=True)
                result['seek_tests'].append(dict(name=name,start=start,requested_frames=CROP,
                    primed_by_full_scan=prime,seek_return=seek_return,actual=stats(actual),
                    versus_sequential=compare(actual,expected)))
    if args.ffmpeg:
        ff,ffmeta=ffmpeg_decode(args.ffmpeg,path)
        result['ffmpeg']=dict(ffmeta,**stats(ff),frames_match_acquisition=len(ff)==native_frames,
                             pcm_hash_matches_acquisition=pcm_sha(ff)==receipt['decoded_native_float32_sha256'],
                             sf_prefix_comparison=compare(ff,sequential[:len(ff)]))
        start=(native_frames-CROP)//2
        result['center_ffmpeg_vs_sf_sequential']=compare(ff[start:start+CROP],sequential[start:start+CROP])
    else:
        result['ffmpeg']=dict(status='not_available_no_binary_supplied')
    if args.ffmpeg_windows:
        result['parent_ffmpeg_window_comparison']=window_comparison(sequential,args.ffmpeg_windows,native_frames)
    assert sha(path)==decoded_sha and sha(original)==EXPECTED_SHA and sha(receipt_path)==receipt_sha
    result['source_and_receipt_unchanged']=True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sf_sequential_blocks','sf_generator_blocks','repeated_full_block_hashes','repeated_generator_blocks','seek_tests')}))


if __name__=='__main__':main()
