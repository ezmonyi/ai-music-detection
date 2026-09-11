#!/usr/bin/env python3
"""Wait for the known local acquisition, audit completed outputs, copy and verify."""
import os
from pathlib import Path
import subprocess
import sys
import time
import fcntl


def main():
    root=Path(__file__).resolve().parent.parent
    target=root/'external_validation/music8k_mureka500_v1'
    remote='/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/music8k_mureka500_v1'
    remote_code='/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/code/verify_music8k_remote_copy.py'
    python='/mnt/nfs-code/users/yi/dynamics_rhythm_external_benchmark_20260903/venv/bin/python'
    audit=root/'audit/music8k_mureka500_physical_v1.json'
    with (target/'finish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (target/'summary.json').is_file():
            try:os.kill(39546,0)
            except ProcessLookupError:raise RuntimeError('Acquisition terminal without completion summary')
            time.sleep(5)
        subprocess.run([sys.executable,str(root/'code/audit_music8k_mureka500.py'),'--output',str(audit)],check=True)
        subprocess.run(['ssh','-o','ConnectTimeout=10','5090-2','mkdir','-p',remote],check=True)
        subprocess.run(['rsync','-a','--exclude=.cache/','--exclude=writer.lock','--exclude=finish.lock',str(target)+'/',f'5090-2:{remote}/'],check=True)
        subprocess.run(['rsync','-a',str(audit),f'5090-2:{remote}/local_physical_audit_v1.json'],check=True)
        subprocess.run(['rsync','-a',str(root/'code/verify_music8k_remote_copy.py'),f'5090-2:{remote_code}'],check=True)
        subprocess.run(['ssh','-o','ConnectTimeout=10','5090-2',python,remote_code,'--root',remote,'--local-audit',remote+'/local_physical_audit_v1.json','--output',remote+'/remote_copy_audit_v1.json'],check=True)
        subprocess.run(['rsync','-a',f'5090-2:{remote}/remote_copy_audit_v1.json',str(root/'audit/music8k_mureka500_remote_copy_v1.json')],check=True)
        print('Completed local audit and fully rehashed NFS-data copy; no classifier admission.',flush=True)


if __name__=='__main__':main()
