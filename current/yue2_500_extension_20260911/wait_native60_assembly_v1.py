"""Wait on the known live producers, then assemble once; never restart producers."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')


def main():
    lock=(ROOT/'native60_assembly_waiter_v1.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dependencies=[(3088876,ROOT/'native60_neural_v1/allinone_COMMIT.json'),
                  (3090026,ROOT/'native60_sdrp_v1/COMMIT.json')]
    for pid,commit in dependencies:
        while not commit.is_file():
            try:os.kill(pid,0)
            except ProcessLookupError:
                raise RuntimeError(f'Producer {pid} stopped without {commit}; no restart attempted')
            time.sleep(30)
        print(f'Producer completion found: {commit}',flush=True)
    assert not (ROOT/'native60_feature_package_v1').exists(), 'Preserve existing package for inspection'
    subprocess.run([sys.executable,str(Path(__file__).with_name('assemble_native60_features_v1.py'))],check=True)
    print('Assembly finished; no scoring automatically authorized',flush=True)


if __name__=='__main__':main()
