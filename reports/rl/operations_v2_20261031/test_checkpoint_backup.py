import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from checkpoint_backup import receive,SENDER


def frame(data,path='route_probe.bin',wrong_sha=False):
    meta={'path':path,'bytes':len(data),'sha256':'0'*64 if wrong_sha else hashlib.sha256(data).hexdigest()}
    return io.BytesIO(b'banner\nMUSIC_FILE_BEGIN:'+json.dumps(meta).encode()+b'\nMUSIC_DATA:'+base64.b64encode(data)+b'\nMUSIC_FILE_END\nMUSIC_TRANSFER_DONE:{"files_sent":1}\n')


class BinaryBackupTests(unittest.TestCase):
    def test_binary_readback_is_verified(self):
        data=bytes(range(256))*50
        with tempfile.TemporaryDirectory() as folder:
            got=receive(frame(data),folder,probe=True)
            self.assertEqual((Path(folder)/'route_probe.bin').read_bytes(),data)
            self.assertEqual(got[0]['sha256'],hashlib.sha256(data).hexdigest())

    def test_corrupt_transfer_is_not_promoted(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):receive(frame(b'payload',wrong_sha=True),folder,probe=True)
            self.assertFalse((Path(folder)/'route_probe.bin').exists())

    def test_parent_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):receive(frame(b'payload',path='../outside.pt'),folder,probe=True)

    def test_real_sender_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'source.bin';data=bytes(range(256))*300;path.write_bytes(data)
            code=SENDER+'\nsend(Path('+repr(str(path))+'),"route_probe.bin")\nprint(\'MUSIC_TRANSFER_DONE:{"files_sent":1}\',flush=True)\n'
            result=subprocess.run([sys.executable,'-c',code],capture_output=True,check=True)
            dest=Path(folder)/'received'
            receive(io.BytesIO(result.stdout),dest,probe=True)
            self.assertEqual((dest/'route_probe.bin').read_bytes(),data)


if __name__=='__main__':unittest.main()
