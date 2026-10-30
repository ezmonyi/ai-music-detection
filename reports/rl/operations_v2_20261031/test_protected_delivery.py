"""Meaningful archive round-trip/corruption and exact allowlist regression tests."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from checkpoint_backup import receive
from protect_outputs import archive_members, verify_archive


class DeliveryTests(unittest.TestCase):
    def test_split_lossless_member_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original=root/'original.bin';original.write_bytes(bytes(range(256))*50)
            members=[{'path':'runs/x/original.bin','source_path':str(original),'bytes':original.stat().st_size,
                'sha256':hashlib.sha256(original.read_bytes()).hexdigest()}]
            parts=archive_members(members,root/'result.tar.gz',limit=101)
            self.assertGreater(len(parts),1);verify_archive(parts,members)
            changed=[{**members[0],'sha256':'0'*64}]
            with self.assertRaises(AssertionError):verify_archive(parts,changed)

    def test_expected_manifest_rejects_unlisted_or_wrong_digest(self):
        data=b'closed result';h=hashlib.sha256(data).hexdigest()
        def frame(path,digest):
            return io.BytesIO(b'MUSIC_FILE_BEGIN:'+json.dumps({'path':path,'bytes':len(data),'sha256':digest}).encode()+b'\n'
                +b'MUSIC_DATA:'+base64.b64encode(data)+b'\nMUSIC_FILE_END\nMUSIC_TRANSFER_DONE:{"files_sent":1}\n')
        with tempfile.TemporaryDirectory() as directory:
            allowed={'proof.txt':{'bytes':len(data),'sha256':h}}
            with self.assertRaises(ValueError):receive(frame('unlisted',h),directory,expected_files=allowed)
            with self.assertRaises(ValueError):receive(frame('proof.txt','0'*64),directory,expected_files=allowed)
            result=receive(frame('proof.txt',h),directory,expected_files=allowed)
            self.assertEqual(result[0]['sha256'],h)


if __name__=='__main__':unittest.main()
