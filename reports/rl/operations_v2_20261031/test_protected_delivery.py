"""Meaningful archive round-trip/corruption and exact allowlist regression tests."""
import base64
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from checkpoint_backup import receive
from protect_outputs import archive_members, verify_archive
import protect_outputs
import closeout_supervisor


class DeliveryTests(unittest.TestCase):
    def test_failed_run_is_completely_archived_without_claiming_test_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);root=folder/'scratch';work=folder/'persistent'
            root.mkdir();work.mkdir()
            (work/'PIPELINE_EXIT.json').write_text(json.dumps({'node_id':2128889,'formal_optimizer_updates':0,'error':'fixture_only'}))
            (work/'ACTIVE_STAGE_PID.json').write_text(json.dumps({'child_pid':99999999}))
            checkpoint=root/'runs/offline-main/checkpoints/failure.pt';checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'unit_fixture_not_a_real_checkpoint')
            loose=root/'runs/offline-main.NUMERICAL_PROTOCOL.json';loose.write_text('{}')
            with patch.object(protect_outputs,'ROOT',root),patch.object(protect_outputs,'WORK',work),contextlib.redirect_stdout(io.StringIO()):
                protect_outputs.main()
            result=json.loads((work/'closeout/DELIVERY_MANIFEST.json').read_text())
            self.assertFalse(result['scientific_success']);self.assertEqual(result['requested_downloads'],[])
            self.assertEqual(result['actual_additional_optimizer_updates'],0)
            self.assertEqual(len(result['checkpoints']),1)
            self.assertTrue(result['latest_persistent_checkpoint']['persistent_copy_sha256_verified'])
            group=[m for m in result['all_output_members'] if m['path']=='runs/offline-main.NUMERICAL_PROTOCOL.json']
            self.assertEqual(len(group),1)
            archive=[a for a in result['archives'] if a['archive_group']=='run-boundaries']
            verify_archive([a['source_path'] for a in archive],group)

    def test_corrupt_local_backup_blocks_any_provider_or_release_action(self):
        with tempfile.TemporaryDirectory() as directory:
            packet=Path(directory)
            artifact=packet/'missing.pt'
            receipt={'all_member_sha256_verified':True,'members':[{'local_path':str(artifact),'bytes':5,'sha256':'0'*64}]}
            (packet/'ALL_CHECKPOINTS_VERIFICATION.json').write_text(json.dumps(receipt))
            with patch.object(closeout_supervisor,'PACKET',packet),patch.object(closeout_supervisor,'acknowledge_delivery') as ack,patch.object(closeout_supervisor,'provider_proof') as provider:
                with self.assertRaises(AssertionError):closeout_supervisor.release_after_delivery({'file_verification_receipts':['ALL_CHECKPOINTS_VERIFICATION.json']})
                ack.assert_not_called();provider.assert_not_called()

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
