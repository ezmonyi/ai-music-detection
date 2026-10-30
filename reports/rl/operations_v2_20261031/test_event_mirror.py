import base64
import hashlib
from pathlib import Path
import tempfile
import unittest
from live_event_mirror import append_verified


def entry(old,new):
    return {'path':'offline-main/tensorboard/run/events.out.tfevents.test','offset':len(old),'total_bytes':len(old+new),
        'prefix_sha256':hashlib.sha256(old).hexdigest(),'snapshot_sha256':hashlib.sha256(old+new).hexdigest(),
        'delta_base64':base64.b64encode(new).decode()}


class EventMirrorTests(unittest.TestCase):
    def test_append_preserves_one_event_file(self):
        with tempfile.TemporaryDirectory() as root:
            append_verified(root,entry(b'',b'first'))
            path=next(Path(root).rglob('events.*'));inode=path.stat().st_ino
            append_verified(root,entry(b'first',b'second'))
            self.assertEqual(path.read_bytes(),b'firstsecond');self.assertEqual(path.stat().st_ino,inode)

    def test_prefix_corruption_cannot_be_hidden(self):
        with tempfile.TemporaryDirectory() as root:
            append_verified(root,entry(b'',b'first'))
            with self.assertRaises(ValueError):append_verified(root,entry(b'other',b'second'))


if __name__=='__main__':unittest.main()
