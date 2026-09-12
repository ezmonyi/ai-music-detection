"""Check configured client invariants without making network requests."""
import unittest
from bounded_hf_http_v1 import factory


class ClientTests(unittest.TestCase):
    def test_timeouts_and_hooks(self):
        with factory() as client:
            self.assertEqual(client.timeout.connect,20)
            self.assertEqual(client.timeout.read,90)
            self.assertEqual(client.timeout.write,90)
            self.assertEqual(client.timeout.pool,60)
            self.assertTrue(client.follow_redirects)
            self.assertTrue(client.event_hooks['request'])


if __name__=='__main__':unittest.main()
