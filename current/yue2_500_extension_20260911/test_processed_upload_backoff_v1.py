"""Local, network-free regression tests for upload quota backoff."""
import unittest
from types import SimpleNamespace
from resume_processed_upload_v1 import retry_delay


def error(message, status=429, headers=None):
    e = RuntimeError(message)
    e.response = SimpleNamespace(status_code=status, headers=headers or {})
    return e


class BackoffTests(unittest.TestCase):
    def test_short_api_quota(self):
        self.assertEqual(retry_delay(error('api quota')), 360)

    def test_hourly_commit_quota(self):
        self.assertEqual(retry_delay(error('repository commits (128 per hour)')), 3900)

    def test_longer_server_delay(self):
        self.assertEqual(retry_delay(error('commits', headers={'Retry-After':'7200'})), 7200)

    def test_short_server_delay_cannot_erase_hourly_floor(self):
        self.assertEqual(retry_delay(error('commits', headers={'Retry-After':'151'})), 3900)

    def test_invalid_header_uses_floor(self):
        self.assertEqual(retry_delay(error('api quota', headers={'Retry-After':'invalid'})), 360)

    def test_other_failures_are_not_retried_as_rate_limits(self):
        self.assertIsNone(retry_delay(error('unauthorized', status=401)))
        self.assertIsNone(retry_delay(RuntimeError('local failure')))


if __name__ == '__main__': unittest.main()
