"""Offline tests of the exact nested retry function, without forking or sleeping."""
import ast
from pathlib import Path
import types
import unittest
from unittest.mock import Mock


class HTTPFailure(Exception):
    def __init__(self, code, retry=''):
        self.response=types.SimpleNamespace(status_code=code,headers={'Retry-After':retry})


def load_retry(original):
    source=Path(__file__).with_name('resume_audio_publication_rate_limited_v1.py')
    tree=ast.parse(source.read_text())
    function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='create_commit')
    sleeper=Mock()
    namespace=dict(original=original,time=types.SimpleNamespace(sleep=sleeper),print=Mock())
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['create_commit'],sleeper


class BackoffTests(unittest.TestCase):
    def test_success_is_returned_without_wait(self):
        result=object();original=Mock(return_value=result)
        retry,sleep=load_retry(original)
        self.assertIs(retry('api',operations=['one']),result)
        sleep.assert_not_called()

    def test_429_replays_same_request_after_wait(self):
        original=Mock(side_effect=[HTTPFailure(429),'committed'])
        retry,sleep=load_retry(original)
        operations=[object()]
        self.assertEqual(retry('api',operations=operations),'committed')
        self.assertEqual(original.call_count,2)
        self.assertEqual(original.call_args_list[0],original.call_args_list[1])
        sleep.assert_called_once_with(3900)

    def test_larger_server_delay_is_respected(self):
        original=Mock(side_effect=[HTTPFailure(429,'7200'),'committed'])
        retry,sleep=load_retry(original);retry('api')
        sleep.assert_called_once_with(7200)

    def test_repeated_429_waits_each_time(self):
        original=Mock(side_effect=[HTTPFailure(429),HTTPFailure(429),'committed'])
        retry,sleep=load_retry(original);retry('api')
        self.assertEqual(sleep.call_count,2)

    def test_other_http_errors_propagate(self):
        for code in (401,403,404,409,500):
            with self.subTest(code=code):
                failure=HTTPFailure(code);original=Mock(side_effect=failure)
                retry,sleep=load_retry(original)
                with self.assertRaises(HTTPFailure) as caught:retry('api')
                self.assertIs(caught.exception,failure)
                sleep.assert_not_called();self.assertEqual(original.call_count,1)

    def test_non_http_errors_propagate(self):
        failure=ValueError('invalid manifest');original=Mock(side_effect=failure)
        retry,sleep=load_retry(original)
        with self.assertRaises(ValueError):retry('api')
        sleep.assert_not_called()


if __name__=='__main__':unittest.main()
