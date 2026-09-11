"""Fail-closed runtime and raw CSV boundary tests; no real data or fitting."""
import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import prepare_evaluation_inputs_v6 as P


class PreparationBoundaryTests(unittest.TestCase):
    def test_no_removable_assertions(self):
        self.assertFalse(any(isinstance(n, ast.Assert) for n in ast.walk(ast.parse(Path(P.__file__).read_text()))))

    def test_optimized_runtime_rejected(self):
        r=subprocess.run([sys.executable,'-O','-c',
            'import prepare_evaluation_inputs_v6 as p; p.checked_runtime()'],
            cwd=Path(P.__file__).parent,capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)
        self.assertIn('Optimized Python is not authorized',r.stderr)

    def test_require_not_assert(self):
        with self.assertRaises(ValueError): P.require(False,'must reject')

    def test_synthetic_cannot_enter_real_validator(self):
        with self.assertRaisesRegex(ValueError,'synthetic'):
            P.validate_package(Path('/nonexistent'),True)

    def test_csv_preserves_tokens_and_missing(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'data.csv'
            P.P5.write_csv(p,[dict(id='a',value='1.00000',missing='')],['id','value','missing'])
            self.assertEqual(P.rows(p),(['id','value','missing'],[dict(id='a',value='1.00000',missing='')]))


if __name__=='__main__': unittest.main()
