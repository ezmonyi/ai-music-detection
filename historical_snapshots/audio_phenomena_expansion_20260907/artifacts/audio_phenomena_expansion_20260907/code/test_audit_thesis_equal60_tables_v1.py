import csv
from pathlib import Path
import unittest
import audit_thesis_equal60_tables_v1 as m

ROOT = Path(__file__).resolve().parent.parent


class TableChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (ROOT / 'latex/final_thesis_completed_transfer_results_en.tex').read_text()
        with (ROOT / 'results/presentation_equal60_v6_v1/primary_macro_overview_255.csv').open(newline='') as stream:
            cls.rows = list(csv.DictReader(stream))

    def test_actual_tables(self):
        result = m.check(self.text, self.rows)
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(result['numeric_fields_checked'], 76)

    def test_changed_display_value(self):
        changed = self.text.replace('0.7882 / 72.07', '0.7882 / 99.99')
        self.assertNotEqual(changed, self.text)
        self.assertEqual(m.check(changed, self.rows)['status'], 'fail')

    def test_missing_target_endpoint(self):
        rows = [r for r in self.rows if not (r['combination'] == 'S' and
                 r['fold_type'] == m.ARMS[0] and r['quantity'] == 'all')]
        with self.assertRaises(KeyError):
            m.check(self.text, rows)

    def test_duplicate_csv_row(self):
        with self.assertRaises(ValueError):
            m.check(self.text, self.rows + [self.rows[0]])

    def test_duplicate_tex_label(self):
        with self.assertRaises(ValueError):
            m.check(self.text + '\\label{tab:equal60-main}', self.rows)


if __name__ == '__main__':
    unittest.main()
