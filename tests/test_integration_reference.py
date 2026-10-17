import importlib.util
from pathlib import Path

import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReferenceComparatorTests(unittest.TestCase):
    def test_reference_comparator_detects_changed_record(self):
        path = ROOT / 'scripts/check_refactor_equivalence.py'
        self.assertTrue(path.is_file(), 'Independent reference comparator is missing')
        spec = importlib.util.spec_from_file_location('reference_comparator', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        record = {name: [1, 2] for name in ['prediction', 'rng', 'shared', 'state', 'gradients']}
        module.compare_records(record, dict(record))
        for field in record:
            with self.subTest(field=field):
                changed = dict(record)
                changed[field] = [1, 3]
                with self.assertRaisesRegex(ValueError, field):
                    module.compare_records(record, changed)
