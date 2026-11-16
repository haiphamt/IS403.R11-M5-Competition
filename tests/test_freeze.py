import sys
import unittest
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from m5.sample import manifest_hash  # noqa: E402


class FreezeConfigTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT / "configs" / "freeze.yaml").read_text(encoding="utf-8"))

    def test_manifest_hash_matches_sample_file(self):
        ids = pd.read_csv(ROOT / self.cfg["sample"]["manifest"])
        self.assertEqual(len(ids), self.cfg["sample"]["n_series"])
        self.assertEqual(manifest_hash(ids), self.cfg["sample"]["manifest_sha256"])

    def test_listed_validation_runs_exist(self):
        for entry in self.cfg["validation_runs"]:
            self.assertTrue((ROOT / "results" / "runs" / entry.split()[0] / "predictions.parquet").exists(), entry)

    def test_test_split_stays_after_validation(self):
        s = self.cfg["splits"]
        self.assertEqual(s["validation"][0], s["train"][1] + 1)
        self.assertEqual(s["test"][0], s["validation"][1] + 1)


if __name__ == "__main__":
    unittest.main()
