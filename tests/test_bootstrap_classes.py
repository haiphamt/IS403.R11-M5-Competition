import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBootstrap(unittest.TestCase):
    def test_paired_difference_is_zero_free_when_one_model_is_always_better(self):
        rng = np.random.default_rng(0)
        base = rng.uniform(0.5, 1.5, 300)
        wide = pd.DataFrame({"ets": base, "croston": base + 0.05, "lightgbm": base + 0.05 + rng.normal(0, 0.2, 300)})
        out = load("bootstrap_ci").summarize(wide, reference="ets", draws=400, seed=1).set_index("model")
        self.assertGreater(out.loc["croston", "diff_ci_low"], 0)                     # always worse than ETS by 0.05
        self.assertLess(out.loc["lightgbm", "diff_ci_low"], out.loc["lightgbm", "diff_ci_high"])
        self.assertAlmostEqual(out.loc["ets", "macro_rmsse"], base.mean())
        self.assertLessEqual(out.loc["croston", "ci_low"], out.loc["croston", "macro_rmsse"])
        self.assertGreaterEqual(out.loc["croston", "ci_high"], out.loc["croston", "macro_rmsse"])

    def test_same_seed_gives_same_draws(self):
        wide = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)[::-1]})
        mod = load("bootstrap_ci")
        np.testing.assert_array_equal(mod.bootstrap_means(wide, 50, 7), mod.bootstrap_means(wide, 50, 7))

    def test_optional_pairs_cover_every_listed_model_and_keep_the_sign(self):
        mod = load("bootstrap_ci")
        base = np.random.default_rng(2).uniform(0.5, 1.5, 200)
        names = {m for pair in mod.OPT_PAIRS for m in pair}
        wide = pd.DataFrame({m: base + 0.02 * i for i, m in enumerate(sorted(names))})  # alphabetical order = increasing error
        out = mod.pair_summary(wide, mod.OPT_PAIRS, draws=200, seed=1)
        self.assertEqual(len(out), len(mod.OPT_PAIRS))
        for r in out.itertuples():
            self.assertEqual(r.diff < 0, r.first < r.second)
            self.assertTrue(r.ci_low <= r.diff <= r.ci_high or abs(r.diff) < 1e-9)


class TestClassRobustness(unittest.TestCase):
    def test_leading_zeros_are_dropped_from_the_after_first_sale_share(self):
        mod = load("robustness_classes")
        history = np.array([[0, 0, 0, 0, 1, 0, 1, 0, 1, 1],    # new item: 4 leading zeros, then 2 zeros in 6 days
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
        all_days, after, first = mod.zero_shares(history)
        np.testing.assert_allclose(all_days, [0.6, 0.0, 1.0])
        np.testing.assert_allclose(after, [2 / 6, 0.0, 1.0])
        self.assertEqual(list(first), [4, 0, 10])
        self.assertEqual(list(mod.classify(all_days)), ["mid_zero(0.5-0.8]", "low_zero(<=0.5)", "high_zero(>0.8)"])
        self.assertEqual(list(mod.classify(after)), ["low_zero(<=0.5)", "low_zero(<=0.5)", "high_zero(>0.8)"])


if __name__ == "__main__":
    unittest.main()
