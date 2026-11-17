import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location("run_experiment", ROOT / "scripts" / "run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSplitLock(unittest.TestCase):
    """Test labels must stay locked behind the freeze tags, also for subset (--max-series) runs."""

    def run_main(self, argv, tags):
        runner = load_runner()
        have = {f"refs/tags/{t}" for t in tags}
        fake_git = lambda *a: "sha" if a and a[-1] in have else ""
        def must_not_run(*a, **k):
            raise AssertionError("lock did not stop the run before data was read")
        with mock.patch.object(runner, "git", fake_git), mock.patch.object(runner, "load_raw_data", must_not_run),                 mock.patch.object(sys, "argv", ["run_experiment.py", *argv]):
            runner.main()

    def test_test_split_needs_freeze_tag_even_for_subset(self):
        with self.assertRaises(SystemExit) as ctx:
            self.run_main(["--split", "test", "--max-series", "2", "--models", "seasonal_naive"], tags=[])
        self.assertIn("locked", str(ctx.exception))

    def test_optional_models_need_second_tag(self):
        with self.assertRaises(SystemExit) as ctx:
            self.run_main(["--split", "test", "--max-series", "2", "--models", "croston_sba"], tags=["config-freeze"])
        self.assertIn("config-freeze-v2", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
