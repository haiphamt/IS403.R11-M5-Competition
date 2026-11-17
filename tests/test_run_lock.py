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
    """Test labels stay locked behind the freeze tags: tag exists, is contained in HEAD, clean tree, config in the tag."""

    def run_main(self, argv, tags=(), contained=(), dirty="", config_in=()):
        runner = load_runner()
        have = {f"refs/tags/{t}" for t in tags}
        ok_contained = {f"refs/tags/{t}" for t in contained}
        in_tag = {f"refs/tags/{t}:configs/models/lightgbm_tuned.json" for t in config_in}

        def fake_git(*a):
            if a[:1] == ("status",):
                return dirty
            return "sha" if a and a[-1] in have else ""

        def fake_git_ok(*a):
            if a[:1] == ("merge-base",):
                return a[2] in ok_contained
            if a[:1] == ("cat-file",):
                return a[2] in in_tag
            return False

        def must_not_run(*a, **k):
            raise AssertionError("lock did not stop the run before data was read")

        with mock.patch.object(runner, "git", fake_git), mock.patch.object(runner, "git_ok", fake_git_ok), \
                mock.patch.object(runner, "resolve_raw_dir", must_not_run), mock.patch.object(runner, "load_raw_data", must_not_run), \
                mock.patch.object(sys, "argv", ["run_experiment.py", *argv]):
            runner.main()

    def locked(self, argv, **kw):
        with self.assertRaises(SystemExit) as ctx:
            self.run_main(argv, **kw)
        return str(ctx.exception)

    def test_test_split_needs_freeze_tag_even_for_subset(self):
        self.assertIn("not found", self.locked(["--split", "test", "--max-series", "2", "--models", "seasonal_naive"]))

    def test_tag_must_be_contained_in_head(self):
        msg = self.locked(["--split", "test", "--models", "seasonal_naive"], tags=["config-freeze"])
        self.assertIn("does not contain", msg)

    def test_dirty_tree_blocks_test(self):
        msg = self.locked(["--split", "test", "--models", "seasonal_naive"], tags=["config-freeze"],
                          contained=["config-freeze"], dirty=" M src/m5/metrics.py")
        self.assertIn("uncommitted", msg)

    def test_optional_models_need_second_tag(self):
        msg = self.locked(["--split", "test", "--models", "croston_sba"], tags=["config-freeze"], contained=["config-freeze"])
        self.assertIn("config-freeze-v2", msg)

    def test_tuned_params_must_be_inside_second_tag(self):
        both = ["config-freeze", "config-freeze-v2"]
        msg = self.locked(["--split", "test", "--models", "lightgbm_tuned"], tags=both, contained=both)
        self.assertIn("lightgbm_tuned.json is not part of tag", msg)


if __name__ == "__main__":
    unittest.main()
