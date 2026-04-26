import unittest

from experiments.robot.libero.run_libero_abstention_sweep import DEFAULT_SUITES, expand_suite_jobs
from experiments.robot.libero.run_libero_progressive_sweep import build_progressive_sweep_configs


class ProgressiveSweepTest(unittest.TestCase):
    def test_builds_progressive_ladder_sweep(self):
        configs = build_progressive_sweep_configs()

        self.assertEqual(len(configs), 146)
        self.assertEqual(len({config["name"] for config in configs}), len(configs))

    def test_progressive_configs_use_progressive_strategy(self):
        configs = {config["name"]: config for config in build_progressive_sweep_configs()}
        config = configs["prog_r_m_s_h_h3p0_m3_medium_burstoff"]
        args = config["args"]

        self.assertEqual(args[args.index("--middle_state_strategy") + 1], "progressive_rewind")
        self.assertEqual(args[args.index("--progressive_rewind_levels") + 1], "retreat,micro_anchor,stable_anchor,home")
        self.assertEqual(args[args.index("--middle_state_max_resets") + 1], "3")
        self.assertEqual(args[args.index("--middle_state_trigger_on_stuck") + 1], "True")
        self.assertEqual(args[args.index("--progress_veto_min_delta") + 1], "0.08")

    def test_expands_progressive_configs_to_all_suites(self):
        configs = build_progressive_sweep_configs()
        jobs = expand_suite_jobs(configs, DEFAULT_SUITES)

        self.assertEqual(len(jobs), len(configs) * len(DEFAULT_SUITES))


if __name__ == "__main__":
    unittest.main()
