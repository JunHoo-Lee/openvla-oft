import unittest

from experiments.robot.libero.run_libero_abstention_sweep import DEFAULT_SUITES, expand_suite_jobs
from experiments.robot.libero.run_libero_guarded_stuck_sweep import build_guarded_stuck_sweep_configs


class GuardedStuckSweepTest(unittest.TestCase):
    def test_builds_guarded_stuck_sweep(self):
        configs = build_guarded_stuck_sweep_configs()

        self.assertEqual(len(configs), 34)
        self.assertEqual(len({config["name"] for config in configs}), len(configs))

    def test_guarded_config_uses_stronger_stuck_gate(self):
        configs = {config["name"]: config for config in build_guarded_stuck_sweep_configs()}
        config = configs["guarded_r_m_s_h_h3p0_m4_guarded_burstoff"]
        args = config["args"]

        self.assertEqual(args[args.index("--middle_state_strategy") + 1], "progressive_rewind")
        self.assertEqual(args[args.index("--middle_state_max_resets") + 1], "4")
        self.assertEqual(args[args.index("--stuck_window_steps") + 1], "80")
        self.assertEqual(args[args.index("--stuck_min_stale_steps") + 1], "48")
        self.assertEqual(args[args.index("--progress_veto_min_delta") + 1], "0.025")
        self.assertEqual(args[args.index("--progressive_reset_on_progress_delta") + 1], "0.025")

    def test_expands_guarded_configs_to_all_suites(self):
        configs = build_guarded_stuck_sweep_configs()
        jobs = expand_suite_jobs(configs, DEFAULT_SUITES)

        self.assertEqual(len(jobs), len(configs) * len(DEFAULT_SUITES))


if __name__ == "__main__":
    unittest.main()
