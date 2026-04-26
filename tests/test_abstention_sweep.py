import unittest

from experiments.robot.libero.run_libero_abstention_sweep import (
    DEFAULT_SUITES,
    build_sweep_configs,
    expand_suite_jobs,
)


class AbstentionSweepTest(unittest.TestCase):
    def test_builds_full_factorial_abstention_sweep(self):
        configs = build_sweep_configs()

        self.assertEqual(len(configs), 76)
        self.assertEqual(len({config["name"] for config in configs}), len(configs))

    def test_no_rewind_configs_disable_recovery(self):
        configs = {config["name"]: config for config in build_sweep_configs()}
        config = configs["nr_h3p0"]
        args = config["args"]

        self.assertIn("--middle_state_max_resets", args)
        self.assertEqual(args[args.index("--middle_state_max_resets") + 1], "0")
        self.assertIn("--middle_state_trigger_on_stuck", args)
        self.assertEqual(args[args.index("--middle_state_trigger_on_stuck") + 1], "False")

    def test_recovery_configs_are_stuck_triggered_not_scheduled(self):
        configs = {config["name"]: config for config in build_sweep_configs()}
        config = configs["pg_critical_h3p0_m2_medium_burstoff"]
        args = config["args"]

        self.assertEqual(args[args.index("--middle_state_strategy") + 1], "critical_rewind")
        self.assertEqual(args[args.index("--middle_state_time_seconds") + 1], "-1.0")
        self.assertEqual(args[args.index("--middle_state_repeat_interval_seconds") + 1], "-1.0")
        self.assertEqual(args[args.index("--middle_state_trigger_on_stuck") + 1], "True")
        self.assertEqual(args[args.index("--progress_veto_min_delta") + 1], "0.08")

    def test_expands_each_config_to_all_suites(self):
        configs = build_sweep_configs()
        jobs = expand_suite_jobs(configs, DEFAULT_SUITES)

        self.assertEqual(len(jobs), len(configs) * len(DEFAULT_SUITES))
        self.assertEqual(len({job["job_id"] for job in jobs}), len(jobs))


if __name__ == "__main__":
    unittest.main()
