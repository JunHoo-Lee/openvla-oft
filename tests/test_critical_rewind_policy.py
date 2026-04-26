import unittest

from experiments.robot.libero.critical_rewind_policy import (
    choose_candidate_with_margin,
    compute_progress_veto,
    recovery_gate_decision,
)


class CriticalRewindPolicyTest(unittest.TestCase):
    def test_progress_veto_blocks_recent_progress(self):
        decision = compute_progress_veto(
            [1.0, 1.02, 1.06, 1.13],
            min_delta=0.08,
        )

        self.assertTrue(decision["veto"])
        self.assertAlmostEqual(decision["progress_delta"], 0.13)

    def test_progress_veto_allows_flat_progress(self):
        decision = compute_progress_veto(
            [2.0, 2.01, 2.015, 2.02],
            min_delta=0.08,
        )

        self.assertFalse(decision["veto"])
        self.assertAlmostEqual(decision["progress_delta"], 0.02)

    def test_recovery_gate_requires_margin_over_progress_loss(self):
        rejected = recovery_gate_decision(
            scene_escape=0.12,
            eef_escape=0.02,
            progress_loss=0.35,
            stability_cost=0.01,
            scene_escape_weight=1.5,
            eef_escape_weight=0.75,
            progress_loss_weight=0.8,
            stability_weight=1.0,
            min_advantage=0.1,
        )
        accepted = recovery_gate_decision(
            scene_escape=0.45,
            eef_escape=0.08,
            progress_loss=0.03,
            stability_cost=0.01,
            scene_escape_weight=1.5,
            eef_escape_weight=0.75,
            progress_loss_weight=0.8,
            stability_weight=1.0,
            min_advantage=0.1,
        )

        self.assertFalse(rejected["apply"])
        self.assertTrue(accepted["apply"])

    def test_candidate_margin_keeps_base_when_noisy_is_not_clearly_better(self):
        self.assertEqual(choose_candidate_with_margin([1.0, 1.02, 0.8], margin=0.05), 0)
        self.assertEqual(choose_candidate_with_margin([1.0, 1.08, 0.8], margin=0.05), 1)


if __name__ == "__main__":
    unittest.main()
