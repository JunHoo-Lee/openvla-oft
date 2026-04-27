"""
Queue-driven LIBERO failed75 sweep for conservative stuck-triggered progressive rewind.

This sweep isolates the question: can higher reset budgets help if recovery only
fires after stronger no-progress evidence and escalation resets after small
scene progress?
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.run_libero_abstention_sweep import (
    BURST_PROFILES,
    DEFAULT_SUITES,
    expand_suite_jobs,
    extend_args,
    float_tag,
    worker_loop,
    write_manifest,
)


HORIZONS = [3.0, 4.0]
MAX_RESETS = [4, 3, 2, 1]
FULL_LADDER = "retreat,micro_anchor,stable_anchor,home"

GUARDED_DETECTOR_PROFILES: Dict[str, Dict[str, object]] = {
    "guarded": {
        "stuck_window_steps": 80,
        "stuck_min_stale_steps": 48,
        "stuck_min_policy_steps": 240,
        "stuck_scene_motion_threshold": 0.018,
        "stuck_eef_speed_threshold": 0.012,
        "stuck_net_scene_delta_threshold": 0.04,
        "stuck_net_eef_delta_threshold": 0.025,
        "stuck_revisit_scene_threshold": 0.035,
        "stuck_revisit_eef_threshold": 0.020,
        "stuck_require_revisit": True,
        "progress_veto_min_delta": 0.025,
        "middle_state_min_steps_between_resets": 160,
        "critical_rewind_min_advantage": 0.20,
    },
    "ultra": {
        "stuck_window_steps": 96,
        "stuck_min_stale_steps": 64,
        "stuck_min_policy_steps": 320,
        "stuck_scene_motion_threshold": 0.014,
        "stuck_eef_speed_threshold": 0.010,
        "stuck_net_scene_delta_threshold": 0.03,
        "stuck_net_eef_delta_threshold": 0.018,
        "stuck_revisit_scene_threshold": 0.028,
        "stuck_revisit_eef_threshold": 0.016,
        "stuck_require_revisit": True,
        "progress_veto_min_delta": 0.018,
        "middle_state_min_steps_between_resets": 240,
        "critical_rewind_min_advantage": 0.25,
    },
}


def build_guarded_stuck_sweep_configs() -> List[Dict[str, object]]:
    configs: List[Dict[str, object]] = []

    for horizon in HORIZONS:
        for max_resets in MAX_RESETS:
            for detector_name, detector_profile in GUARDED_DETECTOR_PROFILES.items():
                for burst_name, burst_profile in BURST_PROFILES.items():
                    config_name = (
                        f"guarded_r_m_s_h_h{float_tag(horizon)}_m{max_resets}_"
                        f"{detector_name}_burst{burst_name}"
                    )
                    config_args = [
                        "--max_steps_multiplier",
                        str(horizon),
                        "--middle_state_strategy",
                        "progressive_rewind",
                        "--middle_state_time_seconds",
                        "-1.0",
                        "--middle_state_repeat_interval_seconds",
                        "-1.0",
                        "--middle_state_max_resets",
                        str(max_resets),
                        "--middle_state_trigger_on_stuck",
                        "True",
                        "--progressive_rewind_levels",
                        FULL_LADDER,
                        "--progressive_require_stale_context",
                        "True",
                        "--progressive_retreat_steps",
                        "10",
                        "--progressive_retreat_z",
                        "0.06",
                        "--progressive_micro_anchor_min_seconds",
                        "0.5",
                        "--progressive_micro_anchor_max_seconds",
                        "2.0",
                        "--progressive_stable_anchor_min_seconds",
                        "2.0",
                        "--progressive_stable_anchor_max_seconds",
                        "6.0",
                        "--progressive_reset_on_progress_delta",
                        "0.025",
                    ]
                    extend_args(config_args, detector_profile)
                    extend_args(config_args, burst_profile)
                    configs.append(
                        {
                            "name": config_name,
                            "family": "guarded_progressive_rewind",
                            "ladder": FULL_LADDER,
                            "detector_profile": detector_name,
                            "burst_profile": burst_name,
                            "args": config_args,
                        }
                    )

    for horizon in HORIZONS:
        configs.append(
            {
                "name": f"nr_h{float_tag(horizon)}",
                "family": "no_rewind",
                "args": [
                    "--max_steps_multiplier",
                    str(horizon),
                    "--middle_state_time_seconds",
                    "-1.0",
                    "--middle_state_repeat_interval_seconds",
                    "-1.0",
                    "--middle_state_max_resets",
                    "0",
                    "--middle_state_trigger_on_stuck",
                    "False",
                    "--post_rewind_exploration_steps",
                    "0",
                ],
            }
        )

    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="./experiments/logs/libero_guarded_stuck_sweep_failed75")
    parser.add_argument(
        "--manifest_path",
        default="./experiments/logs/libero_guarded_stuck_sweep_failed75/manifest.jsonl",
    )
    parser.add_argument("--create_manifest", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker_id", default=f"worker-{os.getpid()}")
    parser.add_argument("--max_jobs", type=int, default=0)
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument(
        "--pretrained_checkpoint",
        default="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
    )
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episode_filter_dir", default="experiments/robot/libero/episode_filters/failed_75")
    parser.add_argument("--libero_repo_root", default="/workspace/TTS_VLA/LIBERO")
    parser.add_argument("--hf_home", default="/root/.cache/huggingface")
    parser.add_argument("--hf_hub_cache", default=None)
    parser.add_argument("--transformers_cache", default=None)
    parser.add_argument("--allow_cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.create_manifest:
        jobs = expand_suite_jobs(build_guarded_stuck_sweep_configs(), args.suites)
        write_manifest(Path(args.manifest_path), jobs)
        print(f"Wrote {len(jobs)} jobs to {args.manifest_path}")
    if args.worker:
        worker_loop(args)
    if not args.create_manifest and not args.worker:
        configs = build_guarded_stuck_sweep_configs()
        jobs = expand_suite_jobs(configs, args.suites)
        print(f"configs={len(configs)} jobs={len(jobs)}")


if __name__ == "__main__":
    main()
