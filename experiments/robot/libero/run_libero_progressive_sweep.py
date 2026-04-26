"""
Queue-driven LIBERO failed75 sweep for progressive adaptive rewind ladders.

This sweep tests whether recovery becomes useful when interventions escalate
from small local retreats to farther rewinds only after repeated stale triggers.
"""

import argparse
import itertools
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
    DETECTOR_PROFILES,
    expand_suite_jobs,
    extend_args,
    float_tag,
    worker_loop,
    write_manifest,
)


BASELINE_HORIZONS = [3.0, 4.0]
PROGRESSIVE_HORIZONS = [3.0, 4.0]
PROGRESSIVE_MAX_RESETS = [2, 3, 4]
PROGRESSIVE_LEVELS = [
    ("r", "retreat"),
    ("m", "micro_anchor"),
    ("s", "stable_anchor"),
    ("h", "home"),
]


def build_progressive_ladders() -> Dict[str, str]:
    """Return every local-to-far progressive ladder, preserving level order."""
    ladders: Dict[str, str] = {}
    for width in range(1, len(PROGRESSIVE_LEVELS) + 1):
        for combo in itertools.combinations(PROGRESSIVE_LEVELS, width):
            name = "_".join(level[0] for level in combo)
            spec = ",".join(level[1] for level in combo)
            ladders[name] = spec
    return ladders


PROGRESSIVE_LADDERS = {
    name: spec
    for name, spec in sorted(
        build_progressive_ladders().items(),
        key=lambda item: (-len(item[0].split("_")), item[0]),
    )
}


def build_progressive_sweep_configs() -> List[Dict[str, object]]:
    configs: List[Dict[str, object]] = []

    for ladder_name, ladder_spec in PROGRESSIVE_LADDERS.items():
        for horizon in PROGRESSIVE_HORIZONS:
            for max_resets in PROGRESSIVE_MAX_RESETS:
                for detector_name, detector_profile in DETECTOR_PROFILES.items():
                    for burst_name, burst_profile in BURST_PROFILES.items():
                        config_name = (
                            f"prog_{ladder_name}_h{float_tag(horizon)}_m{max_resets}_"
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
                            ladder_spec,
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
                            "0.08",
                        ]
                        extend_args(config_args, detector_profile)
                        extend_args(config_args, burst_profile)
                        configs.append(
                            {
                                "name": config_name,
                                "family": "progressive_rewind",
                                "ladder": ladder_spec,
                                "detector_profile": detector_name,
                                "burst_profile": burst_name,
                                "args": config_args,
                            }
                        )

    for horizon in BASELINE_HORIZONS:
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
    parser.add_argument("--output_dir", default="./experiments/logs/libero_progressive_sweep_failed75")
    parser.add_argument(
        "--manifest_path",
        default="./experiments/logs/libero_progressive_sweep_failed75/manifest.jsonl",
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
        jobs = expand_suite_jobs(build_progressive_sweep_configs(), args.suites)
        write_manifest(Path(args.manifest_path), jobs)
        print(f"Wrote {len(jobs)} jobs to {args.manifest_path}")
    if args.worker:
        worker_loop(args)
    if not args.create_manifest and not args.worker:
        configs = build_progressive_sweep_configs()
        jobs = expand_suite_jobs(configs, args.suites)
        print(f"configs={len(configs)} jobs={len(jobs)}")


if __name__ == "__main__":
    main()
