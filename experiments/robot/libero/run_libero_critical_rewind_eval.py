"""
run_libero_critical_rewind_eval.py

Launch LIBERO evaluations with the critical-point rewind and post-rewind uncertainty burst policy.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_SUITES = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretrained_checkpoint",
        default="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
        help="Checkpoint repo id or local path passed through to run_libero_eval.py",
    )
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES, help="LIBERO suites to evaluate")
    parser.add_argument("--num_trials_per_task", type=int, default=50, help="Episodes per task")
    parser.add_argument("--max_steps_multiplier", type=float, default=3.0, help="Episode horizon multiplier")
    parser.add_argument("--seed", type=int, default=7, help="Random seed forwarded to run_libero_eval.py")
    parser.add_argument("--python_bin", default=sys.executable, help="Python executable used to launch evaluations")
    parser.add_argument(
        "--libero_repo_root",
        default="/workspace/TTS_VLA/LIBERO",
        help="LIBERO repository root appended to PYTHONPATH for child evaluation processes",
    )
    parser.add_argument("--hf_home", default="/root/.cache/huggingface", help="HF_HOME for child processes")
    parser.add_argument("--hf_hub_cache", default=None, help="Optional explicit HUGGINGFACE_HUB_CACHE")
    parser.add_argument("--transformers_cache", default=None, help="Optional explicit TRANSFORMERS_CACHE")
    parser.add_argument(
        "--output_dir",
        default="./experiments/logs/libero_critical_rewind",
        help="Directory where summaries and per-suite outputs are stored",
    )
    parser.add_argument("--run_note", default="critical_rewind_3x_burst", help="Run note suffix")
    parser.add_argument(
        "--middle_state_time_seconds",
        type=float,
        default=-1.0,
        help="Initial rewind time trigger; negative disables scheduled rewinds",
    )
    parser.add_argument(
        "--middle_state_repeat_interval_seconds",
        type=float,
        default=-1.0,
        help="Repeat rewind interval after the first trigger; non-positive disables scheduled repeats",
    )
    parser.add_argument("--middle_state_max_resets", type=int, default=2, help="Maximum resets per episode")
    parser.add_argument(
        "--post_rewind_exploration_steps",
        type=int,
        default=8,
        help="Number of noisy policy steps after each rewind",
    )
    parser.add_argument(
        "--post_rewind_xyz_noise_scale",
        type=float,
        default=0.03,
        help="Gaussian noise scale for xyz action dims during the burst",
    )
    parser.add_argument(
        "--post_rewind_rot_noise_scale",
        type=float,
        default=0.06,
        help="Gaussian noise scale for rotation action dims during the burst",
    )
    parser.add_argument(
        "--post_rewind_action_samples",
        type=int,
        default=8,
        help="Number of noisy candidates sampled per burst step",
    )
    parser.add_argument(
        "--post_rewind_base_action_penalty",
        type=float,
        default=0.05,
        help="Small lookahead-score penalty for selecting the unperturbed base action during escape bursts",
    )
    parser.add_argument(
        "--stuck_window_steps",
        type=int,
        default=48,
        help="Recent horizon used for stale-state detection",
    )
    parser.add_argument(
        "--stuck_min_stale_steps",
        type=int,
        default=20,
        help="Minimum stale suffix length before rewinding",
    )
    parser.add_argument(
        "--stuck_require_revisit",
        action="store_true",
        help="Require an explicit state revisit in addition to stale no-progress before rewinding",
    )
    parser.add_argument(
        "--use_failed_episode_filters",
        action="store_true",
        help="Evaluate only failed_episodes_<suite>.json records instead of the full suite",
    )
    parser.add_argument(
        "--episode_filter_dir",
        default=".",
        help="Directory containing failed_episodes_<suite>.json files when --use_failed_episode_filters is set",
    )
    parser.add_argument(
        "--save_rollout_videos",
        action="store_true",
        help="If set, keep per-episode MP4 files during evaluation",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse existing outputs when the expected summary and episode files already exist",
    )
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Allow evaluation to continue even if CUDA is unavailable",
    )
    return parser.parse_args()


def build_env(args: argparse.Namespace, repo_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    hf_home = str(Path(args.hf_home))
    hf_hub_cache = str(Path(args.hf_hub_cache)) if args.hf_hub_cache else str(Path(hf_home) / "hub")
    transformers_cache = (
        str(Path(args.transformers_cache)) if args.transformers_cache else str(Path(hf_home) / "transformers")
    )
    python_paths = [str(repo_root)]
    libero_repo_root = Path(args.libero_repo_root).resolve()
    if libero_repo_root.exists():
        python_paths.append(str(libero_repo_root))

    env["HF_HOME"] = hf_home
    env["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache
    env["TRANSFORMERS_CACHE"] = transformers_cache
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(python_paths)
    return env


def verify_runtime(args: argparse.Namespace, env: Dict[str, str], repo_root: Path) -> None:
    cmd = [
        args.python_bin,
        "-c",
        (
            "import torch; "
            "print('cuda_available=' + str(torch.cuda.is_available())); "
            "print('device_count=' + str(torch.cuda.device_count()));"
        ),
    ]
    completed = subprocess.run(cmd, cwd=repo_root, env=env, check=True, capture_output=True, text=True)
    diagnostics = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.strip().split("=", 1)
            diagnostics[key] = value

    if diagnostics.get("cuda_available") != "True" and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is unavailable in the current runtime. "
            "Refusing to start LIBERO evaluation without --allow_cpu. "
            f"Diagnostics: {diagnostics}"
        )


def run_eval(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    env: Dict[str, str],
    suite: str,
    episode_results_path: Path,
    summary_results_path: Path,
) -> None:
    episode_filter_path = None
    if args.use_failed_episode_filters:
        episode_filter_path = Path(args.episode_filter_dir).resolve() / f"failed_episodes_{suite}.json"
        if not episode_filter_path.exists():
            raise FileNotFoundError(f"Missing failed episode filter for {suite}: {episode_filter_path}")

    cmd = [
        args.python_bin,
        "experiments/robot/libero/run_libero_eval.py",
        "--pretrained_checkpoint",
        args.pretrained_checkpoint,
        "--task_suite_name",
        suite,
        "--num_trials_per_task",
        str(args.num_trials_per_task),
        "--seed",
        str(args.seed),
        "--center_crop",
        "True",
        "--save_rollout_videos",
        "True" if args.save_rollout_videos else "False",
        "--local_log_dir",
        str(summary_results_path.parent),
        "--episode_results_path",
        str(episode_results_path),
        "--summary_results_path",
        str(summary_results_path),
        "--run_id_note",
        f"{args.run_note}_{suite}",
        "--max_steps_multiplier",
        str(args.max_steps_multiplier),
        "--middle_state_strategy",
        "critical_rewind",
        "--middle_state_time_seconds",
        str(args.middle_state_time_seconds),
        "--middle_state_repeat_interval_seconds",
        str(args.middle_state_repeat_interval_seconds),
        "--middle_state_max_resets",
        str(args.middle_state_max_resets),
        "--middle_state_trigger_on_stuck",
        "True",
        "--anchor_min_age_seconds",
        "2.0",
        "--anchor_max_age_seconds",
        "8.0",
        "--critical_anchor_pre_stale_margin_steps",
        "8",
        "--post_rewind_exploration_steps",
        str(args.post_rewind_exploration_steps),
        "--post_rewind_xyz_noise_scale",
        str(args.post_rewind_xyz_noise_scale),
        "--post_rewind_rot_noise_scale",
        str(args.post_rewind_rot_noise_scale),
        "--post_rewind_action_samples",
        str(args.post_rewind_action_samples),
        "--post_rewind_escape_lookahead",
        "True",
        "--post_rewind_base_action_penalty",
        str(args.post_rewind_base_action_penalty),
        "--stuck_window_steps",
        str(args.stuck_window_steps),
        "--stuck_min_stale_steps",
        str(args.stuck_min_stale_steps),
        "--stuck_require_revisit",
        "True" if args.stuck_require_revisit else "False",
    ]
    if episode_filter_path is not None:
        cmd.extend(["--episode_filter_path", str(episode_filter_path)])
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(args, repo_root)
    verify_runtime(args, env, repo_root)

    for suite in args.suites:
        suite_prefix = f"{suite}__{args.run_note}"
        episode_results_path = output_dir / f"{suite_prefix}.episodes.jsonl"
        summary_results_path = output_dir / f"{suite_prefix}.summary.json"
        if args.skip_existing and episode_results_path.exists() and summary_results_path.exists():
            print(f"[skip_existing] Reusing outputs for {suite}")
            continue

        print(f"[critical_rewind] Running suite={suite}")
        run_eval(
            args=args,
            repo_root=repo_root,
            env=env,
            suite=suite,
            episode_results_path=episode_results_path,
            summary_results_path=summary_results_path,
        )


if __name__ == "__main__":
    main()
