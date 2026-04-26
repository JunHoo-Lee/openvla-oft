"""
Queue-driven LIBERO failed75 sweep for abstention-first recovery policies.

The sweep is intentionally centered on the current finding: longer horizons with
minimal intervention are strong, while recovery should only fire under stricter
stale/no-progress evidence.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_SUITES = [
    "libero_goal",
    "libero_spatial",
    "libero_object",
    "libero_10",
]

NO_REWIND_HORIZONS = [1.0, 2.0, 3.0, 4.0]
RECOVERY_HORIZONS = [2.0, 3.0, 4.0]
RECOVERY_STRATEGIES = {
    "joint": "joint_rewind",
    "critical": "critical_rewind",
}
MAX_RESETS = [1, 2]

DETECTOR_PROFILES = {
    "loose": {
        "stuck_window_steps": 24,
        "stuck_min_stale_steps": 8,
        "stuck_require_revisit": False,
        "progress_veto_min_delta": 0.04,
        "middle_state_min_steps_between_resets": 60,
        "critical_rewind_min_advantage": 0.05,
    },
    "medium": {
        "stuck_window_steps": 32,
        "stuck_min_stale_steps": 12,
        "stuck_require_revisit": False,
        "progress_veto_min_delta": 0.08,
        "middle_state_min_steps_between_resets": 80,
        "critical_rewind_min_advantage": 0.10,
    },
    "strict": {
        "stuck_window_steps": 48,
        "stuck_min_stale_steps": 20,
        "stuck_require_revisit": True,
        "progress_veto_min_delta": 0.12,
        "middle_state_min_steps_between_resets": 120,
        "critical_rewind_min_advantage": 0.20,
    },
}

BURST_PROFILES = {
    "off": {
        "post_rewind_exploration_steps": 0,
        "post_rewind_xyz_noise_scale": 0.0,
        "post_rewind_rot_noise_scale": 0.0,
        "post_rewind_action_samples": 1,
        "post_rewind_escape_lookahead": False,
        "post_rewind_score_margin": 0.0,
        "post_rewind_base_action_penalty": 0.0,
    },
    "safe": {
        "post_rewind_exploration_steps": 8,
        "post_rewind_xyz_noise_scale": 0.03,
        "post_rewind_rot_noise_scale": 0.06,
        "post_rewind_action_samples": 8,
        "post_rewind_escape_lookahead": True,
        "post_rewind_score_margin": 0.03,
        "post_rewind_base_action_penalty": 0.0,
    },
}


def float_tag(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def bool_arg(value: bool) -> str:
    return "True" if value else "False"


def extend_args(args: List[str], values: Dict[str, object]) -> None:
    for key, value in values.items():
        args.extend([f"--{key}", bool_arg(value) if isinstance(value, bool) else str(value)])


def build_sweep_configs() -> List[Dict[str, object]]:
    configs: List[Dict[str, object]] = []

    for horizon in NO_REWIND_HORIZONS:
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

    for strategy_name, strategy in RECOVERY_STRATEGIES.items():
        for horizon in RECOVERY_HORIZONS:
            for max_resets in MAX_RESETS:
                for detector_name, detector_profile in DETECTOR_PROFILES.items():
                    for burst_name, burst_profile in BURST_PROFILES.items():
                        config_name = (
                            f"pg_{strategy_name}_h{float_tag(horizon)}_m{max_resets}_"
                            f"{detector_name}_burst{burst_name}"
                        )
                        config_args = [
                            "--max_steps_multiplier",
                            str(horizon),
                            "--middle_state_strategy",
                            strategy,
                            "--middle_state_time_seconds",
                            "-1.0",
                            "--middle_state_repeat_interval_seconds",
                            "-1.0",
                            "--middle_state_max_resets",
                            str(max_resets),
                            "--middle_state_trigger_on_stuck",
                            "True",
                            "--critical_rewind_require_stale_context",
                            "True",
                            "--critical_rewind_progress_loss_weight",
                            "0.8",
                        ]
                        extend_args(config_args, detector_profile)
                        extend_args(config_args, burst_profile)
                        configs.append(
                            {
                                "name": config_name,
                                "family": "progress_gated_recovery",
                                "strategy": strategy,
                                "detector_profile": detector_name,
                                "burst_profile": burst_name,
                                "args": config_args,
                            }
                        )

    return configs


def expand_suite_jobs(configs: Iterable[Dict[str, object]], suites: Iterable[str]) -> List[Dict[str, object]]:
    jobs = []
    for config in configs:
        for suite in suites:
            config_name = str(config["name"])
            jobs.append(
                {
                    "job_id": f"{config_name}__{suite}",
                    "config_name": config_name,
                    "suite": suite,
                    "family": config.get("family", ""),
                    "args": list(config["args"]),
                }
            )
    return jobs


def write_manifest(path: Path, jobs: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for job in jobs:
            f.write(json.dumps(job, sort_keys=True) + "\n")


def read_manifest(path: Path) -> List[Dict[str, object]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def build_env(args: argparse.Namespace, repo_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    hf_home = str(Path(args.hf_home))
    python_paths = [str(repo_root)]
    libero_repo_root = Path(args.libero_repo_root).resolve()
    if libero_repo_root.exists():
        python_paths.append(str(libero_repo_root))
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])

    env["HF_HOME"] = hf_home
    env["HUGGINGFACE_HUB_CACHE"] = str(Path(args.hf_hub_cache)) if args.hf_hub_cache else str(Path(hf_home) / "hub")
    env["TRANSFORMERS_CACHE"] = (
        str(Path(args.transformers_cache)) if args.transformers_cache else str(Path(hf_home) / "transformers")
    )
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env["PYTHONPATH"] = ":".join(python_paths)
    return env


def verify_runtime(args: argparse.Namespace, env: Dict[str, str], repo_root: Path) -> None:
    cmd = [
        args.python_bin,
        "-c",
        "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())",
    ]
    completed = subprocess.run(cmd, cwd=repo_root, env=env, check=True, capture_output=True, text=True)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if (not lines or lines[0] != "True") and not args.allow_cpu:
        raise RuntimeError(f"CUDA unavailable for sweep worker. Diagnostics: {completed.stdout!r}")


def marker_paths(output_dir: Path, job_id: str) -> Dict[str, Path]:
    queue_dir = output_dir / "queue"
    return {
        "queue_dir": queue_dir,
        "lock": queue_dir / f"{job_id}.lock",
        "done": queue_dir / f"{job_id}.done.json",
        "failed": queue_dir / f"{job_id}.failed.json",
    }


def claim_job(output_dir: Path, job: Dict[str, object], worker_id: str) -> bool:
    paths = marker_paths(output_dir, str(job["job_id"]))
    paths["queue_dir"].mkdir(parents=True, exist_ok=True)
    if paths["done"].exists() or paths["failed"].exists():
        return False
    try:
        paths["lock"].mkdir()
    except FileExistsError:
        return False
    (paths["lock"] / "owner.json").write_text(
        json.dumps(
            {
                "worker_id": worker_id,
                "pid": os.getpid(),
                "claimed_at": time.time(),
                "job_id": job["job_id"],
            },
            sort_keys=True,
            indent=2,
        )
    )
    return True


def release_lock(output_dir: Path, job_id: str) -> None:
    paths = marker_paths(output_dir, job_id)
    owner = paths["lock"] / "owner.json"
    if owner.exists():
        owner.unlink()
    if paths["lock"].exists():
        paths["lock"].rmdir()


def output_paths(output_dir: Path, job: Dict[str, object]) -> Dict[str, Path]:
    config_dir = output_dir / str(job["config_name"])
    suite = str(job["suite"])
    return {
        "config_dir": config_dir,
        "episodes": config_dir / f"{suite}.episodes.jsonl",
        "summary": config_dir / f"{suite}.summary.json",
        "launcher_log": config_dir / f"{suite}.launcher.log",
    }


def run_job(args: argparse.Namespace, repo_root: Path, env: Dict[str, str], job: Dict[str, object]) -> int:
    paths = output_paths(Path(args.output_dir), job)
    paths["config_dir"].mkdir(parents=True, exist_ok=True)
    suite = str(job["suite"])
    episode_filter_path = Path(args.episode_filter_dir).resolve() / f"failed_episodes_{suite}.json"
    if not episode_filter_path.exists():
        raise FileNotFoundError(f"Missing failed episode filter: {episode_filter_path}")

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
        "False",
        "--local_log_dir",
        str(paths["config_dir"]),
        "--episode_results_path",
        str(paths["episodes"]),
        "--summary_results_path",
        str(paths["summary"]),
        "--run_id_note",
        f"{job['config_name']}_{suite}",
        "--episode_filter_path",
        str(episode_filter_path),
        *[str(item) for item in job["args"]],
    ]
    with paths["launcher_log"].open("w") as log_file:
        log_file.write(" ".join(cmd) + "\n\n")
        log_file.flush()
        completed = subprocess.run(cmd, cwd=repo_root, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return int(completed.returncode)


def mark_finished(output_dir: Path, job: Dict[str, object], marker: str, payload: Dict[str, object]) -> None:
    paths = marker_paths(output_dir, str(job["job_id"]))
    paths[marker].write_text(json.dumps(payload, sort_keys=True, indent=2))


def worker_loop(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = Path(args.output_dir).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    env = build_env(args, repo_root)
    verify_runtime(args, env, repo_root)

    completed_jobs = 0
    while True:
        claimed = False
        for job in read_manifest(manifest_path):
            paths = output_paths(output_dir, job)
            marker = marker_paths(output_dir, str(job["job_id"]))
            if marker["done"].exists() or marker["failed"].exists():
                continue
            if paths["episodes"].exists() and paths["summary"].exists():
                mark_finished(
                    output_dir,
                    job,
                    "done",
                    {"job_id": job["job_id"], "worker_id": args.worker_id, "already_complete": True},
                )
                continue
            if not claim_job(output_dir, job, args.worker_id):
                continue

            claimed = True
            started_at = time.time()
            try:
                returncode = run_job(args, repo_root, env, job)
                payload = {
                    "job_id": job["job_id"],
                    "worker_id": args.worker_id,
                    "returncode": returncode,
                    "started_at": started_at,
                    "finished_at": time.time(),
                }
                if returncode == 0:
                    mark_finished(output_dir, job, "done", payload)
                else:
                    mark_finished(output_dir, job, "failed", payload)
            except Exception as exc:
                mark_finished(
                    output_dir,
                    job,
                    "failed",
                    {
                        "job_id": job["job_id"],
                        "worker_id": args.worker_id,
                        "exception": repr(exc),
                        "started_at": started_at,
                        "finished_at": time.time(),
                    },
                )
            finally:
                release_lock(output_dir, str(job["job_id"]))

            completed_jobs += 1
            if args.max_jobs and completed_jobs >= args.max_jobs:
                return
            break

        if not claimed:
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="./experiments/logs/libero_abstention_sweep_failed75")
    parser.add_argument(
        "--manifest_path",
        default="./experiments/logs/libero_abstention_sweep_failed75/manifest.jsonl",
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
        jobs = expand_suite_jobs(build_sweep_configs(), args.suites)
        write_manifest(Path(args.manifest_path), jobs)
        print(f"Wrote {len(jobs)} jobs to {args.manifest_path}")
    if args.worker:
        worker_loop(args)
    if not args.create_manifest and not args.worker:
        configs = build_sweep_configs()
        jobs = expand_suite_jobs(configs, args.suites)
        print(f"configs={len(configs)} jobs={len(jobs)}")


if __name__ == "__main__":
    main()
