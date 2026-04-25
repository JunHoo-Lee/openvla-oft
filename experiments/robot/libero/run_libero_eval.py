"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Append repo root and, when present, a sibling LIBERO repo so imports do not depend on launch cwd.
REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_REPO_ROOT = REPO_ROOT.parent / "LIBERO"
for path in (REPO_ROOT, LIBERO_REPO_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.append(str(path))

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
from robosuite.controllers.controller_factory import controller_factory

import wandb
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    control_freq: int = 20                           # Control frequency passed to LIBERO / robosuite env
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)
    max_steps_override: Optional[int] = None         # If set, overrides suite-specific horizon for each episode
    max_steps_multiplier: float = 1.0                # If set > 1, multiplies suite-specific horizon before eval
    middle_state_step: int = -1                      # If >= 0, trigger a one-off mid-episode reset at this policy step
    middle_state_time_seconds: Optional[float] = 10.0  # If >= 0, trigger rewind after this many wall-clock control seconds
    middle_state_repeat_interval_seconds: Optional[float] = None  # If > 0, trigger additional rewinds at this interval
    middle_state_max_resets: int = 1                # Maximum number of mid-episode rewinds to apply
    middle_state_strategy: str = "joint_rewind"      # Reset strategy: "joint_rewind", "anchor_rewind", "critical_rewind", "servo_rewind", "state_restore", "teleport", or "inverse_replay"
    middle_state_settle_steps: int = 5               # Dummy env steps after reset before re-querying the policy
    middle_state_record_transition: bool = True      # If True, record intermediate reset frames in rollout videos
    middle_state_rewind_max_steps: int = 80          # Max env steps spent physically rewinding the arm
    middle_state_gripper_open_steps: int = 5         # Steps spent explicitly opening the gripper before homing
    middle_state_rewind_lift_steps: int = 0          # Optional initial upward retreat steps before homing
    middle_state_rewind_clearance: float = 0.10      # Extra z clearance used before moving back in XY
    middle_state_rewind_xy_tol: float = 0.02         # XY tolerance for phased rewind
    middle_state_rewind_z_tol: float = 0.015         # Z tolerance for phased rewind
    middle_state_rewind_pos_gain: float = 4.0        # P gain from EEF position error to action delta
    middle_state_rewind_rot_gain: float = 2.0        # P gain from EEF rotation error to action delta
    middle_state_rewind_pos_clip: float = 0.08       # Per-step clamp for xyz action during rewind
    middle_state_rewind_rot_clip: float = 0.25       # Per-step clamp for rotation action during rewind
    middle_state_rewind_pos_tol: float = 0.015       # Stop rewind when xyz error norm drops below this
    middle_state_rewind_rot_tol: float = 0.10        # Stop rewind when rot error norm drops below this
    middle_state_rewind_joint_tol: float = 0.05      # Stop joint rewind when arm joint error norm drops below this
    anchor_buffer_seconds: float = 12.0              # How much recent trajectory history to keep for anchor selection
    anchor_min_age_seconds: float = 2.0              # Minimum age of anchor relative to rewind time
    anchor_max_age_seconds: float = 8.0              # Maximum age of anchor relative to rewind time
    anchor_selection_strategy: str = "min_motion"    # Anchor selector: "min_motion" or "latest"
    anchor_eef_speed_weight: float = 0.5             # Weight on EEF speed when scoring candidate anchors
    middle_state_trigger_on_stuck: bool = False      # If True, allow stale-state detection to trigger rewind early
    middle_state_min_steps_between_resets: int = 80  # Minimum step gap between repeated rewinds
    stuck_window_steps: int = 48                     # Number of recent policy steps inspected for stale-state detection
    stuck_min_stale_steps: int = 20                  # Minimum trailing steps that must look stale before rewinding
    stuck_min_policy_steps: int = 120                # Do not trigger stale-state rewinds before this many policy steps
    stuck_scene_motion_threshold: float = 0.02       # Recent scene-motion threshold for stale-state detection
    stuck_eef_speed_threshold: float = 0.015         # Recent EEF speed threshold for stale-state detection
    stuck_net_scene_delta_threshold: float = 0.05    # Net scene displacement threshold across the stale window
    stuck_net_eef_delta_threshold: float = 0.03      # Net EEF displacement threshold across the stale window
    stuck_revisit_scene_threshold: float = 0.04      # Revisit threshold on scene state to confirm stale looping
    stuck_revisit_eef_threshold: float = 0.025       # Revisit threshold on EEF position to confirm stale looping
    stuck_require_revisit: bool = True               # If False, stale no-progress alone can trigger a rewind
    critical_anchor_pre_stale_margin_steps: int = 8  # How far before stale onset the rollback point must be
    critical_anchor_scene_escape_weight: float = 1.5 # Weight on scene-space distance from stale basin
    critical_anchor_stability_weight: float = 1.0    # Weight on anchor stability cost during rollback-point search
    critical_anchor_age_weight: float = 0.25         # Penalty on rolling too far back in time
    critical_anchor_progress_weight: float = 0.15    # Preference for anchors that had already changed the scene
    post_rewind_exploration_steps: int = 0           # Number of policy steps to perturb after rewind
    post_rewind_xyz_noise_scale: float = 0.0         # Gaussian noise added to xyz action dims after rewind
    post_rewind_rot_noise_scale: float = 0.0         # Gaussian noise added to rotation action dims after rewind
    post_rewind_action_samples: int = 1              # Number of noisy action candidates sampled during the post-rewind burst
    post_rewind_escape_lookahead: bool = False       # If True, use one-step simulator lookahead to pick the best burst action
    post_rewind_noise_decay: float = 0.85            # Multiplicative decay of exploration noise across burst steps
    post_rewind_scene_escape_weight: float = 1.0     # Reward for increasing scene-state distance from the stale basin
    post_rewind_eef_escape_weight: float = 0.35      # Reward for increasing EEF distance from the stale basin
    post_rewind_revisit_weight: float = 0.5          # Reward for moving away from recently visited stale states
    post_rewind_action_deviation_weight: float = 0.1 # Penalty on deviating too far from the model action
    post_rewind_anchor_drift_weight: float = 0.1     # Penalty on drifting too far from the selected rollback anchor

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    rollout_dir_override: Optional[str] = None       # If set, save rollout MP4s into this directory
    save_rollout_videos: bool = True                 # Whether to save MP4 rollout videos for each evaluated episode
    episode_results_path: Optional[str] = None       # Optional path for per-episode JSONL results
    summary_results_path: Optional[str] = None       # Optional path for run summary JSON
    episode_filter_path: Optional[str] = None        # Optional JSON/JSONL path listing episodes to evaluate
    rerun_failed_episodes_from: Optional[str] = None # Optional JSON/JSONL path from which failed episodes are selected

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert cfg.max_steps_multiplier > 0, "max_steps_multiplier must be positive!"
    assert not (
        cfg.episode_filter_path is not None and cfg.rerun_failed_episodes_from is not None
    ), "Specify at most one of episode_filter_path and rerun_failed_episodes_from!"
    assert cfg.middle_state_time_seconds is None or isinstance(
        cfg.middle_state_time_seconds, (int, float)
    ), "middle_state_time_seconds must be numeric when set!"
    assert (
        cfg.middle_state_repeat_interval_seconds is None
        or isinstance(cfg.middle_state_repeat_interval_seconds, (int, float))
    ), "middle_state_repeat_interval_seconds must be numeric when set!"
    assert cfg.middle_state_max_resets >= 0, "middle_state_max_resets must be non-negative!"
    assert cfg.middle_state_strategy in {
        "joint_rewind",
        "anchor_rewind",
        "critical_rewind",
        "state_restore",
        "servo_rewind",
        "teleport",
        "inverse_replay",
    }, f"Unsupported middle_state_strategy: {cfg.middle_state_strategy}"
    assert cfg.middle_state_settle_steps >= 0, "middle_state_settle_steps must be non-negative!"
    assert cfg.middle_state_rewind_max_steps >= 0, "middle_state_rewind_max_steps must be non-negative!"
    assert cfg.middle_state_gripper_open_steps >= 0, "middle_state_gripper_open_steps must be non-negative!"
    assert cfg.middle_state_rewind_lift_steps >= 0, "middle_state_rewind_lift_steps must be non-negative!"
    assert cfg.middle_state_rewind_clearance >= 0, "middle_state_rewind_clearance must be non-negative!"
    assert cfg.middle_state_rewind_xy_tol >= 0, "middle_state_rewind_xy_tol must be non-negative!"
    assert cfg.middle_state_rewind_z_tol >= 0, "middle_state_rewind_z_tol must be non-negative!"
    assert cfg.middle_state_rewind_pos_clip > 0, "middle_state_rewind_pos_clip must be positive!"
    assert cfg.middle_state_rewind_rot_clip > 0, "middle_state_rewind_rot_clip must be positive!"
    assert cfg.middle_state_rewind_pos_tol >= 0, "middle_state_rewind_pos_tol must be non-negative!"
    assert cfg.middle_state_rewind_rot_tol >= 0, "middle_state_rewind_rot_tol must be non-negative!"
    assert cfg.middle_state_rewind_joint_tol >= 0, "middle_state_rewind_joint_tol must be non-negative!"
    assert cfg.anchor_buffer_seconds >= 0, "anchor_buffer_seconds must be non-negative!"
    assert cfg.anchor_min_age_seconds >= 0, "anchor_min_age_seconds must be non-negative!"
    assert cfg.anchor_max_age_seconds >= cfg.anchor_min_age_seconds, (
        "anchor_max_age_seconds must be >= anchor_min_age_seconds!"
    )
    assert cfg.anchor_selection_strategy in {"min_motion", "latest"}, (
        f"Unsupported anchor_selection_strategy: {cfg.anchor_selection_strategy}"
    )
    assert cfg.anchor_eef_speed_weight >= 0, "anchor_eef_speed_weight must be non-negative!"
    assert cfg.middle_state_min_steps_between_resets >= 0, (
        "middle_state_min_steps_between_resets must be non-negative!"
    )
    assert cfg.stuck_window_steps > 0, "stuck_window_steps must be positive!"
    assert 0 < cfg.stuck_min_stale_steps <= cfg.stuck_window_steps, (
        "stuck_min_stale_steps must be in [1, stuck_window_steps]!"
    )
    assert cfg.stuck_min_policy_steps >= 0, "stuck_min_policy_steps must be non-negative!"
    assert cfg.stuck_scene_motion_threshold >= 0, "stuck_scene_motion_threshold must be non-negative!"
    assert cfg.stuck_eef_speed_threshold >= 0, "stuck_eef_speed_threshold must be non-negative!"
    assert cfg.stuck_net_scene_delta_threshold >= 0, "stuck_net_scene_delta_threshold must be non-negative!"
    assert cfg.stuck_net_eef_delta_threshold >= 0, "stuck_net_eef_delta_threshold must be non-negative!"
    assert cfg.stuck_revisit_scene_threshold >= 0, "stuck_revisit_scene_threshold must be non-negative!"
    assert cfg.stuck_revisit_eef_threshold >= 0, "stuck_revisit_eef_threshold must be non-negative!"
    assert cfg.critical_anchor_pre_stale_margin_steps >= 0, (
        "critical_anchor_pre_stale_margin_steps must be non-negative!"
    )
    assert cfg.critical_anchor_scene_escape_weight >= 0, (
        "critical_anchor_scene_escape_weight must be non-negative!"
    )
    assert cfg.critical_anchor_stability_weight >= 0, "critical_anchor_stability_weight must be non-negative!"
    assert cfg.critical_anchor_age_weight >= 0, "critical_anchor_age_weight must be non-negative!"
    assert cfg.critical_anchor_progress_weight >= 0, "critical_anchor_progress_weight must be non-negative!"
    assert cfg.post_rewind_exploration_steps >= 0, "post_rewind_exploration_steps must be non-negative!"
    assert cfg.post_rewind_xyz_noise_scale >= 0, "post_rewind_xyz_noise_scale must be non-negative!"
    assert cfg.post_rewind_rot_noise_scale >= 0, "post_rewind_rot_noise_scale must be non-negative!"
    assert cfg.post_rewind_action_samples >= 1, "post_rewind_action_samples must be at least 1!"
    assert 0 < cfg.post_rewind_noise_decay <= 1.0, "post_rewind_noise_decay must be in (0, 1]!"
    assert cfg.post_rewind_scene_escape_weight >= 0, "post_rewind_scene_escape_weight must be non-negative!"
    assert cfg.post_rewind_eef_escape_weight >= 0, "post_rewind_eef_escape_weight must be non-negative!"
    assert cfg.post_rewind_revisit_weight >= 0, "post_rewind_revisit_weight must be non-negative!"
    assert cfg.post_rewind_action_deviation_weight >= 0, (
        "post_rewind_action_deviation_weight must be non-negative!"
    )
    assert cfg.post_rewind_anchor_drift_weight >= 0, "post_rewind_anchor_drift_weight must be non-negative!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize structured result files
    episode_results_path = cfg.episode_results_path or os.path.join(cfg.local_log_dir, run_id + ".episodes.jsonl")
    summary_results_path = cfg.summary_results_path or os.path.join(cfg.local_log_dir, run_id + ".summary.json")
    episode_results_file = open(episode_results_path, "w")
    logger.info(f"Logging per-episode results to: {episode_results_path}")
    logger.info(f"Run summary will be saved to: {summary_results_path}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, episode_results_file, episode_results_path, summary_results_path, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def write_episode_result(result: Dict[str, Any], episode_results_file) -> None:
    """Persist a per-episode result record as JSONL."""
    if episode_results_file is None:
        return
    episode_results_file.write(json.dumps(result) + "\n")
    episode_results_file.flush()


def save_summary(summary: Dict[str, Any], summary_results_path: str) -> None:
    """Persist run summary as JSON."""
    with open(summary_results_path, "w") as f:
        json.dump(summary, f, indent=2)


def serialize_config(cfg: GenerateConfig) -> Dict[str, Any]:
    """Convert config dataclass to a JSON-serializable dictionary."""
    serialized_cfg = {}
    for key, value in vars(cfg).items():
        if isinstance(value, Path):
            serialized_cfg[key] = str(value)
        elif isinstance(value, Enum):
            serialized_cfg[key] = value.value
        else:
            serialized_cfg[key] = value
    return serialized_cfg


def load_records_from_path(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load episode records from a JSON or JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Episode record path does not exist: {path}")

    if path.suffix == ".jsonl":
        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with open(path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "episodes" in payload:
        return payload["episodes"]
    if isinstance(payload, list):
        return payload

    raise ValueError(f"Unsupported episode record format in {path}")


def resolve_episode_filter(cfg: GenerateConfig, log_file=None) -> Optional[Dict[int, List[int]]]:
    """Resolve optional task_id -> episode_idx filter used for targeted reruns."""
    filter_path = cfg.episode_filter_path or cfg.rerun_failed_episodes_from
    if filter_path is None:
        return None

    records = load_records_from_path(filter_path)
    if cfg.rerun_failed_episodes_from is not None:
        records = [record for record in records if record.get("success") is False]

    episodes_by_task = defaultdict(set)
    for record in records:
        if "task_id" not in record or "episode_idx" not in record:
            raise ValueError(
                f"Episode filter record must contain task_id and episode_idx. Offending record: {record}"
            )
        episodes_by_task[int(record["task_id"])].add(int(record["episode_idx"]))

    resolved_filter = {task_id: sorted(episode_indices) for task_id, episode_indices in episodes_by_task.items()}
    num_episodes = sum(len(episode_indices) for episode_indices in resolved_filter.values())
    log_message(
        f"Loaded {num_episodes} filtered episodes across {len(resolved_filter)} tasks from {filter_path}",
        log_file,
    )
    return resolved_filter


def get_episode_max_steps(cfg: GenerateConfig) -> int:
    """Resolve the episode horizon for the current run."""
    base_max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    if cfg.max_steps_override is not None:
        return cfg.max_steps_override
    if cfg.max_steps_multiplier != 1.0:
        return int(np.ceil(base_max_steps * cfg.max_steps_multiplier))
    return base_max_steps


def get_middle_state_trigger_step(cfg: GenerateConfig) -> int:
    """Resolve the policy-step index at which middle-state rewind should trigger."""
    if cfg.middle_state_time_seconds is not None:
        if cfg.middle_state_time_seconds < 0:
            return -1
        return int(round(cfg.middle_state_time_seconds * cfg.control_freq))
    return cfg.middle_state_step


def get_middle_state_repeat_interval_steps(cfg: GenerateConfig) -> Optional[int]:
    """Resolve the policy-step spacing between repeated rewinds."""
    if cfg.middle_state_repeat_interval_seconds is None or cfg.middle_state_repeat_interval_seconds <= 0:
        return None
    return int(round(cfg.middle_state_repeat_interval_seconds * cfg.control_freq))


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def teleport_robot_to_init_pose(env, log_file=None):
    """Teleport the robot arm back to its initial joint configuration without disturbing objects.

    Strategy:
      1. Use set_robot_joint_positions() which writes directly to sim.data.qpos and calls sim.forward().
         This only touches the robot's joint indices — all object qpos entries are left intact.
      2. Zero out joint velocities to prevent residual momentum from yanking the arm on the next step.
      3. Call reset_goal() on the OSC controller so it re-anchors its goal at the new EEF pose;
         without this the controller would compute a large error and cause the arm to "snap" violently.
      4. Force-update observables so the next observation reflects the new arm pose.
    """
    try:
        robot = env.robots[0]
        # 1. Teleport joints to initial config (also calls sim.forward() internally)
        robot.set_robot_joint_positions(np.array(robot.init_qpos))
        # 2. Zero joint velocities
        env.sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
        # 3. Re-anchor OSC controller goal to avoid snap-back torques
        if hasattr(robot, "controller") and hasattr(robot.controller, "reset_goal"):
            robot.controller.reset_goal()
        # 4. Refresh observables so next call to get_observation() is consistent
        env.env._update_observables(force=True)
        log_message("[middle_state] Teleported robot arm to initial joint configuration.", log_file)
    except Exception as e:
        log_message(f"[middle_state] Warning: teleportation failed ({e}). Continuing without reset.", log_file)


def get_eef_state(obs):
    """Extract end-effector state used for physical rewind control."""
    return (
        np.array(obs["robot0_eef_pos"], dtype=np.float64),
        quat2axisangle(np.array(obs["robot0_eef_quat"], dtype=np.float64)),
    )


def get_robot_init_eef_target(env, log_file=None):
    """Measure the true end-effector pose induced by robot.init_qpos without changing episode state."""
    sim_state = env.sim.get_state()
    qvel_snapshot = np.array(env.sim.data.qvel, copy=True)
    try:
        teleport_robot_to_init_pose(env, log_file=None)
        obs = env.env._get_observations(force_update=True)
        target_pos, target_axis_angle = get_eef_state(obs)
    finally:
        env.sim.set_state(sim_state)
        env.sim.forward()
        env.sim.data.qvel[:] = qvel_snapshot
        env.env._update_observables(force=True)

    log_message(
        f"[middle_state] Measured init-arm target pose from robot.init_qpos: pos={target_pos.round(4).tolist()}",
        log_file,
    )
    return target_pos, target_axis_angle


def maybe_record_transition_frame(cfg: GenerateConfig, replay_images, obs, resize_size) -> None:
    """Optionally append an image for reset-transition visualization."""
    if not cfg.middle_state_record_transition:
        return
    _, img = prepare_observation(obs, resize_size)
    replay_images.append(img)


def compute_vector_distance(lhs, rhs) -> float:
    """Compute a robust L2 distance between two optional vectors."""
    if lhs is None or rhs is None:
        return 0.0
    lhs = np.asarray(lhs, dtype=np.float64).reshape(-1)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if lhs.size == 0 or rhs.size == 0 or lhs.size != rhs.size:
        return 0.0
    return float(np.linalg.norm(lhs - rhs))


def get_scene_state_vector(obs) -> np.ndarray:
    """Build a compact scene-state vector from non-robot observation keys."""
    chunks = []
    for key, value in obs.items():
        if key in {"agentview_image", "robot0_eye_in_hand_image"}:
            continue
        if key.startswith("robot0_"):
            continue
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            chunks.append(np.asarray(value, dtype=np.float64).reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def get_anchor_buffer_maxlen(cfg: GenerateConfig) -> int:
    """Resolve ring-buffer length for anchor selection."""
    return max(1, int(np.ceil(cfg.anchor_buffer_seconds * cfg.control_freq)) + 1)


def append_anchor_record(
    anchor_buffer,
    env,
    obs,
    policy_step,
    prev_eef_pos,
    prev_scene_state,
    episode_start_scene_state=None,
):
    """Append one rewind-candidate anchor derived from the current observation."""
    robot = env.robots[0]
    eef_pos, _ = get_eef_state(obs)
    scene_state = get_scene_state_vector(obs)
    eef_speed = 0.0 if prev_eef_pos is None else compute_vector_distance(eef_pos, prev_eef_pos)
    if prev_scene_state is None:
        scene_motion = 0.0
    else:
        scene_motion = compute_vector_distance(scene_state, prev_scene_state)

    anchor_buffer.append(
        {
            "policy_step": policy_step,
            "arm_qpos": get_joint_state(robot),
            "eef_pos": eef_pos,
            "eef_speed": eef_speed,
            "scene_state": scene_state,
            "scene_motion": scene_motion,
            "scene_delta_from_start": compute_vector_distance(scene_state, episode_start_scene_state),
        }
    )
    return eef_pos, scene_state


def detect_stuck_region(
    cfg: GenerateConfig,
    anchor_buffer,
    current_policy_step: int,
    last_reset_policy_step: int,
    log_file=None,
):
    """Detect whether the recent trajectory has entered a stale, revisited region."""
    if not cfg.middle_state_trigger_on_stuck:
        return None
    if current_policy_step < cfg.stuck_min_policy_steps:
        return None
    if current_policy_step - last_reset_policy_step < cfg.middle_state_min_steps_between_resets:
        return None

    recent_records = list(anchor_buffer)[-cfg.stuck_window_steps :]
    if len(recent_records) < cfg.stuck_min_stale_steps:
        return None

    stale_flags = [
        (
            record["scene_motion"] <= cfg.stuck_scene_motion_threshold
            and record["eef_speed"] <= cfg.stuck_eef_speed_threshold
        )
        for record in recent_records
    ]
    stale_fraction = float(np.mean(stale_flags))
    current_record = recent_records[-1]
    window_start = recent_records[0]
    net_scene_delta = compute_vector_distance(current_record["scene_state"], window_start["scene_state"])
    net_eef_delta = compute_vector_distance(current_record["eef_pos"], window_start["eef_pos"])

    revisit_candidates = [
        record
        for record in anchor_buffer
        if current_policy_step - record["policy_step"] >= cfg.stuck_min_stale_steps
    ]
    if not revisit_candidates:
        return None

    revisit_pairs = [
        (
            compute_vector_distance(current_record["scene_state"], record["scene_state"]),
            compute_vector_distance(current_record["eef_pos"], record["eef_pos"]),
            record,
        )
        for record in revisit_candidates
    ]
    closest_scene_revisit, closest_eef_revisit, revisit_record = min(
        revisit_pairs,
        key=lambda item: (item[0], item[1]),
    )
    is_revisited = (
        closest_scene_revisit <= cfg.stuck_revisit_scene_threshold
        and closest_eef_revisit <= cfg.stuck_revisit_eef_threshold
    )
    if (
        stale_fraction < 0.8
        or net_scene_delta > cfg.stuck_net_scene_delta_threshold
        or net_eef_delta > cfg.stuck_net_eef_delta_threshold
        or (cfg.stuck_require_revisit and not is_revisited)
    ):
        return None

    stale_start_idx = len(recent_records) - cfg.stuck_min_stale_steps
    for start_idx in range(len(recent_records) - cfg.stuck_min_stale_steps + 1):
        suffix_flags = stale_flags[start_idx:]
        if float(np.mean(suffix_flags)) >= 0.85:
            stale_start_idx = start_idx
            break

    stale_window = recent_records[stale_start_idx:]
    stale_info = {
        "trigger_step": current_policy_step,
        "stale_start_step": stale_window[0]["policy_step"],
        "stale_fraction": stale_fraction,
        "net_scene_delta": net_scene_delta,
        "net_eef_delta": net_eef_delta,
        "closest_scene_revisit": closest_scene_revisit,
        "closest_eef_revisit": closest_eef_revisit,
        "revisit_step": revisit_record["policy_step"],
        "current_scene_state": np.array(current_record["scene_state"], dtype=np.float64),
        "current_eef_pos": np.array(current_record["eef_pos"], dtype=np.float64),
        "recent_scene_states": [np.array(record["scene_state"], dtype=np.float64) for record in stale_window],
    }
    log_message(
        "[middle_state] stale-state detector fired "
        f"(step={current_policy_step}, stale_start={stale_info['stale_start_step']}, "
        f"stale_fraction={stale_fraction:.2f}, net_scene_delta={net_scene_delta:.4f}, "
        f"net_eef_delta={net_eef_delta:.4f}, revisit_step={stale_info['revisit_step']}).",
        log_file,
    )
    return stale_info


def select_anchor_record(cfg: GenerateConfig, anchor_buffer, current_policy_step, log_file=None):
    """Pick a recent stable anchor for partial rewind."""
    min_age_steps = int(round(cfg.anchor_min_age_seconds * cfg.control_freq))
    max_age_steps = int(round(cfg.anchor_max_age_seconds * cfg.control_freq))
    candidates = []
    for record in anchor_buffer:
        age_steps = current_policy_step - record["policy_step"]
        if age_steps < min_age_steps or age_steps > max_age_steps:
            continue
        stability_cost = record["scene_motion"] + cfg.anchor_eef_speed_weight * record["eef_speed"]
        candidates.append((stability_cost, age_steps, record))

    if not candidates:
        log_message("[middle_state] anchor_rewind found no valid anchor in the age window; falling back to init pose.", log_file)
        return None

    if cfg.anchor_selection_strategy == "latest":
        _, age_steps, record = min(candidates, key=lambda item: item[1])
    else:
        _, age_steps, record = min(candidates, key=lambda item: (item[0], item[1]))

    log_message(
        "[middle_state] anchor_rewind selected anchor "
        f"at step {record['policy_step']} (age_steps={age_steps}, "
        f"scene_motion={record['scene_motion']:.4f}, eef_speed={record['eef_speed']:.4f}).",
        log_file,
    )
    return record


def select_critical_anchor_record(
    cfg: GenerateConfig,
    anchor_buffer,
    current_policy_step: int,
    stale_info: Optional[Dict[str, Any]],
    log_file=None,
):
    """Pick a rollback point just before the stale region while still preserving local scene context."""
    if stale_info is None:
        log_message("[middle_state] critical_rewind has no stale-region context; falling back to stable anchor.", log_file)
        return None

    min_age_steps = int(round(cfg.anchor_min_age_seconds * cfg.control_freq))
    max_age_steps = int(round(cfg.anchor_max_age_seconds * cfg.control_freq))
    latest_allowed_step = stale_info["stale_start_step"] - cfg.critical_anchor_pre_stale_margin_steps
    current_scene_state = stale_info["current_scene_state"]
    current_eef_pos = stale_info["current_eef_pos"]

    candidates = []
    for record in anchor_buffer:
        age_steps = current_policy_step - record["policy_step"]
        if age_steps < min_age_steps or age_steps > max_age_steps:
            continue
        if record["policy_step"] > latest_allowed_step:
            continue

        scene_escape = compute_vector_distance(record["scene_state"], current_scene_state)
        eef_escape = compute_vector_distance(record["eef_pos"], current_eef_pos)
        stability_cost = record["scene_motion"] + cfg.anchor_eef_speed_weight * record["eef_speed"]
        age_penalty = age_steps / max(max_age_steps, 1)
        progress_bonus = record.get("scene_delta_from_start", 0.0)
        score = (
            cfg.critical_anchor_scene_escape_weight * (scene_escape + 0.5 * eef_escape)
            - cfg.critical_anchor_stability_weight * stability_cost
            - cfg.critical_anchor_age_weight * age_penalty
            + cfg.critical_anchor_progress_weight * progress_bonus
        )
        candidates.append(
            (
                score,
                age_steps,
                scene_escape,
                eef_escape,
                stability_cost,
                progress_bonus,
                record,
            )
        )

    if not candidates:
        log_message("[middle_state] critical_rewind found no pre-stale anchor; falling back to stable anchor.", log_file)
        return None

    score, age_steps, scene_escape, eef_escape, stability_cost, progress_bonus, record = max(
        candidates,
        key=lambda item: (item[0], -item[1]),
    )
    log_message(
        "[middle_state] critical_rewind selected anchor "
        f"at step {record['policy_step']} (age_steps={age_steps}, score={score:.4f}, "
        f"scene_escape={scene_escape:.4f}, eef_escape={eef_escape:.4f}, "
        f"stability_cost={stability_cost:.4f}, progress_bonus={progress_bonus:.4f}).",
        log_file,
    )
    return record


def perturb_action_for_exploration(
    action: np.ndarray,
    cfg: GenerateConfig,
    rng,
    scale_multiplier: float = 1.0,
):
    """Inject local exploration noise after rewind to escape deterministic bad branches."""
    perturbed = np.array(action, copy=True)
    if cfg.post_rewind_xyz_noise_scale > 0:
        perturbed[:3] += rng.normal(0.0, cfg.post_rewind_xyz_noise_scale * scale_multiplier, size=3)
    if cfg.post_rewind_rot_noise_scale > 0:
        perturbed[3:6] += rng.normal(0.0, cfg.post_rewind_rot_noise_scale * scale_multiplier, size=3)
    perturbed[:6] = np.clip(perturbed[:6], -1.0, 1.0)
    perturbed[-1] = float(np.clip(perturbed[-1], 0.0, 1.0))
    return perturbed


def score_exploration_candidate(
    cfg: GenerateConfig,
    env,
    base_action: np.ndarray,
    candidate_action: np.ndarray,
    exploration_context: Optional[Dict[str, Any]],
):
    """Score one noisy action candidate using one-step simulator lookahead."""
    sim_state = env.sim.get_state()
    qvel_snapshot = np.array(env.sim.data.qvel, copy=True)
    robot = env.robots[0]
    try:
        processed_action = process_action(np.array(candidate_action, copy=True), cfg.model_family)
        next_obs, _, done, _ = env.step(processed_action.tolist())
        next_scene_state = get_scene_state_vector(next_obs)
        next_eef_pos, _ = get_eef_state(next_obs)

        stale_scene_state = None if exploration_context is None else exploration_context.get("stale_scene_state")
        stale_eef_pos = None if exploration_context is None else exploration_context.get("stale_eef_pos")
        anchor_scene_state = None if exploration_context is None else exploration_context.get("anchor_scene_state")
        recent_scene_states = [] if exploration_context is None else exploration_context.get("recent_scene_states", [])

        scene_escape = compute_vector_distance(next_scene_state, stale_scene_state)
        eef_escape = compute_vector_distance(next_eef_pos, stale_eef_pos)
        revisit_distance = 0.0
        if recent_scene_states:
            revisit_distance = min(compute_vector_distance(next_scene_state, scene_state) for scene_state in recent_scene_states)
        anchor_drift = compute_vector_distance(next_scene_state, anchor_scene_state)
        action_deviation = compute_vector_distance(candidate_action[:6], base_action[:6])
        score = (
            cfg.post_rewind_scene_escape_weight * scene_escape
            + cfg.post_rewind_eef_escape_weight * eef_escape
            + cfg.post_rewind_revisit_weight * revisit_distance
            - cfg.post_rewind_action_deviation_weight * action_deviation
            - cfg.post_rewind_anchor_drift_weight * anchor_drift
        )
        if done:
            score += 10.0
        metadata = {
            "score": score,
            "scene_escape": scene_escape,
            "eef_escape": eef_escape,
            "revisit_distance": revisit_distance,
            "anchor_drift": anchor_drift,
            "action_deviation": action_deviation,
            "would_succeed": bool(done),
        }
        return score, metadata
    finally:
        env.sim.set_state(sim_state)
        env.sim.forward()
        env.sim.data.qvel[:] = qvel_snapshot
        if hasattr(robot, "controller") and robot.controller is not None:
            robot.controller.update(force=True)
            if hasattr(robot.controller, "reset_goal"):
                robot.controller.reset_goal()
        env.env._update_observables(force=True)


def choose_post_rewind_action(
    cfg: GenerateConfig,
    env,
    action: np.ndarray,
    exploration_steps_remaining: int,
    exploration_rng,
    exploration_context: Optional[Dict[str, Any]],
):
    """Optionally sample and score a burst of noisy actions after rewind."""
    if exploration_steps_remaining <= 0:
        return np.array(action, copy=True), None

    total_burst_steps = max(1, cfg.post_rewind_exploration_steps)
    burst_step_idx = total_burst_steps - exploration_steps_remaining
    scale_multiplier = cfg.post_rewind_noise_decay ** burst_step_idx

    if cfg.post_rewind_action_samples <= 1:
        return perturb_action_for_exploration(action, cfg, exploration_rng, scale_multiplier=scale_multiplier), None

    candidates = [np.array(action, copy=True)]
    for _ in range(cfg.post_rewind_action_samples - 1):
        candidates.append(
            perturb_action_for_exploration(action, cfg, exploration_rng, scale_multiplier=scale_multiplier)
        )

    if not cfg.post_rewind_escape_lookahead:
        choice_idx = int(exploration_rng.integers(len(candidates)))
        return candidates[choice_idx], {"choice": choice_idx, "mode": "random_sample"}

    best_idx = 0
    best_score = -np.inf
    best_metadata = None
    for idx, candidate_action in enumerate(candidates):
        score, metadata = score_exploration_candidate(
            cfg,
            env,
            base_action=np.array(action, copy=True),
            candidate_action=candidate_action,
            exploration_context=exploration_context,
        )
        if score > best_score:
            best_idx = idx
            best_score = score
            best_metadata = metadata

    if best_metadata is not None:
        best_metadata["choice"] = best_idx
        best_metadata["mode"] = "lookahead"
    return candidates[best_idx], best_metadata


def get_joint_state(robot):
    """Read the current robot arm joint positions directly from sim."""
    return np.array(robot.sim.data.qpos[robot._ref_joint_pos_indexes], dtype=np.float64)


def build_joint_rewind_controller(robot):
    """Construct a temporary joint-position controller that targets init_qpos."""
    controller_cfg = {
        "sim": robot.sim,
        "eef_name": robot.gripper.important_sites["grip_site"],
        "eef_rot_offset": robot.eef_rot_offset,
        "joint_indexes": {
        "joints": robot.joint_indexes,
        "qpos": robot._ref_joint_pos_indexes,
        "qvel": robot._ref_joint_vel_indexes,
        },
        "actuator_range": robot.torque_limits,
        "policy_freq": robot.control_freq,
        "ndim": len(robot.robot_joints),
        "interpolation": "linear",
        "ramp_ratio": 0.2,
        "input_max": 1,
        "input_min": -1,
        "output_max": 0.05,
        "output_min": -0.05,
        "kp": 50,
        "damping_ratio": 1,
        "impedance_mode": "fixed",
        "qpos_limits": None,
    }
    return controller_factory("JOINT_POSITION", controller_cfg)


def physically_rewind_robot_joints(
    cfg: GenerateConfig,
    env,
    replay_images,
    resize_size,
    target_qpos=None,
    target_label: str = "init pose",
    log_file=None,
):
    """Move only the robot arm back toward robot.init_qpos using joint-space torque control."""
    if cfg.middle_state_rewind_max_steps == 0:
        obs = env.env._get_observations(force_update=True)
        return obs, False

    robot = env.robots[0]
    sim_env = env.env
    joint_controller = build_joint_rewind_controller(robot)
    if target_qpos is None:
        target_qpos = np.array(robot.init_qpos, dtype=np.float64)
    else:
        target_qpos = np.array(target_qpos, dtype=np.float64)
    low, high = robot.torque_limits
    control_steps = int(sim_env.control_timestep / sim_env.model_timestep)
    total_steps = 0

    while total_steps < cfg.middle_state_rewind_max_steps:
        joint_err = target_qpos - get_joint_state(robot)
        joint_err_norm = float(np.linalg.norm(joint_err))
        if joint_err_norm <= cfg.middle_state_rewind_joint_tol:
            log_message(
                f"[middle_state] joint_rewind converged to {target_label} in {total_steps} env steps "
                f"(joint_err={joint_err_norm:.4f}).",
                log_file,
            )
            break

        policy_step = True
        for _ in range(control_steps):
            env.sim.forward()
            if policy_step:
                joint_controller.set_goal(np.zeros(len(target_qpos), dtype=np.float32), set_qpos=target_qpos)

            torques = joint_controller.run_controller()
            robot.torques = np.clip(torques, low, high)
            env.sim.data.ctrl[robot._ref_joint_actuator_indexes] = robot.torques
            if robot.has_gripper:
                robot.grip_action(robot.gripper, np.array([-1.0], dtype=np.float32))

            env.sim.step()
            sim_env._update_observables(force=True)
            policy_step = False

        total_steps += 1
        obs = sim_env._get_observations(force_update=True)
        maybe_record_transition_frame(cfg, replay_images, obs, resize_size)

    if hasattr(robot, "controller") and robot.controller is not None:
        robot.controller.update(force=True)
        if hasattr(robot.controller, "reset_goal"):
            robot.controller.reset_goal()

    obs = sim_env._get_observations(force_update=True)
    final_joint_err = float(np.linalg.norm(target_qpos - get_joint_state(robot)))
    log_message(
        f"[middle_state] joint_rewind finished after {total_steps} env steps toward {target_label} "
        f"(joint_err={final_joint_err:.4f}).",
        log_file,
    )
    return obs, False


def physically_rewind_robot(
    cfg: GenerateConfig,
    env,
    obs,
    initial_eef_pos,
    initial_eef_axis_angle,
    replay_images,
    resize_size,
    log_file=None,
):
    """Move the robot back toward its initial pose using real actions instead of teleportation."""
    if cfg.middle_state_rewind_max_steps == 0:
        return obs, False

    def step_rewind_action(action: np.ndarray, reason: str):
        nonlocal obs, total_steps
        obs, _, done, _ = env.step(action.tolist())
        total_steps += 1
        maybe_record_transition_frame(cfg, replay_images, obs, resize_size)
        if done:
            log_message(f"[middle_state] servo_rewind terminated the episode during {reason}.", log_file)
            return True
        return False

    total_steps = 0

    # Open the gripper first so the rewind tests "manual arm return" more directly than action replay.
    for _ in range(cfg.middle_state_gripper_open_steps):
        open_action = np.zeros(7, dtype=np.float32)
        open_action[-1] = -1.0
        if step_rewind_action(open_action, "gripper-open"):
            return obs, True
        if total_steps >= cfg.middle_state_rewind_max_steps:
            log_message("[middle_state] servo_rewind stopped during gripper-open due to step budget.", log_file)
            return obs, False

    # Optional retreat before phased homing.
    for _ in range(cfg.middle_state_rewind_lift_steps):
        lift_action = np.zeros(7, dtype=np.float32)
        lift_action[2] = cfg.middle_state_rewind_pos_clip
        lift_action[-1] = -1.0
        if step_rewind_action(lift_action, "lift"):
            return obs, True
        if total_steps >= cfg.middle_state_rewind_max_steps:
            log_message("[middle_state] servo_rewind stopped during lift due to step budget.", log_file)
            return obs, False

    current_pos, _ = get_eef_state(obs)
    clearance_z = max(current_pos[2], initial_eef_pos[2]) + cfg.middle_state_rewind_clearance

    while total_steps < cfg.middle_state_rewind_max_steps:
        current_pos, current_axis_angle = get_eef_state(obs)
        rot_error = initial_eef_axis_angle - current_axis_angle
        xy_error = initial_eef_pos[:2] - current_pos[:2]
        z_to_clearance = clearance_z - current_pos[2]
        z_to_home = initial_eef_pos[2] - current_pos[2]
        xy_norm = float(np.linalg.norm(xy_error))
        z_home_abs = float(abs(z_to_home))
        rot_norm = float(np.linalg.norm(rot_error))

        if (
            xy_norm <= cfg.middle_state_rewind_xy_tol
            and z_home_abs <= cfg.middle_state_rewind_z_tol
            and rot_norm <= cfg.middle_state_rewind_rot_tol
        ):
            log_message(
                f"[middle_state] servo_rewind converged in {total_steps} env steps "
                f"(xy_err={xy_norm:.4f}, z_err={z_home_abs:.4f}, rot_err={rot_norm:.4f}).",
                log_file,
            )
            return obs, False

        action = np.zeros(7, dtype=np.float32)
        action[-1] = -1.0

        # Phase 1: go up to a safe clearance before moving laterally.
        if z_to_clearance > cfg.middle_state_rewind_z_tol:
            action[2] = min(cfg.middle_state_rewind_pos_clip, z_to_clearance * cfg.middle_state_rewind_pos_gain)
            if step_rewind_action(action, "clearance-lift"):
                return obs, True
            continue

        # Phase 2: move back in XY while staying near the clearance height.
        if xy_norm > cfg.middle_state_rewind_xy_tol:
            action[0] = np.clip(
                xy_error[0] * cfg.middle_state_rewind_pos_gain,
                -cfg.middle_state_rewind_pos_clip,
                cfg.middle_state_rewind_pos_clip,
            )
            action[1] = np.clip(
                xy_error[1] * cfg.middle_state_rewind_pos_gain,
                -cfg.middle_state_rewind_pos_clip,
                cfg.middle_state_rewind_pos_clip,
            )
            action[2] = np.clip(
                z_to_clearance * cfg.middle_state_rewind_pos_gain,
                -0.5 * cfg.middle_state_rewind_pos_clip,
                0.5 * cfg.middle_state_rewind_pos_clip,
            )
            if step_rewind_action(action, "xy-homing"):
                return obs, True
            continue

        # Phase 3: align orientation close to the original pose while still above the scene.
        if rot_norm > cfg.middle_state_rewind_rot_tol:
            action[3:6] = np.clip(
                rot_error * cfg.middle_state_rewind_rot_gain,
                -cfg.middle_state_rewind_rot_clip,
                cfg.middle_state_rewind_rot_clip,
            )
            action[2] = np.clip(
                z_to_clearance * cfg.middle_state_rewind_pos_gain,
                -0.5 * cfg.middle_state_rewind_pos_clip,
                0.5 * cfg.middle_state_rewind_pos_clip,
            )
            if step_rewind_action(action, "rotation-align"):
                return obs, True
            continue

        # Phase 4: descend once XY and orientation are already near the initial pose.
        action[2] = np.clip(
            z_to_home * cfg.middle_state_rewind_pos_gain,
            -cfg.middle_state_rewind_pos_clip,
            cfg.middle_state_rewind_pos_clip,
        )
        if step_rewind_action(action, "z-descent"):
            return obs, True

    current_pos, current_axis_angle = get_eef_state(obs)
    final_xy_err = float(np.linalg.norm(initial_eef_pos[:2] - current_pos[:2]))
    final_z_err = float(abs(initial_eef_pos[2] - current_pos[2]))
    final_rot_err = float(np.linalg.norm(initial_eef_axis_angle - current_axis_angle))
    log_message(
        f"[middle_state] servo_rewind hit max steps={cfg.middle_state_rewind_max_steps} "
        f"(xy_err={final_xy_err:.4f}, z_err={final_z_err:.4f}, rot_err={final_rot_err:.4f}).",
        log_file,
    )
    return obs, False


def execute_middle_state_reset(
    cfg: GenerateConfig,
    env,
    obs,
    executed_actions,
    anchor_buffer,
    current_policy_step,
    replay_images,
    resize_size,
    initial_sim_state,
    initial_eef_pos,
    initial_eef_axis_angle,
    stale_info: Optional[Dict[str, Any]] = None,
    trigger_reason: str = "scheduled",
    log_file=None,
):
    """Apply the configured middle-state reset and return a fresh observation plus done flag."""
    reset_metadata = {
        "trigger_reason": trigger_reason,
        "trigger_step": current_policy_step,
        "strategy": cfg.middle_state_strategy,
    }
    if stale_info is not None:
        reset_metadata.update(
            {
                "stale_start_step": int(stale_info["stale_start_step"]),
                "stale_fraction": float(stale_info["stale_fraction"]),
                "stale_net_scene_delta": float(stale_info["net_scene_delta"]),
                "stale_net_eef_delta": float(stale_info["net_eef_delta"]),
                "stale_revisit_step": int(stale_info["revisit_step"]),
            }
        )

    selected_anchor_record = None
    if cfg.middle_state_strategy == "joint_rewind":
        log_message("[middle_state] Reset strategy=joint_rewind.", log_file)
        obs, done = physically_rewind_robot_joints(
            cfg,
            env,
            replay_images,
            resize_size,
            target_qpos=None,
            target_label="init pose",
            log_file=log_file,
        )
        if done:
            return obs, True, reset_metadata, selected_anchor_record
    elif cfg.middle_state_strategy == "anchor_rewind":
        log_message("[middle_state] Reset strategy=anchor_rewind.", log_file)
        selected_anchor_record = select_anchor_record(cfg, anchor_buffer, current_policy_step, log_file=log_file)
        obs, done = physically_rewind_robot_joints(
            cfg,
            env,
            replay_images,
            resize_size,
            target_qpos=None if selected_anchor_record is None else selected_anchor_record["arm_qpos"],
            target_label="anchor pose" if selected_anchor_record is not None else "init pose fallback",
            log_file=log_file,
        )
        if done:
            return obs, True, reset_metadata, selected_anchor_record
    elif cfg.middle_state_strategy == "critical_rewind":
        log_message("[middle_state] Reset strategy=critical_rewind.", log_file)
        selected_anchor_record = select_critical_anchor_record(
            cfg,
            anchor_buffer,
            current_policy_step,
            stale_info=stale_info,
            log_file=log_file,
        )
        if selected_anchor_record is None:
            selected_anchor_record = select_anchor_record(cfg, anchor_buffer, current_policy_step, log_file=log_file)
        obs, done = physically_rewind_robot_joints(
            cfg,
            env,
            replay_images,
            resize_size,
            target_qpos=None if selected_anchor_record is None else selected_anchor_record["arm_qpos"],
            target_label="critical anchor pose" if selected_anchor_record is not None else "init pose fallback",
            log_file=log_file,
        )
        if done:
            return obs, True, reset_metadata, selected_anchor_record
    elif cfg.middle_state_strategy == "state_restore":
        log_message("[middle_state] Reset strategy=state_restore.", log_file)
        obs = env.regenerate_obs_from_state(initial_sim_state)
        maybe_record_transition_frame(cfg, replay_images, obs, resize_size)
        return obs, False, reset_metadata, selected_anchor_record
    elif cfg.middle_state_strategy == "inverse_replay":
        log_message(
            "[middle_state] Reset strategy=inverse_replay. This mode is legacy and can disturb object state.",
            log_file,
        )
        for i, action_to_rewind in enumerate(reversed(executed_actions)):
            inverse_action = -action_to_rewind
            obs, _, done, _ = env.step(inverse_action.tolist())
            if i % 5 == 0:
                maybe_record_transition_frame(cfg, replay_images, obs, resize_size)
            if done:
                log_message("[middle_state] inverse_replay terminated the episode during rewind.", log_file)
                return obs, True, reset_metadata, selected_anchor_record
    elif cfg.middle_state_strategy == "servo_rewind":
        log_message("[middle_state] Reset strategy=servo_rewind.", log_file)
        obs, done = physically_rewind_robot(
            cfg,
            env,
            obs,
            initial_eef_pos,
            initial_eef_axis_angle,
            replay_images,
            resize_size,
            log_file=log_file,
        )
        if done:
            return obs, True, reset_metadata, selected_anchor_record
    else:
        log_message("[middle_state] Reset strategy=teleport.", log_file)
        teleport_robot_to_init_pose(env, log_file)
        obs = env.get_observation()
        maybe_record_transition_frame(cfg, replay_images, obs, resize_size)

    for _ in range(cfg.middle_state_settle_steps):
        obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
        maybe_record_transition_frame(cfg, replay_images, obs, resize_size)
        if done:
            log_message("[middle_state] reset settle phase terminated the episode.", log_file)
            return obs, True, reset_metadata, selected_anchor_record

    if selected_anchor_record is not None:
        reset_metadata.update(
            {
                "anchor_step": int(selected_anchor_record["policy_step"]),
                "anchor_scene_motion": float(selected_anchor_record["scene_motion"]),
                "anchor_eef_speed": float(selected_anchor_record["eef_speed"]),
                "anchor_scene_delta_from_start": float(selected_anchor_record.get("scene_delta_from_start", 0.0)),
            }
        )

    return obs, False, reset_metadata, selected_anchor_record


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    max_steps=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(
            f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
            f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
            "both speed and success rate), we recommend executing the full action chunk."
        )
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    replay_images = []
    executed_actions = []
    max_steps = get_episode_max_steps(cfg) if max_steps is None else max_steps
    warmup_steps = 0
    policy_steps_executed = 0
    middle_state_resets_done = 0
    initial_eef_pos = None
    initial_eef_axis_angle = None
    next_middle_state_trigger_step = get_middle_state_trigger_step(cfg)
    middle_state_repeat_interval_steps = get_middle_state_repeat_interval_steps(cfg)
    initial_sim_state = None
    last_reset_policy_step = -max(1, cfg.middle_state_min_steps_between_resets)
    anchor_buffer = deque(maxlen=get_anchor_buffer_maxlen(cfg))
    prev_anchor_eef_pos = None
    prev_anchor_scene_state = None
    episode_start_scene_state = None
    exploration_steps_remaining = 0
    exploration_rng = np.random.default_rng(cfg.seed)
    exploration_context = None
    exploration_events = []
    reset_events = []

    # Run episode
    success = False
    error_message = None
    try:
        while warmup_steps < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            warmup_steps += 1

        initial_sim_state = env.get_sim_state()
        initial_eef_pos, initial_eef_axis_angle = get_robot_init_eef_target(env, log_file=log_file)
        episode_start_scene_state = get_scene_state_vector(obs)
        prev_anchor_eef_pos, prev_anchor_scene_state = append_anchor_record(
            anchor_buffer,
            env,
            obs,
            policy_steps_executed,
            prev_anchor_eef_pos,
            prev_anchor_scene_state,
            episode_start_scene_state=episode_start_scene_state,
        )

        while policy_steps_executed < max_steps:

            # --- Middle-state rewind ---
            scheduled_reset = (
                next_middle_state_trigger_step >= 0
                and middle_state_resets_done < cfg.middle_state_max_resets
                and policy_steps_executed == next_middle_state_trigger_step
            )
            stale_info = None
            if middle_state_resets_done < cfg.middle_state_max_resets:
                stale_info = detect_stuck_region(
                    cfg,
                    anchor_buffer,
                    policy_steps_executed,
                    last_reset_policy_step,
                    log_file=log_file,
                )

            if scheduled_reset or stale_info is not None:
                trigger_reason = "scheduled"
                if stale_info is not None and scheduled_reset:
                    trigger_reason = "scheduled+stuck"
                elif stale_info is not None:
                    trigger_reason = "stuck"
                log_message(
                    f"[middle_state] Reached reset trigger ({trigger_reason}) at step {policy_steps_executed}; "
                    "applying mid-episode reset.",
                    log_file,
                )
                obs, done, reset_metadata, selected_anchor_record = execute_middle_state_reset(
                    cfg,
                    env,
                    obs,
                    executed_actions,
                    anchor_buffer,
                    policy_steps_executed,
                    replay_images,
                    resize_size,
                    initial_sim_state,
                    initial_eef_pos,
                    initial_eef_axis_angle,
                    stale_info=stale_info,
                    trigger_reason=trigger_reason,
                    log_file=log_file,
                )
                reset_events.append(reset_metadata)
                log_message("[middle_state] Finished mid-episode reset.", log_file)
                if done:
                    success = True
                    break

                # Clear stale action chunks so the VLA policy re-queries from the reset state.
                action_queue.clear()
                exploration_steps_remaining = cfg.post_rewind_exploration_steps
                exploration_context = None
                if exploration_steps_remaining > 0:
                    exploration_context = {
                        "total_steps": cfg.post_rewind_exploration_steps,
                        "stale_scene_state": None if stale_info is None else np.array(stale_info["current_scene_state"], copy=True),
                        "stale_eef_pos": None if stale_info is None else np.array(stale_info["current_eef_pos"], copy=True),
                        "anchor_scene_state": None
                        if selected_anchor_record is None
                        else np.array(selected_anchor_record["scene_state"], copy=True),
                        "recent_scene_states": []
                        if stale_info is None
                        else [np.array(scene_state, copy=True) for scene_state in stale_info["recent_scene_states"]],
                    }

                middle_state_resets_done += 1
                last_reset_policy_step = policy_steps_executed
                if scheduled_reset:
                    if (
                        middle_state_repeat_interval_steps is not None
                        and middle_state_resets_done < cfg.middle_state_max_resets
                    ):
                        next_middle_state_trigger_step = policy_steps_executed + middle_state_repeat_interval_steps
                        log_message(
                            f"[middle_state] Scheduled next reset at step {next_middle_state_trigger_step} "
                            f"({middle_state_resets_done}/{cfg.middle_state_max_resets} used).",
                            log_file,
                        )
                    else:
                        next_middle_state_trigger_step = -1
                elif (
                    middle_state_repeat_interval_steps is not None
                    and middle_state_resets_done < cfg.middle_state_max_resets
                ):
                    next_middle_state_trigger_step = policy_steps_executed + middle_state_repeat_interval_steps

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()
            action, exploration_metadata = choose_post_rewind_action(
                cfg,
                env,
                np.array(action, copy=True),
                exploration_steps_remaining,
                exploration_rng,
                exploration_context,
            )
            if exploration_metadata is not None:
                exploration_events.append(
                    {
                        "policy_step": int(policy_steps_executed),
                        "burst_step": int(cfg.post_rewind_exploration_steps - exploration_steps_remaining),
                        "steps_remaining": int(exploration_steps_remaining),
                        **{
                            key: (
                                bool(value)
                                if isinstance(value, (bool, np.bool_))
                                else int(value)
                                if isinstance(value, (int, np.integer))
                                else float(value)
                                if isinstance(value, (float, np.floating))
                                else value
                            )
                            for key, value in exploration_metadata.items()
                        },
                    }
                )

            # Process action
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            executed_actions.append(action)
            policy_steps_executed += 1
            if exploration_steps_remaining > 0:
                exploration_steps_remaining -= 1
                if exploration_steps_remaining == 0:
                    exploration_context = None
            prev_anchor_eef_pos, prev_anchor_scene_state = append_anchor_record(
                anchor_buffer,
                env,
                obs,
                policy_steps_executed,
                prev_anchor_eef_pos,
                prev_anchor_scene_state,
                episode_start_scene_state=episode_start_scene_state,
            )
            if done:
                success = True
                break

    except Exception as e:
        error_message = str(e)
        log_message(f"Episode error: {e}", log_file)

    return {
        "success": success,
        "replay_images": replay_images,
        "policy_steps_executed": policy_steps_executed,
        "max_steps": max_steps,
        "error": error_message,
        "timed_out": (not success) and (error_message is None) and (policy_steps_executed >= max_steps),
        "reset_events": reset_events,
        "exploration_events": exploration_events,
        "num_resets": len(reset_events),
    }


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    episode_filter=None,
    episode_results_file=None,
    log_file=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    episode_indices = episode_filter.get(task_id) if episode_filter is not None else list(range(cfg.num_trials_per_task))
    if episode_indices is None:
        env.close()
        return total_episodes, total_successes, {
            "task_id": task_id,
            "task_description": task_description,
            "episodes_evaluated": 0,
            "successes": 0,
            "success_rate": 0.0,
            "max_steps": get_episode_max_steps(cfg),
        }

    max_steps = get_episode_max_steps(cfg)
    log_message(f"Resolved episode horizon for task {task_id}: {max_steps}", log_file)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(episode_indices):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                write_episode_result(
                    {
                        "task_suite_name": cfg.task_suite_name,
                        "task_id": task_id,
                        "task_description": task_description,
                        "episode_idx": episode_idx,
                        "success": None,
                        "episode_status": "skipped_failed_expert_demo",
                    },
                    episode_results_file,
                )
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        episode_outcome = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            max_steps=max_steps,
            log_file=log_file,
        )
        success = episode_outcome["success"]
        replay_images = episode_outcome["replay_images"]

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        video_path = None
        if cfg.save_rollout_videos:
            video_path = save_rollout_video(
                replay_images,
                total_episodes,
                success=success,
                task_description=task_description,
                log_file=log_file,
                run_id_note=cfg.run_id_note,
                rollout_dir_override=cfg.rollout_dir_override,
            )

        episode_record = {
            "task_suite_name": cfg.task_suite_name,
            "task_id": task_id,
            "task_description": task_description,
            "episode_idx": episode_idx,
            "success": success,
            "episode_status": "evaluated",
            "policy_steps_executed": episode_outcome["policy_steps_executed"],
            "num_steps_wait": cfg.num_steps_wait,
            "max_steps": episode_outcome["max_steps"],
            "timed_out": episode_outcome["timed_out"],
            "error": episode_outcome["error"],
            "num_resets": episode_outcome["num_resets"],
            "reset_events": episode_outcome["reset_events"],
            "exploration_events": episode_outcome["exploration_events"],
            "video_path": video_path,
        }
        write_episode_result(episode_record, episode_results_file)

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    env.close()

    return total_episodes, total_successes, {
        "task_id": task_id,
        "task_description": task_description,
        "episodes_evaluated": task_episodes,
        "successes": task_successes,
        "success_rate": task_success_rate,
        "max_steps": max_steps,
    }


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    (
        log_file,
        local_log_filepath,
        episode_results_file,
        episode_results_path,
        summary_results_path,
        run_id,
    ) = setup_logging(cfg)
    episode_filter = resolve_episode_filter(cfg, log_file)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_ids = sorted(episode_filter.keys()) if episode_filter is not None else list(range(num_tasks))
    for task_id in task_ids:
        assert 0 <= task_id < num_tasks, f"Task id {task_id} is outside valid range [0, {num_tasks})"

    task_results = []
    for task_id in tqdm.tqdm(task_ids):
        total_episodes, total_successes, task_result = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            episode_filter,
            episode_results_file,
            log_file,
        )
        task_results.append(task_result)

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    save_summary(
        {
            "run_id": run_id,
            "task_suite_name": cfg.task_suite_name,
            "pretrained_checkpoint": str(cfg.pretrained_checkpoint),
            "config": serialize_config(cfg),
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "overall_success_rate": final_success_rate,
            "task_results": task_results,
            "episode_results_path": episode_results_path,
            "log_path": local_log_filepath,
        },
        summary_results_path,
    )

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if episode_results_file:
        episode_results_file.close()
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
