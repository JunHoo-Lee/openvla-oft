"""Pure policy helpers for progress-gated critical rewind."""

from typing import Dict, Iterable, List, Sequence, Tuple


PROGRESSIVE_RESET_MODES = {"retreat", "micro_anchor", "stable_anchor", "home"}


def compute_progress_veto(progress_values: Iterable[float], min_delta: float) -> Dict[str, float | bool]:
    """Return whether recent progress is large enough to veto recovery."""
    values = [float(value) for value in progress_values]
    if not values:
        return {"veto": False, "progress_delta": 0.0}

    progress_delta = values[-1] - values[0]
    return {
        "veto": progress_delta >= float(min_delta),
        "progress_delta": progress_delta,
    }


def recovery_gate_decision(
    *,
    scene_escape: float,
    eef_escape: float,
    progress_loss: float,
    stability_cost: float,
    scene_escape_weight: float,
    eef_escape_weight: float,
    progress_loss_weight: float,
    stability_weight: float,
    min_advantage: float,
) -> Dict[str, float | bool]:
    """Score whether rewind has enough state-space advantage to justify intervention."""
    advantage = (
        float(scene_escape_weight) * float(scene_escape)
        + float(eef_escape_weight) * float(eef_escape)
        - float(progress_loss_weight) * max(0.0, float(progress_loss))
        - float(stability_weight) * float(stability_cost)
    )
    return {
        "apply": advantage >= float(min_advantage),
        "advantage": advantage,
        "progress_loss": max(0.0, float(progress_loss)),
    }


def choose_candidate_with_margin(scores: Sequence[float], margin: float) -> int:
    """Choose a non-base candidate only when it beats the base score by margin."""
    if not scores:
        raise ValueError("scores must contain at least one candidate")

    base_score = float(scores[0])
    best_idx = 0
    best_score = base_score
    for idx, score in enumerate(scores[1:], start=1):
        score = float(score)
        if score > best_score:
            best_idx = idx
            best_score = score

    if best_idx != 0 and best_score >= base_score + float(margin):
        return best_idx
    return 0


def parse_progressive_levels(level_spec: str) -> List[str]:
    """Parse and validate a comma-separated progressive reset ladder."""
    levels = [level.strip() for level in str(level_spec).split(",") if level.strip()]
    if not levels:
        raise ValueError("progressive reset ladder must contain at least one level")
    unknown = [level for level in levels if level not in PROGRESSIVE_RESET_MODES]
    if unknown:
        raise ValueError(f"unknown progressive reset mode(s): {unknown}")
    return levels


def choose_progressive_reset_mode(levels: Sequence[str], reset_level: int) -> Tuple[int, str]:
    """Choose the current progressive reset mode, clamping beyond the last level."""
    if not levels:
        raise ValueError("levels must contain at least one reset mode")
    level_idx = max(0, min(int(reset_level), len(levels) - 1))
    return level_idx, str(levels[level_idx])


def should_reset_progressive_ladder(
    *,
    current_progress: float,
    reference_progress: float,
    threshold: float,
) -> bool:
    """Return whether enough new progress occurred to restart the ladder at level 0."""
    return float(current_progress) - float(reference_progress) >= float(threshold)
