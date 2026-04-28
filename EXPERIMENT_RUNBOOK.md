# OpenVLA LIBERO Rewind Experiment Runbook

Last updated: 2026-04-28 KST

This document captures the current project context for the OpenVLA-OFT LIBERO
rewind/recovery experiments: where code is edited, where experiments run, what
algorithms have been tried, where logs live, and how to resume/monitor runs.

## Repository Context

- Local project root:
  - `/Users/demian/Library/Mobile Documents/com~apple~CloudDocs/10_Research/70_openvla_rewind/openvla-oft-1`
- Remote container project root:
  - `/workspace/TTS_VLA/openvla-oft-1`
- GitHub remote used for syncing:
  - `JunHoo-Lee/openvla-oft`
- Current relevant commits:
  - `797101c Add guarded stuck detector sweep`
  - `3159b6b Include reset count one in progressive sweep`
  - `773caa7 Expand progressive rewind ladder sweep`
  - `bde3ac5 Add progressive adaptive rewind sweep`
  - `cfe4ed4 Add abstention sweep queue runner`

The intended workflow is local-first. Edit locally, commit and push, then the
remote container pulls. Do not treat the remote checkout as the source of truth.

## Server And Runtime

- SSH target for the container:
  - `Gwanggyo3-junhoo-container`
- Physical host:
  - `Gwanggyo3-RTXA6000-4X`
- Container:
  - `junhoo_container`
- Remote container hostname seen during latest check:
  - `14b882d74094`
- GPU environment:
  - 4 x NVIDIA RTX A6000
- Remote Python:
  - `/workspace/TTS_VLA/openvla-oft-1/.venv/bin/python`
- Persistent run tmux session:
  - `openvla_run`
  - windows: `goal`, `spatial`, `object`, `libero10`
- Helper attach command from local repo:
  - `./scripts/gwanggyo3_openvla_session.sh`
- Host maintenance attach command:
  - `./scripts/gwanggyo3_openvla_session.sh host`
- Watchdog behavior:
  - Host cron pulls remote repo with `git pull --ff-only origin main`.
  - If `nvidia-smi` fails inside `junhoo_container`, host watchdog restarts the container.
  - Logs:
    - `/home/user/openvla_watchdog/git_pull.log`
    - `/home/user/openvla_watchdog/watchdog.log`

## Evaluation Set

All main comparisons below use the `failed_75` episode filter:

- `libero_goal`: 17 episodes
- `libero_spatial`: 33 episodes
- `libero_object`: 4 episodes
- `libero_10`: 21 episodes
- total: 75 episodes

Filter files live under:

- `experiments/robot/libero/episode_filters/failed_75/`

The common checkpoint is:

- `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`

## Implemented Recovery Algorithms

### No Rewind Baseline

Longer-horizon rerun with no recovery intervention.

Important result:

```text
no_rewind_3x: 39/75 = 52.0%
  goal 12/17, spatial 9/33, object 4/4, libero_10 14/21

no_rewind_4x: 39/75 = 52.0%
  goal 12/17, spatial 9/33, object 4/4, libero_10 14/21
```

This is the current score to beat.

### Anchor Sweep

Earlier anchor-rewind experiments rolled back to recent stable trajectory
anchors and optionally added Gaussian post-rewind noise.

Observed results:

```text
anchor_sweep_3x_a2_det:   32/75 = 42.7%
anchor_sweep_3x_a2_noise: 32/75 = 42.7%
anchor_sweep_3x_a4_det:   29/75 = 38.7%
anchor_sweep_3x_a4_noise: 30/75 = 40.0%
```

Conclusion: stable-anchor rollback often reused bad local basins, especially in
`libero_10`.

### Critical Rewind

Implemented `critical_rewind`: detect stale/no-progress behavior, roll back to
a pre-stale anchor, then optionally use post-rewind noisy candidate sampling.

Main code:

- `experiments/robot/libero/run_libero_eval.py`
- `experiments/robot/libero/critical_rewind_policy.py`
- `experiments/robot/libero/run_libero_critical_rewind_eval.py`

Conclusion: helpful conceptually, but generic stale detection still caused
phase-breaking resets and did not beat no-rewind.

### Progressive Rewind

Implemented `progressive_rewind`: repeated stale triggers escalate through a
ladder:

```text
retreat -> micro_anchor -> stable_anchor -> home
```

Key behavior:

- First intervention is a small upward retreat.
- If still stuck, use a recent micro anchor.
- If still stuck, use a farther stable anchor.
- If still stuck, optionally home/reset the arm.
- If scene progress increases enough, reset the escalation ladder back to level 0.

Main code:

- `experiments/robot/libero/run_libero_eval.py`
- `experiments/robot/libero/critical_rewind_policy.py`
- `experiments/robot/libero/run_libero_progressive_sweep.py`

Important result from max-reset priority sweep:

```text
best progressive max-reset result: 39/75 = 52.0%

h3p0 m1: best 39/75
h3p0 m2: best 39/75
h3p0 m3: best 39/75
h3p0 m4: best 39/75
h4p0 m1: best 39/75
```

Increasing `max_resets` created more opportunities to escape stuck states, but
also increased the chance of breaking long-horizon task phase in `libero_10`.

Example m4 tradeoff:

```text
h3p0 m4 medium/off:
  39/75
  goal 13/17, spatial 9/33, object 4/4, libero_10 13/21
  gained one goal case, lost one libero_10 case
```

### Guarded Stuck Sweep

Implemented a more conservative stuck-trigger sweep:

- `experiments/robot/libero/run_libero_guarded_stuck_sweep.py`

This tests whether higher reset budgets help if reset is only allowed after
stronger no-progress evidence.

Profiles:

- `guarded`
  - longer stale window
  - higher minimum stale duration
  - later minimum trigger step
  - longer reset cooldown
  - smaller progress veto threshold
- `ultra`
  - even more conservative

Current result snapshot:

```text
guarded sweep queue:
  129 done, 0 failed, 4 running, out of 136 jobs

completed guarded configs:
  31 complete configs so far
  all 31 completed configs scored 39/75 = 52.0%
```

Representative completed configs:

```text
guarded_r_m_s_h_h3p0_m4_guarded_burstoff:
  39/75
  goal 12/17, spatial 9/33, object 4/4, libero_10 14/21
  reset_eps 3, total resets 5

guarded_r_m_s_h_h3p0_m4_ultra_burstoff:
  39/75
  reset_eps 2, total resets 3
```

Interpretation:

- Guarded detection successfully suppresses false/phase-breaking resets.
- But it becomes close to no-rewind behavior and still does not beat 39/75.
- The remaining missing ingredient is not just stricter triggering; the policy
  needs to know whether a candidate reset is actually better before committing.

## Current Best Understanding

Current best score:

```text
39/75 = 52.0%
```

Methods at this score:

- no-rewind 3x
- no-rewind 4x
- multiple guarded progressive variants
- multiple max-reset progressive variants

What we learned:

- Blind or scheduled rewind hurts.
- Anchor rewind can preserve local geometry, but can also re-enter a bad local basin.
- Progressive reset reduces the blast radius of reset, but higher reset counts can
  break long-horizon task phase.
- Guarded stuck detection prevents most harmful resets, but also removes most
  useful recovery opportunities.
- Threshold-only stuck detection appears insufficient to beat no-rewind.

Recommended next algorithmic direction:

```text
candidate reset verification before commit
```

That means: when stuck is detected, simulate/evaluate one or more candidate
recoveries briefly, then commit only if a progress/phase proxy improves. This is
different from only tuning thresholds.

## Important Result Directories

Remote paths:

```text
experiments/logs/libero_no_rewind_3x_failed75
experiments/logs/libero_abstention_sweep_failed75
experiments/logs/libero_progressive_maxreset_sweep_failed75
experiments/logs/libero_guarded_stuck_sweep_failed75
experiments/logs/libero_progressive_all_sweep_failed75
```

Visual qualitative checks:

```text
visual_checks/anchor_vs_x1
visual_checks/libero10_rewind_losses
```

Key qualitative examples:

- `libero_10 task 3 ep 24`
  - black bowl bottom drawer + close
  - recovery can break drawer/bowl phase and cause failure
- `libero_10 task 6 ep 28`
  - mug/plate + chocolate pudding
  - repeated reset often loses the second subgoal phase
- `libero_10 task 8 ep 8`
  - moka pots on stove
  - some progressive settings gain this episode

## How To Run Sweeps

All commands below are intended to run inside the remote container:

```bash
cd /workspace/TTS_VLA/openvla-oft-1
```

### Guarded Stuck Sweep

Create manifest:

```bash
OUT=./experiments/logs/libero_guarded_stuck_sweep_failed75
mkdir -p "$OUT"
.venv/bin/python experiments/robot/libero/run_libero_guarded_stuck_sweep.py \
  --create_manifest \
  --output_dir "$OUT" \
  --manifest_path "$OUT/manifest.jsonl"
```

Run one worker:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  experiments/robot/libero/run_libero_guarded_stuck_sweep.py \
  --worker \
  --worker_id guarded_gpu0 \
  --output_dir ./experiments/logs/libero_guarded_stuck_sweep_failed75 \
  --manifest_path ./experiments/logs/libero_guarded_stuck_sweep_failed75/manifest.jsonl
```

Run four workers in `openvla_run` tmux windows:

```bash
for idx in 0 1 2 3; do
  tmux send-keys -t openvla_run:${idx} "cd /workspace/TTS_VLA/openvla-oft-1" Enter
  tmux send-keys -t openvla_run:${idx} \
    "CUDA_VISIBLE_DEVICES=${idx} .venv/bin/python experiments/robot/libero/run_libero_guarded_stuck_sweep.py --worker --worker_id guarded_gpu${idx} --output_dir ./experiments/logs/libero_guarded_stuck_sweep_failed75 --manifest_path ./experiments/logs/libero_guarded_stuck_sweep_failed75/manifest.jsonl 2>&1 | tee experiments/logs/libero_guarded_stuck_sweep_failed75/worker_gpu${idx}.log" Enter
done
```

### Progressive Max-Reset Sweep

Main runner:

```bash
.venv/bin/python experiments/robot/libero/run_libero_progressive_sweep.py
```

This reports the full config/job count. As of the latest code it includes
`max_resets = 1, 2, 3, 4` across progressive ladders.

Priority max-reset output used in the last comparison:

```text
experiments/logs/libero_progressive_maxreset_sweep_failed75
```

## How To Monitor

Queue status:

```bash
OUT=./experiments/logs/libero_guarded_stuck_sweep_failed75
printf "done="; find "$OUT"/queue -name "*.done.json" | wc -l
printf "failed="; find "$OUT"/queue -name "*.failed.json" | wc -l
printf "locks="; find "$OUT"/queue -name "*.lock" | wc -l
find "$OUT"/queue -name "*.lock" -printf "%f\n" | sort
```

GPU status:

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
```

Active episode progress:

```bash
OUT=/workspace/TTS_VLA/openvla-oft-1/experiments/logs/libero_guarded_stuck_sweep_failed75
.venv/bin/python - <<'PY'
from pathlib import Path
import json

OUT = Path("/workspace/TTS_VLA/openvla-oft-1/experiments/logs/libero_guarded_stuck_sweep_failed75")
for lock in sorted((OUT / "queue").glob("*.lock")):
    job = lock.name[:-5]
    if "__" not in job:
        continue
    cfg, suite = job.rsplit("__", 1)
    epfile = OUT / cfg / f"{suite}.episodes.jsonl"
    n = s = reset_eps = resets = 0
    if epfile.exists():
        for line in epfile.read_text().splitlines():
            if not line.strip():
                continue
            ep = json.loads(line)
            n += 1
            s += bool(ep.get("success"))
            evs = ep.get("reset_events") or []
            reset_eps += bool(evs)
            resets += len(evs)
    print(job, "episodes", n, "success", s, "reset_eps", reset_eps, "resets", resets)
PY
```

## How To Summarize Results

Run inside the remote repo. Change `OUT` to the sweep directory of interest.

```bash
OUT=/workspace/TTS_VLA/openvla-oft-1/experiments/logs/libero_guarded_stuck_sweep_failed75
.venv/bin/python - <<'PY'
from pathlib import Path
import json

OUT = Path("/workspace/TTS_VLA/openvla-oft-1/experiments/logs/libero_guarded_stuck_sweep_failed75")
SUITES = ["libero_goal", "libero_spatial", "libero_object", "libero_10"]

bycfg = {}
for p in OUT.glob("*/*.summary.json"):
    cfg = p.parent.name
    suite = p.name.replace(".summary.json", "")
    data = json.loads(p.read_text())
    bycfg.setdefault(cfg, {})[suite] = data

rows = []
for cfg, suites in bycfg.items():
    succ = sum(d["total_successes"] for d in suites.values())
    n = sum(d["total_episodes"] for d in suites.values())
    complete = all(s in suites for s in SUITES)
    parts = []
    for s in SUITES:
        if s in suites:
            parts.append(f"{s.replace('libero_', '')}={suites[s]['total_successes']}/{suites[s]['total_episodes']}")
    rows.append((complete, succ, n, cfg, " ".join(parts)))

for complete, succ, n, cfg, suite_str in sorted(rows, key=lambda x: (-x[1] / max(x[2], 1), x[3])):
    tag = "COMPLETE" if complete else "PARTIAL"
    print(f"{tag:8s} {cfg:60s} {succ:2d}/{n:<2d} {100 * succ / max(n, 1):5.1f}% {suite_str}")
PY
```

## Verification Commands

Run locally before pushing:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m py_compile \
  experiments/robot/libero/run_libero_eval.py \
  experiments/robot/libero/critical_rewind_policy.py \
  experiments/robot/libero/run_libero_progressive_sweep.py \
  experiments/robot/libero/run_libero_guarded_stuck_sweep.py
git diff --check
```

Run remotely after pull:

```bash
cd /workspace/TTS_VLA/openvla-oft-1
.venv/bin/python -m unittest discover -s tests -p "test_*.py"
.venv/bin/python -m py_compile \
  experiments/robot/libero/run_libero_eval.py \
  experiments/robot/libero/critical_rewind_policy.py \
  experiments/robot/libero/run_libero_progressive_sweep.py \
  experiments/robot/libero/run_libero_guarded_stuck_sweep.py
```

## Current Live State At Last Update

Remote guarded sweep:

```text
experiments/logs/libero_guarded_stuck_sweep_failed75
done=129
failed=0
locks=4
```

Active lock examples at last update:

```text
guarded_r_m_s_h_h4p0_m1_ultra_burstsafe__libero_10
nr_h3p0__libero_10
nr_h3p0__libero_spatial
nr_h4p0__libero_goal
```

Remote GPU status at last update showed four RTX A6000 GPUs available with the
guarded sweep still running.
