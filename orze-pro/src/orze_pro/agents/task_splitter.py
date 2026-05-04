"""Multi-task project initialization — detect tasks in GOAL.md and scaffold subfolders.

CALLING SPEC:
    detect_tasks(goal_text) -> list[dict]
        Use LLM to analyze GOAL.md and detect multiple tasks.
        Returns list of dicts with keys: name, title, metric, sort, description,
        directions, dependencies, compute_weight. Falls back to single task on LLM failure.

    propose_gpu_allocation(tasks, n_gpus) -> list[dict]
        Assign GPUs proportionally by task complexity. Each task gets >= 1 GPU.
        Returns tasks with 'gpus' field added.

    generate_task_files(task, project_dir, gpu_info) -> dict
        Call LLM to produce per-task scaffold files. Returns {filename: content}.

    format_proposal(tasks, project_dir) -> str
        Human-readable summary of the proposed multi-task split.

    create_task_folders(tasks, project_dir, env_path) -> None
        Write all generated files, create symlinks, and emit helper scripts.
"""

from __future__ import annotations

from orze_pro._gate import require_license; require_license()

import logging
import os
import re
import stat
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orze.task_splitter")

# ---------------------------------------------------------------------------
# Known metrics and their default sort order
# ---------------------------------------------------------------------------

_METRIC_SORT = {
    "nds": "descending",
    "map": "descending",
    "mAP": "descending",
    "amota": "descending",
    "miou": "descending",
    "mIoU": "descending",
    "test_accuracy": "descending",
    "accuracy": "descending",
    "f1": "descending",
    "auc": "descending",
    "bleu": "descending",
    "rouge": "descending",
    "loss": "ascending",
    "test_loss": "ascending",
    "wer": "ascending",
    "cer": "ascending",
    "fid": "ascending",
}

# Task complexity heuristic for GPU allocation (higher = more GPUs)
_TASK_WEIGHT = {
    "3d_detection": 4,
    "detection": 3,
    "2d_detection": 2,
    "object_detection": 3,
    "segmentation": 2,
    "tracking": 1,
    "classification": 2,
    "image_classification": 2,
    "generation": 3,
    "translation": 2,
    "summarization": 2,
}


# ---------------------------------------------------------------------------
# detect_tasks — LLM-based task detection
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Convert a title to a filesystem-safe slug."""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _infer_sort(metric: str) -> str:
    """Infer sort direction from metric name."""
    key = metric.lower().replace("-", "").replace("_", "")
    for m, sort in _METRIC_SORT.items():
        if m.lower().replace("-", "").replace("_", "") == key:
            return sort
    return "descending"


_DETECT_TASKS_PROMPT = """\
You are an ML project analyzer. Read the following GOAL.md and identify ALL distinct tasks/objectives.

For each task, extract:
- name: filesystem-safe slug (e.g. "3d_detection", "tracking", "image_classification")
- title: human-readable title
- metric: primary evaluation metric (e.g. "NDS", "mAP", "AMOTA", "mIoU", "test_accuracy")
- sort: "descending" if higher is better, "ascending" if lower is better
- description: 1-2 sentence summary of what this task does
- directions: key research approaches to try
- dependencies: list of other task names this depends on (empty list if none)
- compute_weight: integer 1-5, how much GPU compute this task needs relative to others (5=most, 1=least)

Return ONLY a JSON array. No markdown fences, no explanation.

Example output:
[
  {"name": "3d_detection", "title": "3D Object Detection", "metric": "NDS", "sort": "descending", "description": "Detect 3D bounding boxes from LiDAR point clouds", "directions": "CenterPoint, PointPillars, BEVFusion", "dependencies": [], "compute_weight": 4},
  {"name": "tracking", "title": "Multi-Object Tracking", "metric": "AMOTA", "sort": "descending", "description": "Track objects across frames using detection results", "directions": "Greedy matching, Hungarian, Kalman filter", "dependencies": ["3d_detection"], "compute_weight": 1}
]

If there is only ONE task, return a single-element array.

GOAL.md:
---
{goal_text}
---
"""


def detect_tasks(goal_text: str) -> list[dict]:
    """Detect tasks in GOAL.md using an LLM.

    Calls the LLM to understand the goal text and identify distinct tasks,
    their metrics, dependencies, and compute requirements. Falls back to
    a single-task result if the LLM call fails.
    """
    import json as _json

    from orze_pro.agents._llm_utils import (
        call_first_available_llm,
        load_env_file,
    )

    load_env_file()

    prompt = _DETECT_TASKS_PROMPT.replace("{goal_text}", goal_text[:4000])
    raw = call_first_available_llm(prompt)

    if raw:
        # Parse JSON from response
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        try:
            tasks = _json.loads(cleaned)
            if isinstance(tasks, list) and len(tasks) > 0:
                # Validate and normalize each task
                valid_tasks = []
                for t in tasks:
                    if not isinstance(t, dict) or "name" not in t:
                        continue
                    valid_tasks.append({
                        "name": _slugify(t.get("name", "task")),
                        "title": t.get("title", t.get("name", "Task")),
                        "metric": t.get("metric", "test_accuracy"),
                        "sort": t.get("sort", _infer_sort(t.get("metric", ""))),
                        "description": t.get("description", ""),
                        "directions": t.get("directions", ""),
                        "dependencies": t.get("dependencies", []),
                        "compute_weight": t.get("compute_weight", 2),
                    })
                if valid_tasks:
                    logger.info("LLM detected %d tasks: %s",
                                len(valid_tasks),
                                ", ".join(t["name"] for t in valid_tasks))
                    return valid_tasks
        except _json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON for task detection")

    # Fallback: treat entire GOAL.md as a single task
    logger.info("Task detection fallback: treating as single task")
    metric = _extract_metric(goal_text) or "test_accuracy"
    return [{
        "name": "main",
        "title": "Main Task",
        "metric": metric,
        "sort": _infer_sort(metric),
        "description": goal_text[:500].strip(),
        "directions": _extract_directions(goal_text),
        "dependencies": [],
        "compute_weight": 2,
    }]


def _extract_metric(text: str) -> str:
    """Extract a metric name from a text block (fallback helper)."""
    m = re.search(r"\*\*Metric\*\*\s*:\s*(\S+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"-\s*Metric\s*:\s*(\S+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return "test_accuracy"


def _extract_directions(text: str) -> str:
    """Extract research directions (fallback helper)."""
    m = re.search(r"\*\*Directions?\*\*\s*:\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# propose_gpu_allocation
# ---------------------------------------------------------------------------

def propose_gpu_allocation(tasks: list[dict], n_gpus: int) -> list[dict]:
    """Assign GPUs to tasks proportionally by estimated complexity.

    Each task gets at least 1 GPU.  Remaining GPUs are distributed by weight.
    """
    if n_gpus < 1:
        n_gpus = 1

    # Resolve weights — prefer LLM-provided compute_weight, fallback to heuristic
    weights = []
    for t in tasks:
        w = t.get("compute_weight") or _TASK_WEIGHT.get(t["name"], 2)
        weights.append(w)

    total_weight = sum(weights)
    n_tasks = len(tasks)

    if n_gpus <= n_tasks:
        # Not enough GPUs for weighted allocation — 1 each (some may share)
        gpu_idx = 0
        for t in tasks:
            t["gpus"] = [gpu_idx % n_gpus]
            gpu_idx += 1
        return tasks

    # Allocate: 1 base + proportional share of remainder
    remainder = n_gpus - n_tasks
    alloc = []
    for w in weights:
        extra = round(remainder * w / total_weight) if total_weight > 0 else 0
        alloc.append(1 + extra)

    # Fix rounding so sum == n_gpus
    diff = sum(alloc) - n_gpus
    if diff > 0:
        # Remove from largest allocations first
        for _ in range(diff):
            idx = alloc.index(max(alloc))
            alloc[idx] -= 1
    elif diff < 0:
        for _ in range(-diff):
            idx = alloc.index(min(alloc))
            alloc[idx] += 1

    # Assign GPU indices
    gpu_cursor = 0
    for i, t in enumerate(tasks):
        count = alloc[i]
        t["gpus"] = list(range(gpu_cursor, gpu_cursor + count))
        gpu_cursor += count

    return tasks


# ---------------------------------------------------------------------------
# generate_task_files — LLM-assisted file generation
# ---------------------------------------------------------------------------

_TRAIN_PY_TEMPLATE = '''\
#!/usr/bin/env python3
"""Training script for {task_title}.

TODO: Implement your training logic here.

Orze contract:
    python train.py --idea-id <id> --results-dir <dir> --ideas-md <file> --config <yaml>
    Output: results/{{idea_id}}/metrics.json
"""
import argparse
import json
import re
import time
from pathlib import Path
import yaml


def deep_merge(base, override):
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_idea_config(ideas_md, idea_id):
    text = Path(ideas_md).read_text()
    pattern = rf"^##\\s+{{re.escape(idea_id)}}\\b.*?```ya?ml\\s*\\n(.*?)```"
    m = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if not m:
        raise ValueError(f"Idea {{idea_id}} not found in {{ideas_md}}")
    return yaml.safe_load(m.group(1)) or {{}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idea-id", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--ideas-md", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    idea_dir = Path(args.results_dir) / args.idea_id
    idea_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    base_cfg = yaml.safe_load(Path(args.config).read_text()) or {{}}
    idea_cfg = {{}}
    idea_cfg_path = idea_dir / "idea_config.yaml"
    if idea_cfg_path.exists():
        idea_cfg = yaml.safe_load(idea_cfg_path.read_text()) or {{}}
    try:
        idea_cfg = deep_merge(load_idea_config(args.ideas_md, args.idea_id), idea_cfg)
    except ValueError:
        pass
    cfg = deep_merge(base_cfg, idea_cfg)

    t0 = time.time()

    # ========================================
    # TODO: Your training logic here
    #
    # Task: {task_title}
    # Primary metric: {primary_metric}
    # Directions: {directions}
    #
    # 'cfg' contains the merged config from base.yaml + idea YAML
    # Use CUDA_VISIBLE_DEVICES (set by orze) for GPU selection
    #
    # Must write metrics.json when done:
    #   {{"status": "COMPLETED", "{primary_metric}": <value>, ...}}
    #   or {{"status": "FAILED", "error": "<message>"}}
    # ========================================

    metrics = {{
        "status": "FAILED",
        "error": "Training not implemented yet. Edit train.py to add your training logic.",
        "training_time": round(time.time() - t0, 2),
    }}

    (idea_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Metrics: {{json.dumps(metrics, indent=2)}}")


if __name__ == "__main__":
    main()
'''


def _build_file_gen_prompt(task: dict, gpu_info: dict) -> str:
    """Build LLM prompt to generate task-specific scaffold files."""
    gpu_section = ""
    if gpu_info:
        gpu_section = (
            f"GPU: {gpu_info.get('gpu_name', 'unknown')}, "
            f"{gpu_info.get('vram_mb', 0)} MB VRAM, "
            f"{len(task.get('gpus', [1]))} GPUs assigned"
        )

    return f"""\
You are an ML research advisor. Generate scaffold files for an automated experiment task.

Task: {task['title']}
Metric: {task['metric']} ({task['sort']})
Description: {task.get('description', '')}
Research directions: {task.get('directions', '')}
{gpu_section}

Generate the following files as a JSON object where keys are filenames and values are file contents (strings).

1. "GOAL.md" — A focused GOAL.md for this specific task (2-3 paragraphs).

2. "RESEARCH_RULES.md" — Research directions and constraints for the research agent.
   Include published baselines with real numbers, suggest 3-5 research directions.

3. "ideas.md" — 3-5 seed experiment ideas in orze format:
   ## idea-XXXX: Title
   - **Priority**: high|medium|low
   ```yaml
   key: value
   ```

4. "configs/base.yaml" — Reasonable defaults for this task and hardware.
   Include model, training, and data config sections.

Return ONLY valid JSON. No markdown fences, no explanation.
"""


def generate_task_files(task: dict, project_dir: Path, gpu_info: dict) -> dict:
    """Generate per-task scaffold files via LLM.

    Returns dict of {{filename: content}}.  Falls back to templates on LLM failure.
    """
    import json as _json

    from orze_pro.agents._llm_utils import (
        call_first_available_llm,
        load_env_file,
    )

    load_env_file()

    prompt = _build_file_gen_prompt(task, gpu_info)
    raw = call_first_available_llm(prompt)

    files: dict[str, str] = {}

    if raw:
        # Try to parse JSON from response (may be wrapped in ```json ... ```)
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        try:
            files = _json.loads(cleaned)
        except _json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON for task %s — using templates",
                           task["name"])

    # Always generate train.py from template (not LLM)
    files["train.py"] = _TRAIN_PY_TEMPLATE.format(
        task_title=task["title"],
        primary_metric=task["metric"],
        directions=task.get("directions", ""),
    )

    # Ensure critical files exist with fallbacks
    if "GOAL.md" not in files:
        files["GOAL.md"] = (
            f"# {task['title']}\n\n"
            f"{task.get('description', 'TODO: describe this task.')}\n\n"
            f"**Primary metric**: {task['metric']} ({task['sort']})\n\n"
            f"**Directions**: {task.get('directions', 'TODO')}\n"
        )

    if "RESEARCH_RULES.md" not in files:
        files["RESEARCH_RULES.md"] = (
            f"# Research Rules — {task['title']}\n\n"
            f"Primary metric: {task['metric']} ({task['sort']})\n\n"
            f"## Directions\n\n{task.get('directions', 'TODO: add research directions')}\n"
        )

    if "ideas.md" not in files:
        files["ideas.md"] = (
            f"# Ideas — {task['title']}\n\n"
            f"## idea-0001: Baseline\n"
            f"- **Priority**: high\n\n"
            f"```yaml\nlearning_rate: 0.001\nepochs: 10\n```\n"
        )

    if "configs/base.yaml" not in files:
        files["configs/base.yaml"] = (
            f"# Base config — {task['title']}\n"
            f"seed: 42\n"
        )

    return files


# ---------------------------------------------------------------------------
# format_proposal
# ---------------------------------------------------------------------------

def format_proposal(tasks: list[dict], project_dir: Path) -> str:
    """Format the proposed multi-task split as a human-readable string."""
    lines = [f"\nFound {len(tasks)} tasks in GOAL.md:\n"]

    for i, t in enumerate(tasks, 1):
        gpus = t.get("gpus", [])
        if len(gpus) == 0:
            gpu_str = "no GPUs"
        elif len(gpus) == 1:
            gpu_str = f"GPU {gpus[0]}"
        else:
            gpu_str = f"GPUs {gpus[0]}-{gpus[-1]}"

        lines.append(
            f"  {i}. {t['name']}/ ({gpu_str}, primary metric: {t['metric']})"
        )
        if t.get("title"):
            lines.append(f"     {t['title']}")
        lines.append("")

    lines.append("Proceed? [Y/n/edit]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# orze.yaml generator for a task subfolder
# ---------------------------------------------------------------------------

def _generate_orze_yaml(task: dict, project_dir: Path) -> str:
    """Generate orze.yaml content for a single task subfolder."""
    gpu_list = task.get("gpus", [0])
    cuda_devices = ",".join(str(g) for g in gpu_list)

    # Detect LLM backend
    backend = ("anthropic" if os.environ.get("ANTHROPIC_API_KEY") else
               "gemini" if os.environ.get("GEMINI_API_KEY") else
               "openai" if os.environ.get("OPENAI_API_KEY") else "ollama")

    content = f"""\
# orze.yaml — {task['title']}
# Auto-generated by orze-pro task splitter

# --- REQUIRED ---
train_script: train.py
ideas_file: ideas.md
results_dir: results
python: venv/bin/python3

# --- GPU ---
train_extra_env:
  PYTHONUNBUFFERED: "1"
  CUDA_VISIBLE_DEVICES: "{cuda_devices}"

# --- TIMEOUTS ---
timeout: 3600
poll: 30
stall_minutes: 60
max_idea_failures: 2
max_fix_attempts: 2

# --- REPORT ---
report:
  primary_metric: {task['metric']}
  sort: {task['sort']}
  columns:
    - {{key: "{task['metric']}", label: "{task['metric']}", fmt: ".4f"}}
    - {{key: "training_time", label: "Time(s)", fmt: ".0f"}}

# --- RETROSPECTION ---
retrospection:
  enabled: true
  interval: 6
  auto_pause: false
  plateau_window: 20
  fail_window: 10
  fail_threshold: 0.5

# --- ROLES (orze-pro autopilot) ---
# Roles compose from bundled static SOPs. 'orze sop list' to discover.
roles:
  research:
    mode: research
    backend: {backend}
    skills:
      - "@sop:research_base"
      - ./RESEARCH_RULES.md
    cooldown: 120
    timeout: 600
  professor:
    mode: claude
    skills:
      - "@sop:professor_base"
      - "@sop:professor_web_search"
      - "@sop:professor_cross_domain_query"
      - "@sop:professor_idea_review"
      - "@sop:professor_diversity_enforcement"
      - "@sop:professor_gap_closure"
      - "@sop:professor_strategy_review"
      - "@sop:professor_regression_detection"
      - "@sop:professor_steering"
    cooldown: 600
    timeout: 600
    model: opus
    pausable: false
    allowed_tools: "Read,Write,Edit,Glob,Grep,Bash"
"""
    return content


# ---------------------------------------------------------------------------
# create_task_folders
# ---------------------------------------------------------------------------

def create_task_folders(
    tasks: list[dict],
    project_dir: Path,
    env_path: Path,
) -> None:
    """Create per-task subfolders with all scaffold files.

    Also creates symlinks for .env and shared data, plus helper scripts.
    """
    from orze_pro.agents._llm_utils import detect_gpu_info, load_env_file

    load_env_file(str(env_path))
    gpu_info = detect_gpu_info()

    for task in tasks:
        task_dir = project_dir / task["name"]
        task_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Setting up {task['name']}/...")

        # Generate files via LLM
        files = generate_task_files(task, project_dir, gpu_info)

        # Write each file
        for filename, content in files.items():
            fpath = task_dir / filename
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content, encoding="utf-8")
            print(f"    \033[32mcreated\033[0m  {task['name']}/{filename}")

        # orze.yaml
        orze_yaml = _generate_orze_yaml(task, project_dir)
        (task_dir / "orze.yaml").write_text(orze_yaml, encoding="utf-8")
        print(f"    \033[32mcreated\033[0m  {task['name']}/orze.yaml")

        # results dir
        (task_dir / "results").mkdir(exist_ok=True)

        # Symlink .env
        env_link = task_dir / ".env"
        if not env_link.exists() and env_path.exists():
            try:
                env_link.symlink_to(env_path.resolve())
                print(f"    \033[32mlinked\033[0m   {task['name']}/.env -> {env_path}")
            except OSError as e:
                logger.warning("Failed to symlink .env: %s", e)

        # Symlink shared data directories
        for data_dir_name in ("data", "datasets", "checkpoints", "pretrained"):
            src = project_dir / data_dir_name
            dst = task_dir / data_dir_name
            if src.is_dir() and not dst.exists():
                try:
                    dst.symlink_to(src.resolve())
                    print(f"    \033[32mlinked\033[0m   {task['name']}/{data_dir_name}/ -> {src}")
                except OSError:
                    pass

        # Professor behavior ships as bundled static SOPs under
        # orze_pro/sops/professor_*.skill.md — nothing to bootstrap
        # per task. Subtasks that need task-specific professor tailoring
        # author dynamic SOPs at <task>/skills/<name>.skill.md.

    # --- Director setup (cross-task orchestration) ---
    if len(tasks) > 1:
        _setup_director(tasks, project_dir)

    # --- Helper scripts ---
    _write_helper_scripts(tasks, project_dir)


def _setup_director(tasks: list[dict], project_dir: Path) -> None:
    """Set up the cross-task director for multi-task projects."""
    import shutil

    # 1. Copy director_role.py from orze-pro package
    try:
        import orze_pro
        director_src = Path(orze_pro.__file__).parent / "agents" / "director.py"
        director_dst = project_dir / "director_role.py"
        if director_src.exists() and not director_dst.exists():
            shutil.copy2(director_src, director_dst)
            print(f"  \033[32mcreated\033[0m  director_role.py")
    except Exception as e:
        logger.warning("Could not copy director_role.py: %s", e)

    # 2. Create director.yaml with task definitions
    tasks_cfg = {}
    for t in tasks:
        task_entry = {
            "target_metric": t.get("metric", "score"),
            "target_value": 0.0,  # LLM will set expectations
            "default_gpus": t.get("gpus", [0]),
        }
        # Add dependencies
        deps = t.get("dependencies", [])
        if deps:
            task_entry["depends_on"] = {}
            for dep in deps:
                task_entry["depends_on"][dep] = {
                    "metric": "score",
                    "min_value": 0.5,
                    "action": "switch_detection_source",
                }
        tasks_cfg[t["name"]] = task_entry

    director_cfg = {
        "project_dir": str(project_dir),
        "tasks_dir": "tasks",
        "tasks": tasks_cfg,
        "backend": "gemini",
        "model": "gemini-3.1-pro-preview",
        "idle_minutes_for_realloc": 30,
        "plateau_window": 20,
    }

    import yaml as _yaml
    cfg_path = project_dir / "director.yaml"
    if not cfg_path.exists():
        cfg_path.write_text(
            _yaml.dump(director_cfg, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        print(f"  \033[32mcreated\033[0m  director.yaml")

    # 3. Create director_orze.yaml
    orze_cfg_path = project_dir / "director_orze.yaml"
    if not orze_cfg_path.exists():
        orze_cfg_path.write_text(
            "# Director — cross-task orchestrator, no training\n"
            "train_script: /dev/null\n"
            "ideas_file: /dev/null\n"
            "results_dir: director_results\n"
            f"python: {project_dir / 'venv' / 'bin' / 'python3'}\n"
            "timeout: 3600\n"
            "poll: 600\n\n"
            "roles:\n"
            "  director:\n"
            "    mode: script\n"
            "    script: director_role.py\n"
            '    args: ["--once"]\n'
            "    cooldown: 600\n"
            "    timeout: 120\n",
            encoding="utf-8",
        )
        print(f"  \033[32mcreated\033[0m  director_orze.yaml")

    # 4. Create director_results dir
    (project_dir / "director_results").mkdir(exist_ok=True)

    print(f"\n  Director configured for {len(tasks)} tasks.")
    print(f"  Launch with: orze -c director_orze.yaml")


def _write_helper_scripts(tasks: list[dict], project_dir: Path) -> None:
    """Write start_all.sh, stop_all.sh, and status.sh."""
    task_names = [t["name"] for t in tasks]

    # start_all.sh
    start_lines = ["#!/usr/bin/env bash", "# Start all orze tasks", f'DIR="$(cd "$(dirname "$0")" && pwd)"', ""]
    for name in task_names:
        start_lines.append(f'echo "Starting {name}..."')
        start_lines.append(f'cd "$DIR/{name}" && orze start -c orze.yaml &')
    start_lines.append("")
    start_lines.append("wait")
    start_lines.append(f'echo "All {len(task_names)} tasks started."')
    _write_script(project_dir / "start_all.sh", "\n".join(start_lines))

    # stop_all.sh
    stop_lines = ["#!/usr/bin/env bash", "# Stop all orze tasks", f'DIR="$(cd "$(dirname "$0")" && pwd)"', ""]
    for name in task_names:
        stop_lines.append(f'echo "Stopping {name}..."')
        stop_lines.append(f'cd "$DIR/{name}" && orze --stop 2>/dev/null')
    stop_lines.append("")
    stop_lines.append(f'echo "All tasks stopped."')
    _write_script(project_dir / "stop_all.sh", "\n".join(stop_lines))

    # status.sh
    status_lines = ["#!/usr/bin/env bash", "# Show leaderboard for all tasks", f'DIR="$(cd "$(dirname "$0")" && pwd)"', ""]
    for name in task_names:
        status_lines.append(f'echo "=== {name} ==="')
        status_lines.append(f'cd "$DIR/{name}" && orze --leaderboard -c orze.yaml 2>/dev/null || echo "  (not running)"')
        status_lines.append("")
    _write_script(project_dir / "status.sh", "\n".join(status_lines))

    print(f"\n  \033[32mcreated\033[0m  start_all.sh, stop_all.sh, status.sh")


def _write_script(path: Path, content: str) -> None:
    """Write a shell script and make it executable."""
    path.write_text(content + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
