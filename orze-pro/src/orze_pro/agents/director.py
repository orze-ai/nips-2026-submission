#!/usr/bin/env python3
"""Cross-task director role for orze.

Generic: reads task definitions, targets, and dependencies from director.yaml.
Runs as `mode: script` in a top-level orze instance.

Config (director.yaml):
    project_dir: /path/to/project
    tasks_dir: tasks
    tasks:
      task_name:
        target_metric: NDS
        target_value: 0.60
        default_gpus: [0,1,2,3]
        depends_on:
          other_task:
            metric: NDS
            min_value: 0.50
            action: switch_detection_source
    backend: gemini
    model: gemini-3.1-pro-preview
    idle_minutes_for_realloc: 30
    plateau_window: 20
"""
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Action dedup cache — prevent repeating the same action within 30 minutes
# ---------------------------------------------------------------------------
_last_actions: dict[str, float] = {}  # key -> timestamp
_ACTION_COOLDOWN = 1800  # 30 minutes

def _action_key(action: dict) -> str:
    """Create a dedup key for an action."""
    act = action.get("action", "")
    task = action.get("task", "")
    msg = action.get("message", "")[:40]
    return f"{act}:{task}:{msg}"

def _is_action_recent(action: dict) -> bool:
    """Return True if this action was already taken within the cooldown window."""
    key = _action_key(action)
    last = _last_actions.get(key)
    if last is None:
        return False
    return (time.time() - last) < _ACTION_COOLDOWN

def _record_action(action: dict):
    """Record that an action was taken."""
    key = _action_key(action)
    _last_actions[key] = time.time()

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _find_config() -> Path:
    """Find director.yaml in project dir."""
    for name in ["director.yaml", "director_config.yaml"]:
        p = Path(name)
        if p.exists():
            return p
    return Path("director.yaml")


def load_config() -> dict:
    """Load director config. Returns defaults if file not found."""
    cfg_path = _find_config()
    cfg = {}
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}

    # Defaults
    cfg.setdefault("project_dir", str(Path.cwd()))
    cfg.setdefault("tasks_dir", "tasks")
    cfg.setdefault("backend", "gemini")
    cfg.setdefault("model", "gemini-3.1-pro-preview")
    cfg.setdefault("idle_minutes_for_realloc", 30)
    cfg.setdefault("plateau_window", 20)

    # Auto-discover tasks if not configured
    if "tasks" not in cfg:
        cfg["tasks"] = _discover_tasks(Path(cfg["project_dir"]) / cfg["tasks_dir"])

    return cfg


def _discover_tasks(tasks_dir: Path) -> dict:
    """Auto-discover tasks from subdirectories that have orze.yaml."""
    tasks = {}
    if not tasks_dir.exists():
        return tasks
    for d in sorted(tasks_dir.iterdir()):
        if not d.is_dir():
            continue
        orze_yaml = d / "orze.yaml"
        if not orze_yaml.exists():
            continue
        try:
            tcfg = yaml.safe_load(orze_yaml.read_text()) or {}
            report = tcfg.get("report", {})
            metric = report.get("primary_metric", "score")
            tasks[d.name] = {
                "target_metric": metric,
                "target_value": 0.0,  # unknown — LLM will set expectations
                "default_gpus": [],
            }
        except Exception:
            pass
    return tasks


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

PROJECT_DIR = Path.cwd()
LOG_PATH = PROJECT_DIR / "director.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("director")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_env():
    env_path = PROJECT_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# State reading
# ---------------------------------------------------------------------------

def read_task_state(task_name: str, tasks_dir: Path, task_cfg: dict) -> dict:
    task_dir = tasks_dir / task_name
    state = {
        "name": task_name,
        "metric": task_cfg.get("target_metric", "score"),
        "target": task_cfg.get("target_value", 0.0),
        "best": None,
        "completed": 0,
        "failed": 0,
        "queued": 0,
        "running": False,
        "blocker": None,
    }

    # Check if running via PID file
    pid_file = task_dir / "results" / ".orze.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            state["running"] = True
        except (ProcessLookupError, ValueError, OSError):
            pass

    # Parse report.md
    report = task_dir / "results" / "report.md"
    if report.exists():
        text = report.read_text()
        m = re.search(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", text)
        if m:
            state["completed"] = int(m.group(2))
            state["failed"] = int(m.group(3))
            state["queued"] = int(m.group(5))
        # Best result from Results table
        results_section = text.split("## Results")[-1] if "## Results" in text else ""
        results_lines = re.findall(r"^\| \d+ \|.*$", results_section, re.MULTILINE)
        if results_lines:
            cols = [c.strip() for c in results_lines[0].split("|")]
            if len(cols) > 4:
                try:
                    state["best"] = float(cols[4])
                except ValueError:
                    pass

    # Check blocker
    blocker = task_dir / "BLOCKER.md"
    if blocker.exists():
        state["blocker"] = blocker.read_text().strip()[:200]

    return state


def read_all_state(cfg: dict) -> dict:
    tasks_dir = Path(cfg["project_dir"]) / cfg["tasks_dir"]
    tasks = {}
    for name, tcfg in cfg.get("tasks", {}).items():
        task_dir = tasks_dir / name
        if task_dir.exists():
            tasks[name] = read_task_state(name, tasks_dir, tcfg)
    return {"tasks": tasks, "config": cfg}


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def build_prompt(state: dict) -> str:
    cfg = state["config"]
    lines = [
        "You are a research director overseeing multiple ML tasks. Review state and suggest actions.",
        "",
        "## Current State",
    ]

    for name, t in state["tasks"].items():
        best_str = f"{t['best']:.4f}" if t["best"] is not None else "none"
        target = t["target"]
        gap = ""
        if t["best"] is not None and target > 0:
            pct = (target - t["best"]) / target * 100
            gap = f" ({pct:.0f}% below target)" if pct > 0 else " (TARGET MET)"
        status = "RUNNING" if t["running"] else "STOPPED"
        line = f"- {name}: {t['metric']}={best_str}{gap}, target={target}, completed={t['completed']}, failed={t['failed']}, {status}"
        if t["blocker"]:
            line += f" [BLOCKER: {t['blocker'][:60]}]"
        lines.append(line)

    # Dependencies
    deps = []
    for task_name, tcfg in cfg.get("tasks", {}).items():
        for dep_name, dep_cfg in (tcfg.get("depends_on") or {}).items():
            deps.append(f"- {task_name} depends on {dep_name} ({dep_cfg.get('metric','?')} >= {dep_cfg.get('min_value','?')})")
    if deps:
        lines.extend(["", "## Dependencies"] + deps)

    lines.extend([
        "",
        "## Available Actions (return JSON array)",
        '- {"action": "restart_task", "task": "...", "reason": "..."}',
        '- {"action": "pause_task", "task": "...", "reason": "..."}',
        '- {"action": "resume_task", "task": "..."}',
        '- {"action": "flag_blocker", "task": "...", "message": "..."}',
        '- {"action": "switch_detection_source", "task": "...", "source": "/path/to/detections.json"}',
        '- {"action": "no_action", "reason": "..."}',
    ])

    lines.append("")
    lines.append("## GPU Utilization (flag if idle or <30% VRAM used)")
    gpu_warnings = _check_gpu_utilization(cfg)
    if gpu_warnings:
        for w in gpu_warnings:
            lines.append(f"- WARNING: {w}")
    else:
        lines.append("- All GPUs properly utilized")

    lines.extend([
        "",
        "Be conservative. Only act when there's a clear benefit.",
        "IMPORTANT: Keep reason/message fields under 50 characters. Be terse.",
        "Return ONLY a JSON array.",
    ])
    return "\n".join(lines)


def call_llm(prompt: str, cfg: dict) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    model = cfg.get("model", "gemini-3.1-pro-preview")
    if api_key:
        import urllib.request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096},
        }).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read())
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            log.warning("Gemini call failed: %s", e)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        import urllib.request
        url = "https://api.anthropic.com/v1/messages"
        body = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        })
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read())
            return data["content"][0]["text"]
        except Exception as e:
            log.warning("Anthropic call failed: %s", e)

    return '[]'


def parse_actions(response: str) -> list[dict]:
    # Strategy 1: extract from markdown fences
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", response, re.DOTALL)
    if fence_match:
        try:
            actions = json.loads(fence_match.group(1).strip())
            if isinstance(actions, list):
                return actions
        except json.JSONDecodeError:
            pass
    # Strategy 2: parse whole response
    try:
        actions = json.loads(response.strip())
        if isinstance(actions, list):
            return actions
    except json.JSONDecodeError:
        pass
    # Strategy 3: find any JSON array
    m = re.search(r"\[.*\]", response, re.DOTALL)
    if m:
        try:
            actions = json.loads(m.group())
            if isinstance(actions, list):
                return actions
        except json.JSONDecodeError:
            pass
    log.warning("Failed to parse LLM response as JSON")
    return [{"action": "no_action", "reason": "Failed to parse LLM response"}]


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def _stop_task_by_pid(task_dir: Path):
    """Stop a task's orze using `orze stop`, which kills the orchestrator,
    all child training/eval processes, and cleans up GPU orphans.

    Previously used raw SIGTERM, which triggered graceful_shutdown with
    kill_all=False — detaching training processes as orphans that held
    GPU VRAM indefinitely.  `orze stop` uses kill_all=True.
    """
    pid_file = task_dir / "results" / ".orze.pid"
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)  # check if alive
    except (ValueError, ProcessLookupError, OSError):
        log.info("  PID file exists but process gone")
        return

    log.info("  Running 'orze stop' for %s (PID %d)", task_dir.name, pid)
    try:
        result = subprocess.run(
            ["orze", "stop"],
            cwd=task_dir,
            capture_output=True, text=True,
            timeout=60,
        )
        if result.returncode == 0:
            log.info("  orze stop completed for %s", task_dir.name)
        else:
            log.warning("  orze stop exited %d: %s",
                        result.returncode, result.stderr[:200])
    except subprocess.TimeoutExpired:
        log.warning("  orze stop timed out — falling back to SIGKILL")
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    except Exception as e:
        log.warning("  orze stop failed: %s — falling back to SIGTERM", e)
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(20):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.5)
                except ProcessLookupError:
                    return
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _kill_orphans_on_gpus(gpus: str):
    """Kill training processes on specified GPUs that aren't managed by a running orze.

    Detects orphaned train.py / OpenPCDet processes left behind when an orze
    instance shuts down with detach (not kill_all).  Only kills processes whose
    CUDA_VISIBLE_DEVICES intersects the target GPUs.
    """
    target_gpu_set = {g.strip() for g in gpus.split(",")}
    killed = 0
    try:
        result = subprocess.run(
            ["ps", "-u", str(os.getuid()), "-o", "pid,args", "--no-headers"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) < 2:
                continue
            pid_str, cmdline = parts
            if not any(marker in cmdline for marker in
                       ["train.py", "OpenPCDet/tools/train", "ultralytics"]):
                continue
            if "orze" in cmdline and "train.py" not in cmdline:
                continue
            pid = int(pid_str)
            if pid == os.getpid():
                continue
            try:
                env_path = Path(f"/proc/{pid}/environ")
                env_bytes = env_path.read_bytes()
                for entry in env_bytes.split(b"\0"):
                    if entry.startswith(b"CUDA_VISIBLE_DEVICES="):
                        proc_gpus = entry.decode().split("=", 1)[1]
                        proc_gpu_set = {g.strip() for g in proc_gpus.split(",")}
                        if proc_gpu_set & target_gpu_set:
                            os.kill(pid, signal.SIGTERM)
                            log.info("  Killed orphan PID %d on GPUs %s (%s)",
                                     pid, proc_gpus, cmdline[:80])
                            killed += 1
                        break
            except (PermissionError, FileNotFoundError, OSError):
                continue
    except Exception as e:
        log.warning("  Orphan cleanup failed: %s", e)

    if killed:
        time.sleep(5)
        log.info("  Killed %d orphan(s) on GPUs %s", killed, gpus)
    return killed


def _start_task(task_name: str, task_dir: Path, gpus: str):
    """Start an orze instance for a task, fully detached."""
    _kill_orphans_on_gpus(gpus)
    (task_dir / "results" / ".orze_disabled").unlink(missing_ok=True)
    time.sleep(0.5)
    (task_dir / "results" / ".orze_disabled").unlink(missing_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus}
    subprocess.Popen(
        ["setsid", "orze", "-c", "orze.yaml", "--gpus", gpus],
        cwd=task_dir, env=env,
        stdin=subprocess.DEVNULL,
        stdout=open(task_dir / "orze.log", "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.info("  Started %s on GPUs %s", task_name, gpus)


# ---------------------------------------------------------------------------
# Action execution with safety gates
# ---------------------------------------------------------------------------

def gate_action(action: dict, state: dict) -> tuple[bool, str]:
    cfg = state["config"]
    act = action.get("action", "")
    plateau_window = cfg.get("plateau_window", 20)
    idle_threshold = cfg.get("idle_minutes_for_realloc", 30)

    if act in ("no_action", "flag_blocker", "switch_detection_source"):
        return True, "safe"

    if act in ("restart_task", "resume_task"):
        task = action.get("task", "")
        if task not in cfg.get("tasks", {}):
            return False, f"unknown task: {task}"
        return True, "restart/resume is safe"

    if act == "pause_task":
        task = action.get("task", "")
        t = state["tasks"].get(task, {})
        if t.get("completed", 0) < plateau_window:
            return False, f"too few completions ({t.get('completed',0)})"
        return True, "enough completions"

    if act == "reallocate_gpus":
        from_task = action.get("from_task", "")
        ft = state["tasks"].get(from_task, {})
        if ft.get("completed", 0) < plateau_window:
            return False, f"{from_task} too few completions"
        return True, "converged"

    return False, f"unknown action: {act}"


def execute_action(action: dict, state: dict) -> str:
    cfg = state["config"]
    tasks_dir = Path(cfg["project_dir"]) / cfg["tasks_dir"]
    act = action.get("action", "no_action")

    if act == "no_action":
        return action.get("reason", "no action needed")

    if act in ("restart_task", "resume_task"):
        task = action.get("task", "")
        task_dir = tasks_dir / task
        log.info("Restarting %s...", task)
        _stop_task_by_pid(task_dir)
        # Wait for process to fully die
        for _ in range(30):
            time.sleep(1)
            pid_file = task_dir / "results" / ".orze.pid"
            if not pid_file.exists():
                break
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)
            except (ProcessLookupError, ValueError, OSError):
                break
        time.sleep(2)
        # Get GPU assignment (dynamic first, then default)
        tcfg = cfg.get("tasks", {}).get(task, {})
        assignments = cfg.get("gpu_assignments", {})
        gpu_list = assignments.get(task) or tcfg.get("default_gpus", [0])
        gpus = ",".join(str(g) for g in gpu_list)
        if not gpus:
            gpus = "0"
        _start_task(task, task_dir, gpus)
        return f"restarted {task} on GPUs {gpus}"

    if act == "pause_task":
        task = action.get("task", "")
        task_dir = tasks_dir / task
        log.info("Pausing %s: %s", task, action.get("reason", ""))
        _stop_task_by_pid(task_dir)
        return f"paused {task}"

    if act == "flag_blocker":
        task = action.get("task", "")
        msg = action.get("message", "Unknown blocker")
        blocker_path = tasks_dir / task / "BLOCKER.md"
        if blocker_path.exists():
            existing = blocker_path.read_text().strip()
            if len(existing) > 100:
                log.info("  BLOCKER.md already has detailed content (%d chars), not overwriting", len(existing))
                return f"blocker already documented for {task}"
        blocker_path.write_text(f"# Blocker\n\n{msg}\n\nFlagged by director at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        return f"flagged blocker for {task}"

    if act == "switch_detection_source":
        task = action.get("task", "")
        source = action.get("source", "")
        task_dir = tasks_dir / task
        base_yaml = task_dir / "configs" / "base.yaml"
        if base_yaml.exists():
            try:
                base_cfg = yaml.safe_load(base_yaml.read_text()) or {}
                old_source = base_cfg.get("detection_source", "gt")
                base_cfg["detection_source"] = source
                base_yaml.write_text(yaml.dump(base_cfg, default_flow_style=False))
                log.info("  Switched %s detection_source: %s -> %s", task, old_source, source)
                # Remove blocker if present
                blocker_path = task_dir / "BLOCKER.md"
                if blocker_path.exists():
                    blocker_path.unlink()
                    log.info("  Removed blocker for %s", task)
                return f"switched {task} detection_source to {source}"
            except Exception as e:
                return f"failed to switch detection_source: {e}"
        return f"base.yaml not found for {task}"

    if act == "reallocate_gpus":
        log.info("GPU reallocation is now handled automatically by the rule-based engine")
        return "GPU reallocation handled automatically (rule-based)"

    return f"unknown action: {act}"


# ---------------------------------------------------------------------------
# Cross-task dependency checks
# ---------------------------------------------------------------------------

def _find_best_detection_file(tasks_dir: Path, dep_task: str) -> str | None:
    """Find the nuScenes-format detection results JSON from the best run of dep_task."""
    results_dir = tasks_dir / dep_task / "results"
    report = results_dir / "report.md"
    if not report.exists():
        return None
    text = report.read_text()
    # Extract top idea from report table
    results_lines = re.findall(r"^\| \d+ \| (idea-[\w-]+)", text, re.MULTILINE)
    if not results_lines:
        return None
    best_idea = results_lines[0]
    # Check if this idea has a metrics.json with a result_pkl pointing to detection output
    metrics_path = results_dir / best_idea / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        metrics = json.loads(metrics_path.read_text())
        result_pkl = metrics.get("result_pkl", "")
        if result_pkl:
            # results_nusc.json is in the same directory as result.pkl -> final_result/data/
            pkl_dir = Path(result_pkl).parent
            nusc_json = pkl_dir / "final_result" / "data" / "results_nusc.json"
            if nusc_json.exists():
                return str(nusc_json)
    except Exception:
        pass
    return None


def _check_dependencies(state: dict) -> list[dict]:
    """Check cross-task dependencies and return auto-actions for met dependencies."""
    cfg = state["config"]
    tasks_dir = Path(cfg["project_dir"]) / cfg["tasks_dir"]
    actions = []

    for task_name, tcfg in cfg.get("tasks", {}).items():
        for dep_name, dep_cfg in (tcfg.get("depends_on") or {}).items():
            dep_metric = dep_cfg.get("metric", "")
            dep_min = dep_cfg.get("min_value", 0)
            dep_action = dep_cfg.get("action", "")

            # Check if dependency is met
            dep_state = state["tasks"].get(dep_name, {})
            dep_best = dep_state.get("best")
            if dep_best is None or dep_best < dep_min:
                continue

            # Dependency met — check if action already taken
            if dep_action == "switch_detection_source":
                base_yaml = tasks_dir / task_name / "configs" / "base.yaml"
                if base_yaml.exists():
                    try:
                        base_cfg = yaml.safe_load(base_yaml.read_text()) or {}
                        current_source = base_cfg.get("detection_source", "gt")
                        if current_source == "gt":
                            # Find detection file from dependency task
                            det_file = _find_best_detection_file(tasks_dir, dep_name)
                            if det_file:
                                log.info("Dependency met: %s %s=%.4f >= %.2f, switching %s detection_source",
                                         dep_name, dep_metric, dep_best, dep_min, task_name)
                                actions.append({
                                    "action": "switch_detection_source",
                                    "task": task_name,
                                    "source": det_file,
                                })
                            else:
                                log.warning("Dependency met but no detection file found for %s", dep_name)
                    except Exception as e:
                        log.warning("Failed to check dependency for %s: %s", task_name, e)

    return actions


def _query_gpu_info() -> list[dict]:
    """Query nvidia-smi for structured GPU info. Returns list of {index, used_mb, total_mb}."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "index": int(parts[0]),
                    "used_mb": int(parts[1]),
                    "total_mb": int(parts[2]),
                })
        return gpus
    except Exception:
        return []


def _get_free_gpus(gpu_info: list[dict]) -> list[int]:
    """Return GPU indices with <2GB VRAM used (idle).

    Threshold raised from 100MB to 2GB: orphaned processes can hold
    residual VRAM (CUDA context, shared libs) even after the main
    training loop ends. 2GB safely excludes these while still detecting
    GPUs running real training (typically 20-140GB).
    """
    return [g["index"] for g in gpu_info if g["used_mb"] < 2000]


def _check_gpu_utilization(cfg: dict) -> list[str]:
    """Check GPU VRAM utilization and return warnings for underutilized GPUs."""
    gpu_info = _query_gpu_info()
    warnings = []
    idle_gpus = []
    underused_gpus = []
    for g in gpu_info:
        pct = g["used_mb"] / g["total_mb"] * 100 if g["total_mb"] > 0 else 0
        if g["used_mb"] < 100:
            idle_gpus.append(g["index"])
        elif pct < 30:
            underused_gpus.append((g["index"], g["used_mb"], g["total_mb"], pct))
    if idle_gpus:
        warnings.append(f"IDLE GPUs: {idle_gpus} — 0% VRAM, no training running")
    for idx, used, total, pct in underused_gpus:
        warnings.append(f"GPU {idx}: {used}/{total} MiB ({pct:.0f}%) — consider increasing batch_size")
    return warnings


# ---------------------------------------------------------------------------
# GPU allocation engine (rule-based, no LLM)
# ---------------------------------------------------------------------------

def _compute_gpu_plan(state: dict, gpu_info: list[dict], cfg: dict) -> dict[str, list[int]]:
    """Assign free GPUs to tasks that need them. Simple: biggest queue gets free GPUs."""
    free_gpus = _get_free_gpus(gpu_info)
    if not free_gpus:
        return {}  # nothing to reassign
    tasks_cfg = cfg.get("tasks", {})
    current = cfg.get("gpu_assignments", {})

    # Find tasks that can use GPUs: have queued work, not blocked, target not met
    candidates = []
    for name, tcfg in tasks_cfg.items():
        t = state["tasks"].get(name, {})
        if not t:
            continue
        target = tcfg.get("target_value", 0)
        if t["blocker"] and not t["running"]:
            continue  # blocked, can't run
        if t["best"] is not None and target > 0 and t["best"] >= target:
            continue  # target met
        if t["queued"] == 0 and not t["running"]:
            continue  # nothing to do
        candidates.append(name)

    if not candidates:
        return {}

    # Sort by queue size descending
    candidates.sort(key=lambda n: state["tasks"].get(n, {}).get("queued", 0), reverse=True)

    # Build plan: keep existing GPUs, distribute free ones to top candidate
    plan = {}
    top = candidates[0]
    top_current = current.get(top) or tasks_cfg[top].get("default_gpus", [])
    # Keep GPUs already in use (not free) + add all free GPUs
    in_use = [g for g in top_current if g not in free_gpus]
    plan[top] = sorted(in_use + free_gpus)

    return plan


def _persist_gpu_assignments(cfg: dict, plan: dict[str, list[int]]):
    """Write gpu_assignments back to director.yaml."""
    cfg_path = _find_config()
    if not cfg_path.exists():
        return
    text = cfg_path.read_text()
    cfg["gpu_assignments"] = plan
    raw = yaml.safe_load(text) or {}
    raw["gpu_assignments"] = plan
    cfg_path.write_text(yaml.dump(raw, default_flow_style=False, sort_keys=False))
    log.info("Persisted gpu_assignments to %s", cfg_path)


def _execute_gpu_reallocation(plan: dict[str, list[int]], state: dict, cfg: dict):
    """Stop/restart tasks whose GPU assignments changed."""
    tasks_dir = Path(cfg["project_dir"]) / cfg["tasks_dir"]
    tasks_cfg = cfg.get("tasks", {})
    current = cfg.get("gpu_assignments", {})

    for name, new_gpus in plan.items():
        old_gpus = current.get(name) or tasks_cfg.get(name, {}).get("default_gpus", [])
        if sorted(new_gpus) == sorted(old_gpus):
            continue
        task_dir = tasks_dir / name
        if not task_dir.exists():
            continue
        t = state["tasks"].get(name, {})
        # Stop if running, then restart with new GPUs
        if t and t["running"]:
            log.info("  Stopping %s for GPU reassignment...", name)
            _stop_task_by_pid(task_dir)
            time.sleep(3)
        if new_gpus:
            gpus_str = ",".join(str(g) for g in new_gpus)
            log.info("  Starting %s on GPUs %s", name, gpus_str)
            _start_task(name, task_dir, gpus_str)

    _persist_gpu_assignments(cfg, plan)


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------

def run_cycle(cfg: dict, dry_run: bool = False):
    log.info("=" * 60)
    log.info("Director cycle%s", " (DRY RUN)" if dry_run else "")
    log.info("=" * 60)

    state = read_all_state(cfg)

    # GPU utilization check
    gpu_warnings = _check_gpu_utilization(cfg)
    for w in gpu_warnings:
        log.warning("GPU: %s", w)

    # Rule-based GPU reallocation (before LLM call)
    gpu_info = _query_gpu_info()
    if gpu_info:
        plan = _compute_gpu_plan(state, gpu_info, cfg)
        if plan:
            log.info("GPU reallocation: %s", plan)
            if not dry_run:
                _execute_gpu_reallocation(plan, state, cfg)
                state = read_all_state(cfg)

    for name, t in state["tasks"].items():
        best_str = f"{t['best']:.4f}" if t["best"] is not None else "none"
        status = "RUNNING" if t["running"] else "STOPPED"
        extra = f" [BLOCKER]" if t["blocker"] else ""
        log.info("  %s: %s=%s, completed=%d, failed=%d, %s%s",
                 name, t["metric"], best_str, t["completed"], t["failed"], status, extra)

    # Auto-check cross-task dependencies before LLM call
    auto_actions = _check_dependencies(state)
    for aa in auto_actions:
        log.info("  [AUTO] %s", aa.get("action", "?"))
        if dry_run:
            log.info("    (dry run — skipped)")
        else:
            result = execute_action(aa, state)
            log.info("    Result: %s", result)

    log.info("Calling LLM for decisions...")
    prompt = build_prompt(state)
    response = call_llm(prompt, cfg)
    actions = parse_actions(response)
    log.info("LLM suggested %d action(s)", len(actions))

    for i, action in enumerate(actions):
        act = action.get("action", "?")
        allowed, reason = gate_action(action, state)
        if not allowed:
            log.info("  [%d] BLOCKED %s: %s", i + 1, act, reason)
            continue
        if _is_action_recent(action):
            log.info("  [%d] DEDUP %s (same action taken <30 min ago)", i + 1, act)
            continue
        log.info("  [%d] APPROVED %s: %s", i + 1, act, reason)
        if dry_run:
            log.info("    (dry run)")
        else:
            result = execute_action(action, state)
            _record_action(action)
            log.info("    Result: %s", result)

    # ----- Auto-restart stopped tasks with no blocker (rule-based, no LLM) -----
    for name, t in state["tasks"].items():
        if t["running"] or t["blocker"]:
            continue
        # Skip if target already met
        tcfg = cfg.get("tasks", {}).get(name, {})
        target = tcfg.get("target_value", 0)
        if t["best"] is not None and target > 0 and t["best"] >= target:
            log.info("  [AUTO-RESTART] %s skipped (target met: %.4f >= %.4f)", name, t["best"], target)
            continue
        # Skip if no queued work
        if t["queued"] == 0 and t["completed"] > 0:
            log.info("  [AUTO-RESTART] %s skipped (no queued work)", name)
            continue
        restart_action = {"action": "restart_task", "task": name,
                          "reason": "auto-restart: stopped with no blocker"}
        if _is_action_recent(restart_action):
            log.info("  [AUTO-RESTART] %s skipped (restarted <30 min ago)", name)
            continue
        log.info("  [AUTO-RESTART] %s is STOPPED with no blocker — restarting", name)
        if not dry_run:
            result = execute_action(restart_action, state)
            _record_action(restart_action)
            log.info("    Result: %s", result)

    log.info("Director cycle complete. %d actions.", len(actions))


def main():
    load_env()

    import argparse
    parser = argparse.ArgumentParser(description="Cross-task director role")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tasks-dir", default=None)
    parser.add_argument("--goal", default=None)
    args = parser.parse_args()

    cfg = load_config()
    if args.tasks_dir:
        cfg["tasks_dir"] = args.tasks_dir

    if args.once:
        run_cycle(cfg, dry_run=args.dry_run)
    else:
        interval = cfg.get("interval", 600)
        log.info("Director running (interval=%ds)", interval)
        while True:
            try:
                run_cycle(cfg, dry_run=args.dry_run)
            except Exception as e:
                log.error("Cycle failed: %s", e, exc_info=True)
            time.sleep(interval)


if __name__ == "__main__":
    main()
