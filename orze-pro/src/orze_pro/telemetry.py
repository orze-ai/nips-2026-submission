"""Telemetry reporter for orze-admin global panel.

Periodically sends pipeline status to the admin server for customer support
and product improvement. Enabled by default; opt out with telemetry: false
in orze.yaml.

CALLING SPEC:
    start_telemetry(cfg: dict, results_dir: str) -> None
        Start background telemetry thread. No-op if already running or disabled.

    stop_telemetry() -> None
        Stop the telemetry thread.

    report_now(cfg: dict, results_dir: str) -> None
        Send a single telemetry report immediately (blocking, with timeout).
"""

import hashlib
import json
import logging
import os
import socket
import sqlite3
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orze_pro.telemetry")

_ADMIN_SERVER = os.environ.get("ORZE_ADMIN_SERVER", "https://orze.ai/admin")
_REPORT_INTERVAL = 300  # seconds between reports
_NET_TIMEOUT = 10
_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def _is_enabled(cfg: dict) -> bool:
    return cfg.get("telemetry", True) is not False


def _get_key_hash() -> Optional[str]:
    key = os.environ.get("ORZE_PRO_KEY", "").strip()
    if not key:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("ORZE_PRO_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        home_key = Path.home() / ".orze-pro.key"
        if home_key.exists():
            key = home_key.read_text().strip()
    if not key:
        return None
    return hashlib.sha256(key.encode()).hexdigest()


def _get_machine_id() -> str:
    hostname = socket.gethostname()
    mac = __import__("uuid").getnode()
    return hashlib.sha256(f"{hostname}-{mac}".encode()).hexdigest()[:16]


def _get_gpu_info() -> dict:
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return {"gpu_count": 0}
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        gpu_count = len(lines)
        total_util = 0
        gpu_type = ""
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                if not gpu_type:
                    gpu_type = parts[1]
                total_util += int(parts[4])
        return {
            "gpu_count": gpu_count,
            "gpu_type": gpu_type,
            "gpu_utilization_avg": round(total_util / max(gpu_count, 1), 1),
        }
    except Exception:
        return {"gpu_count": 0}


def _get_disk_info(path: str = ".") -> dict:
    import shutil
    try:
        usage = shutil.disk_usage(path)
        return {"disk_free_gb": round(usage.free / (1024**3), 1)}
    except Exception:
        return {"disk_free_gb": -1}


def _read_status_json(results_dir: str) -> dict:
    status_file = Path(results_dir) / "status.json"
    if status_file.exists():
        try:
            return json.loads(status_file.read_text())
        except Exception:
            pass
    return {}


def _read_leaderboard(results_dir: str) -> list:
    # Try multiple leaderboard locations
    candidates = [
        Path(results_dir).parent / ".orze" / "admin" / "leaderboard.json",
        Path(results_dir) / "_leaderboard.json",
    ]
    for lb_file in candidates:
        if lb_file.exists():
            try:
                data = json.loads(lb_file.read_text())
                # Handle {"top": [...], "metric": "..."} format
                if isinstance(data, dict) and "top" in data:
                    entries = data["top"]
                    # Normalize entries to include approach_family + primary_metric
                    result = []
                    for i, e in enumerate(entries[:10]):
                        entry = {
                            "idea_id": e.get("idea_id", ""),
                            "rank": i + 1,
                            "primary_metric": e.get("metric_value") or e.get("primary_metric"),
                            "metrics": e.get("eval_metrics", e.get("metrics", {})),
                            "approach_family": e.get("approach_family", ""),
                        }
                        if not entry["approach_family"]:
                            entry["approach_family"] = _lookup_family(
                                str(Path(results_dir).parent), e.get("idea_id", ""))
                        idea_cfg = _read_idea_config(results_dir, e.get("idea_id", ""))
                        if idea_cfg:
                            entry["config"] = idea_cfg
                        result.append(entry)
                    return result
                elif isinstance(data, list):
                    return data[:10]
            except Exception:
                pass
    return []


def _read_idea_config(results_dir: str, idea_id: str) -> dict:
    if not idea_id:
        return {}
    cfg_path = Path(results_dir) / idea_id / "idea_config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml
        raw = yaml.safe_load(cfg_path.read_text()) or {}
        # Keep it bounded — flatten to top-level scalars + one level of dicts
        trimmed = {}
        for k, v in raw.items():
            if isinstance(v, (str, int, float, bool)):
                trimmed[k] = v
            elif isinstance(v, dict):
                trimmed[k] = {kk: vv for kk, vv in v.items() if isinstance(vv, (str, int, float, bool))}
        return trimmed
    except Exception:
        return {}


def _lookup_family(project_path: str, idea_id: str) -> str:
    """Look up approach_family from idea_lake.db."""
    db_path = Path(project_path) / ".orze" / "idea_lake.db"
    if not db_path.exists() or not idea_id:
        return ""
    try:
        conn = sqlite3.connect(str(db_path), timeout=2)
        row = conn.execute(
            "SELECT approach_family FROM ideas WHERE idea_id = ?", (idea_id,)
        ).fetchone()
        conn.close()
        return row[0] if row else ""
    except Exception:
        return ""


def _read_idea_lake(project_path: str) -> dict:
    db_path = Path(project_path) / ".orze" / "idea_lake.db"
    if not db_path.exists():
        return {"experiments": {"total": 0, "completed": 0, "failed": 0},
                "families": {}, "failures": [], "hit_rate": None}
    try:
        conn = sqlite3.connect(str(db_path), timeout=2)
        conn.row_factory = sqlite3.Row
        total = conn.execute("SELECT COUNT(*) FROM ideas").fetchone()[0]
        completed = conn.execute("SELECT COUNT(*) FROM ideas WHERE status='completed'").fetchone()[0]
        failed = conn.execute("SELECT COUNT(*) FROM ideas WHERE status='failed'").fetchone()[0]
        families = {}
        for row in conn.execute("SELECT approach_family, COUNT(*) as c FROM ideas GROUP BY approach_family"):
            families[row[0] or "other"] = row[1]
        failures = []
        for row in conn.execute(
            "SELECT idea_id, approach_family FROM ideas WHERE status='failed' ORDER BY id_num DESC LIMIT 10"
        ):
            detail = _read_failure_detail(project_path, row[0])
            f_entry = {"idea_id": row[0], "category": detail.get("category", "unknown"),
                       "summary": detail.get("summary", "")}
            if detail.get("traceback"):
                f_entry["traceback"] = detail["traceback"]
            f_cfg = _read_idea_config(str(Path(project_path) / "results"), row[0])
            if f_cfg:
                f_entry["config"] = f_cfg
            failures.append(f_entry)
        conn.close()
        hit_rate = round(completed / total, 2) if total > 0 else None
        return {"experiments": {"total": total, "completed": completed, "failed": failed},
                "families": families, "failures": failures, "hit_rate": hit_rate}
    except Exception:
        return {"experiments": {"total": 0, "completed": 0, "failed": 0},
                "families": {}, "failures": [], "hit_rate": None}


def _read_failure_detail(project_path: str, idea_id: str) -> dict:
    fa = Path(project_path) / "results" / idea_id / "failure_analysis.json"
    if fa.exists():
        try:
            d = json.loads(fa.read_text())
            # Support both formats: {"summary": ...} and {"what": ..., "why": ...}
            summary = d.get("summary") or d.get("error") or ""
            if not summary:
                what = d.get("what", "")
                why = d.get("why", "")
                summary = f"{what}: {why}" if what and why else what or why
            result = {"category": d.get("category", ""), "summary": summary[:200]}
            if d.get("traceback"):
                result["traceback"] = d["traceback"][-500:]
            return result
        except Exception:
            pass
    m = Path(project_path) / "results" / idea_id / "metrics.json"
    if m.exists():
        try:
            d = json.loads(m.read_text())
            return {"category": "runtime", "summary": d.get("error", "")[:200]}
        except Exception:
            pass
    return {}


def _read_receipts_summary(project_path: str) -> dict:
    receipts_dir = Path(project_path) / ".orze" / "receipts"
    if not receipts_dir.exists():
        return {}
    summary = {}
    for f in sorted(receipts_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
        try:
            r = json.loads(f.read_text())
            role = r.get("role", "unknown")
            declared = len(r.get("skills_declared", []))
            evidenced = len(r.get("skills_evidenced", []))
            if role not in summary:
                summary[role] = {"runs": 0, "skills_declared": 0, "skills_evidenced": 0}
            summary[role]["runs"] += 1
            summary[role]["skills_declared"] += declared
            summary[role]["skills_evidenced"] += evidenced
        except Exception:
            pass
    return summary


def _read_intervention_signals(project_path: str) -> dict:
    p = Path(project_path)
    signals = {}
    if (p / ".pause_research").exists():
        signals["paused"] = True
    triggers_dir = p / ".orze" / "triggers"
    if triggers_dir.exists():
        for t in triggers_dir.iterdir():
            if t.is_file():
                signals[t.name] = True
    interv = p / ".orze" / "state" / "interventions.json"
    if interv.exists():
        try:
            signals["interventions"] = len(json.loads(interv.read_text()))
        except Exception:
            pass
    return signals


def _read_role_health(project_path: str) -> dict:
    roles = {}
    logs_dir = Path(project_path) / ".orze" / "logs"
    if not logs_dir.exists():
        return roles
    for role_dir in logs_dir.iterdir():
        if not role_dir.is_dir():
            continue
        log_files = sorted(role_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        status = "idle"
        last_run = None
        errors = 0
        if log_files:
            latest = log_files[0]
            last_run = latest.stat().st_mtime
            age_min = (time.time() - last_run) / 60
            if age_min < 5:
                status = "running"
            try:
                text = latest.read_text(errors="ignore")[-2000:]
                errors = text.lower().count("error") + text.lower().count("traceback")
            except Exception:
                pass
        roles[role_dir.name] = {"status": status, "last_run": last_run, "errors": errors}
    return roles


def _get_config_summary(cfg: dict) -> dict:
    roles_enabled = list(cfg.get("roles", {}).keys())
    return {
        "train_script": cfg.get("train_script", ""),
        "gpu_mode": cfg.get("gpu_scheduling", {}).get("mode", "exclusive"),
        "roles_enabled": roles_enabled,
        "timeout": cfg.get("timeout", 0),
        "poll": cfg.get("poll", 30),
    }


def _build_payload(cfg: dict, results_dir: str) -> dict:
    from .license import check_license, get_activation_status

    license_info = check_license() or {}
    activation = get_activation_status() or {}

    project_path = str(Path(results_dir).parent.resolve())
    project_name = cfg.get("report", {}).get("title", Path(project_path).name)
    project_id = f"{_get_machine_id()}-{hashlib.md5(project_path.encode()).hexdigest()[:8]}"

    status_data = _read_status_json(results_dir)
    lake = _read_idea_lake(project_path)
    experiments = lake["experiments"]
    leaderboard = _read_leaderboard(results_dir)
    roles = _read_role_health(project_path)
    receipts = _read_receipts_summary(project_path)
    signals = _read_intervention_signals(project_path)

    pm_name = cfg.get("report", {}).get("primary_metric", "")
    pm_sort = cfg.get("report", {}).get("sort", "ascending")
    pm_best = None
    if leaderboard and len(leaderboard) > 0:
        first = leaderboard[0]
        if isinstance(first, dict):
            pm_best = first.get("primary_metric") or first.get(pm_name)

    gpu_info = _get_gpu_info()
    disk_info = _get_disk_info(project_path)

    active_count = 0
    if status_data.get("active_runs"):
        active_count = len(status_data["active_runs"]) if isinstance(status_data["active_runs"], list) else 0

    daemon_status = "stopped"
    pid_candidates = [
        Path(results_dir) / ".orze.pid",
        Path(project_path) / ".orze" / "state" / "orze.pid",
    ]
    for pid_file in pid_candidates:
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)
                daemon_status = "running"
                break
            except (ValueError, ProcessLookupError, PermissionError):
                continue

    return {
        "hostname": socket.gethostname(),
        "customer": license_info.get("customer", ""),
        "tier": license_info.get("tier", "pro"),
        "machines_used": activation.get("machines_used", 0),
        "machines_max": activation.get("machines_max", 3),
        "timestamp": time.time(),
        "projects": [{
            "id": project_id,
            "name": project_name,
            "path": project_path,
            "status": daemon_status,
            "primary_metric": {"name": pm_name, "best": pm_best, "sort": pm_sort},
            "experiments": {
                "total": experiments["total"],
                "completed": experiments["completed"],
                "failed": experiments["failed"],
                "active": active_count,
            },
            "gpu_count": gpu_info.get("gpu_count", 0),
            "leaderboard": leaderboard[:5],
            "failures": lake["failures"],
            "families": lake["families"],
            "hit_rate": lake["hit_rate"],
            "roles": roles,
            "receipts": receipts,
            "signals": signals,
            "config_summary": _get_config_summary(cfg),
        }],
        "system": {
            **gpu_info,
            **disk_info,
        },
    }


def _send_report(key_hash: str, machine_id: str, payload: dict) -> bool:
    try:
        data = json.dumps({
            "key_hash": key_hash,
            "machine_id": machine_id,
            "payload": payload,
        }).encode()
        req = urllib.request.Request(
            f"{_ADMIN_SERVER}/api/ingest",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_NET_TIMEOUT) as resp:
            result = json.loads(resp.read())
        logger.debug("Telemetry sent: %s", result)
        return True
    except Exception as e:
        logger.debug("Telemetry send failed: %s", e)
        return False


def report_now(cfg: dict, results_dir: str) -> None:
    if not _is_enabled(cfg):
        return
    key_hash = _get_key_hash()
    if not key_hash:
        return
    machine_id = _get_machine_id()
    try:
        payload = _build_payload(cfg, results_dir)
        _send_report(key_hash, machine_id, payload)
    except Exception as e:
        logger.debug("Telemetry report failed: %s", e)


def _telemetry_loop(cfg: dict, results_dir: str):
    key_hash = _get_key_hash()
    if not key_hash:
        logger.debug("No license key found, telemetry disabled")
        return
    machine_id = _get_machine_id()

    while not _stop_event.is_set():
        try:
            payload = _build_payload(cfg, results_dir)
            _send_report(key_hash, machine_id, payload)
        except Exception as e:
            logger.debug("Telemetry error: %s", e)
        _stop_event.wait(_REPORT_INTERVAL)


def start_telemetry(cfg: dict, results_dir: str) -> None:
    global _thread
    if not _is_enabled(cfg):
        logger.info("Telemetry disabled by config")
        return
    if _thread is not None and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(target=_telemetry_loop, args=(cfg, results_dir), daemon=True)
    _thread.start()
    logger.info("Telemetry reporter started (interval=%ds)", _REPORT_INTERVAL)


def stop_telemetry() -> None:
    global _thread
    _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=5)
        _thread = None
    logger.info("Telemetry reporter stopped")
