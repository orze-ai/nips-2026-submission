"""Tests for orze-dir log path resolution in role_runner."""
import os
import socket
from pathlib import Path

from orze.core.config import orze_path


def test_log_path_lands_under_orze_logs(tmp_path):
    """Test that log path computation lands under .orze/logs/<role>/."""
    orze_dir = tmp_path / ".orze"
    orze_dir.mkdir()
    
    cfg = {
        "_orze_dir": str(orze_dir),
        "_env_ORZE_RESULTS_DIR": str(tmp_path / "orze_results"),
    }
    
    role_name = "engineer"
    cycle_num = 5
    
    # Compute log path using the pattern from role_runner.py line 876-878
    log_dir = orze_path(cfg, "logs", role_name)
    # orze_path creates .orze/logs/ but not .orze/logs/engineer/ — role_runner.py creates that
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{socket.gethostname()}-{os.getpid()}-cycle_{cycle_num:03d}.log"
    
    # Assert shape
    assert log_dir == orze_dir / "logs" / "engineer"
    assert log_path.parent == orze_dir / "logs" / "engineer"
    assert log_path.name == f"{socket.gethostname()}-{os.getpid()}-cycle_005.log"
    
    # orze_path creates logs/ but not logs/engineer/
    assert (orze_dir / "logs").exists()
