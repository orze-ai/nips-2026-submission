"""Round-2 B1/B2/B3: timeout + stall_minutes auto-derivation, warmup."""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Both packages import each other; make src/ importable.
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))


def test_resolve_role_timeout_b1():
    from orze_pro.engine import role_runner as rr

    # Explicit wins.
    assert rr._resolve_role_timeout("x", {"timeout": 999, "mode": "claude"}) == 999

    # Auto-derived for claude: max(300, 60 * len(skills)).
    assert rr._resolve_role_timeout(
        "data_analyst", {"mode": "claude",
                          "skills": [1, 2, 3, 4, 5, 6]}) == 360
    # Floor at 300s.
    assert rr._resolve_role_timeout(
        "tiny", {"mode": "claude", "skills": ["a"]}) == 300
    # Script default: 600s flat.
    assert rr._resolve_role_timeout(
        "scripty", {"mode": "script"}) == 600


def test_resolve_role_stall_minutes_b2():
    from orze_pro.engine import role_runner as rr

    # Explicit wins.
    assert rr._resolve_role_stall_minutes(
        "x", {"stall_minutes": 99, "mode": "claude"}, 5) == 99
    # Auto-derived: max(5, 2 * len(skills)).
    assert rr._resolve_role_stall_minutes(
        "professor", {"mode": "claude",
                       "skills": list(range(10))}, 5) == 20
    # Floor at 5.
    assert rr._resolve_role_stall_minutes(
        "tiny", {"mode": "claude", "skills": []}, 5) == 5
    # script: global default.
    assert rr._resolve_role_stall_minutes(
        "scripty", {"mode": "script"}, 5) == 5


def test_warmup_b3(tmp_path):
    from orze.engine.process import RoleProcess
    from orze.engine.roles import _is_role_stalled

    log = tmp_path / "log"
    log.write_text("")  # 0 bytes
    rp = RoleProcess(
        role_name="r", process=None, start_time=time.time() - 5,
        log_path=log, timeout=600, lock_dir=tmp_path / "lock",
        cycle_num=1,
    )
    rp.stall_warmup_seconds = 60.0
    # Within warmup, no output yet → not stalled.
    assert _is_role_stalled(rp, stall_minutes=1) is False
    # Past warmup, no output, no growth → eventually stalled.
    rp.start_time = time.time() - 3600  # well past warmup
    rp._stall_since = time.time() - 3600  # registered stall a long time ago
    assert _is_role_stalled(rp, stall_minutes=1) is True


if __name__ == "__main__":
    import tempfile
    test_resolve_role_timeout_b1()
    print("B1 OK")
    test_resolve_role_stall_minutes_b2()
    print("B2 OK")
    with tempfile.TemporaryDirectory() as td:
        test_warmup_b3(Path(td))
        print("B3 OK")
