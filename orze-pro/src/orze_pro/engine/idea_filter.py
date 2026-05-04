"""Tier 2 idea filter — validates queued ideas against professor-written rules.

Implements the 'idea_filter' extension point declared in
orze/engine/phases.py line 136-143. Called during _sync_ideas()
before sweep expansion.

CALLING SPEC:
    filter_queued_ideas(results_dir: Path) -> None
        Load _validators/*.yaml, check all queued ideas, skip invalid ones.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import yaml

logger = logging.getLogger("orze")


def filter_queued_ideas(results_dir: Path) -> None:
    """Run Tier 2 validators on all queued ideas in the lake.

    Called from phases.py via get_extension('idea_filter').
    Loads _validators/*.yaml, queries the lake for queued ideas,
    validates each, and sets invalid ones to status='skipped'.
    """
    from orze_pro.engine.sop_tier2 import load_validators, validate_idea_tier2

    validators = load_validators(results_dir)
    if not validators:
        return  # no validators configured

    lake_path = results_dir / "idea_lake.db"
    if not lake_path.exists():
        return

    try:
        conn = sqlite3.connect(str(lake_path), timeout=5)
        # Skip silently when the lake schema isn't initialized yet — this
        # is normal during first-boot or when a project uses a different
        # DB layout. Without this guard the validator spams a WARNING
        # every sync cycle (~30s) until ingest creates the table.
        has_ideas = conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='ideas' LIMIT 1"
        ).fetchone()
        if not has_ideas:
            conn.close()
            return
        cur = conn.execute(
            "SELECT idea_id, config FROM ideas WHERE status = 'queued'")
        rows = cur.fetchall()
    except Exception as e:
        logger.warning("idea_filter: failed to query lake: %s", e)
        return

    skipped = 0
    for idea_id, config_str in rows:
        if not config_str:
            continue
        try:
            config = yaml.safe_load(config_str) or {}
        except yaml.YAMLError:
            continue

        is_valid, err_msg = validate_idea_tier2(config, validators,
                                                idea_id=idea_id)
        if not is_valid:
            try:
                conn.execute(
                    "UPDATE ideas SET status = 'skipped' WHERE idea_id = ?",
                    (idea_id,))
                skipped += 1
                logger.info("Validator skipped %s: %s", idea_id, err_msg)
            except Exception:
                pass

    if skipped:
        conn.commit()
        logger.info("Tier 2 validators skipped %d ideas", skipped)
    conn.close()
