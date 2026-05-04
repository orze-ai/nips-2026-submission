# File Layout SOP

You operate inside an orze project. Respect these env vars:

- `$ORZE_ROLE_SCRATCH` — your per-cycle scratch directory (default cwd for claude roles). Use for throwaway files.
- `$ORZE_METHODS_DIR/<role>/<cycle>/` — reusable analysis-method scripts you want kept. Save *.py, *.sh here (not project root).
- `$ORZE_KNOWLEDGE_DIR/` — distilled learnings (markdown): insights, failure hypotheses, error analysis.
- `$ORZE_RESULTS_DIR/` — project deliverables (experiments, metrics).
- `$ORZE_IDEAS_FILE` — the ideas manifest (read/write with care).
- `$ORZE_RULES_DIR/` — role RULES.md files.

**NEVER** write .py/.sh/.json/.log/.npz/.pkl files to the project root. The stray sweeper will quarantine them to `.orze/stray/` and fire a `needs_intervention` notification.
