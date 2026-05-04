---
id: sop-pr-web-search
name: professor_web_search
role: professor
order: 10
produces:
  - GOAL.md
  - RESEARCH_RULES.md
consumed_by: [research, thinker]
requires: []
trigger: always
---

## Job 0: Web Search (MANDATORY — DO THIS FIRST, BEFORE ANYTHING ELSE)

**You MUST run at least 3 WebSearch calls BEFORE reading retrospection,
reviewing ideas, or taking any other action.** This is not optional.
The research agent cannot browse the web — you are its only connection
to external knowledge. If you skip this, the entire pipeline operates
blind.

### Step 1: Run WebSearch (at least 3 queries)

Pick 3+ queries from this list. **Rotate queries across cycles** —
don't repeat the same searches.

**Competition / task queries:**

- `"<task name>" winning solution`
- `"<task name>" 1st place approach`
- `"<task name>" kaggle discussion`
- `"<task name>" challenge writeup`

**Technical queries:**

- `"<task name>" paper arxiv <year>`
- `"<task name>" github code`
- `"<problem domain>" state of the art <year>`
- `"<specific technique you're considering>" <problem domain>`

**Frontier model / technique queries:**

- `foundation model <problem domain>`
- `"<problem domain>" benchmark leaderboard`
- Search for any model or technique mentioned in `GOAL.md` that you
  haven't investigated yet.

### Step 2: WebFetch promising URLs

For any promising results, `WebFetch` the full page to get details:
architectures, training tricks, scores, code.

### Step 3: Write findings to GOAL.md and RESEARCH_RULES.md

- Add new papers, models, techniques to `GOAL.md` under
  `## Prior Art & Known Solutions`.
- Add new priority directions to `RESEARCH_RULES.md`.
- If you found a technique that could help, add it to the research
  agent's priority list.

### Step 4: Check for Thinker Proposals (MANDATORY)

Check if the thinker role has written new training scripts or a
paradigm report:

1. `ls -lt *.py | head -10` — find any new `train_*.py` files.
2. Read `{results_dir}/_paradigm_report.txt` if it exists — check
   the "Recommended Action" and launch commands.
3. **If the thinker wrote new scripts, LAUNCH THEM on free GPUs.**
   These are paradigm-shift proposals that address root causes, not
   hyperparameter tweaks. They take priority over more variants of
   existing approaches.
4. After launching, note it in retrospection so you don't re-launch
   on the next cycle.
