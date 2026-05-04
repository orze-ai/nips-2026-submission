# orze-pro — Autopilot for GPU Experiments

[![PyPI](https://img.shields.io/badge/pypi-private-blue)](https://pypi.orze.ai/simple/orze-pro/)
[![Requires](https://img.shields.io/badge/requires-orze%20%E2%89%A54.0-green)](https://pypi.org/project/orze/)

The intelligence layer for [orze](https://github.com/ANON/orze).

**orze** (open source) is a complete GPU experiment orchestrator: scheduling, leaderboard, notifications, retrospection, analysis, admin dashboard, and Smart Suggestions.

**orze-pro** adds **autopilot**: LLM-powered research agents that propose ideas, auto-fix failures, and evolve code on plateaus — so experiments run while you sleep.

## Install

```bash
ORZE_PRO_KEY=ORZE-PRO-xxx curl -sL https://orze.ai/install | bash
```

That's it. It installs orze + orze-pro, initializes your project, and starts running.

## What's Included

| Feature | orze (free) | + orze-pro |
|---------|:-----------:|:----------:|
| GPU scheduling & multi-node | ✓ | ✓ |
| Idea queue (ideas.md + SQLite) | ✓ | ✓ |
| Hyperparameter sweep | ✓ | ✓ |
| Leaderboard report | ✓ | ✓ |
| Telegram/Slack notifications | ✓ | ✓ |
| Admin dashboard & MCP server | ✓ | ✓ |
| Retrospection (plateau detection) | ✓ | ✓ |
| Cross-experiment regression analysis | ✓ | ✓ |
| Failure analysis & categorization | ✓ | ✓ |
| Checkpoint GC | ✓ | ✓ |
| Sealed eval protection | ✓ | ✓ |
| Service watchdog (auto-restart + containers) | ✓ | ✓ |
| Smart Suggestions (rule-based ideas) | ✓ | ✓ |
| **Autonomous research agents** (Gemini/GPT/Claude) | | ✓ |
| **The Professor** (paper lake, cross-domain search, strategy) | | ✓ |
| **Engineer** (implement ideas, fix bugs) | | ✓ |
| **Auto-fix failed experiments** | | ✓ |
| **Code evolution on plateau** | | ✓ |
| **Meta-research (strategy adjustment)** | | ✓ |
| **FSM orchestration** (7 procedures) | | ✓ |
| **Data analyst & thinker** (auto-injected) | | ✓ |

## Smart Suggestions vs Research Agents

| | Smart Suggestions (free) | Research Agents (pro) |
|---|---|---|
| How | Rule-based: perturbations, scale sweeps | LLM-driven: reads results, forms hypotheses |
| Requires API key | No | Yes |
| Finds ±10% improvements | Yes | Yes |
| Discovers new approaches | No | Yes |
| Fixes failures | No | Yes (auto-diagnose + patch) |
| Evolves code | No | Yes (modifies train script) |

## Pro Roles

orze-pro ships 6 autonomous roles, each composing from bundled SOPs:

| Role | What it does |
|------|-------------|
| **research** | Reads results, forms hypotheses, designs experiments |
| **professor** | Cross-domain search, idea review, strategy, gap closure |
| **engineer** | Implements ideas, fixes bugs in training code |
| **code_evolution** | Modifies train.py when stuck on a plateau |
| **meta_research** | Adjusts research strategy based on progress |
| **fsm** | Finite state machine that orchestrates all other roles |

```bash
orze sop list                      # see all available SOPs
```

## How It Works

```bash
# Same command — pro features activate automatically:
orze start
# ✓ Research agent reads results and proposes novel ideas
# ✓ Auto-fixes failures and retries
# ✓ Evolves train script when stuck on a plateau
# ✓ The Professor searches literature and reviews strategy
# ✓ You sleep. Experiments run.
```

Zero config change — same `orze.yaml`. Pro features activate on install.

## Compatibility

| orze | orze-pro | Notes |
|------|----------|-------|
| 4.1.x | 0.8.x | Current release |

## License

Proprietary. Contact support@orze.ai for licensing.
