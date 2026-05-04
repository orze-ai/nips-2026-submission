---
id: sop-pr-paper-lake
name: professor_paper_lake
role: professor
order: 12
produces:
  - GOAL.md
  - results/_paper_lake_usage.log
consumed_by: [research, thinker]
requires: []
trigger: always
---

# Professor: Paper Lake (paperdog) — Core Literature Source

**Trigger reason:** {trigger_reason}

`paperdog` is the self-hosted paper-lake HTTP service. It is the
**primary** literature lookup — it sits *above* raw WebSearch. A
curated, local corpus of ~400K arxiv CS papers (last ~3 years)
meant to be queried freely (no rate limit); WebSearch is the
secondary fallback for competition writeups and community posts
not yet in the lake.

- Endpoint: `http://localhost:8000/`
- Container image: `ANON/paperdog:latest`
- Container name: `paperdog` (fixed — enables idempotent self-heal)

## Step 0: Self-heal the container (MANDATORY, first thing each cycle)

**You own the container.** If it's not running, you start it. Do
this BEFORE Step 1. The whole pass below should finish in <5s on
the healthy path and <90s on cold-start (image pull).

### 0a. Fast health probe (should be <2s on the healthy path)

```bash
curl -sf -m 2 http://localhost:8000/ -o /dev/null \
    && echo "paperdog UP" \
    || echo "paperdog DOWN"
```

If `UP` — **skip to Step 1.** Do not waste a cycle restarting a
healthy container.

### 0b. If DOWN — classify and act

```bash
# Does a container with name=paperdog exist at all?
STATE=$(docker inspect -f '{{.State.Status}}' paperdog 2>/dev/null || echo "missing")
echo "paperdog state: $STATE"
```

Three cases:

| `$STATE`          | Action                                                                                      |
|-------------------|---------------------------------------------------------------------------------------------|
| `running`         | Port bound but `/` not serving — it's booting. Wait 5s, re-probe `curl`. If still dead, `docker logs --tail 50 paperdog`, then restart. |
| `exited` / `created` / `paused` | `docker start paperdog` (keeps the same image — fast, <2s).                       |
| `missing`         | First time on this host. `docker run` below (pulls image if absent, ~30–90s).               |

Create command for the `missing` case:

```bash
docker run -d --name paperdog --restart=unless-stopped \
    -p 8000:8000 ANON/paperdog:latest
```

`--restart=unless-stopped` means once you create it, the Docker
daemon keeps it alive across reboots and crashes; future cycles
will almost always hit the `running` fast path.

### 0c. Confirm recovery

```bash
# Give the app up to 20s to accept connections after start.
for i in $(seq 1 10); do
    curl -sf -m 2 http://localhost:8000/ -o /dev/null && break
    sleep 2
done
curl -sf -m 3 http://localhost:8000/ | head -c 4000
```

If the manifest comes back — continue to Step 1. Log to
`results/_paper_lake_usage.log`:

```
{timestamp}\tself_heal\t{prior_state}->running\t<ok>
```

If the container still doesn't serve `/` after 20s:

1. Capture diagnostics: `docker logs --tail 100 paperdog > results/_paperdog_boot.log`.
2. Log `paperdog_down` with the prior state and exit reason.
3. **Fall back to WebSearch** for this cycle. Do not loop on
   restart attempts — one heal attempt per cycle, then move on.
4. If the same failure mode repeats for 3 consecutive cycles,
   surface it to the bug_fixer by writing
   `results/_bug_fixer_diagnosis.json` with
   `{"component": "paperdog", "symptom": <last-log-line>}`.

### 0d. Manifest contents (for reference)

The manifest is ~2.3 KB and contains:

- `purpose`, `mode`, `release` metadata
- `n_papers` — current corpus size
- `endpoints[]` — each with `purpose`, `example_request`,
  `example_response_shape`
- pointer to `/openapi.json` for full schemas
- `agent_quickstart` — the hint sequence to follow

## Step 1: Choose the right endpoint from the manifest

Do NOT guess endpoint names. Read `endpoints[]` from the manifest
and pick the one whose `purpose` matches your need. Common ones
(verify via manifest, names may evolve):

- `/explore_approaches` — survey methods for a problem statement.
  Use this for "what techniques exist for <task>?"
- Other endpoints (search-by-title, cite-by-arxiv-id, neighbor
  papers, etc.) — use the manifest's `example_request` verbatim
  as your starting template.

If the manifest advertises an endpoint that fits better than
`/explore_approaches`, use that one instead.

## Step 2: Query shape

Be concrete. A vague query wastes the corpus. Include:

1. Task type + input modality (e.g., "video classification from
   dashcam clips, 2.25s @ 30fps")
2. Key constraints (dataset size, label regime, compute budget)
3. Exact metric (mAP per TTE group, not just "accuracy")
4. Current baseline + target (e.g., "at 0.853, targeting 0.872")

Example:

```bash
curl -sX POST http://localhost:8000/explore_approaches \
     -H 'content-type: application/json' \
     -d '{"problem": "Binary collision anticipation from 2.25s dashcam video, 2839 train clips, mAP over 500/1000/1500ms TTE groups, V-JEPA2 ViT-L baseline 0.8530, target 0.872"}'
```

## Step 3: Verify every cited paper

The paper lake returns arxiv IDs, titles, and claimed numbers.
**Do not trust numbers without verification.** For each paper the
response names:

1. `WebFetch arxiv.org/abs/<id>` — confirm the paper exists and
   the title matches.
2. Confirm the claimed metric/number in the abstract or
   conclusions. Numbers drift; titles are more reliable.
3. If the arxiv ID doesn't resolve, try `arxiv.org/pdf/<id>` or
   WebSearch the title.

**Unverified papers do NOT go into `GOAL.md`.** Drop them or mark
them `[unverified]`.

## Step 4: Write findings

For every verified paper + technique:

- Append to `GOAL.md` under `## Prior Art & Known Solutions`:
  ```
  - [YYYY-MM-DD] <technique>, <metric>=<number> on <dataset>,
    <arxiv URL>. Source: paper_lake. Relevance: <one sentence>.
  ```
- If the technique maps cleanly to our pipeline, add a priority
  entry to `RESEARCH_RULES.md` under the active-tracks section.
- Log the call in `results/_paper_lake_usage.log`:
  `{timestamp}\t{endpoint}\t{n_results}\t{n_verified}`

## Step 5: Usage guidance

paperdog is self-hosted and not rate-limited. Query as often as
the task genuinely needs — it's both a search engine and a
structured knowledge base, and the retrieval is cheap. The only
rules are:

- **Avoid repeat queries.** Grep `results/_paper_lake_usage.log`
  before calling — if the same query ran in the last 5 cycles,
  reuse the prior findings instead of re-querying.
- **Prefer the manifest's suggested endpoints** over free-form
  `/explore_approaches` when a narrower endpoint fits (e.g., use
  `/deep_dive` for a specific arxiv_id you already know).
- **Every new result goes through Step 3 verification** — don't
  skip the arxiv resolve-check just because paperdog returned a
  convincing record.

## Ordering vs other knowledge sources

Per-cycle order for the professor's BFS pass:

1. `professor_paper_lake` (this SOP) — grounded, verified
   techniques from the curated corpus.
2. `professor_web_search` — fresh competition writeups and
   community posts not yet in the lake.
3. `professor_cross_domain_query` — one structural-analogy query
   into a different field.

Paper lake first because its signal-to-noise ratio is highest.
Web search second to catch what the lake does not yet cover.
Cross-domain last as the "wildcard" slot.

## Non-goals

- Do NOT propose experiments in this SOP. You are a knowledge
  scout; the research agent generates ideas from what you record.
- Do NOT dump raw paperdog responses into `GOAL.md`. Only
  verified papers + one-sentence relevance claims.
- Do NOT call paperdog from the research or thinker roles. They
  consume your distilled findings via `GOAL.md` /
  `RESEARCH_RULES.md`.
