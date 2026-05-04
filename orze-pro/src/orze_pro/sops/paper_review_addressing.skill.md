---
id: sop-paper-review
name: paper_review_addressing
role: professor
order: 50
produces: []
consumed_by: []
requires: []
trigger: manual
---

# Addressing Top-Venue Paper Reviews

**Principle:** No rebuttals. The paper speaks for itself. Every reviewer
concern must be resolved *inside* the paper so a fresh reader never
encounters the same question.

**Trigger reason:** {trigger_reason}

---

## Phase 1: Triage (read everything before touching anything)

1. **Read the review** — extract each distinct complaint into a list.
2. **Read the full paper** — understand what it currently says about
   each complaint.
3. **Read the data** — find result files, logs, JSONs that contain
   numbers relevant to each complaint. Note what exists vs what's
   missing.
4. **Classify each complaint:**
   - **Reframing** — data exists, paper just presents it wrong
     (cheapest fix)
   - **Missing evidence** — data exists on disk but isn't in the paper
     (add it)
   - **Missing experiment** — need to run something new (most
     expensive; avoid if possible)
5. **Check for invisible ablations** — if a reviewer asks for an
   experiment you already ran, the label is wrong, not the experiment.
   Relabel the caption/heading with explicit terms ("schema-matched
   ablation," "causal backbone ablation," etc.).
6. **Deflect "missing experiment" requests** — before running anything,
   check if existing data already constitutes the experiment under a
   different name. Reframe first, run only if nothing fits.

---

## Phase 2: Narrative strategy (decide before editing)

### Core narrative

7. **Identify self-inflicted wounds** — places where the paper
   apologizes for something that's actually fine. These are the #1
   source of reviewer complaints. Remove the apology, replace with the
   positive framing.
8. **Pick numbers that serve the narrative.** Show only the numbers
   that advance the story. Two rows that tell the whole contrast beat
   six rows where four are filler. But: "pick which numbers" means
   choosing what to show, not picking the *most flattering version* of
   a number you do show.
9. **Turn counterevidence into thesis support.** If a result looks bad
   in isolation (e.g., baseline > your method), check if it actually
   supports the thesis from a different angle. If so, say so
   explicitly. Counterevidence you own is stronger than counterevidence
   a reviewer discovers.

### Headline honesty

10. **Abstract numbers must be conservative.** If you report a range in
    the body, the abstract shows the range or the conservative end —
    never the favorable end alone.
11. **Disclose asymmetric comparisons near the claim, including in the
    abstract.** If a result depends on an asymmetry (temporal,
    methodological, resource), state it explicitly next to the claim —
    in the abstract if the claim appears there, not buried 5 pages
    later.

### Definitional challenges

12. **If a reviewer disputes your taxonomy:** (a) acknowledge the
    boundary is fuzzy and show the claim survives reclassification, or
    (b) strengthen the definition with a sharper criterion. Prefer (a).
13. **If a reviewer says "a human/simple baseline could do X
    trivially":** the definition must preempt this. State that the
    boundary is relative to deployment-time knowledge. The relevant
    question is not "could someone have done it" but "did the initial
    setup anticipate it."
14. **Limitations must be argumentative, not just acknowledging.** "No
    X ablation" invites "then run it." Instead: "No X ablation:
    decomposing X requires N arms and tests a different capability; the
    end-to-end pipeline is auditable via released logs." Give the
    reader a reason to accept the gap.

### Capability claims

15. **Distinguish implementation gaps from theoretical
    impossibilities.** "System X cannot do Y" invites "but a future
    version could." Instead: "no current system does Y; the gap is
    empirical, not a theoretical impossibility." This preempts the
    logical-flaw attack.
16. **Disclose the mechanism behind each capability claim.** If a
    system discovered something via web search, say so; if via
    parametric knowledge (training data), say so and note the
    knowledge-cutoff bottleneck. Reviewers will ask.

### Practical applicability

17. **If a reviewer says your tool/diagnostic can't be used in
    practice:** add (a) a low-cost estimation procedure, (b)
    cross-domain priors from your own experiments, and (c) a decision
    rule. Never leave the reader wondering "but how do I actually use
    this?"
18. **Never claim a hard threshold for a continuous diagnostic.** A
    data point near the threshold makes it look arbitrary. Frame as
    regime ranges with noisy boundaries and CIs: "the gap between
    regimes is large even though individual boundaries are noisy."

---

## Phase 3: Edit (surgical, not cosmetic)

19. **Optionally spawn an independent reviewer agent** — give it the
    review, the paper, and data files. Let it independently assess
    which complaints are real vs red herrings.
20. **Edit in order: abstract → intro → contributions → body →
    conclusion → appendix → checklist.** Each downstream section must
    be consistent with upstream claims. Before editing, audit paragraph
    placement: does every paragraph belong in its section? A misplaced
    paragraph signals careless editing to reviewers.
21. **For each complaint, the fix is ONE of:**
    - Remove defensive language (delete the self-inflicted wound)
    - Add a sentence with data
    - Add a small table (only if the contrast needs >1 number)
    - Add a clarifying sentence
    - Relabel an existing experiment/ablation
22. **Never add a new section or paragraph just to address a review.**
    Weave fixes into existing structure.

---

## Phase 4: Verify

### Number & consistency checks

23. **Number consistency** — grep every instance of key numbers and
    verify they match ground truth data files. Round correctly
    (0.9093 → 0.909, not 0.910). If a statistic was computed under
    non-i.i.d. conditions (adaptive sampling, active learning), the
    abstract must lead with the valid (i.i.d.) estimate and explicitly
    flag the inflated one.
24. **Sample-size consistency** — if N differs across sections, label
    the pool composition explicitly. A hostile reviewer will assume
    contradictions are errors, not different pools.
25. **Metric/reward consistency** — if the system uses metric X during
    operation but the paper references metric Y for analysis,
    explicitly state whether Y was used by the system or computed
    retrospectively. Contradictions here look like methodological
    errors.
26. **Cross-table consistency** — if Table A says X is best but Table B
    shows it's worst, explain why (e.g., different regimes, different
    subsets).
27. **Procedural ambiguity** — any step that touches test data must
    state what information is available at test time. A hostile reviewer
    will assume oracle leakage.

### Compilation & compliance

28. **Compile LaTeX** — 2+ passes, zero errors, zero undefined refs.
29. **Verify page limit** — main text within limit, references start on
    the correct page.
30. **Re-read the abstract cold** — does it tell the full story without
    leaving any reviewer complaint as an open question?
31. **Cross-regime pool-composition check.** If comparing a statistic
    across two experimental conditions, verify the candidate pool is
    identical. A "4× drop" that partly reflects removing a dominant
    outlier from the pool is a confound, not a finding. State the pool
    composition difference next to the comparison.
32. **Stress-test headline claims with edge-case arithmetic.** For
    every "X suffices" or "X dominates" claim, check if the data
    contains a counterexample where the suppressed factor still matters
    practically. High variance explained by A doesn't mean B is
    zero — check the absolute gap from B.
33. **Statistical-validity check for information criteria.** If the
    paper reports AIC, BIC, or likelihood-ratio tests, verify the error
    model's assumptions hold. Cumulative-max curves, autocorrelated
    time series, and step functions violate i.i.d. — report these as
    descriptive fit metrics with an explicit caveat, or replace with
    assumption-free alternatives ($R^2$, bootstrap).
34. **Grid-sensitivity caveat for variance decomposition.** If the
    paper reports $\eta^2$, $R^2$, or any ANOVA-based diagnostic,
    acknowledge that effect sizes depend on the chosen factor levels
    (e.g., wider LR range inflates $\eta^2_{\mathrm{lr}}$). Frame the
    diagnostic as ``run YOUR grid, measure, decide'' — not a universal
    constant.
35. **Endogeneity check for diagnostics.** If the paper proposes a
    diagnostic (``when X is high, do A; when low, do B''), check
    whether ``low'' has multiple interpretations. E.g., low variance
    can mean ``factor doesn't matter'' OR ``current levels are
    uniformly weak.'' Add a disambiguation rule (e.g., pair with
    absolute performance check against known SOTA).
36. **Reference accuracy check.** Verify every bibliography entry:
    (a) the cited paper exists at the stated venue/arXiv ID,
    (b) author names are spelled correctly (grep the actual paper),
    (c) any work the codebase explicitly builds on is cited.
    A single uncited dependency or misspelled author name signals
    careless scholarship and invites distrust of all other claims.
37. **Code-paper consistency check.** Before submission, diff every
    factual claim against the released code: (a) if the paper says
    ``no X enters the pipeline,'' grep the code for X; (b) if the
    paper describes model variant A, verify the deployed script uses
    A, not B; (c) if the paper excludes a baseline, the code must not
    contain that baseline at a competitive score without disclosure.
    A reviewer with code access will find every discrepancy.

---

## Phase 5: Retrospective Rebalancing (every 3–5 review rounds)

36. **Audit tone drift.** Diff the paper against the version that
    achieved peak scores. Categorize every change as: defensive caveat,
    compression, offensive addition, or bug fix. If defensive caveats
    outnumber offensive additions 2:1+, the paper has drifted toward
    hedging.
37. **Remove phantom hedges.** A hedge is ``phantom'' if no reviewer has
    ever actually attacked the point it defends. Cut it.
38. **Demote caveats that undermine the contribution.** If a caveat tells
    the reader ``our diagnostic might not mean what we say,'' move it to
    a footnote or appendix.
39. **Restore compressed persuasive evidence.** If compressions removed
    trial counts, effect sizes, or negative results that friendly
    reviewers praised, reverse them. Cut lower-value defensive text
    instead.
40. **Check section demotion.** If Limitations/Conclusion was demoted
    from \texttt{\\section} to \texttt{\\paragraph} for space, restore
    it. Section headings signal completeness.
41. **The ``cold read'' test.** Read abstract and conclusion fresh. Does
    it sound confident and well-bounded, or defensive and uncertain?
    Friendly reviewers give 8 when the paper knows what it proved.
42. **Confound isolation check.** If one method differs from baselines in
    multiple ways (e.g., architecture quality AND feature dimension),
    reviewers will attack the uncontrolled dimension. For every causal
    claim, cite existing controls next to the claim.

---

## Anti-patterns

- Writing a rebuttal document — the paper must stand alone
- Adding defensive hedging ("we acknowledge this limitation but...")
- Showing all available data — pick what serves the narrative
- Creating new sections to address reviewer points — integrate into
  existing flow
- Favorable rounding — always round toward ground truth
- Leaving discrepancies unexplained — one sentence fixes most
- Headline numbers from a biased subset — abstract must use
  conservative estimates or ranges
- Asymmetric comparisons without disclosure — state advantages near
  the claim
- Hard thresholds on continuous diagnostics — use regime ranges
- Acknowledging a limitation without arguing why the gap is
  acceptable — every limitation needs a reason the reader should
  accept it
- Running an ablation the reviewer asked for that you already have —
  relabel, don't re-run
- Claiming "X cannot do Y" when you mean "no current X does Y" — the
  logical-flaw attack is easy to mount
- Claiming a capability without disclosing its mechanism — reviewers
  will ask web search vs parametric knowledge vs human-in-the-loop
- "X suffices" when your own data shows the suppressed factor has a
  large absolute gap — check edge cases before claiming dominance
- Claiming a clean N× transition when the experimental pools differ —
  always verify the same candidates appear in both conditions
- "Strictly worse" on a gap within test-set noise — use "numerically
  worse" or provide a significance test
- Reporting AIC/BIC on non-i.i.d. data (cumulative max, time series)
  without a caveat — reviewers trained in statistics will flag this
- Framing SSC as "just retrieval" — if the agent wrote integration
  code, composed systems, or designed recipes, say so explicitly
- Classifying the same technique as two different categories in
  different paragraphs — grep for each taxonomy label and verify
  consistency across the paper
- Accumulating defensive caveats until the paper sounds uncertain —
  every 3–5 rounds, diff against peak-score version and audit tone
  drift (Phase 5)
- Demoting Limitations/Conclusion to ``\paragraph'' for page space —
  reviewers read section structure as a signal of completeness
- Leading the abstract with ``which inflates X'' or ``this is biased''
  — put the confident number first, caveats in the body
- Releasing code that references uncited work — grep code for paper
  names, arXiv IDs, and method names; cite every dependency
- Code claiming ``no X'' while the script does X — a reviewer with
  code access will check; disclose or fix
- Leaving PII in anonymous repos — scrub notes, build scripts, and
  paths must not contain usernames, hostnames, or org names

---

## Convergence tracking

After each review round, record:

| Review | Score | New edits | Already handled | % pre-handled |
|---|---|---|---|---|

Track the ratio of pre-handled complaints across rounds. If it drops
(regression), identify which new complaint vector the SOP didn't cover
and add the corresponding rule. Convergence means the ratio stabilizes
above ~80% and new edits are structural (require new experiments, not
paper fixes).
