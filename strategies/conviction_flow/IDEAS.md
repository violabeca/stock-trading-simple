# conviction_flow — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why).

## Validation

### Lever-attribution report before believing any study

The five selectivity levers interact — `min_agree` and `min_score` both throttle
opens, quality-gating shrinks the pool `displace_margin` acts on — so a front pick
can credit one lever for another's effect, and five extra dimensions on ~4 years of
daily train is a real overfitting surface. Before a full study is trusted, run the
marginal ablation: each lever alone at its picked value, everything else at
informed_flow defaults, and report the marginal `excess_cagr` per lever. A lever
whose marginal contribution is ~0 inside the joint pick is a free parameter the
optimizer used as noise.

## Signal construction

### Soft quality tilt vs the hard quality veto

`min_quality_pct` is a binary gate: a name below the percentile cannot open,
however strong its flow composite. The competing hypothesis — blend a small quality
weight INTO the composite and let strong flow overrule mediocre quality — is one
`w_quality` param and an A/B against the veto. The gate wins if quality's value is
mostly avoiding disasters (left-tail truncation); the tilt wins if it is monotonic.

### `exit_decay` keyed to composite level, not rank

The decay exit fires on a fall from peak *rank* while held. Rank is relative: a name
can hold its score while peers improve past it. A variant keyed to composite *level*
decay would be consistent with `min_score` (which is level-based) and separates
"this name got worse" from "other names got better" — different theses about when
conviction has actually faded.

## Assessment 2026-09-12 — fold, don't fork

### The fork is stale, and "byte-identical to informed_flow" is no longer true as shipped

`tests/test_conviction_flow.py::test_off_defaults_match_informed_flow` passes only by
setting `displace_margin=1.0` **and `w_recency=0.0`** on the parent — its own docstring says
this strategy "still exits on the eligibility cliff this sleeve exists to remove". Relative
to informed_flow v21 the fork is missing: the `recency` sleeve; the single-axis
`displace_margin` (it still carries the `displace` boolean, whose margin is unidentifiable
when False — the bug that put 33 identical rows on informed_flow's v18 front); `min_events`
is still *searched* 1–3 (the cliff that owned 71% of exits on informed_flow's v20 pick);
`_eligibility` double-counts a day carrying both an insider filing and a rating change;
weight floors are 0.0 (the wasted-dimension problem the parent moved to 0.25); no
`retrim_k`; and it trains 2021–2024 on 100 names against the parent's 2018–2024 on 110, so
the README's "directly comparable in `04_compare`" is also false.

### The quality gate inherits value_factor's diagnosed defect

Gross profitability covers ~40–50% of LiquidLeaders (banks, insurers, energy have no gross
profit) and `_quality` drops a name missing *any* present factor, collapsing the quality
cross-section to ~46 names and excluding every financial. `min_quality_pct > 0` is a sector
veto, not a quality gate, and the composite it ranks on has rank-IC ≈ 0 (2026-07-20
diagnosis in `../value_factor`). The study journal (`conviction-flow-1d-v2`) is empty; 18
searched dimensions on four years of daily data is the overfitting surface the item above
warns about.

### Recommendation

Add the two levers that are genuinely orthogonal to informed_flow — `min_agree`
(independent sleeve agreement) and `exit_decay` (rank-decay exit) — as opt-in parameters
in `informed_flow._select` (~20 lines, default off, with a pinned test that defaults are
bit-identical to today and `[off, on]` sweep-grid columns), then delete this folder and its
test. Drop `min_quality_pct` (IC ≈ 0 evidence) and `min_score` (redundant with `q_in`).
*Wins if* either lever's grid column beats plain informed_flow in §10b·ii's folds; *dies if*
neither does, which is the "the flood of qualifiers was already the right breadth" reading
the README anticipated. Re-syncing the fork is the alternative: more code for the same
information.

## Inherited from informed_flow

The eligibility/lookback confound, best-bar `_coverage`, and the survivorship
caveats all apply here unchanged — see `../informed_flow/IDEAS.md`; fixes should
land there first and be inherited, not forked.
