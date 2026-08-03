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

## Inherited from informed_flow

The eligibility/lookback confound, best-bar `_coverage`, and the survivorship
caveats all apply here unchanged — see `../informed_flow/IDEAS.md`; fixes should
land there first and be inherited, not forked.
