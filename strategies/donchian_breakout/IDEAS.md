# donchian_breakout — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why).

## Data window — the one thing to fix before anything else

`data.start: 2026-05-01` with `end: null` is ~2.5 months of 1m data and **no
train/holdout split at all** — every sibling 1m strategy trains on the shared
2025-07-01 → 2026-04-30 window and keeps the tail as untouched holdout. The README
calls the short window deliberate WIP, but any optimized pick from this config is
in-sample only and not comparable on the 04_compare scoreboard. Set the shared
window and reserve the tail before trusting (or comparing) any study result.

## Signal construction

### Participation gate on the breakout

squeeze_breakout's ~2,000-trial ablation on the same universe found the *volume*
gate is the one that earns its keep (disabling it roughly flips the sign), while
this strategy takes every channel touch. A `vol_mult` gate against the time-of-day
profile baseline (`bt.volume_profile`, already in the engine) is the cheapest
evidence-backed upgrade. Falsify with the gate-on/off sweep pair.

### Breakout-distance filter

Require the close to clear the prior channel extreme by ≥ k·ATR rather than touch
it. Marginal false breaks at the channel edge are the classic Donchian failure
mode; a distance floor trades entry count for follow-through quality — exactly the
trade the ATR trail monetizes.

## Search space

- `atr_window` is a singleton `[14]` in the sweep but searched 7–100 in optuna —
  either sweep a real spread or freeze it as structural (squeeze precedent) and
  drop it from the optuna space.

## Known caveats (documented, not actionable here)

- 2 bps default spread understates 1m HighOctane friction — cost-stress rerun
  mandatory (candle_scalp technique).
- `HighOctaneTop20` is a today-snapshot universe (survivorship).
