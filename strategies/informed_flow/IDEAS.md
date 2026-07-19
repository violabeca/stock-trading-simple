# informed_flow — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why).

## Open questions

### Pessimistic fill pricing — when in the day do positions actually open/close?

Worth checking how positions are opened and closed and at which time of the day.
Today every fill (entry and exit) happens at the **next bar's adjusted open**
(`price=panel.open_` in both engine modes), which is symmetric and neutral —
realism comes only from the per-leg `fees`/`slippage` fractions.

For a deliberately **pessimistic** bound, one should probably open a position at
the **day's high** and close it at the **day's low**. That would answer: does the
edge survive the worst plausible intraday execution, or does it live inside the
open-to-open assumption? A middle ground (e.g. slippage stress via the existing
cost-override technique used for candle_scalp) may be enough; the high/low bound
is the extreme end of that spectrum.

Note: fill pricing lives in the engine (`backtest.run` / `_run_portfolio`), so
actually implementing this is an engine change — proposal-only per repo rules.
