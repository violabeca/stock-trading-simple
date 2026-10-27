# dividend_cycle — buy high-dividend stocks before the payout, short right after

**Verdict (2026-10-27): no edge.** See `RESEARCH_LOG.md` for the pre-registration (§0), the
numbers (§1), the post-hoc diagnostics (§2) and the verdict (§3).

The rule: for a regular quarterly payer whose last four dividends are ≥ 3% of the price, buy at
the open 15 sessions before its *predicted* ex-dividend date (predicted from past ex-dates only),
sell at the open of the actual ex-date, and short from that same open for 10 sessions. Returns
are measured against the equal-weight index of all high-yield names (any cycle phase) and against
the market, on the lab's point-in-time top-500 universes (2016-07 → 2026-10) and on names 501-1000.

| sample | both legs, net bp per event | month-clustered t |
|---|---|---|
| 2016-07 → 2021-12 (TRAIN) | +63.7 | 3.25 (+24.4, t 2.0 without 2020) |
| 2022 → 2026-10, top 500 | −26.1 | −1.64 |
| 2022 → 2026-10, names 501-1000 | −61.3 | −2.64 |

The effect in 2016-2021 comes from 2020's high-yield rebound; it is absent afterwards.

```bash
uv run python strategies/dividend_cycle/research.py fetch   # Yahoo ex-dates -> var/dividend_cycle/ (~30 min, throttled)
uv run python strategies/dividend_cycle/research.py run     # the tables (first run loads daily bars from the DB, ~5 min; then cached)
uv run pytest tests/test_dividend_cycle.py -q               # synthetic: point-in-time prediction, yield, windows
```

Ex-dates come from Yahoo (`Ticker.history(actions=True)`), cached outside the shared DB; the repo
has no dividend table. Yahoo gives no historical pay dates, so "payout" is read as the ex-date
(the day the price drops) and, post hoc, as ex + 15 sessions. If dividends ever become a
`Panel.features` series, it needs a builder in `features.py` (proposal-only).
