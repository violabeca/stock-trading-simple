# fade_movers — the day's steepest moves, to look at

**Research only: there is no `strategy.py`, `config.yaml` or study.** The idea under study: short the
day's biggest gainers and buy its biggest losers, near the close or once the price turns.

| file | what |
|---|---|
| `fade_movers_explore.ipynb` | screens every stock with continuous 1-minute data, finds each day's steepest rise and fall over `WINDOW` minutes (15), lists the week's steepest ones and draws their 1-minute candles |

The scan reads the 1-minute bars week by week and caches each week in `var/fade_movers/` (git-ignored).
The default window ends on 2026-04-30, so the locked 2026-05-01+ holdout stays unread.

## What a first study found (2026-10-14, outside this notebook)

A multi-agent study of daily bars from 2015 to 2026 and 1-minute bars from 2025 to 2026 found no edge.
Shorting the top 10 gainers and buying the top 10 losers at the close, then exiting at the next close,
lost about 12 bp a day before costs on large caps in 2015-2022. It also failed a pre-registered test on
2023 to mid-2025. Entering on a turn did no better than entering at a random minute. The day's
gainers tend to keep rising overnight.
