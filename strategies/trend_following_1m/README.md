# trend_following_1m — the trend follower on 1-minute bars of ten liquid stocks

**The question.** Daily bars are coarse for a trend follower: a breakout is only seen at the
close and a stop only checked once a day. This folder runs **the same code** as
[`trend_following`](../trend_following/README.md) on 1-minute bars. `strategy.py` re-exports
that module, so the two cannot drift apart. The trend start is a Donchian breakout plus ADX
confirmation. The exits are a ratcheting ATR chandelier or an opposite start, and every stock
gets one equal slot. Every window is in minutes. Two intraday rules are added:
`skip_open` / `skip_close` block starts in the first and last 15 minutes of a session, and
`eod_exit` (searched) either closes everything before the session's last bar or holds overnight.

**The stocks** (`Morning10` in `config/settings.yaml`) are HOOD, MU, COIN, SMCI, INTC, ORCL,
GS, NOW, MRVL and SPOT. They are the ten most liquid names of `Morning200`, ranked by median
dollar volume over 2025-01..06, a window that closes **before** the 1-minute backtest opens.
There is one 10% slot per stock, so the relative-strength admission of the daily strategy is
inert here.

**Windows and objective.**
* **Data:** stored 1-minute history 2025-06-18 → 2026-06-26, 100,230 bars per stock.
* **Train:** 2025-07-01 → 2026-04-29, scored warm.
* **Holdout:** 2026-04-30 → 2026-06-26, 39 sessions, read once.
* **Objective:** `[alpha_cagr, volatility]`, as `level_flow` uses. Buy-and-hold of these ten
  stocks returned **+54% over the ten train months and +27% in the two holdout months**, so an
  excess return over it condemns any book that is flat overnight, whatever it selects.
* **Costs:** $5,000 orders (10 slots of a $50k account), so ~8 bp per round trip.

## Result

**One-minute bars do not rescue the strategy.** The optimized 1-minute book is slightly positive
on both train and holdout, but statistically indistinguishable from zero. It rests on trading
very rarely, and slightly higher costs or worst-case fills erase it.

| 1-minute pick (v1 study) | train | holdout |
|---|---|---|
| return over the period | +3.8% | +3.1% |
| buy & hold of the ten stocks | +53.9% | +26.7% |
| alpha (per year) / beta | +4.9% / −0.01 | +24.1% (2 months, annualized) / −0.01 |
| daily P&L, day-level t | +1.7 bp, **t 0.84** | +7.9 bp, **t 0.99** |
| trades | 311 | 51 |
| alpha at 6/5 bps · pessimistic fills | +0.3% · **−3.6%** | +19.4% · +12.5% |

The pick is a 31-minute channel with **ADX ≥ 57**, a 28-ATR chandelier on ATR(47), flat at the
close.
* It only trades the most violent intraday bursts.
* It holds about 0.9 of its 10 slots on average.
* Of its 311 train exits, 272 were the session end; only 34 were the ATR stop and 5 opposite
  starts.
* Alpha peaks at an ADX floor of 54–57. A floor of 50 gives **−5.5%/yr**; above 60 the alpha
  falls and the book trades less than the 250-a-year floor. So the boundary flag on `adx_min`
  was checked, not widened.

## Why

* **Intraday trend starts carry no information worth the costs.** The event test (notebook §4)
  used train only, with channels of 15–240 minutes and ADX floors of 0 and 30. Over the next
  5–120 minutes within the session, forward returns sit within ±5 bp, with every day-clustered
  |t| ≤ 1.2. A round trip costs ~8 bp. No exit rule can turn that into an edge; the study can
  only find a rare subset that happened to pay.
* **These stocks made their money overnight.** Over train, per stock and day, the overnight gap
  averaged **+11.6 bp** and the session itself **−8.0 bp**. A flat-at-close book forfeits the
  only leg that paid. The study's highest-alpha rows hold overnight (+10–18%/yr on train), which
  is exposure the objective charges for, not intraday trend skill.
* **More bars mean more breakouts and more stops hit by noise**, and every one pays the spread.

## Granularity: 1 minute, 5, 15, one day — same stocks, same windows

The 5- and 15-minute studies are scratch runs: candles derived from the same 1-minute history
(`data.resample`), 300 trials each, with the same search spans in time. The daily row is
`trend_following`'s shipped parameters, never fit on these stocks, with one slot per stock and
the same costs.

| bars | trials with alpha > 0 | of which flat at the close | pick | train alpha | train t | holdout return | holdout t |
|---|---|---|---|---|---|---|---|
| 1 minute | 60 / 292 | 31 / 156 | 31 min, ADX ≥ 57, flat at close | +4.9% | 0.84 | +3.1% | 0.99 |
| 5 minutes | 63 / 275 | 30 / 162 | 8.5 h channel, ADX ≥ 38, holds overnight | +13.0% | 1.18 | +2.9% | 0.47 |
| 15 minutes | 25 / 259 | **0 / 167** | 9 h channel, ADX ≥ 38, holds overnight | +4.9% | — | +1.6% (alpha −6%) | — |
| 1 day | — | — | shipped daily config | −0.3% | — | +3.3% (beta 0.5) | — |

No granularity shows an edge distinguishable from zero on these stocks and windows. The 5-minute
pick looks best in-sample, but ORCL made +8.3% of its +9.7% train return. Nothing comes close to
simply owning the ten stocks through a +54% / +27% run. That says as much about the window as
about the strategy.

## Caveats

* The windows are short: ten months of train and 39 holdout sessions. The ten names are
  correlated (fintech and semis/software).
* Each pick is the best of 300 trials, which weakens its in-sample t further. The holdout agrees
  in sign but not in significance.
* `alpha_cagr` is `cagr − beta × benchmark cagr`, so it **credits a net-short book for the
  market rise it stood against**. In the notebook's ablation, a short-only book that holds
  overnight *lost* 19.6%/yr and still reads +25.5% alpha (beta −0.66 against +69%/yr). The pick
  has beta ≈ 0, where alpha is simply its own return. Read `cagr` beside alpha on any one-sided
  row.
* `bt.atr` puts the overnight gap into the first bar's true range, so morning stops are wider
  than intraday volatility warrants (IDEAS.md #3).
* The 5- and 15-minute studies live outside the repo. To reproduce one, set `data.resample: 5`
  (or 15), bump `optuna.study` and divide the window ranges by the bar length.

**Run it:** open `trend_following_1m.ipynb`, or `uv run tsim optimize --strategy trend_following_1m`
(~5 s a trial). Next steps and their falsification tests are in `IDEAS.md`.
