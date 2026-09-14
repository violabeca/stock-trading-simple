# momentum_swing — Qullamaggie-style swing trading, made systematic and testable

Buy a stock that has already led (up 25%+ in three months), rested in a *tightening* base for
a few weeks, and just closed above that base on expanding volume — or a *neglected* stock that
gapped up 8%+ on 3× volume and held the gap. Risk one ATR-scaled stop, sell on the first
close below the 10-day moving average, and do neither unless the market itself is rising.
Long only, daily bars, one shared wallet, a quarter of the free cash per entry.

This folder is the systematic core of the three-setup framework marketed as "Mogen Trades"
and, before that, popularised by Kristjan "Qullamaggie" Kullamägi (breakouts, episodic
pivots, parabolic shorts). Two of the three setups ship; the third is deliberately absent
(see **Why the parabolic short is not shipped**). Every discretionary judgement in the
source — "orderly" base, "tight" range, "big" volume, "how much to sell at 5×" — is a named
parameter here so it can be searched, ablated and falsified instead of remembered.

---

## 1 · The proposed strategy, in detail

### 1a · What the source actually says (and what is community folklore)

Kullamägi's own published rules (his site, FAQ, and the 2021 *Chat With Traders* interview)
are these; everything else circulating is a paraphrase:

| element | his words | community addition (not his) |
|---|---|---|
| Screen | "the 1-2% of stocks up the most over 1, 3, 6 months", "with enough liquidity" | ADR ≥ 5–6%, $10M/day, "30% minimum move" |
| Breakout | prior move "30-100%+" in 1–3 months; "orderly pullback and consolidation with higher lows and tightening range", 2 weeks–2 months; entry at the opening-range high; "stop is always lows of the day", not wider than 1 ATR/ADR; "sell 1/3 to 1/2 after 3-5 days", move stop to break-even, trail the rest on the 10- or 20-day MA, exit on the first *close* below | any numeric tightness metric |
| Episodic pivot | "gap up 10%+" on huge volume (average daily volume in the first 15–30 minutes); best if the stock "has not rallied over the past 3-6 months"; entry ORH, stop low of day ≤ 1–1.5× ADR; trail the 10/20 MA | "5-10× volume" (Stockbee's number) |
| Parabolic short | "up 50-100%+ in a few days" (large cap) or "300-1000%+" (small); "3-5+ days in a row"; short the opening-range low or a failed VWAP bounce; stop = highs of day; target the 10/20 MA; "5-10× risk/reward" | "never day one, rarely day two" |
| Sizing | "risk 0.25-1%" per trade (FAQ: 0.3–0.5%, rarely 1%); positions 10–20%, "never more than 30% overnight"; 15–20 positions ideal | — |
| Regime | qualitative: "breakouts don't exist in a falling market — sit in cash, trade less, smaller size" | the QQQ/SPY "10 SMA over 20 SMA" traffic light |
| Track record | 268% CAGR 2013-19, 50% drawdown 2014, win rate "25%" (2019) / "35%" (2020); ~$5k → $105M peak (Nov 2021), ~$60M given back by mid-2022 | — |

Nothing in the last row is audited. The win rates are worth more than the CAGR: they say the
method's profit sits in a minority of trades, which is the single most important property for
anyone backtesting it.

### 1b · The systematic translation shipped here

| source rule | what `strategy.py` does | why the translation |
|---|---|---|
| leader screen | breakout requires the return over `lead_window` (63) bars *before* the base ≥ `lead_min` (0.25) | measured before the base so a drifting base cannot eat the move; 25% ships because 30% + a 0.75 contraction fire 22 times in 4.5 years on 500 liquid names |
| "tightening range, orderly" | base = last `base_window` (15) bars; its mean daily range ≤ `contraction` (0.9) × the leader-period mean; high-to-low depth ≤ `depth_max` (0.25) | a volatility-contraction ratio is the only reproducible reading of "tight"; the search may switch it off (≥ 1.2) |
| breakout entry | close above the base's highest high, volume ≥ `vol_mult` (1.5) × 20-day average; **filled at the next open** | daily bars have no opening-range high; the next open is the honest daily-bar entry and it *pays* the gap the ORH entry would capture |
| EP | open ≥ `gap_min` (0.08) above prior close, volume ≥ `gap_vol_mult` (3) × average, close ≥ open (the gap held), prior `lead_window` return < `lead_min` (neglect) | one threshold (`lead_min`) divides "leader" from "neglected" |
| stop = low of day, ≤ 1 ATR | initial stop `stop_atr` (1.5) × ATR(14) below entry, **fixed**, evaluated intrabar by the engine, filled at the true open on a gap-through; setups whose stop would exceed `max_stop` (12%) are skipped | the engine's stop machinery models overnight gap risk exactly as it happens; the LOD of a wide breakout bar is 1–2 ATR anyway |
| sell 1/3 at 3–5R, trail the rest | full take-profit at `tp_r` × stop distance (0 = off); exit on the next open after the first close below the `trail_sma` (10) SMA | the engine has no partial sells; a whole-position target is searchable, and the one public rebuild reports the partial sell hurt |
| regime | `regime`: `none` / `fast` (the panel's equal-weight index 10 SMA > 20 SMA) / `slow` (index > 200 SMA) | the index is the benchmark's own composition, so no outside series and no look-ahead |
| risk 0.25–1%, position 10–20% | `cash_frac` 0.25 of the free cash per entry, one shared wallet | fixed-fraction sizing is what the engine expresses; true risk-per-trade sizing needs a per-entry fraction (engine proposal in **IDEAS.md**). Realised risk per trade = 0.25 × stop%, bounded by `max_stop` at 3% of the book |
| 15–20 positions | no hard cap: each entry takes 25% of the *remaining* cash, so sizes decay geometrically (25%, 19%, 14%, 11%, 8%…) | the wallet has no position count; a signal-rich configuration ends up holding dozens of small positions (§2i) |

### 1c · How a trade works, bar by bar

1. **Day t, at the close.** The scanner sees a name pass every filter (setup + gates). The
   stop distance is computed from day t's ATR and close.
2. **Day t+1, at the open.** Buy 25% of the free cash at the adjusted open. Costs: commission,
   half-spread, slippage (the global model). The stop is `stop_atr` × ATR below the fill.
3. **Any later day, intrabar.** If the low touches the stop, the engine sells at the stop —
   or at the *open* if the day opened below it (gap-through). Costs charged again.
4. **Any later day, at the close.** If the close is below the `trail_sma` SMA, sell at the
   next open. If `tp_r` > 0 and the high reaches the target, sell there.
5. Nothing is ever added to a position, nothing is shorted, and an entry signal on a name
   already held is ignored. There is **no position cap**: each entry takes 25% of the
   *remaining* free cash (25%, 19%, 14%, 11%, 8%…), so a day with many signals fills the
   book with progressively smaller positions and a signal-rich parameter set can hold
   dozens of names at once (the v2 pick held up to 82).

---

## 2 · Pitfalls — the deep dive

The brief that prompted this folder listed gap risk, concentration, an unverifiable track
record, regime dependence and a commercial incentive. All five are right. Here is what each
one costs in numbers, plus the pitfalls the brief missed — several of which matter more.

### 2a · The rules do not contain the edge

The only public clean-room rebuild of the published breakout rules
([quant4free](https://quant4free.com/analysis/qullamaggie-strategy-improved/), ~1,500 US
stocks, 2015-2026) earns **1.7%/yr, Sharpe 0.19, −48% drawdown** raw; with an index filter,
half-out at target and stop-scaled sizing it reaches 6.5%/yr, Sharpe 0.45, out-of-sample
Sharpe 0.28 — against SPY at 13.5%/yr, Sharpe 0.80. The author's conclusion: the returns
"live in the discretion the rules don't contain". Every enthusiast rebuild that *beats* the
index (Stonks Capital: 19% CAGR breakouts, 30% EPs, 28% parabolic shorts) withholds its
rules, states no costs, or both. This is the base rate: **a faithful systematic version of
these rules has, so far, lost to buy-and-hold**, and the repo's benchmark-relative objective
will say so if it does here.

### 2b · Low win rate, outlier-carried profit, and what that does to a backtest

Kullamägi's own 25–35% win rate is typical of the family: Concretum's opening-range-breakout
study wins 24% of trades; Zarattini, Pagani & Wilcox (2025, all liquid US stocks 1950-2024,
66,000+ trades) find **under 7% of trend-following trades generate the majority of the
profit**. Two consequences. (i) A 4.5-year window of a 4-slot book holds a few hundred trades,
of which perhaps twenty matter — an optimiser with 13 knobs will fit *those twenty*. (ii) The
pre-registered event test in the notebook (§4) is the only number that does not depend on
which twenty: it scores every setup occurrence, market-demeaned, before any trading. Read it
first, and read `n_trades` beside every Sharpe.

### 2c · Overnight gaps versus day-session stops

The brief's point, and it is structural, not incidental: Lou, Polk & Skouras (JFE 2019) show
that momentum's profit accrues **entirely overnight**, so a swing breakout *must* carry gap
risk to earn its edge. Concretum's SPY test: holding overnight raised long CAGR from 7.3% to
13.4% and took the book's max drawdown from −25.6% to −45.4%. Here the engine evaluates the
stop intrabar against the low and **fills at the true open when a bar gaps through it**, so
realised losses beyond the planned stop are in every number; the notebook's §12 counts them.
Under fixed-fraction sizing a stop at `max_stop` (12%) with `cash_frac` 0.25 risks 3% of the
book *by plan*; a 20% gap through it costs 5%.

### 2d · Concentration and correlation

Four momentum names selected by the same screen are one factor bet. Lou & Polk's
"comomentum" (pairwise abnormal-return correlation inside the momentum book) *predicts* the
severity of the next momentum crash; Daniel & Moskowitz (JFE 2016) document the crashes —
14 of the 15 worst months came after a negative two-year market return, during the rebound,
which is exactly when a 10/20 SMA gate flips back to green. `tracking_error` (the second
objective) rises with concentration, so the front makes the dial visible; the `retrim`
concentration cap of portfolio mode does not exist in wallet mode, so a runaway name simply
grows its share of the book until the trail sells it.

### 2e · Regime dependence

Cooper, Gutierrez & Hameed (2004): momentum earns 0.93%/month after up-markets and
**−0.37%/month** after down-markets. Barroso & Wang: 52-week-high and price momentum have *no
premium at all* after bear markets or high-volatility states. Kullamägi's own ~$60M
give-back was Nov 2021–mid 2022. The regime gate is therefore the best-supported component of
the whole design — and the literature (Faber 2007; Zakamulin 2014, 2018) says it *halves
drawdowns and adds little or no return*, so expect it to lower `tracking_error` rather than
raise `excess_cagr`. §10c in the notebook scores each calendar year separately; 2022 is the
year to read.

### 2f · The high-ADR screen buys the lottery decile

The community's ADR ≥ 5% filter selects, on average, the stocks the literature says
*underperform*: Bali, Cakici & Whitelaw (2011) find the high-MAX decile loses more than
1%/month to the low-MAX decile, Ang et al. (2006) −1.04%/month for high idiosyncratic
volatility. The offset is Han, Yang & Zhou (2013): a moving-average exit adds the most value
in precisely the highest-volatility deciles. So the ADR floor is only defensible *together
with* the trail — which is why `adr_min` searches down to zero and the trail is not optional.

### 2g · Survivorship, twice

The shared database holds **no delisted names**, so `Liquid500` is drawn from today's
listings and its 2020-24 history is survivor-only. For a breakout strategy this is worse than
usual: failed breakouts in names that later delisted are exactly the losses that are missing.
A survivor-only small-cap index overstates momentum returns by ~5 points a year in one
published measurement. The benchmark-relative objective controls for it partially (strategy
and hurdle trade the same survivors); absolute returns and drawdowns are flattered. The
ranking *inside* the pool is point-in-time (`notebooks/01_universe_screener.ipynb` §8b).

### 2h · Look-ahead in "leader" screens

Any leader list built from full-window returns leaks the future; `HighOctaneTop20` in this
repo did so with a +460%/yr hurdle. Here the universe ranking closes before the backtest
opens, the leader move is computed bar by bar from the past, the market index is the panel's
own equal-weight index, and the stop is shifted onto the fill bar so it carries only
signal-bar information.

### 2i · What the engine cannot model, honestly listed

- **Risk-per-trade sizing.** Position = risk ÷ stop distance is the framework's core risk
  rule; the wallet takes one scalar fraction. Consequence: the per-trade risk floats between
  `cash_frac` × (0.5 ATR) and `cash_frac` × `max_stop`. Engine proposal in IDEAS.md.
- **Partial sells.** "Sell a third at 3–5R" is a whole-position `tp_r` here.
- **Opening-range entries.** Daily bars fill at the next open; the source enters intraday
  above the opening range and stops at its low. The 1-minute data in this DB could express it
  (IDEAS.md), at the cost of a much shorter history.
- **Earnings versus non-earnings gaps.** No earnings calendar exists in the DB, so the pivot
  leg pools both populations — and the literature says only the earnings-catalysed gap drifts
  (Brandt et al.), while attention-driven gaps fade (Berkman et al. 2012; Aboody et al. 2018).
- **A position count.** Wallet mode sizes each entry as a fraction of the *remaining*
  cash and never says no, so nothing stops a signal-rich configuration from holding
  forty $200 positions — uneconomic at any real commission floor, and a diluted index in
  disguise (which the benchmark-relative objective then rewards). A per-day signal cap is
  a stateless strategy-side mitigation (IDEAS.md #9); a true concurrency cap is an engine
  feature.
- **Borrow, locates, squeezes.** Not modelled at all — see the parabolic short below.

### 2j · Costs

Liquid500 names traded in $12.5k clips at the global 9 bps round trip are the friendliest
case this style will ever see; the source trades small caps where 30–100 bps is normal and a
$12.5k order can be several percent of a minute's volume. §10b re-prices the pick at 2× and
3× costs and at the repo's two fixed bars. Dai et al. (2021): trailing stops tighter than
10% lose their benefit after costs.

### 2k · The optimiser will overfit thirteen knobs

Bailey, Borwein, López de Prado & Zhu formalise the probability of backtest overfitting; the
practical defences here are the pre-registered event test (§4), the ablation grid (§5: setup
× regime × trail, 18 combos, no tuning), the admissibility floor (`min_trades_per_year: 20`),
the boundary report (§8), the untouched 2025+ holdout opened once (§10d), and the deflated
Sharpe. A front row that only exists at one corner of the space is noise.

---

## 3 · Published evidence — what is real in this family, and what is not

### 3a · Building blocks with peer-reviewed support

| block | evidence | effect size | caveat |
|---|---|---|---|
| Breakout to highs | George & Hwang (JF 2004): buy the top 30% by price ÷ 52-week high | 0.45%/mo raw, 0.85% risk-adjusted, **no long-run reversal**; dominates past-return momentum | Liu, Liu & Ma (2011): not significant in most markets after costs; McLean & Pontiff: −26% OOS, −58% post-publication haircut |
| Breakout *on volume* | Huddart, Lang & Yetman (Mgmt Sci 2009): volume spikes at the crossing of the prior 52-week high; subsequent returns reliably positive | larger the longer since the last high, the smaller the firm | the closest academic analogue to this exact entry |
| Volume shock | Gervais, Kaniel & Mingelgrin (JF 2001): high-volume return premium | 0.5–0.9% over 20 days, t ≥ 3 | *stronger when not accompanied by an extreme return*; wiped out by 0.2% one-way costs on a 1-week horizon |
| Earnings gaps | Brandt, Kishore, Santa-Clara & Venkatachalam: sort on the announcement return | 7.55%/yr abnormal, no reversal | Martineau (2022): PEAD has been absent in large stocks since 2006 |
| Regime gate | Cooper-Gutierrez-Hameed (2004); Faber (2007); Moskowitz-Ooi-Pedersen (2012) | momentum +0.93%/mo after up-markets vs −0.37% after down; 10-month SMA cuts the S&P max drawdown 83.7% → 42.2% | Zakamulin: return indistinguishable from buy-and-hold; drawdown reduction is real |
| Volatility scaling | Barroso & Santa-Clara (JFE 2015) | Sharpe 0.53 → 0.97, kurtosis 18 → 2.7 | the one intervention that removes the left tail — needs per-trade sizing (IDEAS.md) |
| Stop-loss | Han, Zhou & Zhu (2016): 10% stop on momentum | worst month −49.8% → −11.4%, Sharpe doubled | Kaminski & Lo (2014): stops *lower* return under a random walk; help only when returns are autocorrelated |
| MA trail on volatile names | Han, Yang & Zhou (JFQA 2013) | MA timing adds most in the highest-volatility deciles | the reason the ADR floor and the trail must ship together |

### 3b · Building blocks without support, or with evidence against

- **Volatility contraction predicts direction:** no peer-reviewed evidence; volatility
  clusters (Cont 2001), so a quiet stock predicts a quiet stock. Use it for stop placement.
- **Chasing high-volume winners for months:** Lee & Swaminathan (JF 2000) — high-volume
  winners have *less* momentum and reverse in years 2–3.
- **Generic overnight gaps:** Berkman et al. (2012), Aboody et al. (2018) — attention gaps
  reverse intraday and underperform long-term.
- **High-ADR screens:** the MAX / idiosyncratic-volatility discount (§2f).
- **Short-term reversal shorts at retail scale:** Frazzini, Israel & Moskowitz — the
  reversal premium does not survive costs at size; the borrow cost data (§4) settles the rest.

### 3c · Systematic backtests of this specific style

| study | what | result | costs |
|---|---|---|---|
| quant4free (2026) | faithful breakout rules, ~1,500 US names, 2015-26 | 1.7%/yr, Sharpe 0.19, DD −48%; improved 6.5%/0.45; SPY 13.5%/0.80 | not stated; survivorship admitted |
| Stonks Capital (2025) | breakouts / EPs / parabolic shorts, rules withheld | 19% / 30% / 28% CAGR, DD −21% / −29% / −21% | none; "duplicating his results would be unrealistic" |
| Zarattini, Barbon & Aziz (2024) | 5-min ORB on the 20 highest-relative-volume "stocks in play" | 41.6%/yr, Sharpe 2.81, DD 12%, hit 48% | $0.0035/sh; no OOS split; QuantConnect replication degrades outside 2016 |
| Zarattini, Pagani & Wilcox (2025) | all-time-high breakout + ATR trail, all liquid US stocks 1950-2024, survivorship-free | 15.2% CAGR, 6.2% alpha; **< 7% of trades carry the profit** | gross; turnover control needed below ~$1M |
| Wilcox & Crittenden (2005) | close ≥ all-time high, 10-ATR trail, 24,000 stocks incl. delisted | 19.3%/yr, Sharpe 1.24, DD −33.7% | 0.5% round trip |
| Clenow *Stocks on the Move* replications | regression-slope momentum, SPX > 200 SMA, ATR sizing | 8.7–9.0%/yr vs SPX 8.6–12%; DD −11.7% vs −27.3% | none; **wins on risk, loses on return** |
| SetupAlpha (paid) | parabolic short, Russell 1000 incl. delisted, 2000-present | 13.9%/yr, DD −43%, worst year −15.7% (2025) | $0.005/sh + 2% borrow + 0.1% slippage — the only cost-inclusive short test found |

**Reading across the table.** The rigorous, survivorship-free, cost-aware results (Wilcox &
Crittenden; Zarattini, Pagani & Wilcox) say that *long breakouts to highs with a wide ATR
trail* are a real, extremely skewed edge in liquid US stocks. The faithful rule-based
rebuilds of the *Qullamaggie-specific* rules (tight base, LOD stop, 10 SMA trail, partial
sells) underperform the index. The intraday variant on high-relative-volume names posts the
best headline numbers and the weakest validation. The parabolic short is the least tested
and the least implementable.

### 3d · Recommendation

Trade the **long breakout leg, gated by the index trend, with a fixed ATR stop and a
moving-average trail, on liquid names, sized by risk** — and treat the episodic pivot as an
add-on to be kept only if the earnings-gap population can be separated (it cannot yet, here).
Concretely:

1. **Keep** the leader screen, the tight-base breakout on volume, the fixed ATR stop, the
   MA trail, and the regime gate. These are the components with independent support.
2. **Prefer the 200-day gate to the 10/20 cross** if the study cannot separate them; the
   long gate is what every systematic rebuild that survives ends up with, and the fast cross
   re-enters during exactly the rebounds where momentum crashes.
3. **Widen the stop rather than tighten it** (1.5–3 ATR, searched here); Stonks Capital's and
   Dai et al.'s results both say 1 ATR / < 10% stops lose to costs and noise.
4. **Drop the partial sell** unless the study finds a `tp_r` that beats the pure trail.
5. **Size by risk** (the one intervention that removes the left tail — Barroso & Santa-Clara)
   once the engine can take a per-entry fraction; until then keep `max_stop` tight enough
   that `cash_frac` × `max_stop` is a loss you can carry four times over.
6. **Do not ship the parabolic short** at this account size (next section).
7. **Judge it against buy-and-hold of its own universe**, per year, never on an absolute
   Sharpe. In 2020-24 that hurdle was high; a strategy that is 40% invested must earn back
   the 60% it sat out.

Expected magnitude, honestly: the rigorous long-breakout results are +5–6%/yr of alpha
*gross*, carried by a handful of trades, with 30–45% drawdowns. Kullamägi's numbers are a
different population (small caps), a different execution (intraday), and a different
instrument (discretion). Nothing here should be expected to reproduce them.

---

## 4 · Why the parabolic short is not shipped

The engine can express it (`short_entries` in the same wallet), and IDEAS.md carries the
spec. It does not ship because a backtest of it here would be *wrong in the optimistic
direction*, and the repo's rule is never to publish a number it cannot stand behind:

- **Borrow cost is not modelled.** The names that go parabolic are the names that are hard
  to borrow: over 200 US stocks cost more than 100%/yr to borrow in May 2024, sub-$100M caps
  averaged over 30%/yr in 2023, and half of all names are now "special" (> 1%). At 25%/yr
  that is 0.07% a day against a trade whose gross edge is a few percent.
- **Locate availability and recall are not modelled.** Reg SHO requires a locate before
  every short; Rule 201 blocks short sales for the rest of the day and the next after a 10%
  intraday decline; fails must be closed by T+2. Engelberg, Reed & Ringgenberg (JF 2018):
  recall risk is priced and deters shorting for good reason.
- **Loss is unbounded and the tail is the thesis.** A parabolic name is by definition one
  that has just done something the distribution said it would not.
- **Margin.** FINRA 4210 short maintenance is 30% (≥ $5) and 100% or $2.50/share below $5;
  a $50k margin account can carry perhaps two such positions, each capable of a 50% gap
  against it.
- **Liquid500 rarely produces the setup.** Names doing $90M+/day do not go +300% in three
  days; the small-cap population where they do is not in this DB and would not survive the
  cost model.

The single cost-inclusive public test (SetupAlpha, 2% borrow assumed) reports 13.9%/yr with
a −43% drawdown and a losing 2025. That is the ceiling under a *generous* borrow assumption.

---

## 5 · Account mechanics at ~$50,000

Facts as of September 2026; check them against your broker's current terms.

- **Day trades.** A position bought at the open and sold at a later date is never a day
  trade. A same-day stop-out (bought at the open, stopped that afternoon) *is* one. The
  historic Pattern Day Trader rule (4+ day trades in 5 business days requires $25k) does not
  bind a $50k margin account in any case — and **FINRA has replaced it**: SEC approval of
  SR-FINRA-2025-017 (April 2026), effective 4 June 2026 with a firm phase-in to 20 October
  2027, deletes the PDT definition, the $25k minimum and day-trading buying power in favour of
  intraday-margin-deficit monitoring (a customer who repeatedly runs deficits can have credit
  frozen for 90 days). Your broker may still apply the old rule until its phase-in date.
  The notebook's §13 counts the strategy's same-day round trips per 5-session window anyway:
  at the defaults on the train window there were none.
- **Settlement.** T+1 since 28 May 2024. In a **cash** account, selling a security bought
  with unsettled proceeds is a good-faith violation (three in twelve months → 90 days of
  settled-cash-only trading); a swing book therefore has capital locked for one day after
  each exit. A **margin** account has none of this and is the practical choice — the
  strategy never uses leverage (the wallet cannot exceed 100% invested), so margin is a
  settlement convenience, not a risk.
- **Reg T / maintenance.** 50% initial, 25% maintenance long (30% short). Irrelevant at
  ≤ 100% invested; relevant only if a short leg were ever added.
- **Sizing translation.** `cash_frac` 0.25 of a $50k book is a $12.5k first position
  ($9.4k, $7.0k, $5.3k, $4.0k… for the following entries, as the free cash shrinks; at the
  shipped defaults the median position over the train window was $9.7k, the book held 2.6
  names on an average in-market day and up to 24 at the peak, and was in the market on 51%
  of days).
  Planned risk per trade = 0.25 × stop distance: 1.9% of the book at a 7.5% stop, 3.0% at
  the `max_stop` of 12%. The source's own 0.25–1% risk would size those positions at
  $1.7k–$6.7k — smaller, not larger — which is the per-entry sizing IDEAS.md proposes.
- **Costs.** $12.5k in a name trading $90M+/day is invisible; the modelled $0.005/share
  with a $1 minimum plus 2 bps spread and 1 bp slippage is close to what a US discount
  broker charges. It would not be close for the source's small-cap universe.
- **Time.** One scan after the close, orders for the next open, stops resting at the
  broker. No intraday attention is needed, which is the reason the daily-bar translation
  was chosen over the 1-minute one.

---

## 6 · How it is tested (the notebook)

`momentum_swing.ipynb` follows the repo's flow with three additions specific to this style:

- **§4 pre-registered event test** — before any optimisation, every occurrence of each raw
  setup (and of the gated signal actually traded) is scored by its market-demeaned forward
  return from the next open over 5/10/21/42 bars, with a t-statistic on monthly-clustered
  means. **Pass** = the 10- and 21-day means are positive with t > 2 for the gated signal.
  Fail = stop; write the result into IDEAS.md; do not run §6.
- **§5 ablation grid** — `setup` × `regime` × `trail_sma` (18 combos): does either leg carry
  the result, does the gate help, does the slower trail pay.
- **§12 gap-risk audit** — for every closed trade, planned risk (the stop at entry) versus
  realised loss: the share of losers that lost more than planned and the worst gap-through.
- **§13 account translation** — the same trades at $50k: clip sizes, planned dollar risk,
  concurrent positions, same-day round trips per 5-session window.

Plus the mandatory sections every strategy has: baseline overlay against
`strategies/baseline` (§3), boundary check (§8), hurdle on train *and* holdout (§10a), cost
stress (§10b), per-year table (§10c), the untouched 2025+ holdout opened once (§10d).

---

## 7 · Parameters

| param | shipped | searched | grid | meaning |
|---|---|---|---|---|
| `setup` | both | ✓ | breakout / pivot / both | which legs may enter — the ablation |
| `lead_window` | 63 | ✓ (21–252, log) | — | bars of the leader move (before the base) / the neglect window (pivot) / the ADR window |
| `lead_min` | 0.25 | ✓ (0.05–1.0) | — | leader return floor (breakout); a pivot requires less |
| `base_window` | 15 | ✓ (5–60, log) | — | consolidation bars; breakout = close above their highest high |
| `contraction` | 0.9 | ✓ (0.4–1.2) | — | base mean daily range ÷ leader-period mean, at most |
| `depth_max` | 0.25 | — | — | base high-to-low depth, at most |
| `adr_min` | 0.03 | ✓ (0–0.08) | — | average daily range floor |
| `vol_mult` | 1.5 | ✓ (0.5–4) | — | breakout-day volume ÷ 20-day average |
| `gap_min` | 0.08 | ✓ (0.03–0.20) | — | pivot: open above prior close by at least |
| `gap_vol_mult` | 3.0 | ✓ (1–8) | — | pivot: gap-day volume ÷ 20-day average |
| `regime` | fast | ✓ | none / fast / slow | market gate on the panel's equal-weight index |
| `regime_fast` / `regime_slow` / `regime_long` | 10 / 20 / 200 | — | — | the gate's SMAs |
| `avg_window` / `atr_window` | 20 / 14 | — | — | volume average / ATR windows |
| `stop_atr` | 1.5 | ✓ (0.5–4) | — | initial stop in ATRs, fixed |
| `max_stop` | 0.12 | — | — | skip setups with a wider stop; bounds risk per trade at `cash_frac` × this |
| `trail_sma` | 10 | ✓ (5–50, log) | 10 / 20 | exit on the next open after a close below this SMA |
| `tp_r` | 0 | ✓ (0–12) | — | whole-position take-profit in R; 0 = trail only |
| `cash_frac` | 0.25 | — | — | one shared wallet; fraction of free cash per entry |

---

## 8 · Caveats

- **Survivor-only universe** (§2g). The benchmark-relative numbers are the honest ones.
- **Time in market is not normalised away.** A regime-gated, selective scanner sits in cash
  most of the time; against a fully invested equal-weight hurdle that is a real cost, paid
  on purpose.
- **The 2020-24 hurdle is high.** Equal-weight Liquid500 over the train window was a strong
  bull tape with a bear year inside it; beating it long-only with four slots is a tall order,
  and a documented negative is a legitimate outcome (see `turnover_momentum`, `gap_and_go`).
- **Pooled gaps.** The pivot leg cannot tell an earnings gap from a sympathy gap.
- **Fixed-fraction sizing** floats the per-trade risk (§2i).
- **Momentum crashes.** Nothing caps the factor; the trail caps each name.

## Sources

Kullamägi, *My 3 timeless setups* and *Episodic pivots* (qullamaggie.com), FAQ, *Chat With
Traders* ep. 212 (2021) · George & Hwang, JF 2004 · Huddart, Lang & Yetman, Mgmt Sci 2009 ·
Gervais, Kaniel & Mingelgrin, JF 2001 · Lee & Swaminathan, JF 2000 · Brandt, Kishore,
Santa-Clara & Venkatachalam (SSRN 909563) · Martineau, CFR 2022 · Berkman, Koch, Tuttle &
Zhang, JFQA 2012 · Aboody et al., JFQA 2018 · Cooper, Gutierrez & Hameed, JF 2004 · Daniel &
Moskowitz, JFE 2016 · Barroso & Santa-Clara, JFE 2015 · Faber 2007 · Zakamulin 2014, 2018 ·
Bali, Cakici & Whitelaw, JFE 2011 · Ang, Hodrick, Xing & Zhang, JF 2006 · Han, Yang & Zhou,
JFQA 2013 · Han, Zhou & Zhu 2016 (SSRN 2407199) · Kaminski & Lo, JFM 2014 · Dai, Marshall,
Nguyen & Visaltanachoti, IRF 2021 · Lou, Polk & Skouras, JFE 2019 · Frazzini, Israel &
Moskowitz (SSRN 2294498) · Engelberg, Reed & Ringgenberg, JF 2018 · Zarattini & Aziz 2023
(SSRN 4416622) · Zarattini, Barbon & Aziz 2024 (SSRN 4729284) · Zarattini, Pagani & Wilcox
2025 (SSRN 5084316) · Wilcox & Crittenden 2005 · Bailey, Borwein, López de Prado & Zhu
(SSRN 2326253) · quant4free, *Qullamaggie strategy rebuilt with discipline* · Stonks Capital,
*Modeling Kullamägi* parts 1–3 · SEC Release 34-105226 (SR-FINRA-2025-017), FINRA Regulatory
Notice 26-10 · SEC T+1 press release 2024-62 · Acadian, *The incredible cost of short
selling* (S&P Global data).

---

## Result (2026-09-13): a documented negative, as the base rate predicted

Everything below is from the shipped notebook flow on `Liquid500`, train 2020-07-01 →
2024-12-31 (4.5 years, 500 names), holdout 2025-01-02 → 2026-09-11, one shared $10k wallet,
costs on, next-open fills. The hurdle is equal-weight buy-and-hold of the same 500 names:
**+14.7%/yr on train, +21.3%/yr on the holdout.**

**§4, the pre-registered event test — FAIL.** Market-demeaned forward return from the next
open, monthly-clustered t:

| population (shipped defaults) | events | 10d mean | 21d mean | 42d mean | t (21d) |
|---|---|---|---|---|---|
| breakout, raw | 286 | small positive | small positive | +0.8% | ~1.1 |
| pivot, raw | 234 | +0.3% | +0.2% | −1.7% | ≈ 0 |
| gated signal (traded) | 176 | positive, not significant | ≈ 0 | ≈ 0 | < 2 |
| source-faithful breakout (30% / 0.75 contraction / 3.5% ADR) | 22 | **+3.4%** (hit 73%) | +3.0% | +3.3% | 1.2 (10d: 2.0) |

The strict setup shows the shape the source describes — a few percent of market-relative
drift over two to eight weeks — on 22 events in 4.5 years across 500 liquid names. Loosened
to a tradable rate the drift shrinks below a percent; the pivot leg carries nothing past ten
days in this universe (consistent with PEAD's disappearance from large stocks and the fading
of attention gaps).

**§3, shipped defaults:** CAGR +3.9%/yr, **excess −10.8%/yr**, information ratio −0.54,
Sharpe 0.34, max drawdown −26%, 165 trades (37/yr, 95 breakouts / 70 pivots), 36% winners,
median hold 11 days, in the market 51% of days. Holdout: +8.6%/yr vs +21.3%, excess −12.7%.
Per year the gate did exactly what the literature says: **2022 excess +9.5%** (strategy −5%,
hurdle −15%) and negative in every other year.

**§7–§8, the study.** v1: 450 trials from three seeded workers = 300 duplicate parameter
sets and 93% pruned on the 20/yr floor (IDEAS.md #2); 33 scored, none beat the hurdle
(best −8.2%/yr). v2 (narrowed bounds, one worker): 405 trials, 169 scored, front of two,
**none beats the hurdle**. The pick — pivot-only, no gate, 11% gap on 1.5× volume, 1 ATR
stop, 44-day trail, 8R target — reads +12.9%/yr, Sharpe 0.97, max drawdown −24%, 181 trades
on train: **excess −1.8%/yr, information ratio −0.12**; on the holdout +9.2%/yr vs +21.3%,
**excess −12.1%/yr**, information ratio −0.63. The v2 front's best row before the notebook's
resume (a 3%-gap scanner, 715 trades, 2R target) was −3.0%/yr on train and −20.7%/yr on the
holdout with up to 82 concurrent positions. Every well-scoring row moved the same way: no
regime gate, smaller gaps, weaker volume filters, more time in market — the optimizer
drifting toward the index, which the benchmark-relative objective is designed to expose and
the boundary report flagged (three parameters at a bound, all in the "less selective"
direction). The cost stress is the one clean pass: the pick's Sharpe falls only 0.97 → 0.94
at 3× costs, and the breakeven half-spread is beyond 60 bps — liquid names, few trades.

**§13, the gap audit (pick, train):** 50% of losing trades lost more than planned (median
overshoot +2.5 points beyond a 4.5% median stop, worst +9.5 beyond a 6.2% stop; worst single
trade −18%); at the shipped defaults 37% of losers overshot by a median +1.8 points. The
brief's central caution is quantified, not hypothetical: with day-session stops on
overnight holds, roughly a third to a half of the losses land beyond the stop.

**§14, at $50k (pick):** median position $8.1k, median planned risk $310 (0.62% of the
account — inside the source's 0.25–1%), realised +$36k over 4.5 years against +$60k for
buy-and-hold, up to 42 concurrent positions (the no-position-cap effect, §2i), in the market
79% of days, **zero same-day round trips** — no day-trade rule would ever have bound.

**Status: falsified on this data, twice** — the event has no significant market-relative
edge at a tradable rate, and no parameter set beats owning the universe, in or out of
sample. The absolute numbers are respectable (Sharpe ~1, drawdown −15 to −25%, a positive
2022) and that is the Clenow pattern: *wins on risk, loses on return*. What would reopen it,
in order of leverage: an earnings-calendar split of the pivot leg (IDEAS.md #3); a second
point-in-time universe through 2015–2020 so the gate is scored across two bear markets
(#6); per-entry risk sizing (#1) and a per-day signal cap (#9), which together would stop
the front rewarding dilution. The parabolic short stays out regardless (§4).
