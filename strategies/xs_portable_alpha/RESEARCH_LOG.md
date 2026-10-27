# xs_portable_alpha — research log

The cross-strategy round of the proposals lab (`strategies/proposals_lab`, lab v2). The design
was selected by a design panel and a judge (`final.selected[]`, key `xs_portable_alpha`). This
log holds the pre-registration, every run and the declared final. Results are in
`var/proposals_lab/results2/xs_portable_alpha/`. The pre-registrations are SHA-256-snapshotted to
`var/proposals_lab/results2/xs_portable_alpha/prereg/` before the runs they govern.

## §0 Pre-registration (2026-10-23 23:53 CEST, before any number of this design's books)

What existed on lab panels before this section was written:

- The code (`strategy.py`, `research.py`, `config.yaml`) and its synthetic-panel tests (21
  passing).
- A smoke run of `research.py audit` on ONE period of TB and of T1 and of `table` on
  xs_three_book's saved `C_core`, every output suppressed: it reported only "ran, n lines".
  No number of this design was seen.
- `research.py preview` (no label): the judge's return sum of the saved component books. It
  reproduced the design's preview table exactly:
  - TB CAGR 6.05% / B&H 11.22% / alpha +1.11% / beta 0.44 / maxDD −15.7%;
  - T3 17.03% / 35.17% / +1.66% / 0.44 / −4.9%;
  - T1 10.56% / 8.77% / +7.11% / 0.39 / −8.4%;
  - T2 8.04% / 7.63% / +5.55% / 0.33 / −9.5%;
  - TRAIN (TB + T3) raw alpha +1.43%.

  These are the evidence the design was chosen on, not new information.

### Question

How much of buy-and-hold's bull-market CAGR can an index core buy back, while the account still
beats the equal-weight market in every bear episode and keeps a positive alpha net of the core's
own alpha? This is the round's required absolute-performance design.

### Mechanism and evidence (from the design; read-only previews of saved v2 books)

- Under `lock_cash` a hedged book stays at a gross of at most about 0.90, so it cannot reach a
  bull market's CAGR. Beta bought outright (SPY) recovers part of it. The SI + MOM overlay supplies
  alpha and cuts losses in falls.
- The judge's return sum at c = 0.45 (no SPY costs, no netting, daily rebalanced; reproduced
  above):

  | test | CAGR | B&H | SPY | alpha | beta | maxDD |
  |---|---|---|---|---|---|---|
  | T1 | 10.6% | 8.8% | 11.9% | +7.1% | 0.39 | −8.4% |
  | T2 | 8.0% | 7.6% | 11.9% | +5.6% | 0.33 | −9.5% |
  | TB | 6.0% | 11.2% | 8.8% | +1.1% | 0.44 | −15.7% (B&H −37.4%) |
  | T3 | 17.0% | 35.2% | 35.3% | +1.7% | 0.44 | −4.9% |

- Bear episodes in that preview: −11.6 / −15.5 / −6.1 / −8.0% against the market's −18.2 /
  −37.0 / −24.5 / −19.5%. Down months −0.85% on T1, −2.4% on TB.
- **The core is confounded by composition.** SPY's own alpha against the lab's equal-weight
  benchmark: E0 −2.1, TB −1.8, T3 +5.6, T1 +4.4 (P 0.97), T2 +6.1 (P 0.97). So the risk-adjusted
  bars read alpha NET of c × SPY's alpha.
- **No trend gate:** etf_trend I5v2 (SPY's 12-month trend) lost nearly the full fall in all four
  episodes, with VALID alpha −3.59%/yr.
- **The vol rule is only an iteration:** m < 1 on 56-57% of TB/T1 days. At c = 0.45 it cut CAGR
  in TB (1.6% vs 6.0%) and T3 (12.2% vs 17.0%), and helped 2022 (+2.4% vs −6.1%) and drawdowns.

### In-sample cautions (stated beside every result)

1. **TRAIN = TB + T3.** E0 has no short interest. An E0 book of this account would be MOM ×
   0.25 (SI silent: the coverage switch puts its budget in cash) plus 0.45 SPY. That book is
   disclosed beside A4's TRAIN bar as a return sum of MOM's saved `U0v2` E0 book and SPY, not an
   engine book. MOM's own E0 is −8.2%.
2. **T1 is partly in-sample for SI.** informed_flow's studies scored from 2020-07, and
   informed_flow_ls was developed on 2022-26 top-500 data. Only MOM is clean on T1. AH is not in
   this overlay: xs_three_book uses only its `_on_priced_opens` execution helper, so AH's TB + T3
   tune does not touch this book.
3. **TL is fresh only for the combination.** informed_flow_ls §38P already read names 1501-2000
   for SI. TL is read by the orchestrator, once, in one batch, never by me. T4 and TL are not read
   here.
4. **c = 0.45 is the a-priori equal split.** But the judge's c-frontier was computed on
   already-read VALIDATION books. The frontier labels here are information only.

### U0 (label `U0`, overrides `{}`; the config's values)

The construction, in `strategy.py` `signals()`:

1. **Overlay** = `xs_three_book.signals(panel, sleeves="si,mom", gross=0.90 − 0.45)`.
   - SI (`informed_flow_ls_p1` at its config values, lab label `v6_si`) and MOM
     (`resid_momentum` `U0v2`) each get 0.225 of the account: SI × 0.225/0.98, MOM × 0.225/0.90.
   - Each component runs unchanged on the panel without SPY. The sleeves are summed and netted
     per name, the coverage switch applies (SI in cash below 50% coverage), and every target
     change waits for the name's next priced open.
   - xs_three_book's `strategy.py` is loaded by path under `_xspa_tb`. `strategy.SHA256` pins it
     and the six component files it reads, and a changed file refuses to load.
2. **Core:** the SPY column (`lab.with_spy`, the lab's `--hedge`) is set to 0.45 on every bar.
   - The engine buys it on each period's first bar and never re-sends the target, so there are
     no re-trims: about 2 orders a year.
   - `retrim_k` 2 cannot bind on it (it would need SPY at 0.90 of the book).
   - The lab's SPY column has open = high = low = close, so the core fills at the fill bar's
     close.
3. **Gross and costs.**
   - Total nominal gross is 0.90.
   - Borrow is the overlay's (SPY 0.5%, never used); `retrim_k` is 2.
   - Fills at the next open (SPY: that bar's close); `order_notional: 1000`.
   - The declared account is about $150k: overlay slots of 0.56-0.75% ≈ $850-1,100. 2× costs
     bound a $75k account.
4. `data.features: [short_interest, fund_shares]`; nothing is searched (`optuna: n_trials 1`).

### Labels (every one: `lab.py run xs_portable_alpha <label> '<overrides>' TB,T3,T1,T2 --costs2 --hedge`)

| label | overrides | role | counted |
|---|---|---|---|
| U0 | `{}` | the design | variant 1 |
| I1 | `{"core_vol": true}` | pre-registered iteration: the falling-market core | variant 2 |
| F15, F30, F60 | `{"core": 0.15}` / `0.30` / `0.60` | the c-frontier: information only, never adopted | no |
| (C_core) | read-only: xs_three_book's saved `C_core` = this account at c = 0 (the same overlay at 0.90, the same hedged panels and benchmark) | the frontier's c = 0 point | no |

- **Variant budget: 2 (U0, I1).**
- The design's other iteration ideas are parked in `IDEAS.md`, because no budget is left and the
  lab has no RSP:
  - the overlay replaced by xs_three_book's declared final CRISIS (MOM + AH) at 0.45;
  - an equal-weight index ETF as the core.

**I1, operationally** (`core_vol: true`; `strategy.core_scale`):
- SPY's weight is c × m_t, computed on SPY's own rows.
- v_t is the variance (ddof 1) of SPY's 21 close-to-close returns ending at t.
- M_t is the median of v over the trailing 252 sessions, t included, with at least 126.
- m = 1 if v_t ≤ M_t, 0.5 if v_t ≤ 2 M_t, 0 above; m = 1 before the first valid median.
- m is decided only on resid_momentum `_month_ends` rows of SPY's priced rows, shifted one bar,
  and held to the next month end.
- The freed gross sits in cash, and the overlay is never re-traded because of m.
- This is the judge's preview definition (`xs_judge/abs_preview.py`), restated by a loop in the
  tests.

### Definitions fixed now

- **The core's own alpha.** s is SPY's daily close-to-close return (`lab.spy()`) on the label's
  days, and h_t the core multiplier in force at bar t (U0: 1; I1: m_t). α(·) is
  `lab.alpha_of` against the label's OWN benchmark (the period's names plus SPY, equal weight,
  bought and held), on the stitched days of the test.
- **Net alpha (the A4 gate, the design's definition):** net = α(book) − c × α(h·s).
  - P(net > 0) is the share of the lab's 2,000 stationary-bootstrap resamples (`lab.block_index`,
    seed 7, the blocks `lab.bootstrap` uses) on which α_book − c·α_core > 0, both on the same
    resample.
  - Net late-fill, net 2×-cost and net top-5-out subtract the same c × α(h·s) from the lab's late,
    2×-cost and top-5-out alphas.
  - I1's one-bar timing offset is negligible: the engine re-sizes SPY at the close of the bar where
    m changes, so that bar's return is earned at the old size.
- **Disclosed beside it: net of the core-only book.** net' = α(book) − α(core book). The core book
  is an account that holds only SPY at c·h of its value, bought on the first bar, re-sized at the
  close of each target change, the rest in cash, no costs (`research.core_book`).
  - net' is 0 for an overlay that holds only cash.
  - The design's net is not quite 0 there. A core held at weight c has a smaller volatility drag
    than c × SPY's CAGR: α(core) − c·α(SPY) ≈ c(1 − c)σ²/2, about +0.4 pt a year at σ = 18% and
    +0.6 at σ = 22%. This is an analytic expectation, written before any number.
  - **Status rule:** a test where the design's net passes and net' does not is flagged "passes on
    the drag term only". A design carrying that flag on any A4 test cannot be labelled "works".
- **G8 for this design:** the T1 alpha with the daily P&L of the OVERLAY's five biggest
  contributors removed (`lab.top_out`'s rule, SPY never counted as a name), net of c × α(h·s). The
  lab's own G8 (`lab.py verdict`) counts SPY as a name, so there the core is likely the first name
  removed. It is reported as printed.
- **CAGR vs SPY:** SPY's CAGR over the same days. **Sharpe:** the mean of daily returns over their
  sd, annualized, with no risk-free rate. **Max DD ratio:** the book's stitched max DD over
  buy-and-hold's on the same test (B&H from the label's benchmark).
- **The stitched record:** TB + T3 + T1 (2018-07 → 2026-10, top 500 rolling).
- **Episodes:** `lab.episodes`: Q4 2018 and COVID on TB, the 2022 bear and Feb-Apr 2025 on T1. Loss
  share = book return ÷ market return.

### Order of work

1. **Execution audit, before any return is read** (`research.py audit '{}' TB,T3,T1,T2`).
   - Mechanics only: the base, late-fill and 2×-cost books of every period, on lab v2's runner.
   - Every rebuild must equal `lab.book`.
   - It counts refused orders, part-filled orders (below 99.5% of the shares asked, and below
     99%), NaN-price drops, stuck runs, position-days off target, missing gross, core orders, the
     largest realized gross, the smallest free cash and the core's realized weight range.
   - Invariants that must hold exactly, or the code is wrong and is fixed before any run:
     - the overlay equals xs_three_book's `si,mom` book at 0.45 (max |diff| 0);
     - SPY's target is 0.45 on every bar;
     - nominal gross ≤ 0.90;
     - the coverage switch is never off inside a window;
     - rebuild = `lab.book`.
   - **No gross rule.** The spec fixes gross 0.90 and c 0.45. Any refused or part-filled order is
     counted and quantified (step 3), never fixed by a design change inside the budget.
2. **Plumbing** (`research.py reproduce`). The account at `core: 0` must rebuild xs_three_book's
   saved `C_core` exactly (max |daily diff| 0) on T1 2024-H1, TB 2020-H1 and T2 2026-H2.
3. **U0**: the run, then **the reconciliation before reading** (`research.py diagnose U0`, every
   OPEN test). It reports:
   - refused and part-filled orders by class, with free cash;
   - stuck positions with their bars and their P&L from the due open;
   - off-target days and the missing exposure;
   - alpha per test and TRAIN with the stuck P&L removed. If that moves any test's alpha by more
     than 0.5 pt, it is stated beside every number of the label.

   Then the **engine-vs-preview bar**: |engine U0 T1 alpha − preview T1 alpha| ≤ 1.5 pts, both
   against U0's own benchmark (`research.py preview U0`). The design expects about −0.3 pt. A
   larger gap is an implementation error, found and fixed before anything else is read.
4. F15, F30 and F60 (same flags), each with `diagnose`.
5. `lab.py noise xs_portable_alpha U0 5 3`, then `research.py noise U0`.
6. `lab.py luck xs_portable_alpha U0 T1 100` (SPY held fixed: every shuffled book holds the same
   core, so the percentile is net of the core by construction).
7. `lab.py verdict xs_portable_alpha U0 --not-neutral`, then `lab.py bear xs_portable_alpha U0 F30
   F60`.
8. `research.py table U0,F15,F30,F60,xs_three_book/C_core` (the frontier and the design's
   table), `research.py bars U0`, and the E0 disclosure.
9. **I1**: `audit '{"core_vol": true}'`, run, `diagnose`, `research.py bars I1 --base U0`. I1's
   luck run (for its G4) is made only if every other adoption condition holds.
10. **The declared final**, written into this log before any TL read (none here).

### Bars (fixed now; the design's pre-registered bars)

**Absolute bars:**
- **A1:** CAGR > the equal-weight buy-and-hold CAGR on T1 and on T2.
- **A2:** the book beats the equal-weight market in every bear episode (G6's first leg). The loss
  is also reported as a share of the market's loss.
- **A3:** |max drawdown| ≤ ½ of buy-and-hold's on TB, T1 and T2.

**A4, risk-adjusted bars on NET alpha** (the definition above):
- TRAIN (TB + T3) > 0;
- T1 > 0 with P ≥ 0.90;
- T2 > 0;
- G4: luck ≥ 95th percentile, with SPY held fixed by `lab.luck`;
- G5: T1 late-fill and 2×-cost net alpha > 0;
- G8: T1 net top-5-out (overlay names) > 0.

**Reported, not gated:**
- G6's down-month leg (expected negative by design, beta ≈ 0.4);
- CAGR vs SPY;
- TB and T3 CAGR vs buy-and-hold (expected to lose without leverage);
- the stitched 2018-07 → 2026-10 CAGR and Sharpe vs buy-and-hold and SPY;
- the c-frontier table;
- the lab's own `verdict --not-neutral`, as printed;
- the net' disclosure;
- the hair orders at $150k, re-priced at the $1 minimum (orders under $250), beside G5.

**I1 adoption:** only if all of these hold:
- A1-A4 all pass for I1 (its own luck run);
- every episode return of I1 > U0's;
- |max DD| of I1 < U0's on T1 and on T2.

With no adoption, the final is U0. The protocol's generic iteration bar (paired Δ ≥ 0 on TRAIN,
T1 and T2, with a T1 Δ above 2 × U0's T1 noise sd) is reported beside I1 as information. I1 is a
risk iteration, and the design's risk bar replaces the generic one.

**TL** (the orchestrator's, once, for the declared final): A1 and A4 on TL, net of c × SPY's TL
alpha. Small caps lagged in 2022-26, so the core flatters TL.

### Expected (the design's; before any number)

**U0** (an engine haircut of about −0.3 pt vs the preview):
- **T1:** CAGR ≈ 10.3% vs buy-and-hold 8.8% (A1 passes) and SPY 11.9%. Net alpha ≈ +5% (P ≥
  0.95).
- **T2:** CAGR ≈ 7.7% vs 7.6%, so A1 on T2 is a coin flip. Net alpha ≈ +2.9%.
- **TB and T3:** TB ≈ 5.7% vs 11.2%; T3 ≈ 16.7% vs 35.2%.
- **TRAIN** net alpha ≈ +0.6%.
- **Bear episodes:** ≈ −11.6 / −15.5 / −6 / −8%, all beaten; Q4 2018 at about 64% of the market's
  loss.
- **maxDD:** TB ≈ −16%, T1 ≈ −8.5%, T2 ≈ −9.5%.
- **Down months** ≈ −0.9%.
- **Beta** 0.35-0.45. **Luck** ≥ 95th percentile.

**I1:**
- CAGR lower: TB ≈ 1.6%, T1 ≈ 9.2%, T2 ≈ 6.7%, so it likely fails A1 on T2.
- 2022 ≈ +2% instead of −6%, and smaller drawdowns.
- Expected not adopted.

**My own priors, added here:**
- **Mechanics:** nothing refused. The overlay is half of xs_three_book's C_core, which had no
  refusal. The core's gains are unrealized and never raise free cash, while target-percent orders
  are sized on the marked book. So a strong SPY half-year inflates new overlay orders against
  static free cash: about 10% of the book at day 0. I expect free cash ≥ 4% everywhere.
- **The drag term:** the design's net alpha will sit about 0.4-0.6 pt above net' on every test.
  On TRAIN (expected +0.6%) that is the whole margin, so A4's TRAIN bar may pass "on the drag
  term only".

**The honest headline:** no hedged design beats buy-and-hold CAGR in a strong bull (T3) without
leverage.

### Failure modes (the design's, watched, not fixed)

1. **Composition confounding.** SPY beat the equal-weight universe in 2022-26 (+4.4 / +6.1 pts a
   year). That flatters raw alpha and A1: by about +2.0 (T1), +2.7 (T2), and more on TL. Net alpha
   carries the risk gates.
2. **Bull-market CAGR cannot keep up.** T3 ≈ 17% vs 35%, and the stitched 2018-26 CAGR stays below
   buy-and-hold's 13.8%.
3. **A1 on T2 is narrow** (+0.4 pt in the preview).
4. **G6's down-month leg fails by construction.** The user must accept negative down months as the
   price of bull participation.
5. **The overlay inherits SI + MOM's failure modes:** the shared 2026-H2 loss, the junk-rally
   crashes of MOM's shorts, and SI's in-sample T1.
6. **The vol rule (I1) exits V-shaped falls late** and misses rebounds (TB 2019).
7. **Small overlay slots (0.56-0.75%) make hair orders** at small accounts. The declared account is
   ~$150k; 2× costs bound a $75k one.
8. (mechanics, mine) The core's unrealized gains do not free cash. Every lab book restarts from
   cash each half-year, so the lab cannot show the multi-year drift a live account would carry:
   the core's weight would wander far from 0.45 without a rebalancing rule.

### Machine rules

- At most one heavy Python process of mine at a time (two only if `memory_pressure` ≥ 35% free).
- Long jobs run in the background, with logs under `var/proposals_lab/results2/xs_portable_alpha/`.
- No IBKR. No other strategy's folder is touched. `informed_flow_ls` is never edited, imported or
  loaded.

## §1 Execution audit and plumbing (2026-10-23 23:54-23:57; no return read)

**Audit** (`research.py audit '{}' TB,T3,T1,T2`, `results2/xs_portable_alpha/audit_U0.log`):
every period's base, late-fill and 2×-cost books, on lab v2's runner.

| test | periods | orders (base) | refused | part-filled (< 99.5% / < 99%) | NaN-price | stuck | off-target days | core orders | max realized gross | min free cash | core realized weight |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TB | 4 | 712 | 0 | 0 / 0 | 0 | 0 | 0 | 4 | 0.962 | 8.3% | 0.355-0.483 |
| T3 | 3 | 531 | 0 | 0 / 0 | 0 | 0 | 0 | 3 | 0.892 | 9.9% | 0.447-0.494 |
| T1 | 10 | 1,776 | 0 | 0 / 0 | 0 | 0 | 0 | 10 | 0.940 | 8.1% | 0.371-0.483 |
| T2 | 10 | 1,746 | 0 | 0 / 0 | 0 | 0 | 0 | 10 | 0.969 | 6.8% | 0.360-0.495 |

- **Nothing is refused, part-filled, dropped or stuck** in any of the 27 periods. This holds for
  all three books: base, late fills and 2× costs. On the late and 2× books the smallest free
  cash is 6.5% and 6.7% (T2). The late book's T2 has 1,748 orders against the base book's 1,746:
  two more retrims of longs that crossed twice their slot at the worse prices.
- **Every invariant holds exactly**, so the code does what §0 says:
  - every rebuild equals `lab.book` (max |diff| 0.0);
  - the overlay equals xs_three_book's `si,mom` book at 0.45 (max |diff| 0.0, and its SPY column
    is 0);
  - SPY's target is 0.45 on every bar;
  - nominal gross ≤ 0.90;
  - the coverage switch is never off inside a window.
- **The core is bought once per period and never re-traded** (4 / 3 / 10 / 10 orders). Its
  realized weight drifts inside each half-year: down to 0.355 in TB 2020-H1 (COVID) and 0.37 in
  T1 2022-H1, up to 0.495. So the account holds less beta in a fall than its 0.45 target.
- My prior (free cash ≥ 4% everywhere, nothing refused) holds.

**Plumbing** (`research.py reproduce`, `reproduce.log`). The account at `core: 0` rebuilt
xs_three_book's saved `C_core` book with a max |daily diff| of **0.0** on T1 2024-H1, TB 2020-H1
and T2 2026-H2. So these are exact against an independently saved book: the wrapper (loading by
path, the pins, the overlay call, the SPY column, borrow, `retrim_k`) and the lab runner.

## §2 U0 (`lab.py run xs_portable_alpha U0 '{}' TB,T3,T1,T2 --costs2 --hedge`, 2026-10-23 23:57-23:59)

**The reconciliation before reading** (`research.py diagnose U0`; `diagnose_U0.log`,
`diag-U0.pkl`).
- The rebuild with order logs equals the saved returns (0.0) in all 27 periods.
- Refused 0. Part-filled 0 (below 99.5%, and below 99%). NaN-price drops 0. Stuck positions 0.
  Off-target days 0. Missing exposure 0.
- So there is no stuck P&L to remove: **every U0 number below is the account the strategy asked
  for.**
- Hair orders at the declared $150k: 26-55 a year are under $250. Re-pricing them at the $1
  minimum costs 0.01-0.03 pt a year (0.05-0.10 pt for every order under $1,000).
- The net re-pricing of every order is −0.00 to −0.05 pt a year. It is negative: the lab charges
  the core's two large orders a year 10 bp, while a real $67k SPY order pays $1.

**The engine-vs-preview bar passes** (`preview_U0.log`). The engine alpha minus the preview's,
both on U0's own benchmark:

| | TB | T3 | **T1** | T2 | TRAIN |
|---|---|---|---|---|---|
| gap (pts) | −0.90 | −0.11 | **−0.17** (bar: ≤ 1.5) | −0.39 | −0.15 |

The design expected about −0.3. Where the larger gaps come from (`U0_gap_periods.log`):
- **TB is mostly the COVID V.** The engine holds the core: its realized weight fell to 0.355 by
  2020-03-23 and missed part of the rebound. The preview rebalances to 0.45 every day and so
  buys the dip. In 2020-H1 the engine's core leg is −1.84 pts below 0.45 × SPY.
  - −0.42 of that is day 0. The lab's SPY column has open = close, so the core is bought at the
    first day's CLOSE and misses that day's move: SPY +0.94% on 2020-01-02.
  - The rest, −1.42, is buy-and-hold against daily rebalancing through a V.
- **The day-0 convention is a lab-wide fact**, not an error here. It costs the core 0.45 × SPY's
  first-day return each half-year: Σ +2.2% over TB's four periods (≈ −0.5 pt a year of CAGR on
  TB) and +0.75% over T1's and T2's ten (≈ −0.07 pt a year).
- **The remaining gaps are path effects** of an account that drifts against component books that
  each drift on their own (±0.5 pt in a half-year, mostly cancelling).

**Results** (`U0_table.log`, `U0_detail.log`). Alpha is against the equal-weight buy-and-hold of
the period's names plus SPY; net = alpha − 0.45 × SPY's own alpha on the same days and blocks;
net' = net of the core-only book.

| | TB | T3 | **TRAIN** (TB + T3) | **T1** | **T2** | 2018-07 → 2026-10 (TB + T3 + T1) |
|---|---|---|---|---|---|---|
| CAGR | 4.75% | 17.20% | 9.91% | **10.29%** | **7.56%** | 10.12% |
| buy & hold CAGR | 11.24% | 35.18% | 20.93% | 8.73% | 7.62% | 13.71% |
| SPY CAGR | 8.75% | 35.29% | 19.42% | 11.90% | 11.90% | 15.00% |
| raw alpha (P) | +0.20% (0.49) | +1.52% (0.64) | +1.27% (0.63) | +6.95% (0.999) | +5.16% (0.99) | +4.67% |
| SPY's own alpha | −1.82% | +5.52% | +0.13% | +4.44% | +6.14% | +2.82% |
| **net alpha (P)** | +1.02% (0.61) | −0.97% (0.31) | **+1.21% (0.67)** | **+4.96% (0.994)** | **+2.40% (0.92)** | +3.40% (0.98) |
| net' (core book) | +0.98% | +0.02% | +0.99% | +4.77% | +2.23% | +3.21% |
| net late / 2× / top-5-out | −0.48 / +0.66 / −0.69 | −2.43 / −1.32 / −2.79 | −0.26 / +0.86 / −0.08 | **+3.47 / +4.60 / +3.21** | +0.93 / +2.04 / +0.96 | |
| beta | 0.40 | 0.45 | 0.41 | 0.38 | 0.31 | 0.40 |
| vol / Sharpe | 11.4% / 0.46 | 7.8% / 2.07 | 10.0% / 0.99 | 9.2% / 1.11 | 8.7% / 0.88 | 9.5% / 1.06 |
| B&H Sharpe / SPY Sharpe | 0.53 / 0.46 | 2.03 / 2.20 | 0.96 / 0.94 | 0.54 / 0.74 | 0.46 / 0.74 | 0.73 / 0.83 |
| max DD (B&H's) | −14.7% (−37.4%) | −5.1% (−7.7%) | | −8.1% (−25.0%) | −9.5% (−24.0%) | −14.7% (−37.4%; SPY −33.7%) |
| max DD ratio | 0.39 | 0.66 | | 0.32 | 0.39 | 0.39 |
| orders a year | 357 | 354 | | 371 | 365 | |

- **Per period** (raw alpha, annualized):
  - T1: +1.4, +4.2, +12.0, +1.8, +25.5, +1.7, +8.2, +14.7, −1.2, −10.3;
  - T2: +3.8, −4.3, +7.6, −1.7, +24.1, +4.2, +9.5, +5.9, −0.9, +0.1;
  - TB: −3.8, −4.0, 0.0, +4.5;
  - T3: −7.5, +0.4, +9.2.
- **Falling markets** (book; the equal-weight market in brackets; loss share):
  - Q4 2018 −11.7% (−18.1%), 64%;
  - COVID −14.5% (−37.0%), 39%;
  - 2022 bear −6.8% (−24.5%), 28%;
  - Feb-Apr 2025 −7.7% (−19.5%), 40%.

  The market's down months: book −1.16% against −4.11% (n 38).
- **The c × α(SPY) and the core-book definitions differ by −1.0 to +0.2 pt, not by a steady
  +0.4-0.6 as §0 expected.** My analytic prior counted only the volatility drag (which favours
  the c × α(SPY) definition). It missed compounding convexity: in a strong half-year c × SPY's
  compounded return exceeds a core held at weight c. That makes the c × α(SPY) definition the
  stricter one on T3 (−0.97 vs +0.02). On the A4 tests both are positive (TRAIN +1.21 / +0.99,
  T1 +4.96 / +4.77, T2 +2.40 / +2.23), so no "drag term only" flag applies.

**The pre-registered bars on U0** (G4 waits for the luck run, §4):

| bar | value | passed |
|---|---|---|
| A1 T1: CAGR > buy & hold | 10.29% vs 8.73% (+1.56) | yes |
| **A1 T2: CAGR > buy & hold** | **7.563% vs 7.620% (−0.06)** | **NO** |
| A2 every bear episode beaten | all four (loss shares 64 / 39 / 28 / 40%) | yes |
| A3 \|max DD\| ≤ ½ B&H's on TB / T1 / T2 | 0.39 / 0.32 / 0.39 of B&H's | yes |
| A4 TRAIN net > 0 | +1.21% (net' +0.99%) | yes |
| A4 T1 net > 0, P ≥ 0.90 | +4.96% (P 0.994) | yes |
| A4 T2 net > 0 | +2.40% (P 0.92) | yes |
| A4 G5 T1 net late / 2× > 0 | +3.47% / +4.60% | yes |
| A4 G8 T1 net top-5-out (overlay names) > 0 | +3.21% | yes |
| A4 G4 T1 luck ≥ 95th percentile | §4 | — |

- **A1 fails on T2 by 0.06 pt of CAGR.** The design called it a coin flip (+0.4 pt in the
  preview). The engine is 0.48 pt of CAGR below the preview on T2. The miss is smaller than the
  day-0 convention alone (about 0.07 pt a year there) and far inside any path noise (§4). It is
  reported as failed, as written.
- **What U0 is.**
  - Over 2018-07 → 2026-10 it compounds 10.1% a year against 13.7% for buy-and-hold and 15.0% for
    SPY, at about 0.4 of their drawdowns (−14.7% vs −37.4% / −33.7%).
  - Its Sharpe is the best of the three: 1.06 vs 0.73 / 0.83.
  - It beats buy-and-hold CAGR on T1 only. TB and T3 lose by 6.5 and 18.0 pts, as the design
    said: no hedged design keeps up with a strong bull without leverage.
- **The in-sample cautions apply.**
  - T1 is partly in-sample for SI.
  - The raw alphas carry SPY's composition alpha: +4.4 / +6.1 pts a year on T1 / T2, × 0.45 =
    +2.0 / +2.8 of the raw alpha. The net columns remove it.

## §3 The c-frontier (information only, never adopted; runs 00:00-00:07, `frontier_table.log`)

F15, F30 and F60 ran with U0's flags and each reconciled cleanly: rebuild 0.0 in all 27 periods,
0 refused, 0 part-filled, 0 stuck, 0 NaN-price drops (`diagnose_F*.log`). c = 0 is xs_three_book's
saved `C_core` (the same overlay at 0.90, the same hedged panels and benchmark), read only.

| c | TRAIN net | T1 CAGR (B&H 8.73%) | T1 net | T2 CAGR (B&H 7.62%) | T2 net | 2018-26 CAGR (B&H 13.71%, SPY 15.00%) | 2018-26 Sharpe (B&H 0.73) | 2018-26 max DD (B&H −37.4%) | TB max DD ratio | episodes Q4'18 / COVID / 2022 / Feb-Apr'25 | down months |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (C_core) | +2.00% | 9.53% | +9.36% | 4.06% | +4.31% | 6.72% | 0.86 | −12.3% | 0.21 | −6.2 / +1.6 / +10.4 / +1.5 | +0.60% |
| 0.15 (F15) | +1.83% | 9.83% | +7.95% | 5.38% | +3.83% | 7.90% | 1.06 | −9.8% | 0.24 | −8.0 / −3.8 / +4.5 / −1.6 | +0.01% |
| 0.30 (F30) | +1.59% | 10.10% | +6.50% | 6.53% | +3.19% | 9.05% | 1.12 | −10.6% | 0.28 | −9.9 / −9.1 / −1.3 / −4.7 | −0.57% |
| **0.45 (U0)** | **+1.21%** | **10.29%** | **+4.96%** | **7.56%** | **+2.40%** | **10.12%** | **1.06** | **−14.7%** | **0.39** | **−11.7 / −14.5 / −6.8 / −7.7** | **−1.16%** |
| 0.60 (F60) | +0.72% | 10.47% | +3.38% | 8.59% | +1.60% | 11.17% | 0.97 | −20.0% | 0.53 | −13.5 / −19.7 / −12.2 / −10.8 | −1.74% |

- **The core buys CAGR with drawdown and dilutes the net alpha.** Each 0.15 of core adds about
  1.1 pt of 2018-26 CAGR and 0.12 of beta, and takes about 1.5 pt of net alpha off T1. Above
  c = 0.30 it also deepens the drawdown.
- **No c on the frontier passes both A1 on T2 and A3 on TB.**
  - c = 0.45 misses A1 on T2 by 0.06 pt.
  - c = 0.60 passes A1 on T2 (8.59% vs 7.62%), but its TB max DD is 0.53 of buy-and-hold's,
    against the ½ bar (the design predicted 0.56).
  - A c between the two might pass both. Choosing it here would tune c on VALIDATION, so this
    is information only.
- **Return ÷ risk peaks at c ≈ 0.30:** 2018-26 Sharpe 1.12, T1 1.20, max DD −10.6%. The
  design's 0.45 was chosen a priori for CAGR, not for Sharpe.
- **Every c > 0 loses buy-and-hold's 2018-26 CAGR:** 7.9-11.2% against 13.7%, and against 15.0%
  for SPY.
- c = 0 is the only point with positive down months (+0.60%). At c ≥ 0.15 down months turn
  negative: the price of the core's beta.

## §4 U0: noise, luck, the lab's verdict, bear table, bars (00:07-00:14)

**Noise** (`lab.py noise xs_portable_alpha U0 5 3`; `U0_noise_sd.log`). The copies' alpha minus
the label's:

| test | deltas (pts) | sd | RMS |
|---|---|---|---|
| TB | −0.58, −0.03, +0.03 | 0.34 | 0.34 |
| T3 | −1.51, −0.88, −0.59 | 0.47 | 1.07 |
| T1 | −0.32, −0.58, −0.04 | **0.27** | 0.38 |
| T2 | −0.50, −0.52, −0.49 | 0.02 | 0.50 |

- As in xs_three_book, the copies sit below the label on T3 and T2, so the RMS is the honest
  yardstick there.
- **A1 on T2 in the copies** (information; `U0_A1_noise.log`). The CAGR margin over buy-and-hold
  is −0.55, −0.62 and −0.54 pt, against the label's −0.06. On T1 it is +1.25, +0.93 and +1.56
  (label +1.56). So the T2 miss is not a path accident that a slightly different book would
  flip; the perturbed books miss it by more.

**Luck** (`lab.py luck xs_portable_alpha U0 T1 100`, SPY held fixed; `U0_luck.log`).
- Real +6.95%, against a median of +2.21% and a 95th percentile of +3.82% for 100 shuffled-name
  books: **percentile 100**.
- The shuffled books' median is not 0. Every one of them holds the same core, and 0.45 × SPY's
  T1 alpha is +2.0 pts. So the percentile is net of the core by construction.

**The lab's verdict** (`lab.py verdict xs_portable_alpha U0 --not-neutral`; `U0_verdict.log`;
raw alphas, not net):

| gate | value | passed |
|---|---|---|
| G1 TRAIN-era (TB + T3) alpha > 0 | +1.27% | yes |
| G2 T1 alpha > 0, P ≥ 0.90 | +6.95% (P 0.999) | yes |
| G3 T2 alpha > 0 | +5.16% | yes |
| G4 T1 beats ≥ 95% of shuffled books | 100 | yes |
| G5 T1 late-fill and 2×-cost alpha > 0 | +5.47% / +6.60% | yes |
| G6 every bear episode, down months > 0 | episodes all beaten; down months −1.16% | **no** (by construction) |
| G7 | not market-neutral: skipped | — |
| G8 T1 without its top 5 names (the lab counts SPY as a name) | +3.09% | yes |

**Bear table** (`lab.py bear xs_portable_alpha U0 F30 F60`; TB + T3 + T1 stitched;
`bear_U0_F30_F60.log`):

| | Q4 2018 | COVID | 2022 bear | Feb-Apr 2025 | CAGR | max DD |
|---|---|---|---|---|---|---|
| U0 | −11.7% | −14.5% | −6.8% | −7.7% | 10.1% | −14.7% |
| F30 | −9.9% | −9.1% | −1.3% | −4.7% | 9.1% | −10.6% |
| F60 | −13.5% | −19.7% | −12.2% | −10.8% | 11.2% | −20.0% |
| equal-weight market | −18.1% | −37.0% | −24.5% | −19.5% | (13.7%) | (−37.4%) |
| SPY | −18.7% | −33.4% | −24.1% | −18.6% | (15.0%) | (−33.7%) |

The CAGR and max DD in brackets are §2's stitched figures; `lab.py bear` prints them as NaN.

Calendar years, U0 / the equal-weight market:

| | 2018-H2 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 to Oct 16 |
|---|---|---|---|---|---|---|---|---|---|
| U0 | −5.5 | +14.7 | +11.8 | +14.8 | −2.9 | +13.3 | +18.6 | +18.7 | +3.2 |
| market | −6.2 | +35.7 | +25.7 | +21.6 | −18.6 | +22.8 | +15.3 | +16.1 | +11.6 |

U0 beats the market's calendar year in 2018-H2, 2022, 2024 and 2025. It trails it in 2019, 2020,
2021, 2023 and 2026. 2026 is the overlay's shared 2026-H2 loss (T1 2026-H2 raw alpha −10.3%
annualized; design failure mode 5).

**The design's bars on U0** (`research.py bars U0`; `U0_bars.log`): every bar passes except A1
on T2.
- A1 T1 yes; **A1 T2 NO** (7.563% vs 7.620%).
- A2 yes; A3 yes on TB, T1 and T2.
- A4: TRAIN net +1.21%, T1 net +4.96% (P 0.994), T2 net +2.40%, G4 percentile 100, G5 net late /
  2× +3.47% / +4.60%, G8 net top-5-out +3.21%: all yes.
- **all A1-A4: false.**

**The E0 disclosure** (§0 caution 1; `E0_disclosure.log`; a return sum, not an engine book). E0
has no short interest, so SI sits in cash and the account would be MOM × 0.25 + 0.45 SPY:
- CAGR 5.0% against buy-and-hold 18.5%;
- raw alpha −3.29%, SPY's own alpha −2.13%;
- **net alpha −2.33% (P 0.07)**, and −2.94% with late fills.

So if TRAIN were read on E0 + TB + T3, as the lab reads it for price-only books, the net TRAIN bar
would most likely fail: E0 is 2 years at −2.3% against TB + T3's 3.5 years at +1.2%. That is an
estimate, not an engine book. E0 is never run for short-interest designs.

## §5 Iteration I1, the falling-market core (00:12-00:18; `audit_I1.log`, `diagnose_I1.log`, `I1_bars.log`, `I1_table.log`, `report_U0_I1.log`)

**Mechanics.** The audit (`'{"core_vol": true}'`, base, late and 2× books) and the
reconciliation are clean:
- rebuild 0.0 in all 27 periods;
- 0 refused, 0 part-filled, 0 stuck, 0 NaN-price drops, 0 off-target days;
- the overlay identical to U0's;
- the core's target only in {0, 0.225, 0.45}, moving only on the first bar after a month end.

The core trades 5-8 times a year (TB 15 orders, T3 9, T1 25, T2 25). Its mean realized weight is
0.24 / 0.37 / 0.28 / 0.28 on TB / T3 / T1 / T2, against U0's 0.45. The smallest free cash is 7.1%.

| | U0 | I1 |
|---|---|---|
| T1 CAGR (B&H 8.73%) | 10.29% | 8.97% |
| T2 CAGR (B&H 7.62%) | 7.56% | **6.29%** |
| TB / T3 CAGR | 4.75% / 17.20% | 2.20% / 11.79% |
| 2018-26 CAGR / Sharpe / max DD | 10.12% / 1.06 / −14.7% | 7.78% / 1.18 / −6.6% |
| net alpha TRAIN / T1 (P) / T2 | +1.21% / +4.96% (0.994) / +2.40% | +1.29% / +4.58% (0.989) / +2.10% |
| T1 net late / 2× / top-5-out | +3.47 / +4.60 / +3.21 | +2.83 / +4.16 / +2.85 |
| beta T1 / T2 | 0.38 / 0.31 | 0.16 / 0.12 |
| max DD TB / T1 / T2 | −14.7% / −8.1% / −9.5% | −6.6% / −6.3% / −5.4% |
| episodes Q4'18 / COVID / 2022 / Feb-Apr'25 | −11.7 / −14.5 / −6.8 / −7.7 | −5.7 / −1.1 / **+2.1** / −1.1 |
| down months (market −4.11%) | −1.16% | −0.27% |

**The adoption rule: I1 is NOT adopted.** It passes the risk conditions:
- every episode return beats U0's;
- |max DD| is smaller on T1 and T2;
- A2 and A3 pass;
- A4 passes, except G4, which was not run.

But **A1 fails on T2**: 6.29% against 7.62%, 1.33 pts short. A1 on T1 passes by only 0.23 pt.
As pre-registered, I1's luck run is not made: an adoption condition already fails.

- The design expected exactly this: TB ≈ 1.6%, T1 ≈ 9.2%, T2 ≈ 6.7%, 2022 ≈ +2% instead of −6%,
  smaller drawdowns, and "expected not adopted".
- I1 is the better risk book: COVID −1.1% against −14.5%, 2022 +2.1% against −6.8%, a Sharpe of
  1.18. It buys that by holding the core less than two thirds of the time. Over 2018-26 its CAGR
  falls to 7.8%.
- **The protocol's generic iteration bar, information only** (`I1_generic_bar.log`). Paired raw-
  alpha Δ against U0: TRAIN +2.15 (P 0.80), T1 +0.59 (P 0.66), T2 +0.20 (P 0.55). T1 Δ > 2 × U0's
  T1 noise sd (0.54), but not > 2 × its RMS (0.76). These raw deltas compare books whose beta
  differs by 0.2, so they decide nothing here. The design's risk bar replaces the generic one for
  this risk iteration.

Variant count: **2 of 2** (U0, I1). The c-frontier (F15, F30, F60) is information only and not
counted.

## §6 DECLARED FINAL (2026-10-24 00:19 CEST, before any TL or T4 read)

**FINAL = U0** = `{}`, run with `--hedge` (lab label `U0`): SPY 0.45 held as a core, plus the
SI + MOM overlay at 0.225 each (xs_three_book `sleeves: si,mom` at gross 0.45), nominal gross 0.90.

- I1 is not adopted: it fails A1 on T2 (§5). With no adoption, the final is U0, as
  pre-registered.
- No knob was chosen on a result. c = 0.45 is the a-priori split. The frontier is information
  only, and no frontier point passes both A1 on T2 and A3 on TB (§3).

**Status: works to some degree** — not "works":
- **A1 fails on T2**, by 0.06 pt of CAGR in the label and by 0.54-0.62 pt in all three noise
  copies.
- **The lab's G6 fails by construction** (down months −1.16%). The design reports that leg and
  does not gate it.

Everything else passes:
- A2: every bear episode beaten, at 28-64% of the market's loss.
- A3: max DD 0.32-0.39 of buy-and-hold's on TB, T1 and T2.
- A4 on net alpha: TRAIN +1.21%, T1 +4.96% (P 0.994), T2 +2.40% (P 0.92), luck 100th percentile,
  net late / 2× +3.47 / +4.60%, net top-5-out +3.21%.
- The net' disclosure agrees in sign on every A4 test.
- The lab's own verdict passes G1-G5 and G8.

The honest headline:
- **U0 buys back about 3.4 pts of the overlay's lost bull-market CAGR.** Its 2018-26 CAGR is
  10.1%, against 6.7% for the overlay alone (C_core). That is still 3.6 pts short of buy-and-hold
  (13.7%) and 4.9 pts short of SPY (15.0%).
- **It keeps all four bear episodes beaten and the best Sharpe of the three** (1.06 vs 0.73 /
  0.83).
- **It beats buy-and-hold CAGR on T1 only.**

**The caveats that bound every number.**
- **T1 is partly in-sample for SI.** Only MOM is clean there.
- **The raw alphas carry SPY's composition alpha:** about +2.0 / +2.8 pts on T1 / T2. A1 is
  flattered the same way. The net columns remove it.
- **TRAIN is TB + T3.** The E0 disclosure (SI silent, MOM × 0.25 + SPY) reads −2.3% net, so a
  TRAIN read on E0 + TB + T3 would likely be negative.
- **2026 is the overlay's shared loss** (T1 2026-H2 raw alpha −10.3% annualized).
- **The lab's SPY open = close** makes the core miss each period's first-day move: about −0.07 pt
  a year of CAGR on T1 / T2 and −0.5 on TB.
- **A live account would need a core rebalancing rule** (IDEAS.md 3): the lab resets the core to
  0.45 every half-year for free.

**For the orchestrator's one TL batch** (I do not read TL or T4):

| role | strategy / label | overrides | flags |
|---|---|---|---|
| declared final | xs_portable_alpha / U0 | `{}` | `--hedge` (`"hedge": true`) |
| same-read control (the overlay alone at 0.90, c = 0) | xs_three_book / C_core | `{"sleeves": "si,mom"}` | `--hedge` |

**TL bar** (pre-registered, §0):
- A1 on TL: CAGR > the equal-weight buy-and-hold CAGR.
- A4 on TL: net alpha = alpha − 0.45 × SPY's own TL alpha against the same benchmark > 0, with P
  and the late-fill net reported.

Small caps lagged in 2022-26, so the core flatters TL's raw alpha and A1 most of all.
`research.py table` computes both from a saved `U0-LOCK` label if it is pointed at it. TL is fresh
only for the combination: informed_flow_ls §38P already read names 1501-2000 for SI.

## §7 Expected (§0) against observed

| item | expected | observed |
|---|---|---|
| T1 CAGR (B&H / SPY) | ≈ 10.3% (8.8% / 11.9%) | 10.29% (8.73% / 11.90%) |
| T1 net alpha | ≈ +5% (P ≥ 0.95) | +4.96% (P 0.994) |
| T2 CAGR (B&H) | ≈ 7.7% (7.6%): A1 a coin flip | 7.56% (7.62%): **A1 fails by 0.06** |
| T2 net alpha | ≈ +2.9% | +2.40% (P 0.92) |
| TB / T3 CAGR | ≈ 5.7% / 16.7% | 4.75% / 17.20% |
| TRAIN net alpha | ≈ +0.6% | +1.21% (net' +0.99%) |
| episodes | ≈ −11.6 / −15.5 / −6 / −8, all beaten | −11.7 / −14.5 / −6.8 / −7.7, all beaten |
| max DD TB / T1 / T2 | ≈ −16 / −8.5 / −9.5% | −14.7 / −8.1 / −9.5% |
| down months | ≈ −0.9% | −1.16% |
| beta | 0.35-0.45 | 0.31-0.45 |
| luck | ≥ 95th | 100th |
| engine vs preview | ≈ −0.3 pt | T1 −0.17, T2 −0.39, TB −0.90 (the COVID V, §2) |
| I1 | TB ≈ 1.6, T1 ≈ 9.2, T2 ≈ 6.7, 2022 ≈ +2; not adopted | 2.2 / 9.0 / 6.3 / +2.1; not adopted |
| mechanics (mine) | nothing refused, free cash ≥ 4% | nothing refused in 4 labels × 27 periods (+ late / 2× books of U0 and I1); free cash ≥ 6.5% |
| drag term (mine) | the design's net ≈ net' + 0.4-0.6 everywhere | net − net' from −1.0 (T3) to +0.2: **wrong**, compounding convexity dominates in strong trends |

## §8 Conclusion

**The design's question was how much of buy-and-hold's bull-market CAGR an index core can buy
back while still beating every bear episode.** About a third of the gap, at the a-priori
c = 0.45:
- The overlay alone (c = 0) compounds 6.7% over 2018-07 → 2026-10, and U0 10.1%. Buy-and-hold
  compounds 13.7% and SPY 15.0%.
- U0 beats the equal-weight market in all four bear episodes, losing 28-64% of the market's loss.
- Its max DD is 0.32-0.39 of buy-and-hold's.
- Its Sharpe (1.06) is the best of the three.
- The overlay's alpha survives the core: net of 0.45 × SPY's own alpha, T1 +4.96% (P 0.994,
  luck 100th percentile) and T2 +2.40% (P 0.92).

**It does not beat buy-and-hold's CAGR where the design promised it would on both validation
sets.** It wins on T1 (10.3% vs 8.7%) and misses on T2 (7.56% vs 7.62%). The T2 miss holds in
every noise copy. The frontier says that buying more core (c = 0.60) wins T2's CAGR but breaks
the drawdown bar on TB. **No hedged design keeps up with a strong bull** (T3: 17% vs 35%).

The vol-managed core (I1) is the better risk book, with COVID −1.1%, 2022 +2.1% and a Sharpe of
1.18. It gives up 1.3 pts of CAGR on T1 and T2, fails A1 on T2 by 1.3 pts, and is not adopted.

**Final: U0 (`{}`, `--hedge`). Status: works to some degree.** A1 on T2 fails, and the lab's G6
down-month leg fails by construction. Every risk-adjusted, falling-market and drawdown bar
passes.

Nothing is committed to git. `strategies/xs_portable_alpha/`, `tests/test_xs_portable_alpha.py`
and the results are untracked files.

The one-off read-only scripts behind `U0_detail.log`, `U0_gap_periods.log`, `U0_A1_noise.log` and
`I1_generic_bar.log`, plus the §0 smoke run, are kept in
`var/proposals_lab/results2/xs_portable_alpha/scripts/`.
