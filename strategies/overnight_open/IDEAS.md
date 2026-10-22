# overnight_open · ideas (parked, not implemented)

1. **A timed news feed.** The question asks mainly about news *during the night*, and only SEC
   8-Ks are timed. Benzinga's headlines (MarketData.app) carry a date only. Table 5c of the study
   (same-day headlines, hindsight) shows what timing could be worth. If it is large, two options:
   - IBKR's historical news API (`reqHistoricalNews`; the free providers are BRFG, BRFUPDN and
     DJNL; check how far back they go, paced and never parallel);
   - Benzinga through Massive (~$99/month, timed to the second).
2. **Earnings content, not just presence.** Direction needs to know whether the news was good:
   results against estimates, guidance. One proxy is the reaction itself in the after-hours
   (sign of `ah_ret_atr` on an Item 2.02 night), already in the table; V4 tests news vs quiet
   nights only.
3. **More validation data.** Extended-hours 1-minute bars stop on 2026-06-26 for all but 7 names.
   A backfill of July → October 2026 for the ~535 names would give a fresh holdout once V1-V4
   have been read (IBKR, paced, never parallel).
4. **The overnight session itself.** IBKR's OVERNIGHT venue (20:00-03:50) prints for only ~5 names
   in the database. With a backfill, the dark hours would be a fourth part of the night rather
   than a single jump.
5. **Option-implied move on earnings nights.** The end-of-day chains (`var/marketdata/chains`)
   price the expected move. Is a first hour larger than the straddle priced more likely when the
   pre-market already moved past it?
6. **A trading rule, only if V3/V4 pass.** For example: on news nights, enter at 09:31 in the
   night's direction and exit at 10:30. It must clear 20-40 bp of first-half-hour costs at $10k;
   test it with `cash_frac` wallet sizing and pessimistic fills.
7. **Engine (proposal-only): EDGAR's acceptance time.** `providers/sec.py` stores the same JSON
   `acceptanceDateTime` as an insider filing's `known_at`. For about half the filers it is 4-5 h
   late (see the README). There it only makes `known_at` later, which is conservative (no
   look-ahead), but a daily feature can land a day late. The fix is `research.unshift` plus the
   per-filer index-page check, moved into the provider.
