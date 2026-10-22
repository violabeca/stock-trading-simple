# option_signals — parked ideas

Nothing here is implemented. Each needs its own pre-registration before it is scored.

1. **A strategy, if a signal passes discovery and the holdout.** A cross-sectional ranker in
   portfolio mode (like informed_flow) needs the signal as a `Panel.features` frame: one new builder
   in `features.py` (an engine change: proposal first), fed by a daily option snapshot (IBKR after the
   MarketData trial ends), plus the IV code in `research.py`.
2. **Changes, not levels.** Cremers & Weinbaum and An, Ang, Bali & Cakici (2014) also find that
   CHANGES in the call − put spread and in call / put IVs predict. Not tested here (the level test
   came first).
3. **American exercise.** IVs are European Black-Scholes on American quotes. Near the money at 30
   days the early-exercise premium is small but it biases put IVs up a little for every name alike;
   a binomial model would remove it.
