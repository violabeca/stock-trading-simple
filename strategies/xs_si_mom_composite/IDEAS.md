# xs_si_mom_composite — pending ideas / proposals / thoughts

Parking lot for ideas worth keeping but not (yet) implemented. Each idea carries its why and the
falsification that would settle it. Nothing here is a commitment.

## Tested and rejected by their pre-registered bars (RESEARCH_LOG §4, §5, §7)

1. **Sector-neutral composite (I2, `neutral: sector`).** Kept T1's alpha (Δ −0.03), lifted TRAIN
   and T2 by ~3.3 pts each, cut T1 volatility 16.3 → 15.0% and drawdown −21.9 → −16.5%, and halved
   the net sector exposure (Σ|net| 0.48 → 0.27). It failed the T1 noise bar, so it is not the
   final. *Open question:* a risk-oriented protocol (bars on drawdown and alpha ÷ residual vol,
   pre-registered as such) would read it differently; it must not be re-labelled after the fact.
   Note the code knob changed strategy.py's SHA-256 after §0 (default path verified identical).
2. **Profitability junk gate on shorts (I1, `short_gate: prof`).** Positive everywhere (+1.3 /
   +0.95 / +3.2) but inside the T1 noise; it RAISED orders 21% (the gate is re-read daily and
   names near the sector median flip). A gate read only at month ends would test the mechanism
   without the churn.
3. **w_mom 1.5** (the grid's robust maximum): T1 −3.4, PBO 0.75 (9 points × the last six TRAIN
   periods). Not adopted.

## Not tested

- **Tail-preserving entry** (design idea 3): its premise, dilution of SI's extremes, did not
  appear (A_U0 beats C_si on T1 by 3.5 pts), so it was not spent as the written iteration.
- **Gross headroom for squeeze covers.** GME's May-2024 cover part-filled at gross 0.90 (T2): a
  squeeze that gaps far past the +100% stop asks for more than the short's locked proceeds plus
  margin. A per-short free-cash reserve, or a cover re-sent until filled (adaptive_hedge IDEAS 2),
  would remove it; it is a lab-level fix. The stuck 40% happened to make +4.57% of the book from
  its due open as GME fell back (T2 +5.07% as run, +4.15% repaired, RESEARCH_LOG §9 / §12); a
  squeeze that keeps running would charge the same stuck short against the book instead, so the
  failure is a risk, not a source of return. Names 1501-2000 (TL) are more squeeze-prone.

## Engine / lab notes (proposal-only, nothing implemented here)

- **The noise bar's yardstick.** Bar (2) compares a paired Δ with 2 × the LABEL's own noise sd.
  The Δ's own path noise, read from both arms' copies on the same reduced panels (`research.py
  noise_delta`, RESEARCH_LOG §12.3), was larger on T1 (sd 2.2 vs 1.45; copies +3.1 / +3.5 /
  −0.4). Every "Δ > 2 × noise sd" bar in the lab would be better judged on the paired copies'
  spread; `lab.noise` already makes them comparable, since it seeds the dropped names per copy.

- `informed_flow_ls_p1`'s composite with a silent skew sleeve (`w_skew: 1`, no option chains)
  ranks `(si + 0.5) / 2`. Within-group SI percentiles tie across the 5 liquidity groups, and the
  float rounding of that affine map splits the ties differently from `si` alone, so v6_si's
  band-edge entries depend on float noise (TB 2018-H2: 286 weight cells, 5 names). A rank on a
  rounded composite (e.g. `round(12)` before `rank`) would make the book deterministic in the
  tie order; it is p1's code, so it is only noted here.
