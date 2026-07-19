# baseline — equal-weight buy-and-hold

The hurdle made visible. This strategy holds every symbol of the configured
universe in equal weight from the first bar to the last — one shared `$10k`
book, fractional shares, the same costs as everyone else. It is by construction
(and by `tests/test_baseline.py`) identical to the engine's internal
`backtest.benchmark()`, which every strategy's `excess_cagr`/`info_ratio` is
measured against; on its own panel it therefore scores ~0 on both objectives.

Point it at any universe by editing the `data:` block in `config.yaml`. It has
no notebook of its own: `notebooks/04_compare.ipynb` runs it on every
strategy's panel and is the place to compare buy-and-hold across universes and
strategies.
