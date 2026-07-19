# _template — copy me to start a new strategy

1. `cp -r strategies/_template strategies/<your_name>` (folder name = strategy name).
2. Describe the **thesis** here in 3–5 plain sentences: what edge, why it should exist.
3. Implement `signals()` in `strategy.py`; list every parameter in `config.yaml`.
4. Copy an existing strategy notebook, point it at your name, run top to bottom.
5. Gate: `uv run pytest -q && uv run ruff check .` plus `uv run tsim optimize --strategy <your_name> --n-trials 5`.

Document each parameter's meaning in `config.yaml` comments — that file is the
single place someone tunes your strategy from.

Conventions every strategy follows (see "Comparability rules" in the root README):

- **One shared $10k wallet** — keep the `cash_frac` return (or a `weights` frame
  for a cross-sectional ranker); never one book per symbol.
- **`data.end` is the TRAIN cutoff** — the notebook's out-of-sample section
  replays the picked parameters on the untouched tail.
- **Baseline comparison is mandatory** — the notebook must show the strategy
  against `strategies/baseline` (equal-weight buy-and-hold of the same panel):
  equity overlay plus `excess_cagr`/`info_ratio`.
