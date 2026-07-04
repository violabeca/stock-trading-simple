# _template — copy me to start a new strategy

1. `cp -r strategies/_template strategies/<your_name>` (folder name = strategy name).
2. Describe the **thesis** here in 3–5 plain sentences: what edge, why it should exist.
3. Implement `signals()` in `strategy.py`; list every parameter in `config.yaml`.
4. Copy an existing strategy notebook, point it at your name, run top to bottom.
5. Gate: `pytest -q && ruff check .` plus `tsim optimize --strategy <your_name> --n-trials 5`.

Document each parameter's meaning in `config.yaml` comments — that file is the
single place someone tunes your strategy from.
