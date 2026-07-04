"""Configuration: one global YAML file, plus per-strategy YAML files.

Global settings live in ``config/settings.yaml`` and map 1:1 onto the pydantic
models below. Named universes written by the screener notebook live in
``config/universes.yaml`` and are merged in at load time. Environment variables
prefixed ``TRADING_`` (with ``__`` for nesting, e.g. ``TRADING_IBKR__PORT=4002``)
override both, and a ``.env`` file at the repo root is read automatically.

Per-strategy settings live in ``strategies/<name>/config.yaml`` and are loaded
by :mod:`trading_simple.strategies` — they never touch this module.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_FILE = PROJECT_ROOT / "config" / "settings.yaml"
UNIVERSES_FILE = PROJECT_ROOT / "config" / "universes.yaml"


class DBSettings(BaseModel):
    """Connection to the shared TimescaleDB (pre-existing — never created here)."""

    host: str = "localhost"
    port: int = 5432
    user: str = "trading"
    password: str = "trading"
    name: str = "trading"

    @property
    def sqlalchemy_url(self) -> str:
        """SQLAlchemy URL using the psycopg v3 driver."""
        return f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class IBKRSettings(BaseModel):
    """Interactive Brokers Gateway/TWS connection and pacing limits.

    ``extended_hours`` is off by default: backfills store only regular-session
    bars unless you opt in (per run via the CLI flag, or here globally). Extended
    bars are tagged ``rth=false`` in the database either way.
    """

    host: str = "127.0.0.1"
    port: int = 4002          # IB Gateway paper. Live GW=4001, TWS paper=7497, TWS live=7496.
    client_id: int = 1
    max_requests_per_10min: int = 55   # IBKR caps historical requests account-wide (~60/10min)
    pace_seconds: float = 0.5
    extended_hours: bool = False


class CostSettings(BaseModel):
    """Trading friction assumptions used by every backtest."""

    commission_per_share: float = 0.005
    min_commission: float = 1.00
    spread_bps: float = 2.0
    slippage_bps: float = 1.0


class BacktestSettings(BaseModel):
    """Engine defaults shared by all strategies."""

    init_cash: float = 10_000.0
    chunk_size: int = 12       # parameter combos stacked into one vectorbt call per chunk
    engine: str = "auto"       # auto | rust | numba -> vbt.settings['engine']


class Settings(BaseSettings):
    """All global configuration, resolved from YAML + environment."""

    model_config = SettingsConfigDict(
        env_prefix="TRADING_",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    db: DBSettings = DBSettings()
    ibkr: IBKRSettings = IBKRSettings()
    costs: CostSettings = CostSettings()
    backtest: BacktestSettings = BacktestSettings()
    exchanges: dict[str, str] = {}          # per-symbol venue overrides, e.g. {"NESN": "SIX"}
    universe: dict[str, list[str]] = {}     # named symbol lists

    def universe_symbols(self, name: str) -> list[str]:
        """Return the symbols of a named universe.

        Args:
            name: Universe name from ``settings.yaml`` or ``universes.yaml``.

        Returns:
            The list of bare symbols.

        Raises:
            KeyError: If the universe does not exist (message lists known names).
        """
        try:
            return self.universe[name]
        except KeyError as exc:
            raise KeyError(f"Unknown universe {name!r}. Known: {sorted(self.universe)}") from exc


def load_settings(settings_file: Path | None = None) -> Settings:
    """Load global settings from YAML, merge universes, apply env overrides.

    Args:
        settings_file: Alternative settings YAML (used by tests); defaults to
            ``config/settings.yaml``.

    Returns:
        A fully resolved :class:`Settings` instance.
    """
    path = settings_file or SETTINGS_FILE
    values = yaml.safe_load(path.read_text()) or {} if path.exists() else {}
    if UNIVERSES_FILE.exists():
        generated = yaml.safe_load(UNIVERSES_FILE.read_text()) or {}
        generated = generated.get("universe", generated)  # accept flat or nested form
        values.setdefault("universe", {}).update(generated)
    return Settings(**values)
