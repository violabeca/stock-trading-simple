"""Save Interactive Brokers' public short-stock file (borrow fee and shares available per name).

IBKR publishes one pipe-delimited file per country on its anonymous FTP site (user
``shortstock``, no password), refreshed during the day: symbol, currency, name, contract id,
ISIN, rebate rate, FEE RATE (% a year) and shares AVAILABLE to borrow. It is a snapshot only — no
history is served — so this script keeps one dated copy per day in ``var/ibkr/``; run it daily
(before the open) and the copies become the history. ``informed_flow_ls`` reads the newest one
as each name's borrow fee (``research.borrow_fees``).

    uv run python scripts/ibkr_shortstock.py              # usa only
    uv run python scripts/ibkr_shortstock.py canada usa   # several countries
"""
import sys
import urllib.request
from datetime import date
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "var" / "ibkr"
URL = "ftp://shortstock:@ftp2.interactivebrokers.com/{country}.txt"


def save(country: str = "usa", out: Path = OUT) -> Path:
    """Download one country's file to ``out/<country>-<YYYYMMDD>.txt`` (today's copy replaced).

    Args:
        country: IBKR's file name without ``.txt`` (``usa``, ``canada``, ``british`` ...).
        out: Directory for the dated copies.

    Returns:
        The path written.
    """
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{country}-{date.today():%Y%m%d}.txt"
    with urllib.request.urlopen(URL.format(country=country), timeout=120) as resp:
        path.write_bytes(resp.read())
    return path


if __name__ == "__main__":
    for c in sys.argv[1:] or ["usa"]:
        p = save(c)
        print(f"{p} ({p.stat().st_size:,} bytes)")
