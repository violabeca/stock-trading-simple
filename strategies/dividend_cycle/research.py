"""Dividend-cycle research: long high-yield stocks into the ex-dividend date, short just after it.

The question (RESEARCH_LOG §0): does buying high-dividend stocks a few weeks before the payout
and shorting them right after it earn anything? Everything is point in time: the next ex-date
is PREDICTED from past ex-dates, the yield is the trailing 365 days of dividends over the
decision day's close, and the universe is the lab's rolling top 500. The actual ex-date is read
only to exit (it is declared ahead).

    uv run python strategies/dividend_cycle/research.py fetch    # Yahoo -> var/dividend_cycle/
    uv run python strategies/dividend_cycle/research.py run      # the tables of RESEARCH_LOG §1
"""

from __future__ import annotations

import hashlib
import importlib.util
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
VAR = REPO / "var" / "dividend_cycle"
DIV = VAR / "yahoo"

#: Registered rule (RESEARCH_LOG §0): enter N sessions before the predicted ex-date, short H after.
N, H, MIN_YIELD, PLACEBO = 15, 10, 0.03, 30
#: A quarterly payer's gaps in calendar days; the actual ex-date is searched within +-SLACK of P.
GAP, SLACK, HISTORY = (80, 105), 20, 400
COST_BP, BORROW = 8.0, 0.01
GROUPS = {"TRAIN": ["E0", "TB", "T3"], "T1": ["T1"], "T2": ["T2"]}


def _lab():
    spec = importlib.util.spec_from_file_location("lab", REPO / "strategies/proposals_lab/lab.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("lab", mod)
    spec.loader.exec_module(mod)
    return mod


# --- data ---------------------------------------------------------------------------------------

def universe() -> dict[str, list[tuple[str, str, list[str]]]]:
    """Per group, its ``(start, end, names)`` half-year periods (the lab's point-in-time lists)."""
    lab = _lab()
    return {g: [(s, e, lab.names(t, s)) for t in tests for s, e in lab.periods(t)]
            for g, tests in GROUPS.items()}


def fetch_one(sym: str) -> bool:
    """Download one ticker's raw (split-adjusted) close and ex-dividends from Yahoo into
    ``DIV/<sym>.pkl``; ``False`` when Yahoo answered nothing (yfinance swallows errors)."""
    import yfinance as yf
    f = DIV / f"{sym}.pkl"
    if f.exists():
        return True
    for attempt in range(3):
        try:
            h = yf.Ticker(sym.replace(".", "-")).history(
                start="2014-01-01", auto_adjust=False, actions=True)
        except Exception:                                   # noqa: BLE001 - network, retry
            h = pd.DataFrame()
        if not h.empty:
            h.index = h.index.tz_localize(None).normalize()
            div = h["Dividends"][h["Dividends"] > 0]
            f.write_bytes(pickle.dumps({"close": h["Close"], "div": div}))
            return True
        time.sleep(2 ** attempt)
    return False


def fetch(workers: int = 6) -> None:
    """Fetch every name of every point-in-time universe (cached; reruns only the missing)."""
    DIV.mkdir(parents=True, exist_ok=True)
    syms = sorted({n for ps in universe().values() for *_, names in ps for n in names})
    with ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(fetch_one, syms))
    print(f"fetched {sum(ok)} of {len(syms)}; missing: "
          f"{[s for s, k in zip(syms, ok, strict=True) if not k][:40]}")


def prices(syms: list[str], start: str = "2015-01-01") -> pd.DataFrame:
    """Adjusted daily OPEN of ``syms`` (dates x symbols, naive session dates), 200 at a time."""
    from trading_simple.data import load_panel
    from trading_simple.db import make_engine
    key = hashlib.sha1(",".join([start, *syms]).encode()).hexdigest()[:10]
    cache = VAR / f"opens-{key}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    eng, parts = make_engine(), []
    for i in range(0, len(syms), 200):
        o = load_panel(eng, syms[i:i + 200], "1d", pd.Timestamp(start, tz="UTC")).open_
        parts.append(o)
    out = pd.concat(parts, axis=1)
    out.index = out.index.tz_convert(None).normalize()
    cache.write_bytes(pickle.dumps(out))
    return out


# --- point-in-time events -----------------------------------------------------------------------

def predict(ex: pd.Series) -> pd.DataFrame:
    """One row per regular-payer anchor ex-date: the PREDICTED next ex-date and the ACTUAL one.

    An anchor ``a`` is regular when the last 4 ex-dates lie within :data:`HISTORY` days and the
    last two gaps are quarterly (:data:`GAP`). ``pred`` = ``a`` + the median of the last 3 gaps;
    ``ex`` = the Yahoo ex-date nearest ``pred`` within :data:`SLACK` days after ``a``, NaT when
    the dividend never came. Only dates up to ``a`` feed ``pred``, so truncating later history
    cannot change a row (``tests/test_dividend_cycle.py``).

    Args:
        ex: Dividend amounts indexed by ex-date (ascending).

    Returns:
        Columns ``anchor``, ``pred``, ``ex``, ``amount`` (the actual dividend, NaN when none).
    """
    d, rows = ex.index, []
    for j in range(3, len(d)):
        gaps = np.diff(d[j - 3:j + 1].values).astype("timedelta64[D]").astype(int)
        quarterly = ((gaps[1:] >= GAP[0]) & (gaps[1:] <= GAP[1])).all()
        if (d[j] - d[j - 3]).days > HISTORY or not quarterly:
            continue
        p = d[j] + pd.Timedelta(days=int(np.median(gaps)))
        later = d[j + 1:]
        miss = np.abs((later - p).days)
        near = np.flatnonzero(miss <= SLACK)
        k = near[np.argmin(miss[near])] if len(near) else None
        rows.append((d[j], p, later[k] if k is not None else pd.NaT,
                     float(ex.iloc[j + 1 + k]) if k is not None else np.nan))
    return pd.DataFrame(rows, columns=["anchor", "pred", "ex", "amount"])


def ttm_yield(div: pd.Series, close: pd.Series, cal: pd.DatetimeIndex) -> np.ndarray:
    """Yearly dividend over the day's raw close, per session: the last four dividends gone ex
    (within 400 days) - exactly one year for a quarterly payer. A 365-day window counts a FIFTH
    one on the ex-date itself (four quarterly gaps are 364 days): +25% yield exactly there."""
    cs = np.concatenate([[0.0], div.cumsum().to_numpy()])
    hi = div.index.searchsorted(cal, side="right")
    lo = np.maximum(hi - 4, div.index.searchsorted(cal - pd.Timedelta(days=400), side="right"))
    px = close.reindex(cal).ffill(limit=5).to_numpy()
    return (cs[hi] - cs[lo]) / px


def events(opens: pd.DataFrame, store: dict[str, dict]) -> tuple[pd.DataFrame, np.ndarray]:
    """Every regular-payer anchor as session positions, and the yield matrix (sessions x names).

    Args:
        opens: Adjusted opens (sessions x names); its index is the trading calendar.
        store: ``{symbol: {"close": Series, "div": Series}}`` from :func:`fetch`.

    Returns:
        ``(ev, Y)``: ``ev`` has ``col`` (column of ``opens``), ``anchor``, ``P``, ``X`` (session
        positions; ``X = -1`` when no dividend arrived) and ``amount``; ``Y[t, col]`` is the
        yield known at session t's close.
    """
    cal, Y, parts = opens.index, np.full(opens.shape, np.nan), []
    for c, s in enumerate(opens.columns):
        if s not in store or len(store[s]["div"]) < 4:
            continue
        Y[:, c] = ttm_yield(store[s]["div"], store[s]["close"], cal)
        p = predict(store[s]["div"])
        if p.empty:
            continue
        p = p[p["pred"] + pd.Timedelta(days=SLACK) <= cal[-1]]     # outcome known by the data's end
        if p.empty:
            continue
        parts.append(pd.DataFrame({
            "col": c, "anchor": cal.searchsorted(p["anchor"]), "P": cal.searchsorted(p["pred"]),
            "X": np.where(p["ex"].notna(), cal.searchsorted(p["ex"].fillna(cal[0])), -1),
            "amount": p["amount"]}))
    return pd.concat(parts, ignore_index=True), Y


# --- returns, benchmarks, rules -----------------------------------------------------------------

def membership(opens: pd.DataFrame, periods) -> np.ndarray:
    """Boolean (sessions x names): the name is in the period's point-in-time universe."""
    mem = np.zeros(opens.shape, bool)
    for s, e, names in periods:
        rows = (opens.index >= pd.Timestamp(s)) & (opens.index <= pd.Timestamp(e))
        mem[np.ix_(rows, opens.columns.isin(names))] = True
    return mem


def bench_index(R: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cumulative equal-weight open-to-open index of the masked names (1.0 where none)."""
    ok = mask & np.isfinite(R)
    day = np.where(ok, R, 0.0).sum(axis=1) / np.maximum(ok.sum(axis=1), 1)
    return np.cumprod(1 + day)


def chosen(ev: pd.DataFrame, Y: np.ndarray, mem: np.ndarray, n: int = N,
           min_yield: float = MIN_YIELD) -> np.ndarray:
    """Which events a trader would enter: decided at the close before E = P - n, the name in the
    group's universe that day, yield >= ``min_yield``, the anchor already behind, and the
    ex-date not already behind E (it is declared, so the trader would know)."""
    T, E = len(Y), ev["P"].to_numpy() - n
    D, col, X = E - 1, ev["col"].to_numpy(), ev["X"].to_numpy()
    d = np.clip(D, 0, T - 1)
    return ((D >= 0) & (ev["anchor"].to_numpy() <= D) & ~((X >= 0) & (X <= E))
            & mem[d, col] & (Y[d, col] >= min_yield))


def evaluate(ev: pd.DataFrame, op: np.ndarray, Y: np.ndarray, mem: np.ndarray, idx: dict,
             n: int = N, h: int = H, shift: int = 0, min_yield: float = MIN_YIELD,
             lag: int = 0) -> pd.DataFrame:
    """Per-event raw and benchmark-relative returns of the long [E, X) and short [X, X+h) legs.

    E = P - n (decided at the close before, filled at the open); the long exits at the actual
    ex-date's open (at P's open when no dividend came, and then there is no short). ``shift``
    moves both windows later in the dividend cycle (the placebo); the events themselves are
    still those :func:`chosen` on the real decision day. ``lag`` holds the long ``lag`` sessions
    past the ex-date and starts the short there (the PAY date, which Yahoo does not give: a
    post-hoc proxy, ex + 15 sessions is about 3 calendar weeks).

    Returns:
        One row per event: ``col``, ``yield``, ``hit`` (a dividend arrived), ``xl`` (the exit
        session), ``L``/``S`` (raw stock return; a short earns ``-S``), ``Lm``/``Sm`` (minus the
        market index) and ``Lh``/``Sh`` (minus the high-yield index).
    """
    T = len(op)
    ev = ev[chosen(ev, Y, mem, n, min_yield)]
    e, c, X = ev["P"].to_numpy() - n + shift, ev["col"].to_numpy(), ev["X"].to_numpy()
    hit = X >= 0
    xl = np.where(hit, X, ev["P"].to_numpy()) + shift + lag   # long exit = short start
    xs = xl + h
    ok = (e >= 0) & (xs < T)
    e, xl, xs, c, hit = e[ok], xl[ok], xs[ok], c[ok], hit[ok]
    L = op[xl, c] / op[e, c] - 1
    S = np.where(hit, op[xs, c] / op[xl, c] - 1, np.nan)
    out = pd.DataFrame({"col": c, "yield": Y[ev["P"].to_numpy()[ok] - n - 1, c], "hit": hit,
                        "xl": xl, "L": L, "S": S})
    for k, ix in idx.items():
        out["L" + k] = L - (ix[xl] / ix[e] - 1)
        out["S" + k] = S - np.where(hit, ix[xs] / ix[xl] - 1, np.nan)
    return out.dropna(subset=["L"])


def month_t(x: pd.Series, month: pd.Series) -> tuple[float, float, int]:
    """Mean, t-statistic across CALENDAR MONTHS (mean of each month's events) and month count."""
    m = x.groupby(month).mean().dropna()
    if len(m) < 3:
        return float(x.mean()), np.nan, len(m)
    return float(x.mean()), float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m)))), len(m)


def legs(df: pd.DataFrame, cal: pd.DatetimeIndex, bench: str = "h", h: int = H) -> pd.DataFrame:
    """Add net columns (bp): ``long``/``short`` excess vs index ``bench`` after costs, ``both``."""
    d = df.copy()
    d["month"] = cal[d["xl"]].to_period("M")
    d["long"] = (d["L" + bench]) * 1e4 - COST_BP
    d["short"] = (-d["S" + bench]) * 1e4 - COST_BP - BORROW * h / 252 * 1e4
    d["both"] = d["long"] + d["short"].fillna(0.0)
    return d


def summary(d: pd.DataFrame) -> dict:
    """The row of a results table: events, gross legs and net excess with month-clustered t."""
    row = {"n": len(d), "div%": round(100 * d["hit"].mean(), 1),
           "L raw bp": round(1e4 * d["L"].mean(), 1), "S raw bp": round(-1e4 * d["S"].mean(), 1)}
    for k in ("long", "short", "both"):
        m, t, _ = month_t(d[k], d["month"])
        row[f"{k} net bp"], row[f"{k} t"] = round(m, 1), round(t, 2)
    return row


# --- the run ------------------------------------------------------------------------------------

def load_store(syms) -> dict[str, dict]:
    """The cached Yahoo pickles of ``syms``."""
    return {s: pickle.loads((DIV / f"{s}.pkl").read_bytes())
            for s in syms if (DIV / f"{s}.pkl").exists()}


def run() -> None:
    """Print RESEARCH_LOG §1: the registered rule, its legs, buckets, placebo, grid, event time."""
    uni = universe()
    syms = sorted({n for ps in uni.values() for *_, names in ps for n in names})
    store = load_store(syms)
    print(f"{len(store)} of {len(syms)} names have Yahoo history")
    opens = prices([s for s in syms if s in store])
    op = opens.to_numpy()
    R = op / np.vstack([np.full((1, op.shape[1]), np.nan), op[:-1]]) - 1
    ev, Y = events(opens, store)
    cal = opens.index
    print(f"{len(ev):,} regular-payer anchors, {100 * (ev['X'] >= 0).mean():.1f}% followed by a "
          f"dividend")
    Y2 = np.vstack([np.full((2, Y.shape[1]), np.nan), Y[:-2]])        # yield known 2 closes before
    res = {}
    for g, periods in uni.items():
        mem = membership(opens, periods)
        idx = {"m": bench_index(R, mem), "h": bench_index(R, mem & (Y2 >= MIN_YIELD))}
        res[g] = (mem, idx)

    def go(g, **kw):
        mem, idx = res[g]
        return legs(evaluate(ev, op, Y, mem, idx, **kw), cal, "h", kw.get("h", H))

    pd.set_option("display.width", 200)
    print("\n== registered rule (N=15, H=10, yield >= 3%): net bp per event, vs high-yield index")
    print(pd.DataFrame({g: summary(go(g)) for g in res}).T.to_string())
    print("\n== same, excess vs the equal-weight market (style not removed)")
    print(pd.DataFrame({g: summary(legs(evaluate(ev, op, Y, res[g][0], res[g][1]), cal, "m"))
                        for g in res}).T.to_string())
    print(f"\n== placebo: windows moved +{PLACEBO} sessions (mid-cycle)")
    print(pd.DataFrame({g: summary(go(g, shift=PLACEBO)) for g in res}).T.to_string())
    print("\n== registered rule by exit year, both legs net bp per event (POST-HOC diagnostic)")
    yr = {}
    for g in res:
        d, pl = go(g), go(g, shift=PLACEBO)
        for y, x in d.groupby(d["month"].dt.year):
            q = pl[pl["month"].dt.year == y]
            yr[(g, y)] = {"n": len(x), "event": round(x["both"].mean(), 1),
                          "placebo": round(q["both"].mean(), 1),
                          "event-placebo": round(x["both"].mean() - q["both"].mean(), 1)}
    print(pd.DataFrame(yr).T.to_string())
    d = go("TRAIN")
    print("\nTRAIN without 2020 (post-hoc):", summary(d[d["month"].dt.year != 2020]))
    print("\n== pay-date proxy, POST-HOC: long held to ex + 15 sessions, short [ex + 15, ex + 25)")
    print(pd.DataFrame({g: summary(go(g, lag=15)) for g in res}).T.to_string())
    print("\n== by trailing yield (all regular payers)")
    for g in res:
        d = go(g, min_yield=0.0)
        d["bucket"] = pd.cut(d["yield"], [0, .01, .02, .03, .04, 1],
                             labels=["<1", "1-2", "2-3", "3-4", ">=4"])
        print(g)
        rows = {b: summary(x) for b, x in d.groupby("bucket", observed=True)}
        print(pd.DataFrame(rows).T.to_string())
    print("\n== sensitivity grid: both legs, net bp per event (t)")
    for g in res:
        tab = {}
        for n in (10, 15, 20):
            tab[f"N={n}"] = {}
            for h in (5, 10, 20):
                s = summary(go(g, n=n, h=h))
                tab[f"N={n}"][f"H={h}"] = f"{s['both net bp']:+.0f} ({s['both t']:+.1f})"
        print(g)
        print(pd.DataFrame(tab).T.to_string())
    print("\n== event time (matched, yield >= 3%): daily excess vs high-yield index, bp")
    for g in res:
        mem, idx = res[g]
        d = ev[chosen(ev, Y, mem) & (ev["X"] >= 0)]
        X, cc = d["X"].to_numpy(), d["col"].to_numpy()
        rows = []
        for r in range(-30, 31):
            t = X + r
            ok = (t >= 1) & (t < len(op))
            ar = R[t[ok], cc[ok]] - (idx["h"][t[ok]] / idx["h"][t[ok] - 1] - 1)
            rows.append((r, 1e4 * np.nanmean(ar), int(np.isfinite(ar).sum())))
        et = pd.DataFrame(rows, columns=["day", "ar_bp", "n"]).set_index("day")
        et["car_bp"] = et["ar_bp"].cumsum()
        print(g, "(day 0 = ex-date open; day r = open[X+r-1] -> open[X+r])")
        print(et.loc[[-30, -20, -15, -10, -5, -2, -1, 0, 1, 2, 5, 10, 15, 20, 30],
                     ["ar_bp", "car_bp", "n"]].round(1).T.to_string())


if __name__ == "__main__":
    {"fetch": fetch, "run": run}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
