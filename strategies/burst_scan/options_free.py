"""Free options data for a few stocks: collect from every free source, store, compare, analyse.

Loaded by path from ``options_free_data.ipynb`` (like ``research.py``); ``strategy.py`` never
imports it. No API keys, no sign-ups. Two kinds of source:

* **Live snapshots** (about 15 minutes delayed, current chain only; history exists only from the
  day you start collecting):
  - ``cboe``: Cboe's delayed-quotes JSON. One request per stock returns the whole chain with
    bid/ask, last, volume, open interest, IV and Greeks.
  - ``yfinance``: Yahoo, one request per expiry. Bid/ask, last, volume, open interest and IV;
    no Greeks.
  - ``nasdaq``: api.nasdaq.com. Bid/ask, last, volume and open interest; no IV.
  - ``occ_oi``: OCC Series Search. Open interest per strike, as of the previous close.
* **History** (backfilled incrementally, day by day):
  - ``dolthub_chain``: the community DoltHub ``post-no-preference/options`` end-of-day chain
    since 2019. Bid/ask, IV and Greeks for about 100-200 near-the-money contracts per stock,
    1 week to 2 months out. No volume or open interest; some days are missing.
  - ``dolthub_vol``: the same repo's daily IV and historical-vol levels per stock.
  - ``occ_volume``: OCC Volume Query. Daily contract volume per stock, split by account type
    (C customer / F firm / M market maker), put/call and exchange. Rolling 2-year window only.

Every chain source is normalised to one contract key, ``(root, expiration, strike, cp)``, so the
same contract can be compared across sources (:func:`compare`). Storage is gzipped CSV under
``STORE`` (git-ignored), one file per source and day (live: per snapshot).

Terms: the free endpoints are for personal research. Do not redistribute the raw data. The
DoltHub data is CC BY-SA 4.0. Requests are paced and retried politely.
"""

from __future__ import annotations

import io
import math
import os
import re
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SYMBOLS = ["MU", "COIN", "CSCO", "AAL", "ANET"]
STORE = Path(__file__).resolve().parent / "options_data"
UA = {"User-Agent": "Mozilla/5.0 (research; options_free.py)", "Accept": "application/json, */*"}
DOLT = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
CHAIN_COLS = ["source", "fetched_at", "underlying", "root", "expiration", "strike", "cp", "bid",
              "ask", "last", "volume", "open_interest", "iv", "delta", "gamma", "theta", "vega",
              "underlying_price"]
_OCC_RE = re.compile(r"^([A-Z0-9]+?)(\d{6})([CP])(\d{8})$")


# ---------------------------------------------------------------------------------------------
# Plumbing.
# ---------------------------------------------------------------------------------------------
def _get(url: str, *, params=None, tries: int = 3, pause: float = 0.6, timeout: int = 60):
    """GET with a polite pause and retries; returns the Response (raises after ``tries``)."""
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            time.sleep(pause)
            if r.status_code == 200:
                return r
            if r.status_code in (403, 404):
                r.raise_for_status()
        except requests.RequestException:
            if k == tries - 1:
                raise
        time.sleep(2 ** k)
    r.raise_for_status()
    return r


def parse_occ_symbol(sym: str) -> tuple[str, str, str, float] | None:
    """``'MU261002C00005000'`` -> ``('MU', '2026-10-02', 'C', 5.0)``; None if malformed."""
    m = _OCC_RE.match(str(sym).replace(" ", "").replace("-", "").upper())
    if not m:
        return None
    root, ymd, cp, k = m.groups()
    return root, f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:]}", cp, int(k) / 1000.0


def _num(x) -> float:
    """Number from strings like '1,084.65' / '--' / None."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _frame(rows: list[dict], source: str, underlying: str, fetched_at: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in CHAIN_COLS:
        if c not in df:
            df[c] = np.nan
    df["source"], df["underlying"], df["fetched_at"] = source, underlying, fetched_at
    return df[CHAIN_COLS]


def _read(path) -> pd.DataFrame:
    """A stored file; an empty-day marker (no columns) reads as an empty frame."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------------------------
# Live sources (one stock at a time).
# ---------------------------------------------------------------------------------------------
def fetch_cboe(sym: str) -> pd.DataFrame:
    """Cboe delayed quotes: the whole chain with IV and Greeks (IV as a fraction)."""
    j = _get(f"https://cdn-api.cboe.com/api/global/delayed_quotes/options/{sym}.json").json()
    d, ts = j["data"], _now()
    rows = []
    for o in d.get("options", []):
        p = parse_occ_symbol(o.get("option", ""))
        if p is None:
            continue
        rows.append({"root": p[0], "expiration": p[1], "cp": p[2], "strike": p[3],
                     "bid": o.get("bid"), "ask": o.get("ask"), "last": o.get("last_trade_price"),
                     "volume": o.get("volume"), "open_interest": o.get("open_interest"),
                     "iv": o.get("iv") or np.nan, "delta": o.get("delta"), "gamma": o.get("gamma"),
                     "theta": o.get("theta"), "vega": o.get("vega"),
                     "underlying_price": d.get("current_price")})
    return _frame(rows, "cboe", sym, ts)


def fetch_yfinance(sym: str, *, pause: float = 0.5) -> pd.DataFrame:
    """Yahoo via yfinance: one request per expiry; no Greeks."""
    import yfinance as yf

    t, ts, rows = yf.Ticker(sym), _now(), []
    try:
        spot = float(t.fast_info["last_price"])
    except Exception:  # noqa: BLE001 - Yahoo fields come and go
        spot = np.nan
    for exp in t.options:
        ch = t.option_chain(exp)
        time.sleep(pause)
        for cp, part in (("C", ch.calls), ("P", ch.puts)):
            for r in part.itertuples(index=False):
                p = parse_occ_symbol(r.contractSymbol)
                if p is None:
                    continue
                rows.append({"root": p[0], "expiration": p[1], "cp": cp, "strike": p[3],
                             "bid": r.bid, "ask": r.ask, "last": r.lastPrice, "volume": r.volume,
                             "open_interest": r.openInterest, "iv": r.impliedVolatility,
                             "underlying_price": spot})
    return clean_yahoo(_frame(rows, "yfinance", sym, ts))


def clean_yahoo(df: pd.DataFrame) -> pd.DataFrame:
    """Outside US market hours Yahoo serves placeholders: bid = ask = 0, open interest 0 for the
    whole chain and IV 0.00001 / 0.5. Mark them missing so they never count as data."""
    df = df.copy()
    dead = (df["bid"].fillna(0) == 0) & (df["ask"].fillna(0) == 0)
    df.loc[dead, ["bid", "ask", "iv"]] = np.nan
    df.loc[df["iv"] <= 1e-4, "iv"] = np.nan
    if df["open_interest"].fillna(0).sum() == 0:
        df["open_interest"] = np.nan
    return df


def parse_nasdaq(rows_json: list[dict]) -> list[dict]:
    """Nasdaq option-chain rows (one row = call + put at a strike) -> one row per contract."""
    out = []
    for r in rows_json:
        url = r.get("drillDownURL") or ""
        p = parse_occ_symbol(url.rsplit("/", 1)[-1]) if url else None
        if p is None:
            continue
        for cp, pre in (("C", "c_"), ("P", "p_")):
            out.append({"root": p[0], "expiration": p[1], "cp": cp, "strike": p[3],
                        "bid": _num(r.get(pre + "Bid")), "ask": _num(r.get(pre + "Ask")),
                        "last": _num(r.get(pre + "Last")), "volume": _num(r.get(pre + "Volume")),
                        "open_interest": _num(r.get(pre + "Openinterest"))})
    return out


def fetch_nasdaq(sym: str) -> pd.DataFrame:
    """api.nasdaq.com option chain (all expiries); no IV."""
    j = _get(f"https://api.nasdaq.com/api/quote/{sym}/option-chain",
             params={"assetclass": "stocks", "limit": 20000, "fromdate": "all",
                     "todate": "undefined", "excode": "oprac", "callput": "callput",
                     "money": "all", "type": "all"}).json()
    rows = ((j.get("data") or {}).get("table") or {}).get("rows") or []
    return _frame(parse_nasdaq(rows), "nasdaq", sym, _now())


def parse_occ_series(text: str) -> list[dict]:
    """OCC Series Search text -> one row per contract with ``open_interest``."""
    out = []
    for line in text.splitlines():
        f = line.split()
        if len(f) < 10 or not f[1].isdigit() or len(f[1]) != 4:
            continue
        root, y, m, d, k_int, k_dec = f[0], f[1], f[2], f[3], f[4], f[5]
        try:
            call_oi, put_oi = float(f[-3]), float(f[-2])
        except ValueError:
            continue
        strike = float(k_int) + float(f"0.{k_dec}")
        exp = f"{y}-{int(m):02d}-{int(d):02d}"
        out += [{"root": root, "expiration": exp, "cp": "C", "strike": strike,
                 "open_interest": call_oi},
                {"root": root, "expiration": exp, "cp": "P", "strike": strike,
                 "open_interest": put_oi}]
    return out


def fetch_occ_oi(sym: str) -> pd.DataFrame:
    """OCC Series Search: open interest per contract (previous close)."""
    t = _get("https://marketdata.theocc.com/series-search",
             params={"symbolType": "U", "symbol": sym}).text
    return _frame(parse_occ_series(t), "occ_oi", sym, _now())


LIVE = {"cboe": fetch_cboe, "yfinance": fetch_yfinance, "nasdaq": fetch_nasdaq,
        "occ_oi": fetch_occ_oi}


def collect_snapshot(symbols=SYMBOLS, sources=tuple(LIVE), store: Path = STORE) -> pd.DataFrame:
    """Pull every live source for every stock and store one file per source.

    Returns a summary (source, stock, contracts, error). A failing source never stops the others.
    """
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
    log = []
    for src in sources:
        frames = []
        for sym in symbols:
            try:
                df = LIVE[src](sym)
                frames.append(df)
                log.append({"source": src, "underlying": sym, "contracts": len(df), "error": ""})
            except Exception as e:  # noqa: BLE001 - report and continue
                log.append({"source": src, "underlying": sym, "contracts": 0,
                            "error": f"{type(e).__name__}: {str(e)[:120]}"})
        if frames:
            path = store / "live" / src / f"{stamp}.csv.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(frames).to_csv(path, index=False)
    return pd.DataFrame(log)


def load_snapshot(stamp: str | None = None, store: Path = STORE) -> dict[str, pd.DataFrame]:
    """The live files of one collection run (``stamp`` = 'YYYY-MM-DD_HHMMSS'; latest if None)."""
    files = sorted((store / "live").glob("*/*.csv.gz"))
    if not files:
        return {}
    stamp = stamp or max(f.name[:17] for f in files)
    out = {f.parent.name: _read(f) for f in files if f.name.startswith(stamp)}
    if "yfinance" in out:                      # files stored before the cleaning existed
        out["yfinance"] = clean_yahoo(out["yfinance"])
    return out


def snapshot_stamps(store: Path = STORE) -> list[str]:
    return sorted({f.name[:17] for f in (store / "live").glob("*/*.csv.gz")})


# ---------------------------------------------------------------------------------------------
# History (incremental backfill, one file per day).
# ---------------------------------------------------------------------------------------------
def _dolt(sql: str) -> pd.DataFrame:
    j = _get(DOLT, params={"q": sql}, pause=0.3).json()
    if j.get("query_execution_status") != "Success":
        raise RuntimeError(j.get("query_execution_message", "DoltHub query failed"))
    return pd.DataFrame(j.get("rows") or [])


def fetch_dolthub_chain(day: str, symbols=SYMBOLS) -> pd.DataFrame:
    """DoltHub end-of-day chain of ``day`` -> the common chain columns (IV as a fraction).

    DoltHub stores ONE ``vol`` per strike and expiry (the call and the put share it)."""
    syms = ",".join(f"'{s}'" for s in symbols)
    df = _dolt(f"SELECT * FROM option_chain WHERE date='{day}' AND act_symbol IN ({syms})")
    if df.empty:
        return _frame([], "dolthub_chain", "", day)
    out = pd.DataFrame({
        "root": df["act_symbol"], "expiration": df["expiration"],
        "cp": df["call_put"].str[0].str.upper(), "strike": df["strike"].astype(float),
        "bid": df["bid"].astype(float), "ask": df["ask"].astype(float),
        "iv": df["vol"].astype(float), "delta": df["delta"].astype(float),
        "gamma": df["gamma"].astype(float), "theta": df["theta"].astype(float),
        "vega": df["vega"].astype(float)})
    out = _frame(out.to_dict("records"), "dolthub_chain", "", day)
    out["underlying"] = out["root"]
    return out


def fetch_dolthub_vol(day: str, symbols=SYMBOLS) -> pd.DataFrame:
    syms = ",".join(f"'{s}'" for s in symbols)
    return _dolt(f"SELECT * FROM volatility_history WHERE date='{day}' AND act_symbol IN ({syms})")


def fetch_occ_volume(day: str, symbols=SYMBOLS) -> pd.DataFrame:
    """OCC daily volume per stock x account type (C/F/M) x put/call x exchange."""
    frames = []
    for sym in symbols:
        t = _get("https://marketdata.theocc.com/volume-query",
                 params={"reportDate": day.replace("-", ""), "format": "csv",
                         "volumeQueryType": "O", "symbolType": "U", "symbol": sym,
                         "reportType": "D", "accountType": "ALL", "productKind": "ALL",
                         "porc": "BOTH"}, pause=0.4).text
        if "quantity" not in t[:200]:
            continue
        df = pd.read_csv(io.StringIO(t), index_col=False)
        frames.append(df[["quantity", "underlying", "symbol", "actype", "porc", "exchange"]])
    if not frames:
        return pd.DataFrame(columns=["date", "quantity", "underlying", "symbol", "actype",
                                     "porc", "exchange"])
    out = pd.concat(frames)
    out.insert(0, "date", day)
    return out


def marketdata_token() -> str | None:
    """The MarketData.app token: env var ``MARKETDATA_TOKEN``, else the repo's git-ignored ``.env``.
    Never printed or stored."""
    tok = os.environ.get("MARKETDATA_TOKEN")
    env = Path(__file__).resolve().parents[2] / ".env"
    if not tok and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("MARKETDATA_TOKEN="):
                tok = line.split("=", 1)[1].strip().strip('"').strip("'")
    return tok or None


def parse_marketdata(j: dict, day: str) -> pd.DataFrame:
    """A MarketData.app chain response (column arrays) -> the common chain columns, plus
    ``bid_size`` / ``ask_size``. Historical responses carry no IV / Greeks (null)."""
    if j.get("s") != "ok":
        return _frame([], "marketdata_chain", "", day)
    d = pd.DataFrame({k: v for k, v in j.items() if isinstance(v, list)})
    out = pd.DataFrame({
        "root": d["underlying"], "underlying": d["underlying"],
        "expiration": pd.to_datetime(d["expiration"], unit="s").dt.strftime("%Y-%m-%d"),
        "strike": d["strike"].astype(float), "cp": d["side"].str[0].str.upper(),
        "bid": d["bid"], "ask": d["ask"], "last": d["last"], "volume": d["volume"],
        "open_interest": d["openInterest"], "iv": d.get("iv"), "delta": d.get("delta"),
        "gamma": d.get("gamma"), "theta": d.get("theta"), "vega": d.get("vega"),
        "underlying_price": d["underlyingPrice"]})
    out = _frame(out.to_dict("records"), "marketdata_chain", "", day)
    out["underlying"] = out["root"]
    out["bid_size"], out["ask_size"] = d["bidSize"].to_numpy(), d["askSize"].to_numpy()
    return out


def fetch_marketdata_chain(day: str, symbols=SYMBOLS) -> pd.DataFrame:
    """MarketData.app end-of-day chain of ``day`` (all expiries) with OPEN INTEREST and VOLUME
    per contract. Needs ``MARKETDATA_TOKEN``; free / trial accounts reach back 1 year only.
    Costs ~1 credit per 1000 contracts."""
    tok = marketdata_token()
    if not tok:
        raise RuntimeError("MARKETDATA_TOKEN not set (env or .env)")
    frames = []
    for sym in symbols:
        r = requests.get(f"https://api.marketdata.app/v1/options/chain/{sym}/",
                         params={"date": day, "expiration": "all"},
                         headers={**UA, "Authorization": f"Bearer {tok}"}, timeout=180)
        time.sleep(0.3)
        if r.status_code in (402, 401, 403):
            raise RuntimeError(f"MarketData.app {r.status_code}: {r.json().get('errmsg', '')}")
        frames.append(parse_marketdata(r.json(), day))
    return pd.concat(frames, ignore_index=True)


HISTORY = {"dolthub_chain": fetch_dolthub_chain, "dolthub_vol": fetch_dolthub_vol,
           "occ_volume": fetch_occ_volume, "marketdata_chain": fetch_marketdata_chain}


def backfill(source: str, start: str, end: str | None = None, symbols=SYMBOLS,
             store: Path = STORE, *, max_days: int | None = None) -> pd.DataFrame:
    """Fetch every weekday in [start, end] not stored yet (one file per day; holidays are stored
    empty so they are not asked again, except the last 4 days, which may simply not be
    published yet). Returns a log (day, rows, error)."""
    end = end or date.today().isoformat()
    folder = store / source
    folder.mkdir(parents=True, exist_ok=True)
    have = {f.name[:10] for f in folder.glob("*.csv.gz")}
    recent = (date.today() - timedelta(days=4)).isoformat()
    log = []
    for d in pd.bdate_range(start, end).strftime("%Y-%m-%d"):
        if d in have:
            continue
        if max_days is not None and len(log) >= max_days:
            break
        try:
            df = HISTORY[source](d, symbols)
            if len(df) or d < recent:
                df.to_csv(folder / f"{d}.csv.gz", index=False)
            log.append({"day": d, "rows": len(df), "error": ""})
        except Exception as e:  # noqa: BLE001 - keep going; the day is retried next run
            log.append({"day": d, "rows": 0, "error": f"{type(e).__name__}: {str(e)[:120]}"})
    return pd.DataFrame(log, columns=["day", "rows", "error"])


def load_history(source: str, store: Path = STORE) -> pd.DataFrame:
    """Every stored day of a history source in one frame (empty days skipped)."""
    frames = []
    for f in sorted((store / source).glob("*.csv.gz")):
        df = _read(f)
        if len(df):
            if "date" not in df:
                df.insert(0, "date", f.name[:10])
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def inventory(store: Path = STORE) -> pd.DataFrame:
    """What is stored: per source, files, non-empty days, first and last day, rows."""
    out = []
    for src in HISTORY:
        files = sorted((store / src).glob("*.csv.gz"))
        sizes = [(f.name[:10], len(_read(f))) for f in files]
        full = [d for d, n in sizes if n]
        out.append({"source": src, "kind": "history", "files": len(files),
                    "days_with_data": len(full),
                    "first": min(full) if full else None, "last": max(full) if full else None,
                    "rows": sum(n for _, n in sizes)})
    for src in LIVE:
        files = sorted((store / "live" / src).glob("*.csv.gz"))
        out.append({"source": src, "kind": "live snapshots", "files": len(files),
                    "days_with_data": len({f.name[:10] for f in files}),
                    "first": files[0].name[:17] if files else None,
                    "last": files[-1].name[:17] if files else None,
                    "rows": sum(len(_read(f)) for f in files)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------
# Cross-source comparison.
# ---------------------------------------------------------------------------------------------
KEY = ["root", "expiration", "strike", "cp"]
FIELDS = ["bid", "ask", "last", "volume", "open_interest", "iv", "delta"]


def compare(frames: dict[str, pd.DataFrame], fields=FIELDS, *, tol: float = 0.01) -> pd.DataFrame:
    """Every pair of sources x every field both carry: contracts matched on the contract key, the
    share that agree (|difference| <= ``tol`` x max(1, |value|), i.e. 1 cent / 1% / 0.01 vol),
    the median absolute difference, and the rank correlation."""
    out = []
    names = [n for n in frames if len(frames[n])]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A = frames[a].drop_duplicates(KEY).set_index(KEY)
            B = frames[b].drop_duplicates(KEY).set_index(KEY)
            j = A.join(B, how="inner", lsuffix="_a", rsuffix="_b")
            for f in fields:
                x, y = j.get(f + "_a"), j.get(f + "_b")
                if x is None or y is None:
                    continue
                ok = x.notna() & y.notna()
                if ok.sum() < 5:
                    continue
                x, y = x[ok].astype(float), y[ok].astype(float)
                diff = (x - y).abs()
                out.append({"source_a": a, "source_b": b, "field": f, "matched": int(ok.sum()),
                            "agree_share": float((diff <= tol * np.maximum(1, x.abs())).mean()),
                            "median_abs_diff": float(diff.median()),
                            "rank_corr": float(x.rank().corr(y.rank()))})
    return pd.DataFrame(out)


def coverage(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Contracts per source and stock, and the share of each field that is filled."""
    rows = []
    for name, df in frames.items():
        for sym, g in df.groupby("underlying"):
            r = {"source": name, "underlying": sym, "contracts": len(g),
                 "expiries": g["expiration"].nunique()}
            for f in FIELDS:
                r[f] = float(g[f].notna().mean()) if f in g else 0.0
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Analysis (history): daily IV features -> next-day / next-week direction.
# ---------------------------------------------------------------------------------------------
def iv_features(chain: pd.DataFrame, *, target_dte: int = 30) -> pd.DataFrame:
    """Per (date, stock) from an end-of-day chain (DoltHub): the expiry closest to ``target_dte``
    days (7-75 days out), then ``atm_iv`` (IV of the call with delta closest to 0.5), ``skew``
    (IV of the put with delta closest to -0.25 minus IV of the call with delta closest to 0.25),
    and ``term`` (ATM IV of the nearest expiry >= 5 days minus ``atm_iv``). Contracts without a
    real two-sided quote (bid 0, or a spread over half the mid) are dropped first: their IV is a
    model value off a junk quote (a quarter of AAL's and CSCO's DoltHub rows)."""
    c = chain.dropna(subset=["iv", "delta"]).copy()
    c = c[(c["iv"] > 0.01) & (c["iv"] < 5)]
    if {"bid", "ask"} <= set(c.columns):
        mid = (c["bid"] + c["ask"]) / 2
        c = c[(c["bid"] > 0) & ((c["ask"] - c["bid"]) <= 0.5 * mid)]
    c["dte"] = (pd.to_datetime(c["expiration"]) - pd.to_datetime(c["date"])).dt.days
    rows = []
    for (d, sym), g in c.groupby(["date", "underlying"]):
        exps = g[(g["dte"] >= 7) & (g["dte"] <= 75)]["dte"].unique()
        if not len(exps):
            continue
        e = exps[np.argmin(np.abs(exps - target_dte))]
        ge = g[g["dte"] == e]
        calls, puts = ge[ge["cp"] == "C"], ge[ge["cp"] == "P"]
        if calls.empty or puts.empty:
            continue

        def near(frame, target):
            return float(frame.iloc[(frame["delta"] - target).abs().argmin()]["iv"])

        atm = near(calls, 0.5)
        r = {"date": d, "underlying": sym, "dte": int(e), "atm_iv": atm,
             "skew": near(puts, -0.25) - near(calls, 0.25)}
        short = g[(g["dte"] >= 5) & (g["dte"] < e)]
        sc = short[short["cp"] == "C"]
        r["term"] = (near(sc[sc["dte"] == sc["dte"].min()], 0.5) - atm) if len(sc) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def occ_features(vol: pd.DataFrame) -> pd.DataFrame:
    """Per (date, stock): customer put/call volume ratio (log), customer share of all volume, and
    total volume against its own previous 20-day mean (log)."""
    v = vol.groupby(["date", "underlying", "actype", "porc"])["quantity"].sum().unstack(
        ["actype", "porc"], fill_value=0)
    tot = v.sum(axis=1)
    cust_c = v.get(("C", "C"), 0)
    cust_p = v.get(("C", "P"), 0)
    out = pd.DataFrame({"cust_pc": np.log((cust_p + 1) / (cust_c + 1)),
                        "cust_share": (cust_c + cust_p) / tot.replace(0, np.nan),
                        "total": tot}).reset_index()
    out = out.sort_values(["underlying", "date"])
    prev = out.groupby("underlying")["total"].transform(lambda s: s.shift(1).rolling(20, 10).mean())
    out["vol_surge"] = np.log(out["total"] / prev)
    return out.drop(columns="total")


def forward_returns(daily, *, horizons=(1, 5)) -> pd.DataFrame:
    """From a daily Panel: the return from the NEXT session's open to its close (h=1) or to the
    close h sessions later - a feature known at day d's close is traded at d+1's open."""
    o, c = daily.open_, daily.close
    rows = []
    for h in horizons:
        r = np.log(c.shift(-h) / o.shift(-1))
        s = r.stack().rename(f"ret_{h}d").reset_index()
        s.columns = ["date", "underlying", f"ret_{h}d"]
        rows.append(s)
    out = rows[0]
    for r in rows[1:]:
        out = out.merge(r, on=["date", "underlying"])
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    return out


def zscore_by_name(df: pd.DataFrame, cols, *, window: int = 60) -> pd.DataFrame:
    """Each feature minus its own stock's PREVIOUS ``window``-day mean, over that sd (causal)."""
    d = df.sort_values(["underlying", "date"]).copy()
    for c in cols:
        g = d.groupby("underlying")[c]
        m = g.transform(lambda s: s.shift(1).rolling(window, 20).mean())
        s = g.transform(lambda s: s.shift(1).rolling(window, 20).std())
        d[c + "_z"] = (d[c] - m) / s
    return d


def ic_table(df: pd.DataFrame, features, targets) -> pd.DataFrame:
    """Rank IC of each feature with each forward return, pooled over stocks, with a t-statistic
    clustered by day (the stocks share each day's market), plus the per-stock ICs."""
    rows = []
    for f in features:
        for t in targets:
            d = df[["date", "underlying", f, t]].dropna()
            if len(d) < 30:
                continue
            a, b = d[f].rank(), d[t].rank()
            za, zb = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
            p = (za * zb).to_numpy()
            days = d["date"].to_numpy()
            m = p.mean()
            sums = pd.Series(p - m).groupby(days).sum().to_numpy()
            g = len(sums)
            se = math.sqrt((sums ** 2).sum() * g / (g - 1)) / len(p) if g > 2 else np.nan
            r = {"feature": f, "target": t, "n": len(d), "days": g, "ic": m,
                 "t": m / se if se else np.nan}
            for sym, gs in d.groupby("underlying"):
                r[sym] = gs[f].rank().corr(gs[t].rank()) if len(gs) > 20 else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def md_features(chain: pd.DataFrame, atm_iv: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (date, stock) from a chain WITH open interest and volume (MarketData.app history):

    * ``oi_pc`` / ``vol_pc``: log put / call open interest and volume (+1 smoothed);
    * ``d_oi``: log change of total open interest from the previous stored day;
    * ``gex``: dealer gamma imbalance in [-1, 1] - sum(OI x Black-Scholes gamma x S^2) with calls
      counted + and puts - (the usual "dealers long calls, short puts" assumption), divided by the
      same sum without signs. Gamma uses ONE vol per stock-day: ``atm_iv`` (from
      :func:`iv_features`,
      columns date / underlying / atm_iv) when given, else 0.5. Model-based, like every vendor GEX.
    """
    c = chain.dropna(subset=["open_interest", "strike", "underlying_price"]).copy()
    c["dte"] = (pd.to_datetime(c["expiration"]) - pd.to_datetime(c["date"])).dt.days
    c = c[c["dte"] >= 0]
    if atm_iv is not None and len(atm_iv):
        c = c.merge(atm_iv[["date", "underlying", "atm_iv"]], on=["date", "underlying"], how="left")
    else:
        c["atm_iv"] = np.nan
    sig = c["atm_iv"].fillna(0.5).clip(0.05, 3.0)
    S, K = c["underlying_price"].astype(float), c["strike"].astype(float)
    T = np.maximum(c["dte"], 0.5) / 365.0
    d1 = (np.log(S / K) + 0.5 * sig ** 2 * T) / (sig * np.sqrt(T))
    gamma = np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi) / (S * sig * np.sqrt(T))
    w = c["open_interest"] * gamma * S ** 2
    c["g_signed"] = np.where(c["cp"] == "C", w, -w)
    c["g_abs"] = w
    g = c.groupby(["date", "underlying"])
    oi = c.pivot_table(index=["date", "underlying"], columns="cp", values="open_interest",
                       aggfunc="sum").fillna(0)
    vol = c.pivot_table(index=["date", "underlying"], columns="cp", values="volume",
                        aggfunc="sum").fillna(0)
    out = pd.DataFrame({"oi_pc": np.log((oi.get("P", 0) + 1) / (oi.get("C", 0) + 1)),
                        "vol_pc": np.log((vol.get("P", 0) + 1) / (vol.get("C", 0) + 1)),
                        "oi_total": oi.sum(axis=1),
                        "gex": g["g_signed"].sum() / g["g_abs"].sum().replace(0, np.nan)})
    out = out.reset_index().sort_values(["underlying", "date"])
    out["d_oi"] = out.groupby("underlying")["oi_total"].transform(lambda s: np.log(s / s.shift(1)))
    return out.drop(columns="oi_total")
