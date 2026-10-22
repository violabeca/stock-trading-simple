"""Every hyperparameter of the classic_signals strategy in one table, and the settings files.

Three kinds of rows:

* the settings declared here (:data:`DECLARED`): the candles, the indicators and the exit;
* one on/off switch per rule, ``on: <rule name>`` (its starting position from the rule's ``ON``);
* every lever of every rule (the rule's ``PARAMS``; a lever only acts while its rule is on).

A row's ``needs`` says when it acts: a switch's name (its rule must be on), or a
``(setting, values)`` pair (the stop's distance in daily moves only for a daily stop, say).

Each row says what the hyperparameter does, its starting value, the range it may take (the
main notebook's slider), the range the tuner searches, and what it changes (:data:`STAGES`).
Both notebooks read this table: the main notebook builds its sliders from it, the tuning
notebook its search space, so a setting is declared once. :func:`save` and :func:`load` write
and read the settings the tuning notebook exports; :func:`starting` turns one of them into the
main notebook's starting values.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SETTINGS = HERE.parents[1] / "var" / "classic_signals" / "settings"   # exported settings files

#: What a hyperparameter changes: from the most to the least work to recompute.
STAGES = {"candles": "every candle and indicator", "indicators": "one indicator and the rules "
          "that read it", "rules": "the decisions only", "exit": "how the trades end"}

#: The settings declared here. ``choices`` maps each value to its words; numbers have ``low``,
#: ``high`` and ``step`` (the slider) and optionally ``search`` (the tuner's range, else the same).
DECLARED = {
    "candle_size": {
        "group": "candles", "label": "candle size", "value": 1, "stage": "candles",
        "choices": {1: "1 minute", 5: "5 minutes", 15: "15 minutes"},
        "doc": "The candles the strategy trades and every intraday indicator reads."},
    "band_length": {
        "group": "Bollinger Bands", "label": "band: candles in the average", "value": 20,
        "low": 10, "high": 120, "step": 5, "search": (10, 60, 5), "stage": "indicators",
        "doc": "How many candles the bands' middle line averages. Longer: slower bands."},
    "band_width": {
        "group": "Bollinger Bands", "label": "band distance, standard deviations",
        "value": 2.0, "low": 1.0, "high": 3.5, "step": 0.25, "search": (1.5, 3.5, 0.25),
        "stage": "indicators",
        "doc": "How far the bands sit from the middle. Wider: rarer, more extreme signals."},
    "extended_bands": {
        "group": "Bollinger Bands", "label": "bands and average also read the hours outside "
        "the regular session", "value": False, "stage": "indicators",
        "choices": {False: "no: regular hours only", True: "yes: every stored minute"},
        "doc": "Whether the bands and the moving average run through the pre-market, the "
               "after-hours and the night too (earlier, but jumpier)."},
    "average_length": {
        "group": "moving average", "label": "moving average: candles", "value": 60, "low": 10,
        "high": 400, "step": 10, "stage": "indicators",
        "doc": "The minute chart's moving average (read by the moving-average filter)."},
    "trend_length": {
        "group": "daily trend", "label": "trend average, days", "value": 50, "low": 10,
        "high": 200, "step": 5, "stage": "indicators",
        "doc": "The daily average the stock, sector and market trends compare the close with "
               "(read by the three daily trend filters, and the market condition of a trade)."},
    "trend_slope": {
        "group": "daily trend", "label": "its rise is measured over, days", "value": 10,
        "low": 3, "high": 30, "step": 1, "stage": "indicators",
        "doc": "Over how many days the trend average's rise is measured."},
    "trend_flat": {
        "group": "daily trend", "label": "flat zone, ± typical daily moves", "value": 0.5,
        "low": 0.0, "high": 2.0, "step": 0.1, "stage": "indicators",
        "doc": "How little the average may move and still count as flat (sideways)."},
    "level_span": {
        "group": "major levels", "label": "turning point: highest/lowest of ± days",
        "value": 5, "low": 2, "high": 10, "step": 1, "stage": "indicators",
        "doc": "How many days on each side a turning point must top (or undercut)."},
    "level_lookback": {
        "group": "major levels", "label": "memory: sessions back", "value": 120, "low": 40,
        "high": 250, "step": 10, "stage": "indicators",
        "doc": "How old a turning point may be and still count."},
    "level_tolerance": {
        "group": "major levels", "label": "same level if closer than, × typical daily move",
        "value": 0.5, "low": 0.1, "high": 1.5, "step": 0.1, "stage": "indicators",
        "doc": "How close turning points must be to form one level."},
    "level_touches": {
        "group": "major levels", "label": "turning points needed", "value": 2, "low": 2,
        "high": 5, "step": 1, "stage": "indicators",
        "doc": "How many turning points make a major level."},
    "level_mix": {
        "group": "major levels", "label": "a ceiling and a floor may form one level",
        "value": True, "stage": "indicators", "choices": {True: "yes", False: "no"},
        "doc": "Whether swing highs and swing lows group together."},
    "move_size": {
        "group": "typical move", "label": "typical move: candle size", "value": 15,
        "stage": "indicators", "choices": {5: "5 minutes", 15: "15 minutes", 60: "60 minutes"},
        "doc": "The candles the typical move (the stop's yardstick) is measured on."},
    "move_length": {
        "group": "typical move", "label": "typical move: candles averaged", "value": 14,
        "low": 5, "high": 50, "step": 1, "stage": "indicators",
        "doc": "How many of those candles the typical move averages."},
    "stop_kind": {
        "group": "exit", "label": "the stop", "value": "moves", "stage": "exit",
        "choices": {"moves": "trailing: typical moves behind the best price",
                    "daily trailing": "trailing: typical daily moves behind the best price",
                    "daily fixed": "fixed: typical daily moves from the entry"},
        "doc": "How the stop is set: trailing the best price since the entry (by typical moves "
               "of the move candles, or by typical daily moves) or fixed at typical daily moves "
               "from the entry. Daily moves make one setting mean the same for a calm and a "
               "jumpy stock, and in a calm and a wild quarter."},
    "stop_moves": {
        "group": "exit", "label": "trailing stop: typical moves behind the best price",
        "value": 3.0, "low": 1.0, "high": 8.0, "step": 0.5, "stage": "exit",
        "needs": ("stop_kind", ("moves",)),
        "doc": "How far the trailing stop trails, in typical moves."},
    "stop_days": {
        "group": "exit", "label": "stop distance, typical daily moves", "value": 1.0,
        "low": 0.1, "high": 3.0, "step": 0.1, "stage": "exit",
        "needs": ("stop_kind", ("daily trailing", "daily fixed")),
        "doc": "How far the daily stop sits from the best price (trailing) or the entry "
               "(fixed), in typical daily moves (the average daily range of the last 14 "
               "sessions)."},
    "max_sessions": {
        "group": "exit", "label": "holding limit", "value": 0, "stage": "exit",
        "choices": {0: "no limit", 1: "close by the end of the entry session",
                    2: "at most 2 sessions", 5: "at most 5 sessions"},
        "doc": "Close a trade at the end of its n-th session (1: never overnight)."},
    "target_kind": {
        "group": "exit", "label": "profit target", "value": "none", "stage": "exit",
        "choices": {"none": "none", "major": "the next major level",
                    "both": "the next major level or previous high / low",
                    "daily moves": "typical daily moves from the entry",
                    "option walls": "the next option wall"},
        "doc": "Close a trade at the next level in its direction (at least one typical move "
               "from the entry: a major level, the previous high / low, or an option wall), or "
               "at a number of typical daily moves from the entry."},
    "target_days": {
        "group": "exit", "label": "profit target, typical daily moves", "value": 1.0,
        "low": 0.1, "high": 3.0, "step": 0.1, "stage": "exit",
        "needs": ("target_kind", ("daily moves",)),
        "doc": "How far the daily-move profit target sits from the entry."},
}


def active(row: dict, settings: dict) -> bool:
    """Whether a hyperparameter acts with these settings: always, or while the switch or the
    choice its ``needs`` names allows it."""
    need = row.get("needs")
    if need is None:
        return True
    if isinstance(need, str):
        return bool(settings[need])
    name, allowed = need
    return settings[name] in allowed


def needs_words(row: dict) -> str:
    """When a hyperparameter acts, in words ("" for always)."""
    need = row.get("needs")
    if need is None or isinstance(need, str):
        return need or ""
    return f"{need[0]}: {' / '.join(need[1])}"


def stop_args(settings: dict) -> dict:
    """The exit settings ``simulate.run`` takes (the profit target goes in as ``targets``)."""
    return {k: settings[k] for k in ("stop_kind", "stop_moves", "stop_days", "max_sessions")}


def switch_key(rule_name: str) -> str:
    """The hyperparameter that switches a rule on or off: ``"on: <rule name>"``."""
    return f"on: {rule_name}"


def table(rules: dict) -> dict:
    """Every hyperparameter: name -> its row (``group``, ``label``, ``value``, ``stage``, ``doc``
    and either ``choices`` or ``low`` / ``high`` / ``step`` / ``search``; a rule lever also has
    ``needs``: the switch it acts under). Declared settings first, then the rule switches, then
    the levers, so a switch always comes before its levers."""
    out = {}
    for name, row in DECLARED.items():
        r = dict(row)
        if "choices" not in r:
            r.setdefault("search", (r["low"], r["high"], r["step"]))
        out[name] = r
    for name, rule in rules.items():
        out[switch_key(name)] = {
            "group": f"switch ({rule.KIND})", "label": name, "value": bool(getattr(rule, "ON",
                                                                                    True)),
            "stage": "rules", "choices": {True: "on", False: "off"}, "doc": rule.DESCRIPTION}
    for name, rule in rules.items():
        for key, p in rule.PARAMS.items():
            out[key] = {"group": f"lever of {name}", "label": p["label"], "value": p["value"],
                        "low": p["start"], "high": p["stop"], "step": p["step"],
                        "search": (p["start"], p["stop"], p["step"]), "stage": "rules",
                        "needs": switch_key(name),
                        "doc": f"A lever of '{name}': it only acts while that rule is on."}
    return out


def defaults(rules: dict) -> dict:
    """The built-in starting value of every hyperparameter."""
    return {name: row["value"] for name, row in table(rules).items()}


def frame(rules: dict, settings: dict | None = None) -> pd.DataFrame:
    """The table as rows for reading: name, group, what it does, the value (``settings`` or the
    built-in one), the slider range, the searched range and what a change rebuilds."""
    settings = settings or {}
    rows = []
    for name, r in table(rules).items():
        if "choices" in r:
            rng = searched = " / ".join(str(v) for v in r["choices"].values())
        else:
            rng = f"{r['low']} to {r['high']}, step {r['step']}"
            searched = f"{r['search'][0]} to {r['search'][1]}, step {r['search'][2]}"
        rows.append({"hyperparameter": name, "group": r["group"], "what it is": r["label"],
                     "value": settings.get(name, r["value"]), "may take": rng,
                     "tuner searches": searched, "a change rebuilds": STAGES[r["stage"]],
                     "acts only while": needs_words(r), "what it does": r["doc"]})
    return pd.DataFrame(rows)


def default_space(rules: dict) -> dict:
    """The search space with the optimizer free everywhere: every number across its ``search``
    range as ``(from, to, step)``, every choice allowed (a list; a switch: ``[True, False]``)."""
    return {name: list(r["choices"]) if "choices" in r else tuple(r["search"])
            for name, r in table(rules).items()}


def clean_space(space: dict, rules: dict) -> dict:
    """A configured space made usable: a number's ``(from, to, step)`` kept inside its range,
    in order and on its grid (``from == to``: fixed at that value); a choice's allowed values
    in the table's order (none allowed: all of them). Missing names get the default space."""
    out = {}
    for name, r in table(rules).items():
        spec = space.get(name)
        if "choices" in r:
            allowed = [v for v in r["choices"] if spec is None or v in spec]
            out[name] = allowed or list(r["choices"])
            continue
        lo, hi, st = spec if spec is not None else r["search"]
        whole = all(float(x).is_integer() for x in (r["low"], r["high"], r["step"]))
        st = st if st and st > 0 else r["step"]
        lo, hi = sorted(min(max(float(v), r["low"]), r["high"]) for v in (lo, hi))
        hi = lo + math.floor((hi - lo) / st + 1e-9) * st
        out[name] = ((int(round(lo)), int(round(hi)), int(round(st))) if whole
                     else (round(lo, 9), round(hi, 9), round(float(st), 9)))
    return out


def fixed_space(settings: dict, rules: dict, free=()) -> dict:
    """A space with every hyperparameter fixed at its value in ``settings``, except the ``free``
    ones (their default range)."""
    out = default_space(rules)
    for name, r in table(rules).items():
        if name not in free:
            v = settings[name]
            out[name] = [v] if "choices" in r else (v, v, r["search"][2])
    return clean_space(out, rules)


def is_free(spec) -> bool:
    """Whether the optimizer chooses this hyperparameter (more than one value to choose)."""
    return len(spec) > 1 if isinstance(spec, list) else spec[0] != spec[1]


def values(spec) -> int:
    """How many values a hyperparameter's space holds."""
    if isinstance(spec, list):
        return len(spec)
    return int(round((spec[1] - spec[0]) / spec[2])) + 1


def project(settings: dict, space: dict, rules: dict) -> dict:
    """``settings`` moved into the space: a fixed hyperparameter takes its one value, a number
    is moved onto its searched grid, a choice the space does not allow becomes its first
    allowed one. The tuner's first trial."""
    out = dict(settings)
    for name in table(rules):
        spec, v = space[name], settings[name]
        if isinstance(spec, list):
            out[name] = v if v in spec else spec[0]
        else:
            lo, hi, st = spec
            g = min(max(lo + round((float(v) - lo) / st) * st, lo), hi)
            out[name] = int(round(g)) if isinstance(lo, int) else round(g, 9)
    return out


def describe_space(space: dict, rules: dict) -> pd.DataFrame:
    """The space in words, one row per hyperparameter: what the optimizer may do with it, how
    many values that is, what a change rebuilds and which switch a lever acts under."""
    rows = []
    for name, r in table(rules).items():
        spec = space[name]
        if name.startswith("on: "):
            what = ("optimizer decides (on or off)" if len(spec) > 1
                    else f"always {'on' if spec[0] else 'off'}")
        elif isinstance(spec, list):
            words = [str(r["choices"][v]) for v in spec]
            what = (f"optimizer decides among: {' / '.join(words)}" if len(spec) > 1
                    else f"fixed: {words[0]}")
        else:
            what = (f"optimizer decides: {spec[0]} to {spec[1]}, step {spec[2]}"
                    if spec[0] != spec[1] else f"fixed: {spec[0]}")
        rows.append({"hyperparameter": name, "group": r["group"], "what it is": r["label"],
                     "the optimizer": what, "values": values(spec),
                     "a change rebuilds": STAGES[r["stage"]],
                     "acts only while": needs_words(r)})
    return pd.DataFrame(rows)


def switches(settings: dict, rules: dict) -> dict:
    """The rule switches of a settings dict: rule name -> on / off."""
    return {name: bool(settings[switch_key(name)]) for name in rules}


def widget(name: str, row: dict, value):
    """A marimo control for one hyperparameter, starting at ``value``: a switch for yes / no, a
    dropdown for other choices, a slider for numbers."""
    import marimo as mo

    if "choices" in row and set(row["choices"]) == {True, False}:
        return mo.ui.switch(value=bool(value), label=row["label"])
    if "choices" in row:
        options = {label: v for v, label in row["choices"].items()}
        return mo.ui.dropdown(options, value=row["choices"].get(value, next(iter(options))),
                              label=row["label"])
    return mo.ui.slider(row["low"], row["high"], step=row["step"], value=value,
                        show_value=True, label=row["label"])


def save(stocks: dict, *, meta: dict, name: str, folder: Path = SETTINGS) -> Path:
    """Write tuned settings: one complete settings dict per stock, and what made them
    (``meta``: the procedure, its scores, the data). Returns the file."""
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    path = folder / f"{stamp}-{name}.json"
    path.write_text(json.dumps({"made": stamp, "meta": meta, "stocks": stocks}, indent=1,
                               default=str))
    return path


def load(path: Path) -> dict:
    """A settings file: ``made``, ``meta`` and ``stocks`` (stock -> settings)."""
    return json.loads(Path(path).read_text())


def files(folder: Path = SETTINGS) -> list[Path]:
    """The exported settings files, newest first."""
    return sorted(folder.glob("*.json"), reverse=True) if folder.exists() else []


def starting(rules: dict, path: Path | None = None, stock: str | None = None) -> dict:
    """The starting values: the built-in ones, overwritten by one stock's settings from an
    exported file (a value the table no longer has, or one out of its range, is left out)."""
    out = defaults(rules)
    if path is None:
        return out
    rows = table(rules)
    for name, value in load(path)["stocks"][stock].items():
        r = rows.get(name)
        if r is None:
            continue
        if "choices" in r:
            if value in r["choices"]:
                out[name] = value
        elif r["low"] <= value <= r["high"]:
            out[name] = value
    return out
