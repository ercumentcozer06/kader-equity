"""Canonical six-family GEX playbook event backtest.

Rules are locked in CANONICAL_SPEC.md. This script uses prior-EOD option levels
and next-session one-minute paths. Signal bars are five-minute OHLC; fills occur
at the following minute open, so a confirmation close is never used as its own fill.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, time, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SE_DIR = ROOT / "backtest" / "scenario_engine"
sys.path.insert(0, str(SE_DIR))
import se_panel  # noqa: E402

SYMS = ("SPY", "QQQ")
COSTS = (1.0, 2.0, 5.0)
ACCEPT_EM = 0.02
STOP_EM = 0.10
CUTOFF = time(14, 30)
HOLDOUT = pd.Timestamp("2026-02-02")
MIN_N = 30


def bars5(day: pd.DataFrame) -> pd.DataFrame:
    origin = day.index[0].normalize() + pd.Timedelta(hours=9, minutes=30)
    return day.resample("5min", origin=origin, label="left", closed="left").agg(
        o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum")
    ).dropna(subset=["c"])


def fill_after_bar(day: pd.DataFrame, bar_start: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    ts = bar_start + pd.Timedelta(minutes=5)
    future = day[day.index >= ts]
    if future.empty or ts.time() > CUTOFF:
        return None
    return future.index[0], float(future["o"].iloc[0])


def simulate(day: pd.DataFrame, entry_ts: pd.Timestamp, entry: float, side: int,
             stop: float, target: float | None) -> dict:
    path = day[day.index >= entry_ts]
    if path.empty:
        raise ValueError("empty execution path")
    risk = side * (entry - stop)
    if not np.isfinite(risk) or risk <= 0:
        raise ValueError(f"invalid risk entry={entry} stop={stop} side={side}")
    exit_px, reason, exit_ts = float(path["c"].iloc[-1]), "eod", path.index[-1]
    for ts, b in path.iterrows():
        if side > 0:
            hit_s = float(b["l"]) <= stop
            hit_t = target is not None and float(b["h"]) >= target
        else:
            hit_s = float(b["h"]) >= stop
            hit_t = target is not None and float(b["l"]) <= target
        if hit_s:  # conservative same-bar ordering
            exit_px, reason, exit_ts = stop, "stop", ts
            break
        if hit_t:
            exit_px, reason, exit_ts = float(target), "target", ts
            break
    used = path[path.index <= exit_ts]
    if side > 0:
        mfe = (float(used["h"].max()) - entry) / risk
        mae = (entry - float(used["l"].min())) / risk
    else:
        mfe = (entry - float(used["l"].min())) / risk
        mae = (float(used["h"].max()) - entry) / risk
    return {
        "exit_ts": exit_ts, "exit": exit_px, "reason": reason,
        "gross_bps": side * (exit_px / entry - 1.0) * 1e4,
        "r_multiple_gross": side * (exit_px - entry) / risk,
        "mfe_r": mfe, "mae_r": mae, "risk_pct": risk / entry,
    }


def target_from(entry: float, side: int, risk: float, policy: str, levels: dict) -> float | None:
    if policy.endswith("R"):
        return entry + side * float(policy[:-1]) * risk
    if policy == "EOD":
        return None
    level = levels.get(policy.lower())
    if level is None or not np.isfinite(level) or side * (level - entry) <= 0:
        raise KeyError(policy)
    return float(level)


def add_trade(out: list[dict], *, sym: str, D: pd.Timestamp, N: pd.Timestamp, row: pd.Series,
              setup: str, variant: str, primary: bool, day: pd.DataFrame,
              entry_ts: pd.Timestamp, entry: float, side: int, stop: float,
              target: float | None, extra: dict | None = None) -> None:
    try:
        sim = simulate(day, entry_ts, entry, side, stop, target)
    except ValueError:
        return
    rec = {
        "sym": sym, "D": D, "N": N, "setup": setup, "variant": variant,
        "primary": primary, "entry_ts": entry_ts, "entry_hour": entry_ts.hour,
        "entry": entry, "side": side, "stop": stop, "target": target,
        "regime_idx": int(row["regime_idx"]), "regime_own": int(row["regime_own"]),
        **sim,
    }
    if extra:
        rec.update(extra)
    out.append(rec)


def first_accept(b5: pd.DataFrame, level: float, em: float, side: int):
    threshold = level + side * ACCEPT_EM * em
    prev = float(b5["o"].iloc[0])
    for ts, b in b5.iterrows():
        if (ts + pd.Timedelta(minutes=5)).time() > CUTOFF:
            break
        accepted = float(b["c"]) > threshold if side > 0 else float(b["c"]) < threshold
        was_not = prev <= threshold if side > 0 else prev >= threshold
        if accepted and was_not:
            return ts, b
        prev = float(b["c"])
    return None


def first_reject(b5: pd.DataFrame, level: float, em: float, side_after: int):
    # side_after=-1 at upper level; +1 at lower level.
    for ts, b in b5.iterrows():
        if (ts + pd.Timedelta(minutes=5)).time() > CUTOFF:
            break
        if side_after < 0:
            ok = float(b["h"]) >= level and float(b["c"]) <= level - ACCEPT_EM * em
        else:
            ok = float(b["l"]) <= level and float(b["c"]) >= level + ACCEPT_EM * em
        if ok:
            return ts, b
    return None


def wall_family(out: list[dict], sym: str, D: pd.Timestamp, N: pd.Timestamp,
                row: pd.Series, day: pd.DataFrame, pdh: float, pdl: float) -> None:
    b5 = bars5(day)
    em, cw, pw = float(row["em1"]), float(row["call_wall"]), float(row["put_wall"])
    levels = {"ghost": row["ghost"], "flip": row["flip"], "max_pain": row["max_pain"],
              "pdh": pdh, "pdl": pdl, "vu": row["vu"], "vd": row["vd"]}

    # P1 call-wall fade.
    ev = first_reject(b5, cw, em, -1)
    if ev:
        ts, _ = ev
        fill = fill_after_bar(day, ts)
        if fill:
            et, entry = fill; stop = cw + STOP_EM * em; risk = entry - stop
            # short risk distance is stop-entry
            risk = stop - entry
            if risk > 0:
                for pol in ("1R", "2R", "3R", "EOD", "ghost", "flip", "max_pain", "pdl"):
                    try: target = target_from(entry, -1, risk, pol, levels)
                    except KeyError: continue
                    add_trade(out, sym=sym, D=D, N=N, row=row, setup="P1_call_wall_fade",
                              variant=pol, primary=pol == "2R", day=day, entry_ts=et,
                              entry=entry, side=-1, stop=stop, target=target)

    # P3 forced buying.
    ev = first_accept(b5, cw, em, +1)
    if ev:
        ts, _ = ev; fill = fill_after_bar(day, ts)
        if fill:
            et, entry = fill; stop = cw - STOP_EM * em; risk = entry - stop
            if risk > 0:
                for pol in ("1R", "2R", "3R", "EOD", "pdh", "vu"):
                    try: target = target_from(entry, +1, risk, pol, levels)
                    except KeyError: continue
                    add_trade(out, sym=sym, D=D, N=N, row=row, setup="P3_forced_buying",
                              variant=pol.upper(), primary=pol == "2R", day=day, entry_ts=et,
                              entry=entry, side=+1, stop=stop, target=target)

    # P4 forced selling.
    ev = first_accept(b5, pw, em, -1)
    if ev:
        ts, _ = ev; fill = fill_after_bar(day, ts)
        if fill:
            et, entry = fill; stop = pw + STOP_EM * em; risk = stop - entry
            if risk > 0:
                for pol in ("1R", "2R", "3R", "EOD", "pdl", "vd"):
                    try: target = target_from(entry, -1, risk, pol, levels)
                    except KeyError: continue
                    add_trade(out, sym=sym, D=D, N=N, row=row, setup="P4_forced_selling",
                              variant=pol.upper(), primary=pol == "2R", day=day, entry_ts=et,
                              entry=entry, side=-1, stop=stop, target=target)

    # P2 cascade: negative regime, accepted PW break, next 5m close below break low.
    base = first_accept(b5, pw, em, -1)
    if base:
        bts, bb = base; loc = b5.index.get_loc(bts)
        if loc + 1 < len(b5):
            nts, nb = b5.iloc[loc + 1].name, b5.iloc[loc + 1]
            cascade = float(nb["c"]) < float(bb["l"])
            if cascade and (nts + pd.Timedelta(minutes=5)).time() <= CUTOFF:
                fill = fill_after_bar(day, nts)
                if fill:
                    et, entry = fill; stop = pw + STOP_EM * em; risk = stop - entry
                    if risk > 0:
                        for regsrc in ("idx", "own"):
                            if int(row[f"regime_{regsrc}"]) >= 0:
                                continue
                            for pol in ("2R", "3R", "EOD", "pdl"):
                                try: target = target_from(entry, -1, risk, pol, levels)
                                except KeyError: continue
                                add_trade(out, sym=sym, D=D, N=N, row=row,
                                          setup="P2_negative_gex_cascade",
                                          variant=f"{regsrc}_{pol.upper()}",
                                          primary=(regsrc == "idx" and pol == "3R"), day=day,
                                          entry_ts=et, entry=entry, side=-1, stop=stop, target=target,
                                          extra={"regime_source": regsrc})


def pin_family(out: list[dict], sym: str, D: pd.Timestamp, N: pd.Timestamp,
               row: pd.Series, day: pd.DataFrame) -> None:
    em, cw, pw = float(row["em1"]), float(row["call_wall"]), float(row["put_wall"])
    candidates = {"flip": row["flip"], "ghost": row["ghost"], "max_pain": row["max_pain"]}
    for decision, stop_em in ((time(11, 0), .50), (time(12, 0), .75)):
        before = day[day.index.time < decision]
        at = day[day.index.time >= decision]
        if before.empty or at.empty or float(before["h"].max()) >= cw or float(before["l"].min()) <= pw:
            continue
        et, entry = at.index[0], float(at["o"].iloc[0])
        valid = {k: float(v) for k, v in candidates.items()
                 if pd.notna(v) and pw < float(v) < cw and abs(float(v) - entry) >= .20 * em}
        if not valid:
            continue
        selectors = {**valid, "nearest": min(valid.values(), key=lambda x: abs(x - entry))}
        for name, target in selectors.items():
            side = 1 if target > entry else -1
            stop = entry - side * stop_em * em
            variant = f"{name}_{decision.strftime('%H%M')}_stop{stop_em:.2f}"
            add_trade(out, sym=sym, D=D, N=N, row=row, setup="P5_unresolved_pinning",
                      variant=variant, primary=(name == "nearest" and decision == time(11, 0)),
                      day=day, entry_ts=et, entry=entry, side=side, stop=stop, target=target,
                      extra={"pin_target": name, "decision_time": decision.strftime("%H:%M")})


def pd_family(out: list[dict], sym: str, D: pd.Timestamp, N: pd.Timestamp,
              row: pd.Series, day: pd.DataFrame, pdh: float, pdl: float) -> None:
    b5, em = bars5(day), float(row["em1"])
    specs = [
        ("PDH_rejection_short", first_reject(b5, pdh, em, -1), -1, pdh + STOP_EM * em),
        ("PDL_rejection_long", first_reject(b5, pdl, em, +1), +1, pdl - STOP_EM * em),
        ("PDH_acceptance_long", first_accept(b5, pdh, em, +1), +1, pdh - STOP_EM * em),
        ("PDL_acceptance_short", first_accept(b5, pdl, em, -1), -1, pdl + STOP_EM * em),
    ]
    for name, ev, side, stop in specs:
        if not ev:
            continue
        ts, _ = ev; fill = fill_after_bar(day, ts)
        if not fill:
            continue
        et, entry = fill; risk = side * (entry - stop)
        if risk <= 0:
            continue
        for pol in ("1R", "2R", "3R", "EOD"):
            target = target_from(entry, side, risk, pol, {})
            add_trade(out, sym=sym, D=D, N=N, row=row, setup=f"P6_{name}",
                      variant=pol, primary=pol == "2R", day=day, entry_ts=et,
                      entry=entry, side=side, stop=stop, target=target)


def collect() -> pd.DataFrame:
    records: list[dict] = []
    for sym in SYMS:
        panel = pd.read_parquet(SE_DIR / f"panel_{sym.lower()}.parquet")
        mins = se_panel.rth_minutes(sym)
        days = {pd.Timestamp(d): g.sort_index() for d, g in mins.groupby("date")}
        for D, row in panel.iterrows():
            D, N = pd.Timestamp(D), pd.Timestamp(row["N"])
            day, prev = days.get(N), days.get(D)
            required = ("em1", "call_wall", "put_wall")
            if day is None or prev is None or any(pd.isna(row[k]) for k in required):
                continue
            if len(day) < 350 or len(prev) < 350:
                continue
            pdh, pdl = float(prev["h"].max()), float(prev["l"].min())
            wall_family(records, sym, D, N, row, day, pdh, pdl)
            pin_family(records, sym, D, N, row, day)
            pd_family(records, sym, D, N, row, day, pdh, pdl)
    return pd.DataFrame(records).sort_values(["N", "sym", "setup", "variant"]).reset_index(drop=True)


def block_boot(x: np.ndarray, n_boot: int = 3000, block: int = 5) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    if len(x) < 10:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(16072026)
    m = math.ceil(len(x) / block); offs = np.arange(block); vals = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, len(x), m)
        ix = ((starts[:, None] + offs) % len(x)).ravel()[:len(x)]
        vals[i] = x[ix].mean()
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975)), float((vals > 0).mean())


def metrics(g: pd.DataFrame, cost: float) -> dict:
    net = g["gross_bps"].to_numpy(float) - cost
    n = len(net); sd = float(np.std(net, ddof=1)) if n > 1 else np.nan
    t = float(np.mean(net) / (sd / np.sqrt(n))) if n > 1 and sd > 0 else np.nan
    p1 = float(stats.t.sf(t, n - 1)) if np.isfinite(t) else np.nan
    lo, hi, prob = block_boot(net)
    gains, losses = net[net > 0].sum(), -net[net < 0].sum()
    daily = pd.Series(net / 1e4, index=pd.to_datetime(g["N"])).groupby(level=0).sum().sort_index()
    eq = (1 + daily).cumprod()
    total = net.sum()
    top3 = float(np.sort(net)[-3:].sum() / total) if total > 0 and n >= 3 else np.nan
    return {
        "n": n, "mean_net_bps": float(np.mean(net)), "median_net_bps": float(np.median(net)),
        "t": t, "p_one_sided": p1, "hit": float((net > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else None,
        "mean_r_gross": float(g["r_multiple_gross"].mean()),
        "mean_mfe_r": float(g["mfe_r"].mean()), "mean_mae_r": float(g["mae_r"].mean()),
        "block5_ci95_bps": [lo, hi], "block5_probability_mean_gt0": prob,
        "cum_return": float(eq.iloc[-1] - 1) if len(eq) else None,
        "max_drawdown": float((eq / eq.cummax() - 1).min()) if len(eq) else None,
        "top3_pnl_share": top3, "underpowered_n_lt_30": n < MIN_N,
        "exit_reasons": g["reason"].value_counts().to_dict(),
    }


def bh(pairs: list[tuple[str, float]]) -> dict[str, float]:
    x = sorted([(k, p) for k, p in pairs if np.isfinite(p)], key=lambda z: z[1]); m = len(x)
    out, run = {}, 1.0
    for rank in range(m, 0, -1):
        k, p = x[rank - 1]; run = min(run, p * m / rank); out[k] = run
    return out


def summarize(trades: pd.DataFrame) -> dict:
    result: dict = {"cells": {}, "primary_breakdowns": {}}
    pvals = []
    for (sym, setup, variant), g in trades.groupby(["sym", "setup", "variant"]):
        key = f"{sym}|{setup}|{variant}"; result["cells"][key] = {}
        for period, mask in {
            "full": np.ones(len(g), bool), "train": pd.to_datetime(g["D"]) < HOLDOUT,
            "holdout": pd.to_datetime(g["D"]) >= HOLDOUT,
        }.items():
            sub = g.loc[mask]
            result["cells"][key][period] = {str(c): metrics(sub, c) for c in COSTS} if len(sub) else {}
        if bool(g["primary"].iloc[0]):
            h = g[pd.to_datetime(g["D"]) >= HOLDOUT]
            if len(h):
                pvals.append((key, metrics(h, 2.0)["p_one_sided"]))
            br = {}
            for col in ("regime_idx", "regime_own", "entry_hour"):
                br[col] = {str(v): metrics(s, 2.0) for v, s in g.groupby(col)}
            result["primary_breakdowns"][key] = br
    result["primary_holdout_bh_fdr"] = bh(pvals)
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    trades = collect()
    if trades.empty:
        raise RuntimeError("no canonical trades generated")
    if not (pd.to_datetime(trades["D"]) < pd.to_datetime(trades["N"])).all():
        raise AssertionError("PIT violation: level date must precede trade date")
    if not (((trades["side"] > 0) & (trades["stop"] < trades["entry"])) |
            ((trades["side"] < 0) & (trades["stop"] > trades["entry"]))).all():
        raise AssertionError("invalid stop placement")
    targeted = trades["target"].notna()
    if not (trades.loc[targeted, "side"] *
            (trades.loc[targeted, "target"] - trades.loc[targeted, "entry"]) > 0).all():
        raise AssertionError("target is not ahead of entry")
    if not (pd.to_datetime(trades["exit_ts"]) >= pd.to_datetime(trades["entry_ts"])).all():
        raise AssertionError("exit precedes entry")
    trades.to_parquet(OUT / "canonical_trades.parquet", index=False)
    result = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "spec": str(HERE / "CANONICAL_SPEC.md"),
        "n_trade_records_all_variants": len(trades),
        "date_range": [str(pd.Timestamp(trades["N"].min()).date()), str(pd.Timestamp(trades["N"].max()).date())],
        **summarize(trades),
    }
    (OUT / "canonical_results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"CANONICAL GEX PLAYBOOK — records={len(trades)} {result['date_range'][0]}..{result['date_range'][1]}")
    for key, periods in result["cells"].items():
        sym, setup, variant = key.split("|")
        sub = trades[(trades.sym == sym) & (trades.setup == setup) & (trades.variant == variant)]
        if not bool(sub.primary.iloc[0]):
            continue
        f = periods["full"]["2.0"]; h = periods["holdout"].get("2.0")
        hs = "none" if not h else f"n={h['n']} mean={h['mean_net_bps']:+.1f} t={h['t']:+.2f}"
        print(f"{key:65s} FULL n={f['n']:3d} mean={f['mean_net_bps']:+6.1f} t={f['t']:+5.2f} "
              f"PF={f['profit_factor'] if f['profit_factor'] is not None else float('nan'):.2f} | HOLDOUT {hs}")
    print(f"results: {OUT / 'canonical_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
