"""Prospective validator for the frozen canonical GEX candidates."""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PROSPECTIVE_START = pd.Timestamp("2026-07-17")

spec = importlib.util.spec_from_file_location("canonical_playbook", HERE / "canonical_playbook.py")
C = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(C)


def live_inputs() -> pd.DataFrame:
    path = ROOT / "output" / "gex_forward_eod_levels.parquet"
    if not path.exists():
        return pd.DataFrame()
    levels = pd.read_parquet(path).copy()
    levels["as_of"] = pd.to_datetime(levels["as_of"])
    levels = levels[levels["ticker"].isin(["SPY", "QQQ"])].sort_values(["ticker", "as_of"])
    levels = levels.drop_duplicates(["ticker", "as_of"], keep="last")
    return levels


def complete_days(sym: str) -> dict[pd.Timestamp, pd.DataFrame]:
    mins = C.se_panel.rth_minutes(sym)
    return {pd.Timestamp(d): g.sort_index() for d, g in mins.groupby("date") if len(g) >= 350}


def next_session(days: dict[pd.Timestamp, pd.DataFrame], D: pd.Timestamp) -> pd.Timestamp | None:
    future = [d for d in days if d > D]
    return min(future) if future else None


def add_rule(out: list[dict], *, sym: str, D: pd.Timestamp, N: pd.Timestamp,
             day: pd.DataFrame, setup: str, ev, side: int, level: float, em: float,
             regime: int) -> None:
    if not ev:
        return
    ts, _ = ev
    fill = C.fill_after_bar(day, ts)
    if not fill:
        return
    et, entry = fill
    stop = level - side * C.STOP_EM * em
    risk = side * (entry - stop)
    if risk <= 0:
        return
    target = C.target_from(entry, side, risk, "2R", {})
    row = pd.Series({"regime_idx": regime, "regime_own": regime})
    C.add_trade(out, sym=sym, D=D, N=N, row=row, setup=setup, variant="2R",
                primary=True, day=day, entry_ts=et, entry=entry, side=side,
                stop=stop, target=target, extra={"source": "live_gamma_ledger"})


def collect_forward() -> pd.DataFrame:
    levels = live_inputs()
    if levels.empty:
        return pd.DataFrame()
    out: list[dict] = []
    for sym in ("SPY", "QQQ"):
        days = complete_days(sym)
        lv = levels[levels["ticker"] == sym]
        for _, r in lv.iterrows():
            D = pd.Timestamp(r["as_of"])
            prior = days.get(D)
            N = next_session(days, D)
            if prior is None or N is None:
                continue
            day = days[N]
            prior_close = float(prior["c"].iloc[-1])
            ledger_spot = float(r["spot"])
            if not np.isfinite(ledger_spot) or ledger_spot <= 0:
                continue
            scale = prior_close / ledger_spot
            cw, pw = float(r["call_wall"]) * scale, float(r["put_wall"]) * scale
            regime = -1 if "SHORT" in str(r["regime"]).upper() else 1
            ivpct = float(r["atm_iv_30d"])
            em = prior_close * (ivpct / 100.0) / np.sqrt(252)
            if not all(np.isfinite(x) and x > 0 for x in (cw, pw, em)):
                continue
            b5 = C.bars5(day)
            add_rule(out, sym=sym, D=D, N=N, day=day, setup="P3_forced_buying",
                     ev=C.first_accept(b5, cw, em, +1), side=+1, level=cw, em=em, regime=regime)
            add_rule(out, sym=sym, D=D, N=N, day=day, setup="P4_forced_selling",
                     ev=C.first_accept(b5, pw, em, -1), side=-1, level=pw, em=em, regime=regime)
            if sym == "QQQ":
                pdl = float(prior["l"].min())
                add_rule(out, sym=sym, D=D, N=N, day=day, setup="P6_PDL_acceptance_short",
                         ev=C.first_accept(b5, pdl, em, -1), side=-1, level=pdl, em=em, regime=regime)
    if not out:
        return pd.DataFrame()
    t = pd.DataFrame(out).sort_values(["N", "sym", "setup"]).reset_index(drop=True)
    if not (pd.to_datetime(t["D"]) < pd.to_datetime(t["N"])).all():
        raise AssertionError("forward PIT violation")
    return t


def top5_share(net: np.ndarray) -> float | None:
    total = float(net.sum())
    return float(np.sort(net)[-5:].sum() / total) if len(net) >= 5 and total > 0 else None


def gate(g: pd.DataFrame) -> dict:
    if g.empty:
        return {"status": "COLLECTING", "n": 0, "reason": "no prospective event yet"}
    m2, m5 = C.metrics(g, 2.0), C.metrics(g, 5.0)
    net5 = g["gross_bps"].to_numpy(float) - 5.0
    half = len(g) // 2
    halves = [float(net5[:half].mean()), float(net5[half:].mean())] if half else [None, None]
    # 99% circular block bootstrap lower bound.
    rng = np.random.default_rng(17072026); x = net5; vals = []
    if len(x) >= 10:
        block = 5; nb = int(np.ceil(len(x) / block)); off = np.arange(block)
        for _ in range(5000):
            starts = rng.integers(0, len(x), nb)
            ix = ((starts[:, None] + off) % len(x)).ravel()[:len(x)]
            vals.append(float(x[ix].mean()))
    lower99 = float(np.quantile(vals, .01)) if vals else None
    upper95 = float(np.quantile(vals, .95)) if vals else None
    concentration = top5_share(net5)
    pass_core = (len(g) >= 200 and m5["mean_net_bps"] > 0 and lower99 is not None and lower99 > 0
                 and (m5["profit_factor"] or 0) >= 1.20 and all(v is not None and v > 0 for v in halves)
                 and concentration is not None and concentration <= .35)
    hard_fail = ((len(g) >= 100 and upper95 is not None and upper95 <= 0)
                 or (len(g) >= 200 and (m5["mean_net_bps"] <= 0 or (m5["profit_factor"] or 0) < 1.0)))
    return {"status": "PASS" if pass_core else ("FAIL" if hard_fail else "COLLECTING"),
            "n": len(g), "metrics_2bps": m2, "metrics_5bps": m5,
            "bootstrap_lower99_mean_bps_5bps": lower99,
            "bootstrap_upper95_mean_bps_5bps": upper95,
            "half_means_bps_5bps": halves, "top5_share_5bps": concentration}


def required_n(mean: float, sd: float, alpha: float = .01, power: float = .90) -> int:
    if mean <= 0 or sd <= 0:
        return 0
    z_alpha = 2.326347874  # one-sided 99%
    z_power = 1.281551566  # 90% power
    return int(np.ceil(((z_alpha + z_power) * sd / mean) ** 2))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t = collect_forward()
    if not t.empty:
        t.to_parquet(OUT / "forward_validation_trades.parquet", index=False)
    result = {"run_utc": datetime.now(timezone.utc).isoformat(),
              "prospective_start": str(PROSPECTIVE_START.date()), "shadow": {}, "prospective": {},
              "power_reference": {}}
    candidates = [("SPY", "P4_forced_selling"), ("QQQ", "P4_forced_selling"),
                  ("QQQ", "P3_forced_buying"), ("QQQ", "P6_PDL_acceptance_short")]
    hist = pd.read_parquet(OUT / "canonical_trades.parquet")
    for sym, setup in candidates:
        key = f"{sym}|{setup}|2R"
        sg = t[(t["sym"] == sym) & (t["setup"] == setup)] if not t.empty else pd.DataFrame()
        result["shadow"][key] = C.metrics(sg, 2.0) if len(sg) else {"n": 0}
        pg = sg[pd.to_datetime(sg["D"]) >= PROSPECTIVE_START] if len(sg) else pd.DataFrame()
        result["prospective"][key] = gate(pg)
        hg = hist[(hist["sym"] == sym) & (hist["setup"] == setup) & (hist["variant"] == "2R")]
        gross = hg["gross_bps"].to_numpy(float) - 2.0
        result["power_reference"][key] = {
            "historical_mean_2bps": float(gross.mean()), "historical_sd_bps": float(gross.std(ddof=1)),
            "n_for_99pct_one_sided_90pct_power_if_effect_stable": required_n(float(gross.mean()), float(gross.std(ddof=1)))
        }
    # Forced-selling family cannot pass unless both legs independently have at
    # least a positive 95% bootstrap lower bound. Individual gates remain visible.
    (OUT / "forward_validation.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print("CANONICAL GEX FORWARD VALIDATION")
    for k in result["prospective"]:
        sh = result["shadow"][k]; pr = result["prospective"][k]; pw = result["power_reference"][k]
        print(f"{k:45s} shadow n={sh.get('n',0):2d} mean={sh.get('mean_net_bps',float('nan')):+.2f}bps "
              f"| prospective {pr['status']} n={pr['n']} | power-n≈{pw['n_for_99pct_one_sided_90pct_power_if_effect_stable']}")
    print(f"results: {OUT / 'forward_validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
