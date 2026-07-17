"""Literature-aligned intraday gamma-feedback test.

Precommitted information timing:
  * source GEX is the last SqueezeMetrics EOD observation strictly before session N;
  * early return is known at 10:30 ET;
  * a strategy may then trade 10:30 ET -> 16:00 ET;
  * no same-day option-chain value is used.

Primary statistical claim: the coefficient on early_return * lagged_gex_z is
negative. Higher GEX should turn continuation into reversal; lower GEX should
amplify continuation. The volatility claim expects a negative coefficient on
lagged_gex_z after controlling for early-session realized volatility.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
SYMS = ("SPY", "QQQ")
COSTS_BPS = (0.0, 1.0, 2.0, 5.0)
EXTREME_Z = 1.0
HAC_LAGS = 5


def rth_minutes(sym: str) -> pd.DataFrame:
    raw = pd.read_parquet(ROOT / "data" / "historical_bars" / f"alpaca_{sym.lower()}_1m.parquet")
    idx = pd.to_datetime(raw.index.get_level_values("timestamp"), utc=True).tz_convert("America/New_York")
    x = pd.DataFrame({
        "o": raw["open"].to_numpy(float), "h": raw["high"].to_numpy(float),
        "l": raw["low"].to_numpy(float), "c": raw["close"].to_numpy(float),
        "v": raw["volume"].to_numpy(float),
    }, index=idx).sort_index()
    tm = x.index.time
    x = x[(tm >= pd.Timestamp("09:30").time()) & (tm < pd.Timestamp("16:00").time())]
    x = x[~x.index.duplicated(keep="last")]
    x["date"] = pd.to_datetime(x.index.date)
    return x


def session_panel(sym: str) -> pd.DataFrame:
    mins = rth_minutes(sym)
    rows: list[dict] = []
    for day, g in mins.groupby("date", sort=True):
        # Require enough observations in every fixed bucket; half-days are excluded.
        early = g[(g.index.time >= pd.Timestamp("09:30").time()) &
                  (g.index.time < pd.Timestamp("10:30").time())]
        middle = g[(g.index.time >= pd.Timestamp("10:30").time()) &
                   (g.index.time < pd.Timestamp("15:00").time())]
        late = g[(g.index.time >= pd.Timestamp("15:00").time()) &
                 (g.index.time < pd.Timestamp("16:00").time())]
        if len(early) < 50 or len(middle) < 240 or len(late) < 50:
            continue
        p_open = float(early["o"].iloc[0])
        p_1030 = float(early["c"].iloc[-1])
        p_1500 = float(middle["c"].iloc[-1])
        p_close = float(late["c"].iloc[-1])
        eret = p_1030 / p_open - 1.0
        mret = p_1500 / p_1030 - 1.0
        lret = p_close / p_1500 - 1.0
        rest = p_close / p_1030 - 1.0
        e_lr = np.diff(np.log(early["c"].to_numpy(float)))
        r_lr = np.diff(np.log(pd.concat([early.tail(1), middle, late])["c"].to_numpy(float)))
        rows.append({
            "date": day, "early_ret": eret, "middle_ret": mret,
            "last_hour_ret": lret, "rest_ret": rest,
            "early_rv": float(np.sqrt(np.sum(e_lr * e_lr))),
            "rest_rv": float(np.sqrt(np.sum(r_lr * r_lr))),
            "early_dollar_volume": float((early["c"] * early["v"]).sum()),
        })
    p = pd.DataFrame(rows).set_index("date").sort_index()

    sq = pd.read_parquet(ROOT / "data" / "cache" / "squeeze_dix_gex.parquet").sort_index()
    sq.index = pd.to_datetime(sq.index).tz_localize(None)
    gm = sq["gex"].rolling(252, min_periods=60).mean()
    gs = sq["gex"].rolling(252, min_periods=60).std()
    source = pd.DataFrame({"source_date": pd.DatetimeIndex(sq.index).astype("datetime64[ns]"), "gex": sq["gex"].values,
                           "gex_z": ((sq["gex"] - gm) / gs).values}).sort_values("source_date")
    left = p.reset_index().rename(columns={"date": "session_date"}).sort_values("session_date")
    left["session_date"] = pd.to_datetime(left["session_date"]).astype("datetime64[ns]")
    merged = pd.merge_asof(left, source, left_on="session_date", right_on="source_date",
                           direction="backward", allow_exact_matches=False)
    merged = merged.set_index("session_date")
    if not (merged["source_date"] < merged.index.to_series()).all():
        raise AssertionError("PIT violation: GEX source is not strictly before the traded session")
    merged["gex_sign"] = np.sign(merged["gex"])
    return merged.dropna(subset=["gex_z", "early_ret", "rest_ret", "rest_rv"])


def split_name(index: pd.DatetimeIndex) -> pd.Series:
    # Dates are fixed before results are calculated. The final period is the OOS test.
    v = np.where(index <= pd.Timestamp("2023-12-31"), "discovery",
        np.where(index <= pd.Timestamp("2024-12-31"), "validation", "test"))
    return pd.Series(v, index=index)


def hac_regression(df: pd.DataFrame, future: str) -> dict:
    d = df[[future, "early_ret", "gex_z", "early_rv"]].dropna().copy()
    d["interaction"] = d["early_ret"] * d["gex_z"]
    x = sm.add_constant(d[["early_ret", "gex_z", "interaction", "early_rv"]])
    fit = sm.OLS(d[future], x).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return {
        "n": int(fit.nobs), "r2": float(fit.rsquared),
        "interaction_beta": float(fit.params["interaction"]),
        "interaction_t_hac": float(fit.tvalues["interaction"]),
        "interaction_p_hac": float(fit.pvalues["interaction"]),
        "early_beta": float(fit.params["early_ret"]),
        "gex_z_beta": float(fit.params["gex_z"]),
    }


def volatility_regression(df: pd.DataFrame) -> dict:
    d = df[["rest_rv", "early_rv", "gex_z"]].replace(0, np.nan).dropna().copy()
    d["log_rest_rv"] = np.log(d["rest_rv"])
    d["log_early_rv"] = np.log(d["early_rv"])
    x = sm.add_constant(d[["gex_z", "log_early_rv"]])
    fit = sm.OLS(d["log_rest_rv"], x).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return {"n": int(fit.nobs), "gex_z_beta": float(fit.params["gex_z"]),
            "gex_z_t_hac": float(fit.tvalues["gex_z"]),
            "gex_z_p_hac": float(fit.pvalues["gex_z"]), "r2": float(fit.rsquared)}


def slope_cells(df: pd.DataFrame, future: str) -> dict:
    out = {}
    masks = {"negative_extreme": df["gex_z"] <= -EXTREME_Z,
             "neutral": df["gex_z"].abs() < EXTREME_Z,
             "positive_extreme": df["gex_z"] >= EXTREME_Z}
    for name, mask in masks.items():
        d = df.loc[mask, ["early_ret", future]].dropna()
        if len(d) < 20:
            out[name] = {"n": len(d)}
            continue
        fit = sm.OLS(d[future], sm.add_constant(d["early_ret"])).fit(
            cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
        out[name] = {"n": len(d), "slope": float(fit.params["early_ret"]),
                     "t_hac": float(fit.tvalues["early_ret"]),
                     "p_hac": float(fit.pvalues["early_ret"])}
    return out


def block_bootstrap_mean(x: np.ndarray, block: int = 5, n_boot: int = 3000) -> tuple[float, float, float]:
    """Circular fixed-block bootstrap of the daily mean; deterministic seed."""
    x = np.asarray(x, float)
    if len(x) < 20:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(71237)
    means = np.empty(n_boot)
    need = int(np.ceil(len(x) / block))
    offsets = np.arange(block)
    for i in range(n_boot):
        starts = rng.integers(0, len(x), need)
        idx = ((starts[:, None] + offsets) % len(x)).ravel()[:len(x)]
        means[i] = x[idx].mean()
    return float(np.quantile(means, .025)), float(np.quantile(means, .975)), float((means > 0).mean())


def perf(gross: pd.Series, active: pd.Series, cost_bps: float) -> dict:
    net = gross - active.astype(float) * cost_bps / 1e4
    net = net.dropna()
    if len(net) == 0:
        return {"n": 0}
    sd = float(net.std(ddof=1))
    eq = (1.0 + net).cumprod()
    act = active.reindex(net.index)
    active_net = net[act]
    ci_lo, ci_hi, boot_p = block_bootstrap_mean(net.to_numpy(float))
    return {
        "n": len(net), "active_days": int(act.sum()),
        "mean_bps": float(net.mean() * 1e4),
        "mean_active_bps": float(active_net.mean() * 1e4) if len(active_net) else None,
        "t": float(net.mean() / (sd / np.sqrt(len(net)))) if sd > 0 else None,
        "sharpe": float(net.mean() / sd * np.sqrt(252)) if sd > 0 else None,
        "hit_active": float((active_net > 0).mean()) if len(active_net) else None,
        "block5_mean_ci95_bps": [ci_lo * 1e4, ci_hi * 1e4],
        "block5_probability_mean_gt_zero": boot_p,
        "cum_return": float(eq.iloc[-1] - 1),
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
    }


def strategy_table(df: pd.DataFrame) -> dict:
    es = np.sign(df["early_ret"])
    rules = {
        "unconditional_momentum": es,
        "unconditional_reversal": -es,
        "negative_gex_momentum": es.where(df["gex_z"] <= -EXTREME_Z, 0.0),
        "positive_gex_reversal": (-es).where(df["gex_z"] >= EXTREME_Z, 0.0),
        "extreme_regime_switch": es.where(df["gex_z"] <= -EXTREME_Z,
                                           (-es).where(df["gex_z"] >= EXTREME_Z, 0.0)),
        "continuous_regime_switch": es * (-df["gex_z"]).clip(-1.0, 1.0),
    }
    out = {}
    splits = split_name(df.index)
    for name, pos in rules.items():
        gross = pos * df["rest_ret"]
        active = pos.abs() > 1e-12
        out[name] = {}
        for period in ("all", "discovery", "validation", "test"):
            m = pd.Series(True, index=df.index) if period == "all" else splits.eq(period)
            out[name][period] = {str(c): perf(gross[m], active[m], c) for c in COSTS_BPS}
    return out


def bh_adjust(pvals: dict[str, float]) -> dict[str, float]:
    valid = [(k, v) for k, v in pvals.items() if v is not None and np.isfinite(v)]
    valid.sort(key=lambda kv: kv[1])
    m = len(valid)
    ans: dict[str, float] = {}
    running = 1.0
    for rank in range(m, 0, -1):
        key, p = valid[rank - 1]
        running = min(running, p * m / rank)
        ans[key] = float(running)
    return ans


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "spec": {"gex_timing": "last source EOD strictly before session",
                 "early": "09:30-10:30 ET", "trade": "10:30-16:00 ET",
                 "extreme_z": EXTREME_Z, "hac_lags": HAC_LAGS,
                 "costs_bps_round_trip": COSTS_BPS,
                 "splits": {"discovery_end": "2023-12-31", "validation": "2024",
                            "test_start": "2025-01-01"}},
        "symbols": {},
    }
    primary_p = {}
    panels = []
    for sym in SYMS:
        p = session_panel(sym)
        p["split"] = split_name(p.index)
        p.to_parquet(OUT / f"panel_{sym.lower()}.parquet")
        panels.append((sym, p))
        feedback_rest = hac_regression(p, "rest_ret")
        feedback_last = hac_regression(p, "last_hour_ret")
        vol = volatility_regression(p)
        result["symbols"][sym] = {
            "role": "primary" if sym == "SPY" else "correlated sensitivity",
            "coverage": {"start": str(p.index.min().date()), "end": str(p.index.max().date()),
                         "n": len(p), "by_split": p["split"].value_counts().to_dict(),
                         "median_gex_staleness_days": float((p.index.to_series() - p["source_date"]).dt.days.median())},
            "feedback_rest": feedback_rest,
            "feedback_last_hour": feedback_last,
            "slopes_rest": slope_cells(p, "rest_ret"),
            "slopes_last_hour": slope_cells(p, "last_hour_ret"),
            "volatility": vol,
            "strategies": strategy_table(p),
        }
        primary_p[f"{sym}_feedback_rest"] = feedback_rest["interaction_p_hac"]
        primary_p[f"{sym}_feedback_last"] = feedback_last["interaction_p_hac"]
        primary_p[f"{sym}_volatility"] = vol["gex_z_p_hac"]
    result["bh_fdr_all_primary_cells"] = bh_adjust(primary_p)
    (OUT / "literature_intraday.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print("GEX LITERATURE INTRADAY — strict t-1 EOD GEX")
    for sym, p in panels:
        r = result["symbols"][sym]
        fr, fl, vv = r["feedback_rest"], r["feedback_last_hour"], r["volatility"]
        print(f"{sym} {p.index.min().date()}..{p.index.max().date()} n={len(p)}")
        print(f"  early×GEX -> rest: beta={fr['interaction_beta']:+.4f} t={fr['interaction_t_hac']:+.2f} p={fr['interaction_p_hac']:.4g}")
        print(f"  early×GEX -> last: beta={fl['interaction_beta']:+.4f} t={fl['interaction_t_hac']:+.2f} p={fl['interaction_p_hac']:.4g}")
        print(f"  GEX -> log(rest RV): beta={vv['gex_z_beta']:+.4f} t={vv['gex_z_t_hac']:+.2f} p={vv['gex_z_p_hac']:.4g}")
        for rule in ("negative_gex_momentum", "positive_gex_reversal", "extreme_regime_switch"):
            m = r["strategies"][rule]["test"]["2.0"]
            print(f"  OOS {rule:25s} active={m['active_days']:3d} mean={m['mean_bps']:+.2f}bps Sharpe={m['sharpe']:+.2f} t={m['t']:+.2f}")
    print(f"results: {OUT / 'literature_intraday.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
