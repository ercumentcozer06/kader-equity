"""
backtest/scenario_engine/se_panel.py — ortak panel + bar yardımcıları (tanımlar: se_config docstring).
Panel satırı (sym, D): level_series_livematch[D] + regime_own + regime_idx + VU/VD (ham zincirden) +
mid_up/mid_dn + D+1 seans kimliği ve OHLC. Çalışmalar dakika yolunu rth_minutes()/bars15() ile kendisi okur.
  & <venv python> backtest/scenario_engine/se_panel.py   → panel_{spy,qqq}.parquet + sanity çıktısı
"""
from __future__ import annotations

import glob
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import se_config as SE  # noqa: E402

N_EXP = 5
BAND = 0.15


def rth_minutes(sym: str) -> pd.DataFrame:
    """1-dk barlar, ET RTH 09:30-16:00; kolonlar o/h/l/c/v + date (seans günü) + t (ET zaman)."""
    b = pd.read_parquet(ROOT / "data" / "historical_bars" / f"alpaca_{sym.lower()}_1m.parquet")
    ts = pd.to_datetime(b.index.get_level_values("timestamp")).tz_convert("America/New_York")
    df = pd.DataFrame({"o": b["open"].values, "h": b["high"].values, "l": b["low"].values,
                       "c": b["close"].values, "v": b["volume"].values}, index=ts).sort_index()
    t = df.index.time
    df = df[(t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())]
    df["date"] = pd.to_datetime(pd.Series(df.index.date, index=df.index))
    return df


def bars15(minutes: pd.DataFrame) -> pd.DataFrame:
    """09:30-anchored 15-dk barlar (label=left: bar 09:30 = 09:30-09:44 dk'ları)."""
    g = minutes.groupby("date")
    out = []
    for d, day in g:
        r = day.resample("15min", origin=pd.Timestamp(d.date()).tz_localize(day.index.tz) + pd.Timedelta("9h30min"))
        bar = r.agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum")).dropna(subset=["c"])
        bar["date"] = d
        out.append(bar)
    return pd.concat(out)


def _vu_vd_one(path: str, call_wall: float, put_wall: float):
    """Ham gz → front-5 expiry, OI>0, band ±%15 strike-grid → VU/VD (indikatör: 'next strike step')."""
    try:
        o = json.load(gzip.open(path, "rt", encoding="utf-8"))
    except Exception:
        return None, None
    j = o["resp"]
    if not j.get("strike"):
        return None, None
    K = np.asarray(j["strike"], float)
    oi = np.asarray(j["openInterest"], float)
    side = np.asarray(j["side"])
    exp = np.asarray(j["expiration"])
    S = float(np.median(j["underlyingPrice"]))
    m = (oi > 0) & (np.abs(K / S - 1) <= BAND)
    exps = np.sort(np.unique(exp[m]))[:N_EXP]
    m &= np.isin(exp, exps)
    vu = vd = None
    if call_wall == call_wall and call_wall is not None:
        ks = np.unique(K[m & (side == "call") & (K > call_wall)])
        vu = float(ks.min()) if len(ks) else None
    if put_wall == put_wall and put_wall is not None:
        ks = np.unique(K[m & (side == "put") & (K < put_wall)])
        vd = float(ks.max()) if len(ks) else None
    return vu, vd


def build_panel(sym: str) -> pd.DataFrame:
    lv = SE.level_series(sym, "livematch")
    idx_full = SE.level_series(SE.INDEX_FLAG_MAP[sym], "fullsurface")
    mins = rth_minutes(sym)
    daily = mins.groupby("date").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    sessions = list(daily.index)
    raw_dir = ROOT / "data" / "raw_chains" / sym
    rows = []
    lo, hi = pd.Timestamp(SE.PANEL_START), pd.Timestamp(SE.PANEL_END)
    for D in lv.index:
        if D < lo or D > hi:
            continue
        nxt = [s for s in sessions if s > D]
        if not nxt or D not in daily.index:
            continue
        N = nxt[0]
        r = lv.loc[D]
        f = raw_dir / f"{D.date()}.json.gz"
        vu, vd = _vu_vd_one(str(f), r["call_wall"], r["put_wall"]) if f.exists() else (None, None)
        reg_idx = int(idx_full.loc[D, "regime"]) if D in idx_full.index else 0
        rows.append({
            "D": D, "N": N, "regime_own": int(r["regime"]), "regime_idx": reg_idx,
            "net_gex": r["net_gex"], "flip": r["flip"], "ghost": r["ghost"],
            "call_wall": r["call_wall"], "put_wall": r["put_wall"], "hvl": r["hvl"],
            "max_pain": r["max_pain"], "atm_iv": r["atm_iv"], "em1": r["em1"], "spot_D": r["spot"],
            "vu": vu, "vd": vd,
            "mid_up": (r["flip"] + r["call_wall"]) / 2 if pd.notna(r["flip"]) and pd.notna(r["call_wall"]) else np.nan,
            "mid_dn": (r["put_wall"] + r["flip"]) / 2 if pd.notna(r["flip"]) and pd.notna(r["put_wall"]) else np.nan,
            "c0": daily.loc[D, "c"], "o1": daily.loc[N, "o"], "h1": daily.loc[N, "h"],
            "l1": daily.loc[N, "l"], "c1": daily.loc[N, "c"],
        })
    return pd.DataFrame(rows).set_index("D").sort_index()


def main():
    for sym in SE.SYMS:
        p = build_panel(sym)
        out = SE.panel_path(sym)
        p.to_parquet(out)
        vu_ok = p["vu"].notna().mean()
        print(f"=== {sym}: {len(p)} gün → {out.name} ===")
        print(f"  rejim_own −γ %{100*(p['regime_own']==-1).mean():.0f} | rejim_idx −γ %{100*(p['regime_idx']==-1).mean():.0f}"
              f" | own-vs-idx uyum %{100*(p['regime_own']==p['regime_idx']).mean():.0f}")
        print(f"  VU dolu %{100*vu_ok:.0f}, VU-CW adım medyan {float((p['vu']-p['call_wall']).median()):.2f}"
              f" | VD-PW adım medyan {float((p['vd']-p['put_wall']).median()):.2f}")
        print(f"  ghost<CW %{100*(p['ghost']<p['call_wall']).mean():.0f} | open-vs-ghost: üstte %"
              f"{100*(p['o1']>p['ghost']).mean():.0f} | em1 medyan %{100*(p['em1']/p['spot_D']).median():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
