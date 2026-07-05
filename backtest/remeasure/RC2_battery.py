"""
backtest/remeasure/RC2_battery.py — RC2.5 P&L BATTERY (kilitli liste, TEK harness; config = tek-gerçek-kaynak).
Re-run sınıfı: replacement (reason=instrument-fix); INDEX-FLAG üyeleri = amendment (new-trial, K config'te).
Üyeler/bayrak-setleri/maliyet/train-holdout/metrikler: config.py. YENİ ÜYE/EŞİK/PARAMETRE YOK.
PIT: level[D] (D-EOD) → D+1 open→close; gap = D+1-open/D-close − 1. Panel ≤ PANEL_END (06-09/10 panel-DIŞI).
  & <venv python> backtest/remeasure/RC2_battery.py
→ backtest/remeasure/RC2_battery_results.json (+ stdout tablo)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from math import erf, exp as mexp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "backtest"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import config as CFG                         # noqa: E402
from spine_diagnostic import daily_rth       # noqa: E402

EULER = 0.5772156649015329


def _ncdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def _nppf(p):
    """Φ⁻¹ (Acklam yaklaşımı — scipy'siz)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    if p < plow:
        q = sqrt(-2 * log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - plow:
        return -_nppf(1 - p)
    q = p - 0.5; r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def dsr(net: np.ndarray, K: int) -> float:
    """Deflated Sharpe (Bailey-LdP; D6 ile aynı formül). Döndürür: DSR olasılığı (0..1)."""
    x = net[~np.isnan(net)]
    N = len(x)
    if N < 10 or x.std() == 0:
        return float("nan")
    sr = x.mean() / x.std()                                  # günlük SR
    g3 = float(pd.Series(x).skew()); g4 = float(pd.Series(x).kurt() + 3)
    var = (1 - g3 * sr + (g4 - 1) / 4 * sr * sr) / (N - 1)
    if var <= 0:
        return float("nan")
    sr0 = sqrt(var) * ((1 - EULER) * _nppf(1 - 1 / K) + EULER * _nppf(1 - 1 / (K * np.e)))
    return _ncdf((sr - sr0) / sqrt(var))


def sharpe(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / x.std() * sqrt(252) if len(x) > 2 and x.std() > 0 else 0.0


def metrics(net: pd.Series) -> dict:
    """net: günlük NET pnl (tarih-indeksli)."""
    v = net.values
    n = len(v)
    sr_ann = sharpe(v)
    t = (v.mean() / v.std() * sqrt(n)) if n > 2 and v.std() > 0 else 0.0
    blocks = [round(sharpe(b), 2) for b in np.array_split(v, CFG.N_BLOCKS)] if n >= CFG.N_BLOCKS * 5 else []
    roll = net.rolling(CFG.ROLL_WIN).apply(lambda w: sharpe(w.values), raw=False).dropna()
    rng = np.random.default_rng(CFG.BOOT_SEED)
    bs = np.array([sharpe(v[rng.integers(0, n, n)]) for _ in range(CFG.BOOT_N)]) if n > 20 else np.array([np.nan])
    tot = v.sum()
    top3 = float(np.sort(v)[-CFG.TOPK_CONC:].sum() / tot) if tot > 0 else float("nan")
    tr = net[net.index <= pd.Timestamp(CFG.TRAIN_END)]
    ho = net[net.index >= pd.Timestamp(CFG.HOLDOUT_START)]
    return {
        "n": n, "sharpe": round(sr_ann, 2), "t": round(t, 2), "dsr_K": CFG.K_CURRENT,
        "dsr": round(dsr(v, CFG.K_CURRENT), 3),
        "mean_bps": round(1e4 * v.mean(), 1), "hit": round(float((v > 0).mean()), 2),
        "blocks6": blocks, "blocks_pos": int(sum(b > 0 for b in blocks)),
        "roll63_pct_pos": round(float((roll > 0).mean()), 2) if len(roll) else None,
        "boot_p_gt0": round(float((bs > 0).mean()), 2), "boot_ci5": round(float(np.percentile(bs, 5)), 2),
        "boot_ci95": round(float(np.percentile(bs, 95)), 2), "top3_share": round(top3, 2) if top3 == top3 else None,
        "train_sharpe": round(sharpe(tr.values), 2), "train_n": len(tr),
        "holdout_sharpe": round(sharpe(ho.values), 2), "holdout_n": len(ho),
    }


def build_panel(sym: str, flag_set: str) -> pd.DataFrame | None:
    """own-level (duvar/atm) + flag-src (regime) + D+1 RTH getirileri. PANEL_END'e kırpılır."""
    own_mode = "livematch" if flag_set == "livematch_own" else "fullsurface"
    own = pd.read_parquet(CFG.level_path(own_mode, sym))
    if flag_set == "index_flag":
        fsrc = pd.read_parquet(CFG.level_path("fullsurface", CFG.INDEX_FLAG_MAP[sym]))
    else:
        fsrc = own
    rth = daily_rth(sym)
    sess = list(rth.index)
    end = pd.Timestamp(CFG.PANEL_END)
    rows = []
    for D in own.index:
        if D > end or D not in rth.index or D not in fsrc.index:
            continue
        nxt = [s for s in sess if s > D]
        if not nxt:
            continue
        N = nxt[0]
        o1, h1, l1, c1 = rth.loc[N, ["o", "h", "l", "c"]]
        c0 = rth.loc[D, "c"]
        rows.append({"D": D, "gap": o1 / c0 - 1, "intraday": c1 / o1 - 1,
                     "regime": int(fsrc.loc[D, "regime"]),
                     "call_wall": own.loc[D, "call_wall"], "put_wall": own.loc[D, "put_wall"],
                     "atm_iv": own.loc[D, "atm_iv"], "h1": h1, "l1": l1, "c1": c1})
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("D").sort_index()


def member_pnl(p: pd.DataFrame, member: str) -> pd.Series:
    """Günlük NET pnl. Tanımlar config docstring + D-FAZ disentangle/spine ile birebir."""
    sg = np.sign(p["gap"].values)
    reg = p["regime"].values
    ret = p["intraday"].values
    if member == "gamma_txt":
        pos = np.where(reg > 0, -sg, sg)
    elif member == "gamma_inv":
        pos = np.where(reg > 0, sg, -sg)
    elif member == "vol_only":
        volhigh = p["atm_iv"].values > np.nanmedian(p["atm_iv"].values)   # in-sample medyan (D-FAZ ile aynı; flag'li)
        pos = np.where(volhigh, -sg, sg)
    elif member == "hep_mom":
        pos = sg
    elif member == "hep_rev":
        pos = -sg
    elif member == "M3_setup":
        pnl = np.zeros(len(p))
        traded = np.zeros(len(p), bool)
        for i, (_, r) in enumerate(p.iterrows()):
            legs = []
            if pd.notna(r["call_wall"]) and r["h1"] >= r["call_wall"]:
                legs.append((r["call_wall"] - r["c1"]) / r["call_wall"])     # MR (fade-yönü getirisi)
            if pd.notna(r["put_wall"]) and r["l1"] <= r["put_wall"]:
                legs.append((r["c1"] - r["put_wall"]) / r["put_wall"])
            if not legs:
                continue
            mr = float(np.mean(legs))
            pnl[i] = mr if r["regime"] > 0 else -mr                          # +γ fade / −γ breakout
            traded[i] = True
        net = pnl - np.where(traded, CFG.COST_DAILY, 0.0)
        return pd.Series(net, index=p.index)
    else:
        raise ValueError(member)
    return pd.Series(pos * ret - CFG.COST_DAILY, index=p.index)


def measurements(p: pd.DataFrame) -> dict:
    """M1 (gap→intraday corr, rejim-başına) + M2 (duvar-MR ort, rejim-başına) — spine tanımları."""
    out = {}
    for reg, lab in ((1, "+g"), (-1, "-g")):
        sub = p[p["regime"] == reg]
        out[f"M1_corr_{lab}"] = round(float(sub["gap"].corr(sub["intraday"])), 3) if len(sub) > 5 else None
        mrs = []
        for _, r in sub.iterrows():
            legs = []
            if pd.notna(r["call_wall"]) and r["h1"] >= r["call_wall"]:
                legs.append((r["call_wall"] - r["c1"]) / r["call_wall"])
            if pd.notna(r["put_wall"]) and r["l1"] <= r["put_wall"]:
                legs.append((r["c1"] - r["put_wall"]) / r["put_wall"])
            if legs:
                mrs.append(np.mean(legs))
        if mrs:
            a = np.array(mrs)
            tt = a.mean() / (a.std(ddof=1) / sqrt(len(a))) if len(a) > 2 and a.std() > 0 else 0.0
            out[f"M2_MR_{lab}"] = {"n": len(a), "mean_bps": round(1e4 * a.mean(), 1), "t": round(tt, 2)}
        else:
            out[f"M2_MR_{lab}"] = {"n": 0}
    return out


def main():
    results = {"config_sha": CFG.config_sha(), "run_utc": datetime.now(timezone.utc).isoformat(),
               "K": CFG.K_CURRENT, "amendment": CFG.AMENDMENT["id"], "rows": [], "measurements": {}}
    for sym in CFG.TRADE_SYMS:
        for flag_set in CFG.FLAG_SETS:
            p = build_panel(sym, flag_set)
            if p is None:
                results["rows"].append({"sym": sym, "flag": flag_set, "error": "panel kurulamadı"})
                continue
            for m in CFG.BATTERY_PNL:
                if m in CFG.FLAG_FREE and flag_set != "livematch_own":
                    continue                                   # bayrak-bağımsız üyeler tek koşum
                net = member_pnl(p, m)
                row = {"sym": sym, "flag": flag_set if m not in CFG.FLAG_FREE else "none(livematch-panel)",
                       "member": m, "class": "amendment" if flag_set == "index_flag" and m in CFG.FLAG_DEPENDENT
                       else "replacement", **metrics(net)}
                results["rows"].append(row)
            results["measurements"][f"{sym}|{flag_set}"] = {"panel_n": len(p), **measurements(p)}
    out = HERE / "RC2_battery_results.json"
    out.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    # stdout özet tablo
    print(f"RC2 BATTERY — config_sha={results['config_sha']} K={CFG.K_CURRENT} (panel ≤ {CFG.PANEL_END})")
    hdr = f"{'sym':4}{'flag':18}{'member':11}{'n':>5}{'Sharpe':>8}{'t':>7}{'DSR':>7}{'blk+':>5}{'roll+':>6}{'bootP':>6}{'top3':>6}{'trS':>6}{'hoS':>6}"
    print(hdr); print("-" * len(hdr))
    for r in results["rows"]:
        if "error" in r:
            print(f"{r['sym']:4}{r['flag']:18}ERROR {r['error']}"); continue
        print(f"{r['sym']:4}{r['flag']:18}{r['member']:11}{r['n']:>5}{r['sharpe']:>8}{r['t']:>7}{r['dsr']:>7}"
              f"{r['blocks_pos']:>5}{str(r['roll63_pct_pos']):>6}{r['boot_p_gt0']:>6}{str(r['top3_share']):>6}"
              f"{r['train_sharpe']:>6}{r['holdout_sharpe']:>6}")
    print(f"\nölçümler (M1 corr / M2 wall-MR) → {out.name} içinde; tüm satırlar rapora (seçicilik yok).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
