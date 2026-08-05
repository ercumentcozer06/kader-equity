"""
screen/candidate_es_basis_v2 — Emir push-back sonrasi v1'in UC BOSLUGUNU kapatir:
  A) HORIZON LADDER (slow-signal dersi, M3 emsali): LEVEL + z, fwd {21,42,63,126}g, TAM tarih 2011+
  B) LONG-ADD / momentum (v1 yalniz trim-only idi; d10z>+2 bucket POZITIFTI): tide-flat gunlerde ekle
  C) seviye x donus etkilesimi (mania'dan DONUS = unwind) + funding-oncu probu (basis->VIX/HY-OAS)
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from spine import contract as C, tide as T              # noqa: E402
from backtest import engine as E                         # noqa: E402
from screen._util import load_price_csv, paired_win_prob, fdr_bh   # noqa: E402

DESK = Path(r"C:\Users\admin\Desktop\backtesting")
LAB = DESK / "es_basis_lab"
PRICES = {"SPX": "SPX_daily.csv", "NDX": "NASDAQ_daily.csv"}


def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")
def _dd(r): eq = (1+r.dropna()).cumprod(); return float((eq/eq.cummax()-1).min())


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def roll_z(s, win=252, minp=126):
    return (s - s.rolling(win, min_periods=minp).mean()) / s.rolling(win, min_periods=minp).std()


def main():
    bas = pd.read_parquet(LAB / "es_basis_series.parquet")
    sp = bas["spread_5dm"].dropna()
    z = roll_z(sp).dropna()
    dz = roll_z(sp.diff(10)).dropna()
    pctl = sp.rolling(504, min_periods=252).apply(lambda w: (w < w.iloc[-1]).mean())
    closes = {a: load_price_csv(DESK / fn) for a, fn in PRICES.items()}

    # ── A) HORIZON LADDER — LEVEL bucket x fwd-h, tam tarih ──
    print("=" * 100)
    print("  A) HORIZON LADDER — spread LEVEL (bps) -> fwd h-gun mutlak getiri (tam tarih 2011+; %neg parantez)")
    print("=" * 100)
    lev_edges = [(-9999, 0), (0, 50), (50, 100), (100, 150), (150, 9999)]
    for a in PRICES:
        cl = closes[a]
        idx = sp.index.intersection(cl.index)
        cb = cl.reindex(idx)
        spx_i = sp.reindex(idx)
        print(f"  [{a}]  {'bucket':<14}" + "".join(f"{'fwd'+str(h)+'g':>16}" for h in (21, 42, 63, 126)) + f"{'n':>7}")
        for lo, hi in lev_edges:
            m = (spx_i >= lo) & (spx_i < hi)
            cells = []
            for h in (21, 42, 63, 126):
                f = (cb.shift(-h) / cb - 1)[m].dropna()
                cells.append(f"{100*f.mean():+7.1f}%({100*(f<0).mean():3.0f}%)" if len(f) > 10 else "      n/a")
            print(f"        [{lo if lo>-9999 else '-inf'},{hi if hi<9999 else 'inf'}) " + "".join(f"{c:>16}" for c in cells) + f"{int(m.sum()):>7}")

    # ── B) LONG-ADD momentum (2019+, tide-flat gunlerde) ──
    print("\n" + "=" * 100)
    print("  B) LONG-ADD (v1 test ETMEMISTI) — tide-FLAT gunlerde d10z-momentum ile long ac; strict FDR")
    print("=" * 100)
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    zt = z.reindex(idx, method="ffill")
    dzt = dz.reindex(idx, method="ffill")
    pct_t = pctl.reindex(idx, method="ffill")
    bases = {a: strat_ret(tdir, prices[a]) for a in PRICES}
    print(f"  base tide: SPX {_sh(bases['SPX']):+.3f} / NDX {_sh(bases['NDX']):+.3f}   "
          f"(tide-flat gun: {int((tdir==0).sum())}/{len(tdir)})")
    addons = {
        "B1 flat&dz>+1.5 -> long1.0": ((tdir == 0) & (dzt > 1.5)).astype(float),
        "B2 flat&dz>+1.5 -> long0.5": 0.5 * ((tdir == 0) & (dzt > 1.5)).astype(float),
        "B3 flat&z>+1   -> long0.5":  0.5 * ((tdir == 0) & (zt > 1.0)).astype(float),
    }
    print(f"  {'variant':<28}{'SPX dSh':>9}{'SPX P':>7}{'NDX dSh':>9}{'NDX P':>7}{'SPX dd':>8}{'NDX dd':>8}{'FDR':>6}")
    for label, add in addons.items():
        res = {}
        for a in PRICES:
            v = strat_ret((tdir + add).clip(0, 1).reindex(idx), prices[a])
            res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]), _dd(v))
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in PRICES) else "—"
        print(f"  {label:<28}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}"
              f"{100*res['SPX'][2]:>+7.0f}%{100*res['NDX'][2]:>+7.0f}%{both:>6}")

    # ── C) seviye x DONUS etkilesimi + funding-oncu probu ──
    print("\n" + "=" * 100)
    print("  C1) MANIA'DAN DONUS (p75+ VE dz<-1.5) trim — incremental over tide 2019+; strict FDR")
    print("=" * 100)
    combos = {
        "C1a flat (p75+ & dz<-1.5)": (~((pct_t > 0.75) & (dzt < -1.5))).astype(float),
        "C1b flat (p90+ & dz<-1.0)": (~((pct_t > 0.90) & (dzt < -1.0))).astype(float),
    }
    print(f"  {'variant':<28}{'SPX dSh':>9}{'SPX P':>7}{'NDX dSh':>9}{'NDX P':>7}{'FDR':>6}")
    for label, vfac in combos.items():
        res = {}
        for a in PRICES:
            v = strat_ret((tdir * vfac.reindex(idx, method='ffill').fillna(1.0)).reindex(idx), prices[a])
            res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]))
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in PRICES) else "—"
        print(f"  {label:<28}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}{both:>6}")

    print("\n  C2) FUNDING-ONCU probu (betimsel): basis_z_t vs fwd-21g degisimler (Spearman, tam tarih)")
    try:
        import requests
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2", timeout=30)
        hy = pd.read_csv(io.StringIO(r.text))
        hy.columns = ["date", "v"]
        hy["date"] = pd.to_datetime(hy["date"])
        hy = pd.to_numeric(hy.set_index("date")["v"], errors="coerce").dropna() * 100
        import yfinance as yf
        vix = yf.download("^VIX", period="max", auto_adjust=False, progress=False)["Close"]
        if isinstance(vix, pd.DataFrame):
            vix = vix.iloc[:, 0]
        for name, s in (("dHY-OAS(bp)", hy), ("dVIX", vix)):
            si = s.reindex(z.index, method="ffill")
            fwd = si.shift(-21) - si
            j = pd.concat([z, fwd], axis=1, keys=["z", "f"]).dropna()
            rho = float(j["z"].rank().corr(j["f"].rank()))
            # yalniz ZENGIN bolge (z>1): mania -> ileride stres mi?
            jr = j[j["z"] > 1]
            rho_r = float(jr["z"].rank().corr(jr["f"].rank())) if len(jr) > 50 else float("nan")
            print(f"    {name:<12} tum-gunler rho {rho:+.3f} (n={len(j)}) | z>1 bolgesi rho {rho_r:+.3f} (n={len(jr)})")
    except Exception as e:
        print(f"    proba erisilemedi: {e}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
