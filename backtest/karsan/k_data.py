"""
backtest/karsan/k_data.py — Faz 0: veri envanteri + KALİTE KAPISI + birleşik yükleyici (spec Faz 0).
- yfinance OHLC (^GSPC, ^NDX, ^VIX9D) çek + cache (deterministik; 1 kez indir, sonra parquet'ten oku).
- Mevcut CBOE serileri (VIX/SKEW/VVIX/VIX3M/VXN/COR1M) data/cache'ten yükle.
- Kalite kapısı: neg/sıfır fiyat, |log-ret|>0.25 FLAG, volume>0, dup-timestamp, ffill≤1, index-ETF corr≥0.98.
- Coverage & regime raporu.
  & <venv> backtest/karsan/k_data.py [build|report]
"""
from __future__ import annotations
import sys, json
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
import k_config as K

K.KDATA.mkdir(parents=True, exist_ok=True)
K.KRESULTS.mkdir(parents=True, exist_ok=True)


def fetch_yf():
    """yfinance OHLC çek + cache (idempotent: parquet varsa atla)."""
    import yfinance as yf
    out = {}
    for name, tk in K.YF_TICKERS.items():
        p = K.KDATA / f"yf_{name}.parquet"
        if p.exists():
            out[name] = pd.read_parquet(p)
            continue
        d = yf.download(tk, start="1990-01-01", end="2026-06-11", progress=False, auto_adjust=False)
        d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
        d = d.rename(columns={"Open": "o", "High": "h", "Low": "l", "Close": "c", "Volume": "v"})
        d = d[["o", "h", "l", "c", "v"]].copy()
        d.index = pd.to_datetime(d.index)
        d.to_parquet(p)
        out[name] = d
    return out


def quality_gate(name: str, df: pd.DataFrame, price_col: str = "c", has_vol: bool = True) -> dict:
    """spec Faz 0 kalite kapısı. FLAG'leri döndürür (atmaz)."""
    flags = {}
    px = df[price_col]
    flags["neg_or_zero_px"] = int((px <= 0).sum())
    lr = np.log(px / px.shift(1))
    big = lr.abs() > K.QG_MAX_ABS_LOGRET
    flags["big_logret_gt0.25"] = int(big.sum())
    flags["big_logret_dates"] = [str(d.date()) for d in df.index[big.fillna(False)]][:10]
    if has_vol and "v" in df.columns:
        flags["zero_vol_days"] = int((df["v"] <= 0).sum())
    flags["dup_timestamps"] = int(df.index.duplicated().sum())
    flags["n"] = len(df)
    flags["span"] = [str(df.index.min().date()), str(df.index.max().date())]
    return flags


def load_all():
    """Tüm Faz-1 serilerini tek sözlükte döndür (cache + yfinance)."""
    yf = fetch_yf()
    cache = ROOT / "data" / "cache"
    series = {
        "SPX_ohlc": yf["SPX"], "NDX_ohlc": yf["NDX"], "VIX9D": yf["VIX9D"],
        "VIX": pd.read_parquet(cache / "vixcls.parquet")["vix"],
        "SKEW_VVIX": pd.read_parquet(cache / "skew_vvix.parquet"),
        "VOLSURF": pd.read_parquet(cache / "vol_surface.parquet"),   # vix, vix3m, vxn, ts_ratio
        "CORR": pd.read_parquet(cache / "corr_pc.parquet"),          # COR1M, COR3M, ...
        "BREADTH": pd.read_parquet(cache / "breadth.parquet"),       # SPY, QQQ close (ETF proxy)
    }
    return series


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    S = load_all()
    rep = {"generated": "Faz0", "series": {}, "quality": {}, "coverage": {}}

    # --- envanter + kalite ---
    qg = {}
    qg["SPX_ohlc"] = quality_gate("SPX", S["SPX_ohlc"])
    qg["NDX_ohlc"] = quality_gate("NDX", S["NDX_ohlc"])
    qg["VIX9D"] = quality_gate("VIX9D", S["VIX9D"])
    # index vs ETF corr (SPX↔SPY, NDX↔QQQ) ortak pencerede
    br = S["BREADTH"]
    def idx_etf_corr(idx_ohlc, etf_col):
        a = idx_ohlc["c"].pct_change(); b = br[etf_col].pct_change()
        j = pd.concat([a, b], axis=1, join="inner").dropna()
        return round(float(j.iloc[:, 0].corr(j.iloc[:, 1])), 4), len(j)
    c_spx, n_spx = idx_etf_corr(S["SPX_ohlc"], "SPY")
    c_ndx, n_ndx = idx_etf_corr(S["NDX_ohlc"], "QQQ")
    qg["SPX_vs_SPY_corr"] = {"corr": c_spx, "n": n_spx, "flag": c_spx < K.QG_ETF_INDEX_MINCORR}
    qg["NDX_vs_QQQ_corr"] = {"corr": c_ndx, "n": n_ndx, "flag": c_ndx < K.QG_ETF_INDEX_MINCORR}
    rep["quality"] = qg

    # --- coverage / regime statement ---
    def span(x):
        idx = x.index
        return f"{pd.to_datetime(idx.min()).date()}..{pd.to_datetime(idx.max()).date()} (n={len(x)})"
    cov = {
        "SPX daily OHLC (^GSPC)": span(S["SPX_ohlc"]) + "  → 1990+ : GFC2008, 2011-euro, 2017-lowvol, 2018Q4, 2020-COVID, 2022-bear, 2023-SVB, 2025-tariff = MULTI-REGIME",
        "NDX daily OHLC (^NDX)": span(S["NDX_ohlc"]) + "  → MULTI-REGIME",
        "VIX (vixcls)": span(S["VIX"]) + "  → 1990+ MULTI-REGIME",
        "SKEW+VVIX": span(S["SKEW_VVIX"]) + "  → SKEW 1990+, VVIX (CBOE 2007+ ama dosya 1990 başlangıç → VVIX erken NaN olabilir, kontrol)",
        "VIX3M+VXN (vol_surface)": span(S["VOLSURF"]) + "  → 2007-12+ (VIX3M/VXN). NDX-vol = VXN VAR.",
        "COR1M/COR3M (corr_pc)": span(S["CORR"]) + "  → 2006+ (implied corr)",
        "VIX9D (^VIX9D)": span(S["VIX9D"]) + "  → 2011+; 0DTE-bozulma bayrağı 2023+ ayrı",
        "1-min bars (alpaca SPY/QQQ)": "data/historical_bars/alpaca_{spy,qqq}_1m.parquet ~2020-09..2026-06 (~5.8y) → SINGLE-REGIME (C5/C6 caveat)",
        "FOMC dates": f"{len(K.FOMC_DATES)} scheduled decisions 2015-2026 (elle-giriş, public FOMC takvimi; 2020-03 irregular çıkarıldı) → C4 penceresi",
    }
    rep["coverage"] = cov

    # VVIX erken NaN kontrolü
    vv = S["SKEW_VVIX"]["VVIX"].dropna()
    rep["VVIX_first_valid"] = str(vv.index.min().date()) if len(vv) else None
    sk = S["SKEW_VVIX"]["SKEW"].dropna()
    rep["SKEW_first_valid"] = str(sk.index.min().date()) if len(sk) else None
    vxn = S["VOLSURF"]["vxn"].dropna()
    rep["VXN_first_valid"] = str(vxn.index.min().date()) if len(vxn) else None

    (K.KRESULTS / "phase0_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")

    # --- stdout ---
    print("=" * 92); print("  FAZ 0 — VERİ ENVANTERİ + KALİTE KAPISI"); print("=" * 92)
    print("\n  [COVERAGE & REGIME]")
    for k, v in cov.items():
        print(f"   • {k}\n       {v}")
    print(f"\n   VVIX ilk geçerli: {rep['VVIX_first_valid']} | SKEW: {rep['SKEW_first_valid']} | VXN: {rep['VXN_first_valid']}")
    print("\n  [KALİTE KAPISI]")
    for k, v in qg.items():
        if "corr" in k:
            print(f"   • {k}: corr={v['corr']} (n={v['n']})  {'⚠FLAG<0.98' if v['flag'] else 'OK'}")
        else:
            fl = []
            if v.get("neg_or_zero_px"): fl.append(f"neg/0 px={v['neg_or_zero_px']}")
            if v.get("big_logret_gt0.25"): fl.append(f"|logret|>0.25: {v['big_logret_gt0.25']}g {v['big_logret_dates']}")
            if v.get("zero_vol_days"): fl.append(f"zero-vol={v['zero_vol_days']}")
            if v.get("dup_timestamps"): fl.append(f"dup-ts={v['dup_timestamps']}")
            print(f"   • {k}: n={v['n']} {v['span']}  {'⚠ '+'; '.join(fl) if fl else 'temiz'}")
    print(f"\n  → results/phase0_report.json yazıldı.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
