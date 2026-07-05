"""
backtest/remeasure/R2_keydiag — FAZ-R / R2 ÖN-DİAGNOSTİK (157/243 gün; tam battery yarın full-capture'da).
ASIL SORULAR (D-FAZ birebir, OLD→NEW; yeni strateji/eşik YOK — replacement):
  1. KAPSAMA: gamma$ old(tek-expiry) → LIVE-MATCH(front5) → FULL-SURFACE.
  2. ② SIGN: net_gex-işaret vs SqueezeMetrics-gex-işaret — overall + |gex| tercile. D2'nin %30/%49 üst-tercile
     çöküşü KAPANIYOR mu? KRİTİK TEST: SPX-FULL (gerçek SPX-market gamma) vs squeeze (SPX-market) ≥%90 mi
     (ikisi aynı şeyi ölçüyorsa yüksek olmalı) → "sapma truncation/ETF-havuzdandı" tezinin doğrudan testi.
  3. BAYRAK İSTİKRAR: old-işaret vs LIVE-MATCH-işaret flip-gün %.
  4. EVENT-EDGE 8-hücre (sym×rejim×duvar): ters-desen (tez-dışı) KALIYOR mu yoksa ders-kitabına DÖNÜYOR mu.
TIDE/overlay frozen; P&L üretilmez (event-edge ölçüm, D-FAZ ile aynı mean_reversion_return).
  & <venv python> backtest/remeasure/R2_keydiag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backtest"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from spine_diagnostic import daily_rth, mean_reversion_return     # noqa: E402

CACHE = ROOT / "data" / "cache"


def load_lv(name):
    p = CACHE / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def load_squeeze():
    p = CACHE / "squeeze_dix_gex.parquet"
    if not p.exists():
        return None
    s = pd.read_parquet(p)["gex"].dropna()
    s.index = pd.to_datetime(s.index)
    return s


def sign_agreement(lv, sq):
    """net_gex-işaret vs squeeze-gex-işaret: overall + |net_gex| tercile."""
    if lv is None or sq is None:
        return None
    df = pd.DataFrame({"ng": lv["net_gex"]}).join(pd.DataFrame({"sq": sq}), how="inner").dropna()
    if len(df) < 10:
        return None
    df["ag"] = (np.sign(df["ng"]) == np.sign(df["sq"]))
    overall = df["ag"].mean()
    try:
        df["t"] = pd.qcut(df["ng"].abs(), 3, labels=["alt", "orta", "üst"])
        terc = {t: g["ag"].mean() for t, g in df.groupby("t", observed=True)}
    except Exception:
        terc = {}
    return overall, terc, len(df)


def panel(lv, sym):
    rth = daily_rth(sym)
    sess = list(rth.index)
    rows = []
    for D in lv.index:
        if D not in rth.index:
            continue
        nxt = [s for s in sess if s > D]
        if not nxt:
            continue
        N = nxt[0]
        r = lv.loc[D].to_dict()
        r.update({"call_wall": lv.loc[D, "call_wall"], "put_wall": lv.loc[D, "put_wall"],
                  "h1": rth.loc[N, "h"], "l1": rth.loc[N, "l"], "c1": rth.loc[N, "c"], "regime": lv.loc[D, "regime"]})
        rows.append(r)
    return pd.DataFrame(rows)


def event_cells(lv, sym):
    p = panel(lv, sym)
    if p.empty:
        return None
    p["mr"] = mean_reversion_return(p)
    out = {}
    for reg, rl in ((1, "+γ"), (-1, "−γ")):
        sub = p[p["regime"] == reg]
        for wall, sel in (("call", sub["h1"] >= sub["call_wall"]), ("put", sub["l1"] <= sub["put_wall"])):
            ev = sub[sel]["mr"].dropna()
            out[(rl, wall)] = (len(ev), 1e4 * ev.mean() if len(ev) else np.nan)
    return out


def main():
    sq = load_squeeze()
    print("=" * 96)
    print("  FAZ-R / R2 ÖN-DİAGNOSTİK (onarılmış enstrüman, 157/243 gün — tam battery full-capture'da)")
    print("=" * 96)
    if sq is not None:
        print(f"  SqueezeMetrics cache: {len(sq)}g, {sq.index.min().date()}→{sq.index.max().date()}")

    # 1+2: kapsama + sign-agreement (old → livematch → full); SPX-full = squeeze'in apples-to-apples testi
    print("\n  [1+2] KAPSAMA (gamma$ medyan) + ② SIGN-AGREEMENT (vs SqueezeMetrics SPX-gex)")
    print("  " + "-" * 92)
    rows = [("SPY", "level_series_spy", "level_series_livematch_spy", "level_series_fullsurface_spy"),
            ("QQQ", "level_series_qqq", "level_series_livematch_qqq", "level_series_fullsurface_qqq"),
            ("SPX", None, "level_series_livematch_spx", "level_series_fullsurface_spx")]
    for sym, old_n, lm_n, full_n in rows:
        for tag, name in (("OLD(tek-exp)", old_n), ("LIVE-MATCH", lm_n), ("FULL-SURF", full_n)):
            if name is None:
                continue
            lv = load_lv(name)
            if lv is None:
                print(f"    {sym:3} {tag:13}: (parquet yok — build sürüyor?)"); continue
            gd = lv["gamma_dollar"].median() / 1e9 if "gamma_dollar" in lv else float("nan")
            ag = sign_agreement(lv, sq)
            if ag:
                ov, terc, n = ag
                ts = " ".join(f"{k}%{100*v:.0f}" for k, v in terc.items())
                print(f"    {sym:3} {tag:13}: gamma$ {gd:6.2f}bn | sign-agr GENEL %{100*ov:.0f} (n{n}) | tercile {ts}")
            else:
                print(f"    {sym:3} {tag:13}: gamma$ {gd:6.2f}bn | sign-agr ölçülemedi (squeeze overlap yok)")

    # 3: bayrak istikrar (old vs livematch işaret)
    print("\n  [3] BAYRAK İSTİKRAR (old-işaret vs LIVE-MATCH-işaret flip-gün %)")
    for sym in ("SPY", "QQQ"):
        old = load_lv(f"level_series_{sym.lower()}"); lm = load_lv(f"level_series_livematch_{sym.lower()}")
        if old is None or lm is None:
            print(f"    {sym}: (parquet eksik)"); continue
        j = pd.DataFrame({"o": old["regime"], "n": lm["regime"]}).dropna()
        flip = (j["o"] != j["n"]).mean()
        print(f"    {sym}: {len(j)} ortak gün, işaret-flip %{100*flip:.0f} (onarım rejimi ne kadar değiştirdi)")

    # 4: event-edge 8-hücre OLD vs LIVE-MATCH
    print("\n  [4] EVENT-EDGE 8-hücre (ort MR bps; +>0=duvar-tuttu, <0=kırıldı) — ters-desen kalıyor mu?")
    for sym in ("SPY", "QQQ"):
        for tag, name in (("OLD", f"level_series_{sym.lower()}"), ("LIVE-MATCH", f"level_series_livematch_{sym.lower()}")):
            lv = load_lv(name)
            if lv is None:
                print(f"    {sym} {tag}: (parquet yok)"); continue
            ec = event_cells(lv, sym)
            if not ec:
                continue
            cells = " | ".join(f"{r}-{w} n{ec[(r,w)][0]} {ec[(r,w)][1]:+.0f}bps" for (r, w) in
                               [("+γ","call"),("+γ","put"),("−γ","call"),("−γ","put")])
            print(f"    {sym} {tag:11}: {cells}")
    print("\n  TEZ: +γ→MR>0 (duvar tutar) / −γ→MR<0 (kırılır). OLD'da TERS'ti (+γ MR≈−20bps). NEW'de düzeliyor mu?")
    print("  NOT: 157/243 gün ön-sonuç; ⑦ DSR/t tam-battery yarın full-capture'da (bu ön-bakış güç ÜRETMEZ).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
