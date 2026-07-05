"""
backtest/gex_playbook_v0 — GÖREV 3. GEX = EXECUTION katmanı (sinyal DEĞİL; sinyal TIDE'da). Üç alt-analiz;
G3 TOPLAMI DSR trial sayacına 1 ANALİZ olarak işlenir. YENİ EŞİK YOK (gex_shield sign+z, thr=1.0).

 a) KOŞULLANDIRMA: TIDE LONG-run gün-1 (ve g1-3) getirisini SqueezeMetrics GEX rejimiyle ayrıştır.
    −0.22% gün-1 negatif-gamma başlangıçlı run'larda mı yoğunlaşıyor? Run-başı rejim dağılımı?
 b) KURAL-1: negatif-gamma'da (z<−thr) girişi ilk normal-gamma kapanışına ERTELE (max 3 gün, sonra koşulsuz).
    PIT: rejim t kapanışından, işlem t+1 (book_returns +1g lag). Metrik: gün-1 değişimi, run-yakalama, Sharpe farkı.
 c) ENTEGRASYON: kural-1'li seriyi prop_sim'e besle → playbook'lu vs playbook'suz pass/kill/$. Kabul ölçütü =
    prop_sim İYİLEŞMESİ (standalone performans değil).

YAPILMAZ: Kural-2 (call-wall trim) / Kural-3 (put-wall/HVL stop) — tarihsel wall yok (G1 forward doldurunca).
  pyramid/trail/sentetik-convexity/SMC/intraday YOK.

  & <venv python> backtest/gex_playbook_v0.py
"""
from __future__ import annotations

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

from spine import contract as C, tide as T          # noqa: E402
from backtest import engine as E                      # noqa: E402
from backtest import prop_sim as PS                   # noqa: E402
from modules.cor1m_froth import froth_factor_series   # noqa: E402
from modules.gex_shield import gex_zscore, shield_factor_series  # noqa: E402

THR = 1.0                # gex_shield mevcut eşiği (YENİ DEĞİL)
MAX_DELAY = 3            # a priori SABİT (taranmaz)


def _load():
    scores, prices, vector, _ = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector)); idx = tdir.index
    cor = pd.read_parquet(ROOT/"data"/"cache"/"corr_pc.parquet")["COR1M"].dropna()
    gex = pd.read_parquet(ROOT/"data"/"cache"/"squeeze_dix_gex.parquet")["gex"].dropna()
    froth = froth_factor_series(cor.reindex(idx, method="ffill"), 8, 11, 0.0)
    zg = gex_zscore(gex).reindex(idx, method="ffill")
    shield = shield_factor_series(zg, 0.5, 1.0, 0.4)
    return tdir, froth, shield, zg, prices, idx


def apply_rule1(tdir: pd.Series, zg: pd.Series) -> pd.Series:
    """Negatif-gamma'da (z<−THR) LONG girişini ilk normal-gamma'ya ertele, max MAX_DELAY gün sonra koşulsuz."""
    d = tdir.values.astype(float); z = zg.values; out = d.copy()
    i = 1
    while i < len(d):
        if d[i] == 1 and d[i-1] == 0:                  # LONG giriş
            delay = 0
            while (delay < MAX_DELAY and i+delay < len(d) and d[i+delay] == 1
                   and not np.isnan(z[i+delay]) and z[i+delay] < -THR):
                out[i+delay] = 0.0                     # ertele (flat kal)
                delay += 1
            i += delay + 1
        else:
            i += 1
    return pd.Series(out, index=tdir.index)


def _metrics(pos, prices, idx, asset="SPX"):
    ret = E.fwd_ret(prices[asset], idx); p = pos.astype(float).values
    p = np.concatenate([[0.0], p[:-1]])
    r = pd.Series(p*ret, index=idx).dropna()
    sh = float(r.mean()/r.std()*np.sqrt(252)) if r.std() else float("nan")
    eq = (1+r).cumprod(); dd = float((eq/eq.cummax()-1).min())
    return sh, dd, float((1+r).prod()-1)


def conditioning(tdir, zg, prices, idx):
    print("  [a] KOŞULLANDIRMA — TIDE LONG-girişi gün-1/g1-3 getirisi × GEX rejimi (SPX)")
    d = tdir.values; z = zg.values
    cl = prices["SPX"].reindex(idx, method="ffill").values
    entries = [k for k in range(1, len(d)) if d[k] == 1 and d[k-1] == 0]
    short, normal = [], []
    short3, normal3 = [], []
    n_short_start = 0
    for k in entries:
        if k+1 >= len(cl):
            continue
        r1 = cl[k+1]/cl[k]-1
        r3 = (cl[min(k+3, len(cl)-1)]/cl[k]-1)
        if not np.isnan(z[k]) and z[k] < -THR:
            short.append(r1); short3.append(r3); n_short_start += 1
        else:
            normal.append(r1); normal3.append(r3)
    print(f"      giriş say {len(entries)} | negatif-gamma başlangıç {n_short_start} (%{100*n_short_start/max(1,len(entries)):.0f})")
    print(f"      gün-1 ort:  negatif-γ {100*np.mean(short) if short else float('nan'):+.2f}%  "
          f"normal-γ {100*np.mean(normal) if normal else float('nan'):+.2f}%")
    print(f"      g1-3 ort:   negatif-γ {100*np.mean(short3) if short3 else float('nan'):+.2f}%  "
          f"normal-γ {100*np.mean(normal3) if normal3 else float('nan'):+.2f}%")
    print(f"      → gün-1 zayıflığı {'NEGATİF-γ girişlerde yoğun' if (short and normal and np.mean(short)<np.mean(normal)) else 'rejimle ayrışmıyor'}")
    return n_short_start, len(entries)


def main():
    tdir, froth, shield, zg, prices, idx = _load()
    print("=" * 100)
    print("  GÖREV 3 — GEX EXECUTION PLAYBOOK v0 (kural-1 entry-erteleme; sinyal TIDE, GEX=execution)")
    print("=" * 100)
    conditioning(tdir, zg, prices, idx)

    # [b] kural-1
    tdir_r1 = apply_rule1(tdir, zg)
    base_book = (tdir * froth * shield).reindex(idx)
    r1_book = (tdir_r1 * froth * shield).reindex(idx)
    n_changed = int((tdir != tdir_r1).sum())
    print(f"\n  [b] KURAL-1 — negatif-γ girişi ertele (max {MAX_DELAY}g). Değişen gün: {n_changed}")
    for a in ("SPX", "NDX"):
        sb, db, pb = _metrics(base_book, prices, idx, a)
        sr, dr, pr = _metrics(r1_book, prices, idx, a)
        print(f"      {a}: base Sharpe {sb:.2f}/maxDD {100*db:.0f}%/PnL {100*pb:+.0f}%  →  "
              f"kural-1 Sharpe {sr:.2f}/maxDD {100*dr:.0f}%/PnL {100*pr:+.0f}%  (ΔSharpe {sr-sb:+.2f})")

    # [c] entegrasyon: prop_sim base vs kural-1 (temsili pos=0.7)
    print(f"\n  [c] ENTEGRASYON — prop_sim base vs kural-1 (pos 0.7; kabul ölçütü = prop_sim İYİLEŞMESİ)")
    print(f"      {'kitap':<8}{'pass-base':>11}{'pass-r1':>10}{'kill-base':>11}{'kill-r1':>10}{'gün-base':>10}{'gün-r1':>9}")
    for book in ("SPX", "NDX", "50/50"):
        b = PS.run_book(book, 0.7)
        r = PS.run_book(book, 0.7, pos_override=r1_book)
        print(f"      {book:<8}{100*b['total_pass']:>10.0f}%{100*r['total_pass']:>9.0f}%"
              f"{len(b['kills']):>11}{len(r['kills']):>10}{(b['median_days'] or 0):>10.0f}{(r['median_days'] or 0):>9.0f}")
    print("\n" + "=" * 100)
    print("  NOT: kural-1 yalnız prop_sim metriklerini İYİLEŞTİRİRSE kabul (standalone Sharpe değil). G3=1 DSR-trial.")
    return 0


if __name__ == "__main__":
    main()
