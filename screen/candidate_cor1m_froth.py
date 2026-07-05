"""
screen/candidate_cor1m_froth — SpotGamma COR1M sinyalini DOĞRU yönde test (Emir 2026-06-09 düzeltmesi).

SpotGamma (Kochuba): COR1M < ~8 = single-stock call froth / complacency = KONTRARİAN BEARISH (~30g spazm).
ÖNCEKİ testim TERSTİ (COR1M yüksekken trim = sistemik-stres). Bu sefer DÜŞÜK-COR1M = de-risk.

1) BUCKET (absolute fwd getiri, his 'rank≠absolute' dersi): COR1M seviyesine göre forward 21g SPX/NDX getiri.
2) EVENT-STUDY: COR1M 8-altına geçişten sonra fwd 5/21/42g getiri.
3) INCREMENTAL (2019+ tide): COR1M DÜŞÜKKEN tide-long'u kes/kıs → strict FDR + son episode'lar.
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

from spine import contract as C, tide as T              # noqa: E402
from backtest import engine as E                         # noqa: E402
from screen._util import load_price_csv, paired_win_prob, fdr_bh   # noqa: E402

DESK = Path(r"C:\Users\admin\Desktop\backtesting")
PRICES = {"SPX": "SPX_daily.csv", "NDX": "NASDAQ_daily.csv"}


def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")
def _dd(r): eq = (1+r.dropna()).cumprod(); return float((eq/eq.cummax()-1).min())
def _ep(r, s, e): w = r[(r.index >= s) & (r.index <= e)]; return _sh(w)


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def main():
    cor = pd.read_parquet(ROOT / "data" / "cache" / "corr_pc.parquet")["COR1M"].dropna()

    # ── 1) BUCKET: COR1M seviyesi → forward 21g getiri (absolute) ──
    print("=" * 90)
    print("  1) BUCKET — COR1M seviyesine göre forward 21g getiri (DÜŞÜK COR1M = froth → bearish mi?)")
    print("=" * 90)
    print(f"  {'COR1M bucket':<14}{'n':>6}{'SPX fwd21 ort':>15}{'SPX %neg':>10}{'NDX fwd21 ort':>15}{'NDX %neg':>10}")
    edges = [(0, 8), (8, 11), (11, 14), (14, 18), (18, 25), (25, 100)]
    for lo, hi in edges:
        line = [f"[{lo},{hi})", 0, None, None, None, None]
        for ai, (a, fn) in enumerate(PRICES.items()):
            close = load_price_csv(DESK / fn)
            idx = cor.index.intersection(close.index)
            cb = close.reindex(idx, method="ffill")
            f21 = (cb.shift(-21) / cb - 1).reindex(idx)
            c = cor.reindex(idx)
            m = (c >= lo) & (c < hi)
            sub = f21[m].dropna()
            if ai == 0:
                line[1] = int(len(sub)); line[2] = sub.mean(); line[3] = (sub < 0).mean()
            else:
                line[4] = sub.mean(); line[5] = (sub < 0).mean()
        print(f"  {line[0]:<14}{line[1]:>6}{(100*line[2] if line[2]==line[2] else float('nan')):>+14.1f}%"
              f"{(100*line[3]):>+9.0f}%{(100*line[4] if line[4]==line[4] else float('nan')):>+14.1f}%{(100*line[5]):>+9.0f}%")

    # ── 2) EVENT-STUDY: COR1M 8-altına geçiş ──
    print("\n" + "=" * 90)
    print("  2) EVENT-STUDY — COR1M 8-altına DÜŞÜŞ (cross<8) sonrası forward getiri")
    print("=" * 90)
    cross = (cor < 8) & (cor.shift(1) >= 8)
    ev = cor.index[cross]
    print(f"  cross<8 olay sayısı: {len(ev)}  (2006-2026)")
    for a, fn in PRICES.items():
        close = load_price_csv(DESK / fn)
        cb = close.reindex(close.index)
        outs = []
        for h in (5, 21, 42):
            rr = []
            for d in ev:
                p0 = cb.asof(d)
                p1 = cb.asof(d + pd.Timedelta(days=int(h * 1.45)))
                if p0 and p1 and p0 > 0:
                    rr.append(p1 / p0 - 1)
            outs.append((h, np.mean(rr) if rr else float("nan"), np.mean([x < 0 for x in rr]) if rr else float("nan")))
        s = "  ".join(f"{h}g: {100*m:+.1f}% ({100*n:.0f}%neg)" for h, m, n in outs)
        print(f"  {a}: {s}")

    # ── 3) INCREMENTAL over TIDE (2019+): COR1M DÜŞÜKKEN de-risk ──
    print("\n" + "=" * 90)
    print("  3) INCREMENTAL over TIDE (2019+) — COR1M DÜŞÜKKEN (froth) tide-long'u kes/kıs. STRICT FDR {SPX,NDX}")
    print("=" * 90)
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    cor_t = cor.reindex(idx, method="ffill")
    variants = {
        "flat<8 (froth off)": (cor_t >= 8).astype(float),                          # COR1M<8 → FLAT
        "trim<10 soft":       (1.0 - 0.5 * np.clip((10 - cor_t) / 4, 0, 1)).clip(0.5, 1.0),
        "flat<9":             (cor_t >= 9).astype(float),
    }
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    print(f"  base: SPX {_sh(bases['SPX']):+.3f} / NDX {_sh(bases['NDX']):+.3f}   (FDR geç = ikisi de P(v>b)≥95%)")
    print(f"  {'variant':<20}{'SPX ΔSh':>9}{'SPX P':>7}{'NDX ΔSh':>9}{'NDX P':>7}{'SPX dd':>8}{'2026 SPX':>10}{'FDR':>6}")
    for label, vfac in variants.items():
        res = {}
        for a in ("SPX", "NDX"):
            v = strat_ret((tdir * vfac.reindex(idx, method="ffill")).reindex(idx), prices[a])
            res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]), _dd(v), v)
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
        e26 = _ep(res["SPX"][3], "2026-01-01", "2026-06-08")
        print(f"  {label:<20}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}"
              f"{100*res['SPX'][2]:>+7.0f}%{e26:>+10.2f}{both:>6}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
