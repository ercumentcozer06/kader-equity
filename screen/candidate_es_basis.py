"""
screen/candidate_es_basis — equity futures basis ("equity repo" spread, Conks 2026-07) adayı.

Seri: Desktop/backtesting/es_basis_lab/es_basis_series.parquet -> spread_5dm
(bps over o/n FF; (F1/S-1)*365/dte + divY - DFF, 5g median; Barchart dolmus-kontrat arsivi).

Hipotezler (pre-register):
  SOL kuyruk  : spread cokuk/negatif = funding stress / de-grossing  -> tide-long trim (shield)
  SAG kuyruk  : spread asiri zengin (mania, >~+100-150bps / z>+1.5) = kaldirac frothu -> trim
  DELTA       : spread'in hizli cokusu (10g) = unwind nowcast -> trim

Protokol (FINDING-5 disiplini): 1) bucket BOTH orientations + tails (mutlak fwd-21g getiri),
2) event-study, 3) incremental over TIDE 2019+ (strict BH-FDR {SPX,NDX}),
4) incremental over FULL STACK (tide x COR1M-froth x GEX-shield) = marjinal deger + epizot cetveli.
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
from modules.cor1m_froth import froth_factor_series      # noqa: E402
from modules.gex_shield import gex_zscore, shield_factor_series    # noqa: E402

DESK = Path(r"C:\Users\admin\Desktop\backtesting")
LAB = DESK / "es_basis_lab"
PRICES = {"SPX": "SPX_daily.csv", "NDX": "NASDAQ_daily.csv"}

EPISODES = [("COVID", "2020-02-15", "2020-04-30"),
            ("2022 ayi", "2022-01-01", "2022-10-31"),
            ("SVB", "2023-02-15", "2023-04-15"),
            ("Aug'24 spazm", "2024-07-10", "2024-08-30"),
            ("late'24 mania->25Q1", "2024-12-01", "2025-03-31"),
            ("Jun'26 mania", "2026-05-01", "2026-07-31")]


def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")
def _dd(r): eq = (1+r.dropna()).cumprod(); return float((eq/eq.cummax()-1).min())
def _ep(r, s, e): w = r[(r.index >= s) & (r.index <= e)]; return _sh(w)


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def roll_z(s: pd.Series, win=252, minp=126) -> pd.Series:
    mu = s.rolling(win, min_periods=minp).mean()
    sd = s.rolling(win, min_periods=minp).std()
    return (s - mu) / sd


def main():
    bas = pd.read_parquet(LAB / "es_basis_series.parquet")
    sp = bas["spread_5dm"].dropna()
    z = roll_z(sp).dropna()
    dz = roll_z(sp.diff(10)).dropna()          # 10g degisimin z'si (unwind ekseni)
    print(f"seri: spread_5dm {sp.index.min().date()}->{sp.index.max().date()} n={len(sp)} | z n={len(z)}")

    closes = {a: load_price_csv(DESK / fn) for a, fn in PRICES.items()}

    # ── 1) BUCKET both orientations + tails (mutlak fwd-21g) ──
    print("\n" + "=" * 96)
    print("  1) BUCKET — spread z-quintile + tails -> forward 21g mutlak getiri")
    print("=" * 96)
    for label, sig in (("z(spread)", z), ("z(d10 spread)", dz)):
        qs = [(-99, -2), (-2, -1), (-1, 1), (1, 2), (2, 99)]
        print(f"  {label:<14}{'bucket':<12}{'n':>6}{'SPX fwd21':>12}{'%neg':>7}{'NDX fwd21':>12}{'%neg':>7}")
        for lo, hi in qs:
            row = [f"[{lo},{hi})", 0, np.nan, np.nan, np.nan, np.nan]
            for ai, a in enumerate(PRICES):
                cl = closes[a]
                idx = sig.index.intersection(cl.index)
                cb = cl.reindex(idx)
                f21 = (cb.shift(-21) / cb - 1).reindex(idx)
                m = (sig.reindex(idx) >= lo) & (sig.reindex(idx) < hi)
                sub = f21[m].dropna()
                if ai == 0:
                    row[1] = len(sub); row[2] = sub.mean(); row[3] = (sub < 0).mean()
                else:
                    row[4] = sub.mean(); row[5] = (sub < 0).mean()
            print(f"  {'':<14}{row[0]:<12}{row[1]:>6}{100*row[2]:>+11.1f}%{100*row[3]:>+6.0f}%"
                  f"{100*row[4]:>+11.1f}%{100*row[5]:>+6.0f}%")
    # mutlak seviye tail'leri
    print(f"\n  mutlak seviye:  spread<0bps  n={int((sp<0).sum())} | spread>+100  n={int((sp>100).sum())}"
          f" | spread>+150  n={int((sp>150).sum())}")
    for a in PRICES:
        cl = closes[a]
        idx = sp.index.intersection(cl.index)
        cb = cl.reindex(idx); f21 = (cb.shift(-21)/cb - 1)
        for tag, m in (("spread<0", sp.reindex(idx) < 0), (">+100", sp.reindex(idx) > 100), (">+150", sp.reindex(idx) > 150)):
            sub = f21[m].dropna()
            if len(sub) > 5:
                print(f"    {a} {tag:<10} n={len(sub):>4}  fwd21 {100*sub.mean():+.1f}%  ({100*(sub<0).mean():.0f}%neg)")

    # ── 2) EVENT-STUDY ──
    print("\n" + "=" * 96)
    print("  2) EVENT-STUDY — z esik-gecisleri sonrasi forward getiri")
    print("=" * 96)
    for tag, cross in (("z cross < -1.5 (stress)", (z < -1.5) & (z.shift(1) >= -1.5)),
                       ("z cross > +1.5 (mania)", (z > 1.5) & (z.shift(1) <= 1.5))):
        ev = z.index[cross]
        print(f"  {tag}: {len(ev)} olay")
        for a in PRICES:
            cb = closes[a]
            outs = []
            for h in (5, 21, 42):
                rr = []
                for d in ev:
                    p0 = cb.asof(d); p1 = cb.asof(d + pd.Timedelta(days=int(h * 1.45)))
                    if p0 and p1 and p0 > 0:
                        rr.append(p1/p0 - 1)
                outs.append((h, np.mean(rr) if rr else np.nan, np.mean([x < 0 for x in rr]) if rr else np.nan))
            print(f"    {a}: " + "  ".join(f"{h}g {100*m:+.1f}% ({100*n:.0f}%neg)" for h, m, n in outs))

    # ── 3) INCREMENTAL over TIDE (2019+) ──
    print("\n" + "=" * 96)
    print("  3) INCREMENTAL over TIDE (2019+) — strict BH-FDR {SPX,NDX} (gec = ikisi de P(v>b)>=95%)")
    print("=" * 96)
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    zt = z.reindex(idx, method="ffill")
    dzt = dz.reindex(idx, method="ffill")
    spt = sp.reindex(idx, method="ffill")
    variants = {
        "L1 flat z<-1.5 (stress)":  (zt >= -1.5).astype(float),
        "L2 soft z<-1 k.5 f.4":     (1.0 - 0.5*np.clip(-zt - 1.0, 0, 3)).clip(0.4, 1.0),
        "L3 flat spread<0":         (spt >= 0).astype(float),
        "R1 flat z>+2 (mania)":     (zt <= 2.0).astype(float),
        "R2 soft z>+1.5 k.5 f.4":   (1.0 - 0.5*np.clip(zt - 1.5, 0, 3)).clip(0.4, 1.0),
        "R3 flat spread>+120":      (spt <= 120).astype(float),
        "D1 flat dz<-2 (unwind)":   (dzt >= -2.0).astype(float),
    }
    bases = {a: strat_ret(tdir, prices[a]) for a in PRICES}
    print(f"  base tide: SPX {_sh(bases['SPX']):+.3f} / NDX {_sh(bases['NDX']):+.3f}")
    print(f"  {'variant':<26}{'SPX dSh':>9}{'SPX P':>7}{'NDX dSh':>9}{'NDX P':>7}{'SPX dd':>8}{'NDX dd':>8}{'FDR':>6}")
    keep = {}
    for label, vfac in variants.items():
        res = {}
        for a in PRICES:
            v = strat_ret((tdir * vfac.reindex(idx, method="ffill").fillna(1.0)).reindex(idx), prices[a])
            res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]), _dd(v), v)
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in PRICES) else "—"
        keep[label] = (vfac, res)
        print(f"  {label:<26}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}"
              f"{100*res['SPX'][2]:>+7.0f}%{100*res['NDX'][2]:>+7.0f}%{both:>6}")

    # ── 4) INCREMENTAL over FULL STACK + epizot cetveli ──
    print("\n" + "=" * 96)
    print("  4) FULL STACK marjinal (tide x froth x GEX-shield + basis-gate) + epizot Sharpe")
    print("=" * 96)
    cor = pd.read_parquet(ROOT/"data"/"cache"/"corr_pc.parquet")["COR1M"].dropna()
    gex = pd.read_parquet(ROOT/"data"/"cache"/"squeeze_dix_gex.parquet")["gex"].dropna()
    froth = froth_factor_series(cor.reindex(idx, method="ffill"), 8, 11, 0.0)
    zg = gex_zscore(gex).reindex(idx, method="ffill")
    shield = shield_factor_series(zg, 0.5, 1.0, 0.4)
    stack = tdir * froth * shield
    stack_r = {a: strat_ret(stack.reindex(idx), prices[a]) for a in PRICES}
    print(f"  stack base: SPX {_sh(stack_r['SPX']):+.3f} (dd {100*_dd(stack_r['SPX']):+.0f}%) / "
          f"NDX {_sh(stack_r['NDX']):+.3f} (dd {100*_dd(stack_r['NDX']):+.0f}%)")
    print(f"  {'stack + variant':<26}{'SPX dSh':>9}{'SPX P':>7}{'NDX dSh':>9}{'NDX P':>7}{'SPX dd':>8}{'NDX dd':>8}{'FDR':>6}")
    stack_v = {}
    for label, (vfac, _) in keep.items():
        res = {}
        for a in PRICES:
            v = strat_ret((stack * vfac.reindex(idx, method="ffill").fillna(1.0)).reindex(idx), prices[a])
            res[a] = (paired_win_prob(stack_r[a], v), _sh(v) - _sh(stack_r[a]), _dd(v), v)
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in PRICES) else "—"
        stack_v[label] = res
        print(f"  {label:<26}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}"
              f"{100*res['SPX'][2]:>+7.0f}%{100*res['NDX'][2]:>+7.0f}%{both:>6}")

    print("\n  epizot SPX Sharpe (stack vs stack+variant):")
    hdr = "  " + f"{'epizot':<22}" + f"{'stack':>8}" + "".join(f"{k.split()[0]:>8}" for k in stack_v)
    print(hdr)
    for name, a, b in EPISODES:
        line = f"  {name:<22}{_ep(stack_r['SPX'], a, b):>+8.2f}"
        for k, res in stack_v.items():
            line += f"{_ep(res['SPX'][3], a, b):>+8.2f}"
        print(line)
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
