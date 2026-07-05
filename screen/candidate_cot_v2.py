"""
screen/candidate_cot_v2 — COT'u PRO'ların yöntemiyle (Emir 2026-06-09 push). z-score DEĞİL:
  • COT INDEX (Williams): net'in 3yıl(156hafta) aralığındaki yeri 0-100 (pro standardı)
  • Commercials (smart money) + Non-commercials/specs (dumb money), legacy rapor
  • ÇOKLU HORIZON 21/42/63g (COT yavaş sinyal)
  • İKİ UÇ contrarian (spec-index yüksek=kalabalık long=bearish; comm-index yüksek=smart-long=bullish)

1) BUCKET: COT-index 5-kova → fwd 21/42/63g (eşleşen endeks). Uçlar keskin mi?
2) INCREMENTAL over tide: revealed yönde de-risk, çoklu horizon-mantığı, strict BH-FDR.
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


def cot_index(net, win=156):
    lo = net.rolling(win, min_periods=52).min()
    hi = net.rolling(win, min_periods=52).max()
    return ((net - lo) / (hi - lo) * 100).clip(0, 100)


def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def main():
    cot = pd.read_parquet(ROOT / "data" / "cache" / "cot_legacy.parquet")
    cot.index = cot.index + pd.Timedelta(days=3)          # PIT publish lag
    ci = {c: cot_index(cot[c].dropna()) for c in cot.columns}

    # ── 1) BUCKET: COT-index → fwd 21/42/63g ──
    print("=" * 104)
    print("  1) COT INDEX (0-100, 3yıl aralık) → forward getiri (eşleşen endeks). Uçlar (0-20 / 80-100) keskin mi?")
    print("=" * 104)
    SIG = {"SPX": ("ES_comm_net", "ES_spec_net"), "NDX": ("NQ_comm_net", "NQ_spec_net")}
    edges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]
    for a, fn in PRICES.items():
        close = load_price_csv(DESK / fn)
        for sig in SIG[a]:
            s = ci[sig].dropna()
            idx = s.index.intersection(close.index)
            cb = close.reindex(idx, method="ffill")
            row = []
            for h in (21, 42, 63):
                fh = (cb.shift(-h) / cb - 1).reindex(idx)
                sv = s.reindex(idx)
                vals = [100 * fh[(sv >= lo) & (sv < hi)].mean() for lo, hi in edges]
                row.append(vals)
            nm = sig.replace("_net", "")
            print(f"  {nm:<12} h21: " + " ".join(f"{v:>+5.1f}" for v in row[0]) +
                  f"   h42: " + " ".join(f"{v:>+5.1f}" for v in row[1]) +
                  f"   h63: " + " ".join(f"{v:>+5.1f}" for v in row[2]))
    print("  (kovalar 0-20|20-40|40-60|60-80|80-100. comm yüksek=smart-long; spec yüksek=kalabalık-long=bearish bekle.)")

    # ── 2) INCREMENTAL over tide (2019+): revealed yönlerde, strict FDR ──
    print("\n" + "=" * 104)
    print("  2) INCREMENTAL over TIDE (2019+): de-risk uç COT-index'te. STRICT BH-FDR {SPX,NDX}")
    print("=" * 104)
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    print(f"  base SPX {_sh(bases['SPX']):+.3f}/NDX {_sh(bases['NDX']):+.3f}")
    print(f"  {'kural':<30}{'SPX ΔSh':>9}{'SPX P':>7}{'NDX ΔSh':>9}{'NDX P':>7}{'FDR':>6}")
    rules = {
        "spec-index>80 → flat":     (("ES_spec_net", "NQ_spec_net"), lambda v: (v <= 80).astype(float)),
        "spec-index>80 → trim.5":   (("ES_spec_net", "NQ_spec_net"), lambda v: np.where(v > 80, 0.5, 1.0)),
        "comm-index<20 → flat":     (("ES_comm_net", "NQ_comm_net"), lambda v: (v >= 20).astype(float)),
        "spec-idx soft (lin 60-90)": (("ES_spec_net", "NQ_spec_net"), lambda v: (1 - np.clip((v-60)/30, 0, 1)*0.6)),
    }
    for label, (sigs, fn) in rules.items():
        res = {}
        for a, sigc in (("SPX", sigs[0]), ("NDX", sigs[1])):
            fac = pd.Series(fn(ci[sigc].reindex(idx, method="ffill").values), index=idx)
            v = strat_ret((tdir * fac).reindex(idx), prices[a])
            res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]))
        passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
        both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
        print(f"  {label:<30}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}{both:>6}")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
