"""
reproduce_baseline — FAITHFUL HARNESS: donmuş tide spine, kader-macro İMPORT'SUZ + AĞ'SIZ,
çapa Sharpe SPX 1.43 / NQ 1.49'u byte-yakın reprodüce ediyor mu?

Akış: spine/frozen (module_scores + vector) → tide_score_series → tide_dir_series → engine
(+1g lag) → SPX/NDX Sharpe == provenance.anchor. Bu, runtime'ın canlı kader-macro'ya bağlı
OLMADAN spine'ı ürettiğini kanıtlar (kader-btc reproduce_baseline disiplini).

  python backtest/reproduce_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from spine import contract as C, tide as T          # noqa: E402
from backtest import engine as E                     # noqa: E402


def main() -> int:
    scores, prices, vector, prov = C.read_frozen()
    score = T.tide_score_series(scores, vector)
    tdir = T.tide_dir_series(score)

    win = prov.get("window", {})
    print("=" * 84)
    print(f"  KADER-EQUITY — FAITHFUL HARNESS | frozen tide spine, NO kader-macro / NO network")
    print(f"  window {win.get('start')}..{win.get('end')}  ({win.get('n_days')} gün)  "
          f"vector: {[ (k, round(v,3)) for k,v in sorted(vector.items(), key=lambda t:-abs(t[1])) if abs(v)>1e-9 ]}")
    print("=" * 84)
    print(f"  {'asset':<7}{'Sharpe':>9}{'maxDD':>8}{'CVaR':>9}{'total':>9}{'expo':>7}   {'anchor':>8}{'  match':>8}")

    ok = True
    for a in ("SPX", "NDX"):
        res = E.backtest_dir(tdir, prices[a], lag=1)
        st = res["strat"]
        anc = float(prov["anchor"][a]["sharpe"])
        match = bool(np.isclose(round(st["sharpe"], 3), anc, atol=0.002))
        ok = ok and match
        print(f"  {a:<7}{st['sharpe']:>+9.3f}{100*st['maxdd']:>+7.0f}%{100*st['cvar']:>+8.1f}%"
              f"{100*st['total']:>+8.0f}%{100*res['expo']:>+6.0f}%   {anc:>+8.3f}{('PASS' if match else 'FAIL'):>8}")

    print("=" * 84)
    print(f"  {'[OK] tide spine reprodüce edildi (1.43/1.49) — runtime self-contained' if ok else '[!] FAIL — incele'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
