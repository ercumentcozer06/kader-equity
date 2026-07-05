"""screen/candidate_econ — #9 ekonomik momentum (free CESI yok → FRED WEI weekly economic index proxy).
WEI = real-activity nowcast. z(WEI) + z(WEI değişim) tide overlay, iki yön, strict FDR. (Tide-M0 ile redundant beklenir.)"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from dotenv import load_dotenv                           # noqa: E402
load_dotenv(Path(r"C:\Users\admin\Downloads\kader-macro") / ".env")
KEY = os.environ.get("FRED_API_KEY")
from spine import contract as C, tide as T              # noqa: E402
from backtest import engine as E                         # noqa: E402
from screen._util import paired_win_prob, fdr_bh         # noqa: E402


def fred(sid, start="2008-01-01"):
    r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                     params={"series_id": sid, "api_key": KEY, "file_type": "json", "observation_start": start}, timeout=30)
    o = r.json().get("observations", [])
    return pd.Series({pd.Timestamp(x["date"]): float(x["value"]) for x in o if x["value"] not in (".", "")}).sort_index()


def z(s, win): return (s - s.rolling(win, min_periods=win//3).mean()) / s.rolling(win, min_periods=win//3).std()
def _sh(r): r = r.dropna(); return float(r.mean()/r.std()*np.sqrt(252)) if (len(r) > 20 and r.std() > 0) else float("nan")


def strat_ret(pos, close, lag=1):
    idx = pos.index; ret = E.fwd_ret(close, idx).values; p = pos.astype(float).values
    if lag:
        p = np.concatenate([np.zeros(lag), p[:-lag]])
    return pd.Series(p * ret, index=idx).dropna()


def main():
    try:
        wei = fred("WEI"); print(f"  WEI: {len(wei)} hafta {wei.index.min().date()}..{wei.index.max().date()}  son {wei.iloc[-1]:.2f}")
    except Exception as e:
        print(f"  WEI fetch hata: {e}"); return 1
    wei_lvl = z(wei, 104); wei_chg = z(wei.diff(4), 104)   # seviye + momentum
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    tidx = tdir.index
    bases = {a: strat_ret(tdir, prices[a]) for a in ("SPX", "NDX")}
    print(f"  base SPX {_sh(bases['SPX']):+.3f}/NDX {_sh(bases['NDX']):+.3f}   (PIT-lag +1hafta)")
    print(f"  {'kural':<26}{'SPX ΔSh':>9}{'SPX P':>7}{'NDX ΔSh':>9}{'NDX P':>7}{'FDR':>6}")
    for nm, sig in (("WEI-level", wei_lvl), ("WEI-momentum", wei_chg)):
        st = sig.copy(); st.index = st.index + pd.Timedelta(days=7)   # PIT publish lag
        st = st.reindex(tidx, method="ffill")
        for sgn, ys in ((+1, "düşük→trim"), (-1, "yüksek→trim")):     # zayıf-aktivite→de-risk
            fac = (1 - 0.5*np.clip(-sgn*st - 1, 0, 3)).clip(0.4, 1)
            res = {}
            for a in ("SPX", "NDX"):
                v = strat_ret((tdir * fac).reindex(tidx), prices[a])
                res[a] = (paired_win_prob(bases[a], v), _sh(v) - _sh(bases[a]))
            passed = fdr_bh({a: 1.0 - res[a][0] for a in res if res[a][0] is not None}, alpha=0.05)
            both = "PASS" if all(passed.get(a, False) for a in ("SPX", "NDX")) else "—"
            print(f"  {nm+' '+ys:<26}{res['SPX'][1]:>+9.2f}{res['SPX'][0]:>7.0%}{res['NDX'][1]:>+9.2f}{res['NDX'][0]:>7.0%}{both:>6}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
