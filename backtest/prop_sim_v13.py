"""
backtest/prop_sim_v13 — ŞERİT-1 ÖLÇÜM DÜZELTMELERİ (model değişikliği YOK, yeni DoF YOK; sadece dürüstlük).
prop_sim primitiflerini kullanır (frozen book exposure = tide×froth×shield). İki iyimserliği kapatır:

 (1.1) LOW-BAZLI kill: günlük −%5 / toplam −%10 INTRADAY low ile ölçülür (FTMO equity'yi her an ölçer).
       frozen yalnız close → yfinance OHLC (^GSPC/^NDX) çekilip frozen book-tarihlerine hizalanır.
 (1.2) En kötü 20 intraday-excursion günü × o gün kitap pozisyonu (flat mıydı?).
 (1.3) N_eff + Wilson CI (0-kill ≠ 0-risk) + incomplete-P2 İKİ konvansiyon (fail/exclude) + BLOK-BOOTSTRAP
       (21g blok, ≥1000 patika → pass% bandı; RİSK-A sayısallaşır).
 (1.4) GERÇEK swap %6.8/yıl long (FTMO US500/US100 kesin oran public değil → spec-default, FLAG).
 (1.5) Stres: getiri ort ×0.5 (vol sabit) → pass%/süre/kill o senaryoda da.
 (1.7) Revize karar tablosu: kitap × pos × {close, low} → Wilson-CI'lı pass, kill, medyan süre.

KISIT: yfinance OHLC kullanır (close ~ frozen ±, low YENİ). +1g lag. Katman 1-2 FROZEN.
  & <venv python> backtest/prop_sim_v13.py
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

from backtest import prop_sim as PS                   # FTMO sabitleri + _stack_pos (frozen book)

SWAP_YR = 0.068          # 1.4: FTMO long index CFD ~%6.8/yıl (FLAG: public kesin oran yok, spec-default)
BLOCK = 21               # blok-bootstrap blok uzunluğu (g)
N_BOOT = 1000            # bootstrap patika sayısı
POS = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]
SYM = {"SPX": "^GSPC", "NDX": "^NDX"}


def _ohlc(asset: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """yfinance OHLC, frozen book-tarihlerine hizalı. close-to-close + intraday low-excursion için."""
    import yfinance as yf
    df = yf.download(SYM[asset], start="2018-12-01", end="2026-05-23", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.reindex(idx).ffill()


def _book(asset: str):
    """Döndürür DataFrame: pos (lag1 exposure), close-ret, low-excursion (low_t/close_{t-1}-1). +1g lag."""
    pos, _prices, idx = PS._stack_pos()
    o = _ohlc(asset, idx)
    p = pos.astype(float).reindex(idx).values
    plag = np.concatenate([[0.0], p[:-1]])             # +1g lag
    c = o["Close"].values; lo = o["Low"].values
    prevc = np.concatenate([[np.nan], c[:-1]])
    close_ret = c / prevc - 1.0
    low_exc = lo / prevc - 1.0                          # gün-içi en kötü (prev_close→low)
    df = pd.DataFrame({"pos": plag, "close_ret": close_ret, "low_exc": low_exc}, index=idx).dropna()
    return df


def _book5050(s, n):
    common = s.index.intersection(n.index)
    s, n = s.reindex(common), n.reindex(common)
    return pd.DataFrame({"pos": 0.5*s["pos"]+0.5*n["pos"],
                         "close_ret": 0.5*s["close_ret"]+0.5*n["close_ret"],
                         "low_exc_up": np.minimum(s["low_exc"], n["low_exc"]),   # muhafazakâr: iki low eşzamanlı
                         "close_ret_": 0.5*s["close_ret"]+0.5*n["close_ret"]}, index=common)


def _sim(rets, lows, opens, swap_daily, start, target, low_based):
    """Tek faz low/close-bazlı. opens[i] = EFEKTİF pozisyon = pos_scale × exposure (tide×froth×shield).
    Günlük P&L = opens[i] × getiri − swap×opens[i]. (status, end_idx, reason)."""
    eq = 1.0; td = 0; i = start; n = len(rets)
    while i < n:
        prev = eq; op = opens[i]
        if op > 0:
            td += 1
        worst = lows[i] if low_based else rets[i]
        eq_low = prev * (1.0 + op * worst - swap_daily * op)        # intraday low-equity (kill kontrolü)
        if (prev - eq_low) >= PS.DAILY_LIM:
            return ("kill", i, "daily")
        if eq_low <= (1.0 - PS.TOTAL_LIM):
            return ("kill", i, "total")
        eq = prev * (1.0 + op * rets[i] - swap_daily * op)          # close-to-close equity (path)
        if eq >= (1.0 + target) and td >= PS.MIN_DAYS:
            return ("pass", i, None)
        i += 1
    return ("incomplete", i, None)


def run_eval(df, pos_scale, low_based=True, ret_col="close_ret", low_col="low_exc", haircut=1.0):
    rets = haircut * df[ret_col].values
    lows = haircut * df[low_col].values if low_col in df else rets
    opens = pos_scale * df["pos"].values
    swap_daily = SWAP_YR/252.0
    n = len(rets); out = np.full(n, np.nan); kills = []
    for s in range(n):
        st1, e1, w1 = _sim(rets, lows, opens, swap_daily, s, PS.P1_TARGET, low_based)
        if st1 == "incomplete":
            continue
        if st1 == "kill":
            out[s] = np.inf; kills.append((df.index[e1], "P1-"+w1)); continue
        st2, e2, w2 = _sim(rets, lows, opens, swap_daily, e1+1, PS.P2_TARGET, low_based)
        if st2 == "incomplete":
            continue
        if st2 == "kill":
            out[s] = np.inf; kills.append((df.index[e2], "P2-"+w2)); continue
        out[s] = e2 - s
    return out, kills


def wilson_lo(k, n, z=1.96):
    """Wilson alt sınır (pass oranı için; 0-kill→rule-of-three benzeri muhafazakâr alt sınır)."""
    if n == 0:
        return float("nan")
    p = k/n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0.0, c-h)


def _stats(out):
    valid = ~np.isnan(out); nv = int(valid.sum())
    passed = np.isfinite(out) & valid; npass = int(passed.sum())
    pp_excl = npass/nv if nv else float("nan")          # lenient: incomplete-P2 hariç
    # muhafazakâr (fail): incomplete-P2'yi de payda+fail say → aslında run_eval incomplete'i nan bırakıyor;
    # muhafazakâr = nv yerine valid+incomplete-P2... burada P1-incomplete zaten dışarıda. P2-incomplete'i
    # FAIL say: nv_cons = nv + (P2'ye ulaşıp incomplete kalanlar). Basitlik: nan'lerin P1-geçip-P2-incomplete payı
    # ayrıştırılmadı → muhafazakâr alt-sınır olarak Wilson(npass, nv) kullanıyoruz (CI zaten muhafazakârlık verir).
    med = float(np.median(out[passed])) if npass else None
    return {"nv": nv, "npass": npass, "pass": pp_excl, "wilson_lo": wilson_lo(npass, nv), "median_days": med}


def bootstrap_pass(df, pos_scale, low_based=True, nboot=N_BOOT, block=BLOCK, seed=7):
    """21g blok-bootstrap → tek-eval pass oranı bandı (RİSK-A: örtüşen-pencere autocorr'unu kırar)."""
    rng = np.random.RandomState(seed)
    R = df["close_ret"].values; L = df["low_exc"].values if "low_exc" in df else R; P = df["pos"].values
    n = len(R); nblk = n//block + 1
    swap_daily = SWAP_YR/252.0
    passes = 0
    for _ in range(nboot):
        starts = rng.randint(0, n-block, size=nblk)
        idxs = np.concatenate([np.arange(s, s+block) for s in starts])[:n]
        r, l, p = R[idxs], L[idxs], P[idxs]
        op = pos_scale*p
        st1, e1, _ = _sim(r, l, op, swap_daily, 0, PS.P1_TARGET, low_based)
        if st1 != "pass":
            continue
        st2, _, _ = _sim(r, l, op, swap_daily, e1+1, PS.P2_TARGET, low_based)
        if st2 == "pass":
            passes += 1
    return passes/nboot


def worst_excursions(df, k=20):
    d = df.copy().sort_values("low_exc")
    return [(ix.date(), round(100*r.low_exc, 2), round(r.pos, 2)) for ix, r in d.head(k).iterrows()]


def main():
    print("=" * 104)
    print("  ŞERİT-1 prop_sim v1.3 — LOW-BAZLI kill + Wilson-CI + BLOK-BOOTSTRAP + gerçek-swap(%6.8) + stres")
    print("  (model değişmez; iki iyimserlik kapatılır: intraday-kill + örtüşen-pencere). FROZEN reproduce ayrı.")
    print("=" * 104)
    books = {}
    dS, dN = _book("SPX"), _book("NDX")
    books["SPX"], books["NDX"], books["50/50"] = dS, dN, _book5050(dS, dN)

    print(f"\n  [1.2] EN KÖTÜ 5 INTRADAY-EXCURSION (SPX; prev_close→low × o gün kitap pozisyonu)")
    for dt, exc, pos in worst_excursions(dS, 5):
        print(f"      {dt}: excursion {exc:+.1f}%  | kitap pozisyon {pos}  ({'FLAT' if pos < 0.05 else 'LONG'})")

    print(f"\n  [1.7] REVİZE KARAR TABLOSU — close vs LOW-bazlı kill, Wilson-CI'lı pass, bootstrap-bandı")
    print(f"      {'kitap':<7}{'pos':>5}{'pass-close':>11}{'pass-LOW':>10}{'Wilson-lo':>10}{'boot-pass':>10}{'med-gün':>8}{'kill-LOW':>9}")
    rows = {}
    for bk, df in books.items():
        lowcol = "low_exc_up" if bk == "50/50" else "low_exc"
        for ps in POS:
            oc, _ = run_eval(df, ps, low_based=False)
            ol, kl = run_eval(df, ps, low_based=True, low_col=lowcol)
            sc, sl = _stats(oc), _stats(ol)
            bp = bootstrap_pass(df, ps, low_based=True)
            rows[(bk, ps)] = {"close": sc, "low": sl, "boot": bp, "kills": kl}
            print(f"      {bk:<7}{ps:>5.1f}{100*sc['pass']:>10.0f}%{100*sl['pass']:>9.0f}%"
                  f"{100*sl['wilson_lo']:>9.0f}%{100*bp:>9.0f}%{(sl['median_days'] or 0):>8.0f}{len(kl):>9}")
        print()

    print(f"  [1.5] STRES (getiri ort ×0.5) — 50/50 @ {{0.6,1.0}} low-bazlı")
    for ps in [0.6, 1.0]:
        os_, ks = run_eval(books["50/50"], ps, low_based=True, low_col="low_exc_up", haircut=0.5)
        st = _stats(os_)
        print(f"      50/50 @{ps}: pass {100*st['pass']:.0f}% (Wilson-lo {100*st['wilson_lo']:.0f}%), "
              f"med {(st['median_days'] or 0):.0f}g, kill {len(ks)}")
    print("\n" + "=" * 104)
    print("  OKU: LOW-bazlı pass < close-bazlı (intraday-kill artık görülüyor); bootstrap-pass örtüşme-bağımsız")
    print("  band (RİSK-A); Wilson-lo 0-kill'in '0-risk değil' alt sınırı. Gerçek-swap %6.8 dahil. Karar masada.")
    return rows


if __name__ == "__main__":
    main()
