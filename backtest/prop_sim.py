"""
backtest/prop_sim — GÖREV 2. Frozen TIDE kitabı (deployed stack = tide×COR1M-froth×GEX-shield) FTMO Swing
2-Step kural setinden geçiyor mu? Rolling-start EXHAUSTIVE (her başlangıç tarihinden ileri koş, pass/kill'e dek).

FTMO Swing 2-Step (2026-06, ftmo.com/trading-objectives doğrulandı):
  Faz1 hedef +%10, Faz2 +%5; günlük kayıp −%5 (00:00 CET reset, initial-bakiye bazlı mutlak); toplam −%10
  STATİK (initial); faz başına min 4 işlem günü; süre limiti yok; swing = hafta-sonu/haber serbest, consistency yok.
KISIT (dürüst): günlük seri (close-to-close) → INTRADAY günlük-limit ihlali GÖRÜLMEZ → kill SAYISI az tahmin,
  pass olasılığı YUKARI-yanlı. Granülarite sınırı; gerçek için intraday gerekir.

Parametreler (a priori / parametrik): pozisyon taraması 0.4–1.0×0.1 (SIZING taraması, sinyal-fit DEĞİL);
  swap drag %3/yıl (taşınan notional, taşınan gün); profit split %80; eval fee $170 (FTMO 10k Swing ~, FLAG).
  +1g exec lag korunur. Yöntem birincil = rolling-start exhaustive; blok-bootstrap ikincil (eklenebilir).

  & <venv python> backtest/prop_sim.py
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
from modules.cor1m_froth import froth_factor_series   # noqa: E402
from modules.gex_shield import gex_zscore, shield_factor_series  # noqa: E402

# FTMO Swing 2-Step (parametrik — web-doğrulandı 2026-06)
P1_TARGET, P2_TARGET = 0.10, 0.05
DAILY_LIM = 0.05          # initial'in %5'i, MUTLAK (gün-başı bakiye − equity ≥ 0.05 → kill)
TOTAL_LIM = 0.10          # initial'in %10'u statik → equity ≤ 0.90 → kill
MIN_DAYS = 4              # faz başına min işlem günü
SWAP_YR = 0.03           # %/yıl taşıma drag (parametrik)
PROFIT_SPLIT = 0.80      # trader payı (parametrik)
EVAL_FEE_USD = 170.0     # FTMO 10k Swing ~ (FLAG: parametrik, güncel fiyatla doğrula)
FUNDED_USD = 10000.0     # 10k funded (payout $ için)
POS_GRID = [round(x, 1) for x in np.arange(0.4, 1.0001, 0.1)]


def _stack_pos():
    scores, prices, vector, _ = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector)); idx = tdir.index
    cor = pd.read_parquet(ROOT/"data"/"cache"/"corr_pc.parquet")["COR1M"].dropna()
    gex = pd.read_parquet(ROOT/"data"/"cache"/"squeeze_dix_gex.parquet")["gex"].dropna()
    froth = froth_factor_series(cor.reindex(idx, method="ffill"), 8, 11, 0.0)
    zg = gex_zscore(gex).reindex(idx, method="ffill")
    shield = shield_factor_series(zg, 0.5, 1.0, 0.4)
    pos = (tdir * froth * shield).reindex(idx)        # deployed book exposure ∈ [0,1]
    return pos, prices, idx


def book_returns(book: str, pos_override: pd.Series | None = None):
    """Günlük kitap getirisi (1× pos-scale öncesi) + holding maskesi. +1g lag (strat_ret)."""
    pos, prices, idx = _stack_pos()
    if pos_override is not None:
        pos = pos_override.reindex(idx)
    def sr(asset):
        ret = E.fwd_ret(prices[asset], idx)            # t->t+1 (look-ahead-free)
        p = pos.astype(float).values
        p = np.concatenate([[0.0], p[:-1]])            # +1g lag
        return p * ret, p                              # (account_ret_1x, lagged_pos)
    if book == "SPX":
        r, p = sr("SPX")
    elif book == "NDX":
        r, p = sr("NDX")
    else:                                              # 50/50
        rS, pS = sr("SPX"); rN, pN = sr("NDX")
        r = 0.5*np.nan_to_num(rS) + 0.5*np.nan_to_num(rN); p = 0.5*(pS+pN)
    df = pd.DataFrame({"ret1x": r, "pos": p}, index=idx).dropna()
    return df


def _sim_phase(rets, opens, start, target):
    """Tek faz: (status, end_idx, reason). status ∈ pass/kill/incomplete."""
    eq = 1.0; tdays = 0; i = start
    n = len(rets)
    while i < n:
        prev = eq
        eq = eq * (1.0 + rets[i])
        if opens[i] > 0:
            tdays += 1
        if (prev - eq) >= DAILY_LIM:                   # günlük −%5 mutlak (gün-başı − equity)
            return ("kill", i, "daily")
        if eq <= (1.0 - TOTAL_LIM):                    # toplam −%10 statik
            return ("kill", i, "total")
        if eq >= (1.0 + target) and tdays >= MIN_DAYS:
            return ("pass", i, None)
        i += 1
    return ("incomplete", i, None)


def run_book(book: str, pos_scale: float, pos_override=None):
    df = book_returns(book, pos_override)
    swap_daily = SWAP_YR / 252.0
    # pos-scale + swap drag (taşınan notional)
    rets = (pos_scale * df["ret1x"].values) - (swap_daily * pos_scale * df["pos"].values)
    opens = (pos_scale * df["pos"].values)
    idx = df.index
    n = len(rets)
    passes, p1pass, kills, days = 0, 0, [], []
    starts = 0
    for s in range(n):
        # Faz 1
        st1, e1, why1 = _sim_phase(rets, opens, s, P1_TARGET)
        if st1 == "incomplete":
            continue                                   # ileri veri yetmiyor → başlangıç sayılmaz
        starts += 1
        if st1 == "kill":
            kills.append((str(idx[e1].date()), "P1-"+why1)); continue
        p1pass += 1
        # Faz 2 (taze hesap, e1+1'den)
        st2, e2, why2 = _sim_phase(rets, opens, e1+1, P2_TARGET)
        if st2 == "incomplete":
            continue
        if st2 == "kill":
            kills.append((str(idx[e2].date()), "P2-"+why2)); continue
        passes += 1
        days.append(e2 - s)
    pp = passes/starts if starts else float("nan")
    p1 = p1pass/starts if starts else float("nan")
    return {"book": book, "pos": pos_scale, "n_starts": starts,
            "p1_pass": p1, "p2_cond": (passes/p1pass if p1pass else float("nan")),
            "total_pass": pp, "avg_days": (float(np.mean(days)) if days else None),
            "median_days": (float(np.median(days)) if days else None),
            "kills": kills}


def _annual_book_return(book, pos_scale):
    """Funded fazda beklenen yıllık kitap getirisi (pos-scale + swap sonrası, frozen örnek annualize)."""
    df = book_returns(book)
    swap_daily = SWAP_YR/252.0
    r = pos_scale*df["ret1x"].values - swap_daily*pos_scale*df["pos"].values
    return float(np.nanmean(r)*252)


def main():
    print("=" * 100)
    print("  GÖREV 2 — FTMO Swing 2-Step KAPI SİMÜLASYONU (frozen TIDE kitap, rolling-start exhaustive)")
    print("  KISIT: close-to-close → intraday günlük-limit ihlali görülmez → kill az tahmin / pass yukarı-yanlı")
    print("=" * 100)
    allres = {}
    for book in ("SPX", "NDX", "50/50"):
        print(f"\n  [{book}]   {'pos':>4}{'n_start':>9}{'P1%':>7}{'P2|P1%':>8}{'TOPLAM%':>9}{'ort-gün':>9}{'med-gün':>9}"
              f"{'kill-daily':>11}{'kill-total':>11}")
        for ps in POS_GRID:
            res = run_book(book, ps)
            allres[(book, ps)] = res
            kd = sum(1 for _, w in res["kills"] if w.endswith("daily"))
            kt = sum(1 for _, w in res["kills"] if w.endswith("total"))
            print(f"       {ps:>4.1f}{res['n_starts']:>9}{100*res['p1_pass']:>6.0f}%{100*res['p2_cond']:>7.0f}%"
                  f"{100*res['total_pass']:>8.0f}%{(res['avg_days'] or 0):>9.0f}{(res['median_days'] or 0):>9.0f}{kd:>11}{kt:>11}")
        ann = _annual_book_return(book, 1.0)
        print(f"       funded yıllık kitap getirisi (1.0×): {100*ann:+.1f}% → payout(80%, $10k): ${FUNDED_USD*ann*PROFIT_SPLIT:,.0f}/yıl")
    print("\n" + "=" * 100)
    print(f"  payout ekonomisi: eval-fee ${EVAL_FEE_USD:.0f} (FLAG parametrik). 'eval-fee başına beklenen $' =")
    print(f"  total_pass × funded-yıllık-payout$ − eval-fee. Önerilen config + tam kill-listesi: rapor bölümünde.")
    return allres


if __name__ == "__main__":
    main()
