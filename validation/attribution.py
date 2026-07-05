"""
validation/attribution — GÖREV 6a. m9 vektörün ~%56'sı; çürürse kompozit Sharpe'tan fark etmek ~1 yıl sürer.
Modül-bazlı erken-uyarı: dominant modüllerin (m9/m5/m2) skoru + rolling-60g corr(skor, ertesi-gün-getiri).

İKİ kanal:
  • TARİHSEL BANT (frozen module_scores + prices, locked/reproducible): her modül için 60g-rolling corr serisi
    → bandı (mean, p5, p95, işaret) ve modül-skorunun tarihsel dağılımı (p1/p99 uç).
  • CANLI (live_tide_latest.json): bugünkü m9/m5/m2 skoru. ALARM: (a) skor tarihsel [p1,p99] dışına çıkarsa
    (eşi-görülmemiş bölge), (b) ileride forward-ledger 60g+ gerçekleşen getiri biriktirince rolling-corr
    İŞARET değiştirir ya da banttan düşerse. Şimdilik corr "frozen-son (05-22) itibarıyla" + canlı skor-uç alarmı.

NOT: corr forward gerçekleşen getiri gerektirir (1g lag) → bugün corr frozen-son değeridir; forward-ledger
dolunca canlıya genişler. Hızlı/bugün-çalışan alarm = canlı skor vs tarihsel skor-bandı.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODS = ["m9", "m5", "m2"]            # dominant ağırlıklar (0.563/0.214/0.118); m9 ucuzsa m5/m2 de izlenir
CORR_WIN = 126                      # rolling-corr penceresi (g) — a priori ~6ay; 60g'lik pencere 21g-getiri ile
#                                     yalnız ~3 bağımsız gözlem içerir (gürültü); 126g daha kararlı.
HORIZON = 21                        # PRİMER corr horizonu (g) — A PRİORİ: tide YAVAŞ sinyal (GÖREV 2: edge
#                                     gün 2-45'te, gün-1 NEGATİF; LONG-run ort 35g) → m9'un edge'i ~21g'de,
#                                     1g'de DEĞİL. [[feedback_slow_signal_horizon]]. 1g referans olarak da raporlanır.
LIVE_CACHE = ROOT / "data" / "cache" / "live_tide_latest.json"
REF_ASSET = "SPX"


def _fwd_ret(close: pd.Series, h: int) -> pd.Series:
    return close.pct_change(h).shift(-h)             # h-gün ileri getiri (look-ahead-free)


def _hist_corr(scores: pd.DataFrame, fwd: pd.Series, m: str, win: int = CORR_WIN) -> pd.Series:
    return scores[m].astype(float).rolling(win).corr(fwd)


def evaluate() -> dict:
    """Döndürür {modules: {m: {...}}, asof_frozen, asof_live}. Frozen bant + canlı skor-uç alarmı.
    Corr PRİMER horizon=21g (yavaş sinyal); 1g referans için de hesaplanır."""
    from spine import contract as C
    scores, prices, _vec, _prov = C.read_frozen()
    close = prices[REF_ASSET].reindex(scores.index, method="ffill")
    fwdH = _fwd_ret(close, HORIZON)                   # 21g ileri getiri (primer)
    fwd1 = _fwd_ret(close, 1)                         # 1g referans

    live = json.loads(LIVE_CACHE.read_text(encoding="utf-8")) if LIVE_CACHE.exists() else {}
    live_scores = (live.get("scores_row") or {})

    out = {}
    for m in MODS:
        if m not in scores.columns:
            continue
        corr = _hist_corr(scores, fwdH, m).dropna()    # primer: 21g
        corr1 = _hist_corr(scores, fwd1, m).dropna()   # referans: 1g
        sc = scores[m].astype(float).dropna()
        live_s = live_scores.get(m)
        # canlı skorun tarihsel yüzdelik konumu
        pctile = float((sc < live_s).mean() * 100) if live_s is not None else None
        score_p1, score_p99 = float(sc.quantile(0.01)), float(sc.quantile(0.99))
        score_extreme = (live_s is not None and (live_s < score_p1 or live_s > score_p99))
        cmean = float(corr.mean())
        c_p5, c_p95 = float(corr.quantile(0.05)), float(corr.quantile(0.95))
        last_corr = float(corr.iloc[-1]) if len(corr) else None
        # CANLI corr (forward gerçekleşen getiri gerektirir) HENÜZ YOK → frozen-son = in-sample seed.
        # corr-bant/işaret alarmı yalnız FORWARD canlı corr gelince ANLAMLI (seed'in uçluğunu işaretlemek
        # yanlış-alarm). Bugün çalışan alarm = SKOR-uç (canlı skor vs tarihsel bant). corr = bilgilendirici.
        live_corr = None                              # forward-ledger 60g+ dolunca hesaplanacak
        corr_band_exit = (live_corr is not None and (live_corr < c_p5 or live_corr > c_p95))
        out[m] = {
            "live_score": (None if live_s is None else round(float(live_s), 3)),
            "score_pctile": (None if pctile is None else round(pctile, 1)),
            "score_band_p1_p99": [round(score_p1, 2), round(score_p99, 2)],
            "score_extreme": bool(score_extreme),
            "hist_corr_mean_21d": round(cmean, 3),
            "hist_corr_band_p5_p95_21d": [round(c_p5, 3), round(c_p95, 3)],
            "last_corr_frozen_21d": (None if last_corr is None else round(last_corr, 3)),
            "hist_corr_mean_1d": (round(float(corr1.mean()), 3) if len(corr1) else None),
            "live_corr_21d": live_corr,
            "corr_alarm_status": "dormant (forward 60g+ gerçekleşen getiri gerekir)",
            "alarm": bool(score_extreme or corr_band_exit),   # bugün = skor-uç; corr-bant forward'da aktif
        }
    return {"modules": out, "asof_frozen": str(scores.index[-1].date()),
            "asof_live": live.get("as_of"), "corr_win": CORR_WIN, "horizon": HORIZON, "ref": REF_ASSET}


def render(att: dict | None = None) -> str:
    att = att or evaluate()
    L = [f"  MODÜL-ATTRIBUTION (corr-{att['corr_win']}g × {att['horizon']}g-getiri {att['ref']}; "
         f"frozen-son {att['asof_frozen']}, canlı {att['asof_live']})"]
    for m, d in att["modules"].items():
        flag = " ⚠ALARM" if d["alarm"] else ""
        L.append(f"    {m}: skor {d['live_score']} (tarihsel %{d['score_pctile']}, bant {d['score_band_p1_p99']}) | "
                 f"corr21g-ort {d['hist_corr_mean_21d']:+.2f} bant {d['hist_corr_band_p5_p95_21d']} "
                 f"son {d['last_corr_frozen_21d']} (1g-ref {d['hist_corr_mean_1d']}){flag}")
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(render())
