"""
validation/regime_transition_lab — BULGU-1 takibi: tide-flip rejim-geçişinde Sharpe yarıya düşüyor
(+0.85 vs stabil +2.07, günlerin %33'ü). Binary kapının (score>0) düşük-conviction whipsaw'ı azaltılabilir mi?

MÜDAHALE AİLESİ (hepsi trailing/PIT, lag=1 nihai poz):
  • baseline   : dir = (score>0)                          — mevcut binary kapı
  • hysteresis : long↑ score>+δ, flat↓ score<−δ, arası HOLD (dead-band, flip-frekansı↓)
  • soft_ramp  : dir = clip(score/w, 0, 1)                — sıfır civarı KISMİ poz (binary yerine rampa)
  • debounce   : yeni tarafta N ardışık gün sonra flip    — gürültü-flip debounce
  • ema_smooth : (EMA_span(score) > 0)                     — skoru yumuşat, sonra kapıla

DİSİPLİN: (1) HER İKİ varlıkta (SPX,NDX) baseline'ı geçmeli — strict paired win-prob + BH-FDR.
(2) Kazanan δ/w/N/span TEK-NOKTA değil PLATO olmalı (knife-edge=overfit). (3) train(ilk %60)→test(son %40)
split ile OOS teyit. (4) full stack üstünde: variant_dir × cor1m_froth × gex_shield. Look-ahead YOK.
Bu SADECE araştırma — kanıtlı Sharpe artışı + Emir onayı olmadan spine'a DOKUNULMAZ.
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

from spine import contract as C, tide as T                            # noqa: E402
from backtest import engine as E                                       # noqa: E402
from modules.cor1m_froth import froth_factor_series, fetch_cor1m_live  # noqa: E402
from modules.gex_shield import gex_zscore, shield_factor_series        # noqa: E402
from screen._util import paired_win_prob, fdr_bh                       # noqa: E402

ANN = np.sqrt(252)


def _sh(r):
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    return float(r.mean() / r.std() * ANN) if (len(r) > 20 and r.std() > 0) else np.nan


def _dd(r):
    eq = np.cumprod(1 + np.asarray(r, float)); return float((eq / np.maximum.accumulate(eq) - 1).min())


# ── kapı varyantları (hepsi trailing) ─────────────────────────────────
def g_baseline(score):
    return (score > 0).astype(float)


def g_hysteresis(score, delta):
    s = score.values; out = np.zeros(len(s)); st = 0.0
    for i, v in enumerate(s):
        if v > delta:
            st = 1.0
        elif v < -delta:
            st = 0.0
        out[i] = st
    return pd.Series(out, index=score.index)


def g_soft_ramp(score, w):
    return (score / w).clip(0.0, 1.0)


def g_debounce(score, n):
    s = score.values; out = np.zeros(len(s)); st = 0.0; pend = None; cnt = 0
    for i, v in enumerate(s):
        side = 1.0 if v > 0 else 0.0
        if side == pend:
            cnt += 1
        else:
            pend = side; cnt = 1
        if cnt >= n:
            st = pend
        out[i] = st
    return pd.Series(out, index=score.index)


def g_ema(score, span):
    return (score.ewm(span=span, adjust=False).mean() > 0).astype(float)


def build():
    scores, prices, vector, prov = C.read_frozen()
    score = T.tide_score_series(scores, vector)
    idx = score.index
    cor = fetch_cor1m_live()
    cf = froth_factor_series(cor.reindex(idx, method="ffill"), 8.0, 11.0, 0.0)
    sg = pd.read_parquet(ROOT / "data" / "cache" / "squeeze_dix_gex.parquet")
    zg = gex_zscore(sg["gex"], 252).reindex(idx, method="ffill")
    gf = shield_factor_series(zg, 0.5, 1.0, 0.4)
    return score, cf, gf, prices


def stack_ret(dir_series, cf, gf, price):
    pos = (dir_series * cf * gf)
    idx = pos.index
    ret = E.fwd_ret(price, idx).values
    p = pos.astype(float).values
    p = np.concatenate([np.zeros(1), p[:-1]])          # lag=1
    return pd.Series(p * ret, index=idx).dropna()


def flips_mask(dir_series, win=10):
    fl = dir_series.diff().abs() > 1e-9
    near = pd.Series(False, index=dir_series.index)
    for k in np.where(fl.values)[0]:
        near.iloc[max(0, k - win):min(len(near), k + win + 1)] = True
    return near, int(fl.sum())


def eval_variant(gate_dir, cf, gf, prices):
    """SPX+NDX Sharpe, maxDD, geçiş/stabil Sharpe, flip-sayısı."""
    out = {}
    near, nfl = flips_mask(gate_dir)
    out["n_flips"] = nfl
    for a in ("SPX", "NDX"):
        r = stack_ret(gate_dir, cf, gf, prices[a])
        nr = near.reindex(r.index).fillna(False).values
        out[a] = {"sh": _sh(r), "dd": _dd(r), "sh_tr": _sh(r[nr]), "sh_st": _sh(r[~nr]), "ret": r}
    return out


def main():
    score, cf, gf, prices = build()
    base_dir = g_baseline(score)
    base = eval_variant(base_dir, cf, gf, prices)
    print("=" * 96)
    print("  REGIME-TRANSITION LAB — tide-flip whipsaw azaltılabilir mi? (full stack, 2019+, lag=1, READ-ONLY)")
    print("=" * 96)
    print(f"  BASELINE (binary score>0): flips {base['n_flips']} | "
          f"SPX Sh {base['SPX']['sh']:+.3f} (geçiş {base['SPX']['sh_tr']:+.2f}/stabil {base['SPX']['sh_st']:+.2f}) | "
          f"NDX Sh {base['NDX']['sh']:+.3f} (geçiş {base['NDX']['sh_tr']:+.2f}/stabil {base['NDX']['sh_st']:+.2f})")
    base_r = {a: base[a]["ret"] for a in ("SPX", "NDX")}

    families = {
        "hysteresis δ": (g_hysteresis, [0.5, 1.0, 2.0, 3.0, 5.0]),
        "soft_ramp w":  (g_soft_ramp,  [1.0, 2.0, 4.0, 8.0]),
        "debounce N":   (g_debounce,   [2, 3, 5, 8]),
        "ema_smooth sp":(g_ema,        [3, 5, 10, 20]),
    }
    print(f"\n  {'variant':<18}{'param':>7}{'flips':>7}{'SPX Sh':>9}{'ΔSPX':>7}{'NDX Sh':>9}{'ΔNDX':>7}"
          f"{'SPX geçiş':>11}{'P>b SPX':>9}{'P>b NDX':>9}{'FDR':>6}")
    winners = []
    for fam, (fn, grid) in families.items():
        for pr in grid:
            gd = fn(score, pr)
            ev = eval_variant(gd, cf, gf, prices)
            wps = {a: paired_win_prob(base_r[a], ev[a]["ret"]) for a in ("SPX", "NDX")}
            passed = fdr_bh({a: 1.0 - w for a, w in wps.items() if w is not None}, alpha=0.05)
            both = all(passed.get(a, False) for a in ("SPX", "NDX"))
            dsp, dnd = ev["SPX"]["sh"] - base["SPX"]["sh"], ev["NDX"]["sh"] - base["NDX"]["sh"]
            wsp = f"{wps['SPX']:.0%}" if wps['SPX'] is not None else "n/a"
            wnd = f"{wps['NDX']:.0%}" if wps['NDX'] is not None else "n/a"
            print(f"  {fam:<18}{pr:>7}{ev['n_flips']:>7}{ev['SPX']['sh']:>+9.3f}{dsp:>+7.2f}"
                  f"{ev['NDX']['sh']:>+9.3f}{dnd:>+7.2f}{ev['SPX']['sh_tr']:>+11.2f}"
                  f"{wsp:>9}{wnd:>9}{('PASS' if both else '-'):>6}")
            if both and dsp > 0 and dnd > 0:
                winners.append((fam, pr, dsp, dnd))
        print()

    print("-" * 96)
    if not winners:
        print("  SONUÇ: hiçbir varyant HER İKİ varlıkta baseline'ı strict-FDR geçemedi → rejim-geçiş drag'i")
        print("         İNDİRGENEMEZ görünüyor (binary yön-kapısının doğası). Model dokunulmadan DURUR.")
    else:
        print(f"  ADAY(lar) strict-FDR geçti: {len(winners)} — PLATO + train/test teyidi ŞART (aşağı).")
        for w in winners:
            print(f"     {w[0]} @ {w[1]}  ΔSPX {w[2]:+.2f} ΔNDX {w[3]:+.2f}")
    print("-" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
