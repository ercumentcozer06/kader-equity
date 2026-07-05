"""
backtest/gex_swing/T4_short — "derin short-gamma'da SHORT aç" testi (Emir 2026-06-12).
Mevcut model deep-short-gamma'da yalnız KISIYOR (shield floor 0.4). Soru: kısmak yerine SHORT açsak?
Standalone SPX 2011-26 (çok-rejim) + modele konunca (2019+ frozen stack). PIT: sinyal[D]→getiri[D+1] (lag=1).
trim-only DİSİPLİNİ BİLEREK kırılıyor (bu testin amacı short eklemek) — ana model FROZEN, bu yalnız ölçüm.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import gxs_config as G


def met(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return dict(sharpe=0.0, maxdd=0.0, cum=0.0, cvar5=0.0, n=len(r))
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    sh = float(r.mean() / r.std() * np.sqrt(252))
    k = max(1, int(0.05 * len(r)))
    cv = float(np.sort(r.values)[:k].mean())
    return dict(sharpe=round(sh, 3), maxdd=round(dd, 4), cum=round(float(eq.iloc[-1] - 1), 3),
                cvar5=round(cv, 5), n=len(r))


def pnl(pos: pd.Series, ret: pd.Series) -> pd.Series:
    """pos[D] sonraki gün getirisini kazanır → pos.shift(1)*ret (look-ahead yok)."""
    return (pos.shift(1) * ret).dropna()


def main():
    g = G.load_squeeze()
    ret = g["price"].pct_change()
    z = G.gex_zscore(g["gex"])
    sma200 = g["price"].rolling(200).mean()
    bull = (g["price"] > sma200)                       # standalone tide-proxy (gerçek tide 2019+ ablation'da)
    shield = (1 - 0.5 * np.clip(-z - 1, 0, 3)).clip(0.4, 1)

    print("=" * 88)
    print("  T4 — DEEP SHORT-GAMMA'DA SHORT  (standalone SPX 2011-26, çok-rejim)")
    print("=" * 88)
    base = {
        "B&H (hep long)": pd.Series(1.0, index=g.index),
        "mevcut shield (trim 0.4)": shield,
    }
    rows = {}
    for nm, pos in base.items():
        rows[nm] = met(pnl(pos, ret))

    for thr in (1.0, 1.5, 2.0):
        deep = (z <= -thr)
        nd = int(deep.sum())
        # short günlerinin ertesi-gün getiri ortalaması (short kârlı mı? neg ise short kazanır)
        nxt = ret.shift(-1)
        avg_next = float(nxt[deep].mean()) if nd else float("nan")
        win_short = float((nxt[deep] < 0).mean()) if nd else float("nan")
        variants = {
            f"SHORT -1.0 @z≤-{thr}":   pd.Series(1.0, index=g.index).mask(deep, -1.0),
            f"SHORT -0.5 @z≤-{thr}":   pd.Series(1.0, index=g.index).mask(deep, -0.5),
            f"FLAT 0 @z≤-{thr}":       pd.Series(1.0, index=g.index).mask(deep, 0.0),
            f"SHORT-if-bear @z≤-{thr}": pd.Series(1.0, index=g.index).mask(deep & ~bull, -1.0).mask(deep & bull, 0.4),
        }
        print(f"\n  --- eşik z≤-{thr}: {nd} gün ({100*nd/len(g):.0f}%); short-günü ertesi getiri ort %{100*avg_next:+.2f}"
              f" | short-kazanır(ertesi<0) %{100*win_short:.0f} ---")
        for nm, pos in variants.items():
            rows[nm] = met(pnl(pos, ret))

    hdr = f"  {'varyant':30}{'Sharpe':>9}{'maxDD':>9}{'cumPnL':>10}{'CVaR5':>9}"
    print("\n" + hdr); print("  " + "-" * (len(hdr) - 2))
    for nm, m in rows.items():
        print(f"  {nm:30}{m['sharpe']:>+9.2f}{100*m['maxdd']:>+8.0f}%{100*m['cum']:>+9.0f}%{100*m['cvar5']:>+8.2f}%")

    # alt-dönem: short-full @z≤-1.5 vs shield vs B&H, maxDD + cumPnL
    print("\n  --- alt-dönem (SHORT -1.0 @z≤-1.5 vs mevcut shield vs B&H): maxDD | cumPnL ---")
    deep15 = (z <= -1.5)
    sfull = pd.Series(1.0, index=g.index).mask(deep15, -1.0)
    for nm, (a, b) in G.SUBPERIODS.items():
        if nm == "full_2011_26":
            continue
        sl = slice(a, b)
        r_s = pnl(sfull, ret).loc[sl]; r_sh = pnl(shield, ret).loc[sl]; r_b = pnl(pd.Series(1.0, index=g.index), ret).loc[sl]
        def dd(r):
            r = r.dropna(); eq = (1 + r).cumprod(); return 100 * float((eq / eq.cummax() - 1).min()) if len(r) else 0.0
        def cu(r):
            r = r.dropna(); return 100 * float((1 + r).prod() - 1) if len(r) else 0.0
        print(f"  {nm:14} SHORT {dd(r_s):+5.0f}%/{cu(r_s):+5.0f}% | shield {dd(r_sh):+5.0f}%/{cu(r_sh):+5.0f}%"
              f" | B&H {dd(r_b):+5.0f}%/{cu(r_b):+5.0f}%")

    # ---------------- IN-MODEL ABLATION (2019+ frozen stack) ----------------
    print("\n" + "=" * 88)
    print("  T4 — MODELE KONUNCA (2019+ frozen stack; shield-trim YERİNE short)")
    print("=" * 88)
    from spine import contract as C, tide as T
    from backtest import engine as E
    from modules.cor1m_froth import froth_factor_series
    from modules.gex_shield import gex_zscore as gz_mod, shield_factor_series
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))
    idx = tdir.index
    cor = pd.read_parquet(ROOT / "data/cache/corr_pc.parquet")["COR1M"].dropna()
    gex = pd.read_parquet(ROOT / "data/cache/squeeze_dix_gex.parquet")["gex"].dropna()
    froth = froth_factor_series(cor.reindex(idx, method="ffill"), 8, 11, 0.0)
    zg = gz_mod(gex).reindex(idx, method="ffill")
    shield_m = shield_factor_series(zg, 0.5, 1.0, 0.4)

    def strat(pos, close, lag=1):
        r = E.fwd_ret(close, pos.index).values
        p = pos.astype(float).values
        p = np.concatenate([np.zeros(lag), p[:-lag]])
        return pd.Series(p * r, index=pos.index).dropna()

    for a in ("SPX", "NDX"):
        print(f"\n  [{a}]   {'stack':38}{'Sharpe':>8}{'maxDD':>8}{'cumPnL':>9}")
        base_stack = tdir * froth * shield_m
        deep_m = (zg <= -1.0)
        deep15_m = (zg <= -1.5)
        # short varyantları: deep'te tide_dir×floor YERİNE short. froth yine çarpılır (köpük varsa short'u da büyütür mantıklı değil → froth yalnız long tarafta; short'ta 1.0 al)
        def stack_with_short(deep_mask, short_size, only_when_flat):
            pos = (tdir * froth * shield_m).copy()
            if only_when_flat:
                m = deep_mask & (tdir == 0)        # yalnız tide FLAT iken short (makroya karşı gitme)
            else:
                m = deep_mask
            pos[m] = -short_size
            return pos
        variants = {
            "base (mevcut: trim)":              base_stack,
            "SHORT -1 @z≤-1 (her zaman)":       stack_with_short(deep_m, 1.0, False),
            "SHORT -1 @z≤-1.5 (her zaman)":     stack_with_short(deep15_m, 1.0, False),
            "SHORT -1 @z≤-1 (yalnız tide-FLAT)":stack_with_short(deep_m, 1.0, True),
            "SHORT -0.5 @z≤-1 (yalnız FLAT)":   stack_with_short(deep_m, 0.5, True),
        }
        for nm, pos in variants.items():
            r = strat(pos.reindex(idx), prices[a])
            m = met(r)
            print(f"       {nm:38}{m['sharpe']:>+8.2f}{100*m['maxdd']:>+7.0f}%{100*m['cum']:>+8.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
