# -*- coding: utf-8 -*-
"""GECE vs SEANS-ICI ATTRIBUTION LABI (2026-08-11, Emir onayi).

SORU: kader-equity'nin edge'i GECE mi (kapanis_t -> acilis_t+1) yoksa SEANS ICINDE mi
(acilis_t+1 -> kapanis_t+1) olusuyor?

NEDEN ONEMLI: cevap ICRA ZAMANLAMASINI degistirir. Edge tamamen gecedeyse kapanistan ONCE
pozisyonda olmak ZORUNLU demektir; ertesi sabah girmek edge'in tamamini kacirir.

ILERI-BAKIS YOK: pozisyon[t] kapanis t'de bilinir; hem ON[t+1] hem ID[t+1] ondan SONRA gerceklesir.
(1+ON)(1+ID) = kapanis_t -> kapanis_t+1 = modelin zaten skorlandigi getiri. Ayrisim EKSIKSIZ.

KRITIK KONTROL: SPX'in gece-driftl olgusu LITERATURDE bilinen bir PIYASA ozelligi. O yuzden
"stratejinin gecede cok kazanmasi" tek basina modelin ozelligi DEGIL. Bu yuzden her segmentte
ZAMANLAMA EDGE'i ayrica olculur:
    timing_edge[seg] = ort(pos * r_seg) - expo * ort(r_seg)
yani "modelin sectigi gunler, o segmentte ortalama gunden daha mi iyi?" Bu, piyasanin gece
driftini modelin becerisi sanmayi engeller.

Istatistik: fark-CI'lari BLOK-BOOTSTRAP (gunluk otokorelasyon + rejim kumelenmesi icin).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EQ = Path(r"C:\Users\admin\Downloads\kader-equity")
sys.path.insert(0, str(EQ))

from spine import contract as C, tide as T   # noqa: E402

SEED = 7
N_BOOT = 5000
BLOCK = 21          # ~1 ay; gunluk seri + rejim kumelenmesi
ANN = np.sqrt(252.0)


# ── veri ────────────────────────────────────────────────────────────────────────────────
def ohlc(sym: str, start="2018-12-01") -> pd.DataFrame:
    import yfinance as yf
    h = yf.Ticker(sym).history(start=start, auto_adjust=False)
    h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
    return h[["Open", "Close"]].dropna()


def segments(px: pd.DataFrame) -> pd.DataFrame:
    """kapanis_t -> acilis_t+1 (ON) ve acilis_t+1 -> kapanis_t+1 (ID); ikisi t'ye HIZALI
    (yani satir t = 'pozisyon t ile kazanilan t+1 getirisi')."""
    c, o = px["Close"], px["Open"]
    on = (o.shift(-1) / c - 1.0)
    idr = (c.shift(-1) / o.shift(-1) - 1.0)
    cc = (c.shift(-1) / c - 1.0)
    return pd.DataFrame({"on": on, "id": idr, "cc": cc}).dropna()


# ── istatistik ──────────────────────────────────────────────────────────────────────────
def sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    return float(ANN * r.mean() / sd) if sd > 0 else float("nan")


def block_boot_ci(x: np.ndarray, fn, n=N_BOOT, block=BLOCK, seed=SEED):
    """x = (T,k) matris; fn(x_resample) -> skaler. Blok-bootstrap CI (p5,p50,p95) + P(>0)."""
    rng = np.random.default_rng(seed)
    T_ = len(x)
    nb = int(np.ceil(T_ / block))
    out = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, max(1, T_ - block), size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T_]
        out[i] = fn(x[idx])
    return float(np.percentile(out, 5)), float(np.percentile(out, 50)), \
        float(np.percentile(out, 95)), float((out > 0).mean())


# ── pozisyon serileri ───────────────────────────────────────────────────────────────────
def positions():
    """(a) kilitli SPINE tide_dir (capa 1.425/1.489) ve (b) TAM STACK (run.py frozen kolunun
    birebir tekrari: tide x dispersion x GEX-shield x es_basis, + SPX'te uc-gun boost)."""
    scores, prices, vector, prov = C.read_frozen()
    tdir = T.tide_dir_series(T.tide_score_series(scores, vector))

    from modules import dispersion_ensemble as DE, gex_shield as GS, es_basis_unwind as EB
    cache = EQ / "data" / "cache"
    idx = tdir.index

    disp = pd.read_parquet(cache / "dispersion.parquet")
    corr = pd.read_parquet(cache / "corr_pc.parquet")["COR1M"].dropna()
    froth_pct = DE.froth_pct_series(corr, disp["spread"].dropna(), disp["dspx"].dropna(), 756, 252)
    f_disp = froth_pct.reindex(idx, method="ffill").map(
        lambda v: DE.ensemble_factor(v, 0.70, 0.95, 0.0) if pd.notna(v) else 1.0)

    gex = pd.read_parquet(cache / "squeeze_dix_gex.parquet")["gex"].dropna()
    f_gex = GS.shield_factor_series(GS.gex_zscore(gex, 252).reindex(idx, method="ffill"),
                                    0.5, 1.0, 0.4).fillna(1.0)

    bpp = cache / "es_basis_daily.parquet"
    if bpp.exists():
        f_es = EB.unwind_factor_series(pd.read_parquet(bpp)["spread_bps"],
                                       p_thr=0.75, dz_thr=-1.5, floor=0.0
                                       ).reindex(idx, method="ffill").fillna(1.0)
    else:
        f_es = pd.Series(1.0, index=idx)

    stack = (tdir.astype(float) * f_disp * f_gex * f_es).clip(0.0, 1.0)
    return {"spine (kilitli capa)": tdir.astype(float), "tam stack": stack}, prices


# ── ana ─────────────────────────────────────────────────────────────────────────────────
def run_asset(asset: str, sym: str, pos_map: dict, frozen_px: pd.Series):
    px = ohlc(sym)
    seg = segments(px)

    # SAGLIK: yfinance kapanisi ile donmus panel kapanisi ayni varligi mi olcuyor?
    ov = frozen_px.dropna().index.intersection(px.index)
    if len(ov) > 100:
        rel = (px.loc[ov, "Close"] / frozen_px.loc[ov] - 1.0).abs()
        print(f"  [saglik] yfinance vs donmus kapanis: medyan |fark| {rel.median()*100:.4f}%  "
              f"maks {rel.max()*100:.3f}%  (n={len(ov)})")

    rows = []
    for label, pos in pos_map.items():
        idx = pos.index.intersection(seg.index)
        p = pos.reindex(idx).astype(float).shift(1).fillna(0.0)     # +1g icra gecikmesi (run.py ile ayni)
        s = seg.reindex(idx)
        m = np.isfinite(p) & np.isfinite(s["on"]) & np.isfinite(s["id"])
        p, s = p[m].values, s[m]
        on, idr, cc = s["on"].values, s["id"].values, s["cc"].values
        expo = float(p.mean())

        strat_on, strat_id, strat_cc = p * on, p * idr, p * cc
        # zamanlama edge'i = modelin sectigi gunler o segmentte ortalamadan iyi mi
        te_on = strat_on.mean() - expo * on.mean()
        te_id = strat_id.mean() - expo * idr.mean()

        X = np.column_stack([p, on, idr])
        ci_on = block_boot_ci(X, lambda z: z[:, 0].mean() * z[:, 1].mean() * 0 +
                              (z[:, 0] * z[:, 1]).mean() - z[:, 0].mean() * z[:, 1].mean())
        ci_id = block_boot_ci(X, lambda z: (z[:, 0] * z[:, 2]).mean() - z[:, 0].mean() * z[:, 2].mean())

        rows.append({
            "pozisyon": label, "n": len(p), "expo": expo,
            "toplam_getiri_cc": float(np.prod(1 + strat_cc) - 1),
            "pay_gece": float(strat_on.sum() / (strat_on.sum() + strat_id.sum()))
            if (strat_on.sum() + strat_id.sum()) != 0 else np.nan,
            "Sh_cc": sharpe(strat_cc), "Sh_gece": sharpe(strat_on), "Sh_seans": sharpe(strat_id),
            "BH_Sh_gece": sharpe(on), "BH_Sh_seans": sharpe(idr),
            "BH_pay_gece": float(np.sum(on) / (np.sum(on) + np.sum(idr)))
            if (np.sum(on) + np.sum(idr)) != 0 else np.nan,
            "timing_gece_bp": te_on * 1e4, "timing_seans_bp": te_id * 1e4,
            "tg_p5": ci_on[0] * 1e4, "tg_p95": ci_on[2] * 1e4, "tg_P>0": ci_on[3],
            "ts_p5": ci_id[0] * 1e4, "ts_p95": ci_id[2] * 1e4, "ts_P>0": ci_id[3],
        })
    return pd.DataFrame(rows)


def main():
    pos_map, frozen_prices = positions()
    print("=" * 100)
    print("  GECE vs SEANS-ICI ATTRIBUTION — kader-equity")
    print("  pozisyon[t] kapanista bilinir; ON=kapanis_t->acilis_t+1, ID=acilis_t+1->kapanis_t+1")
    print("=" * 100)
    for asset, sym in (("SPX", "^GSPC"), ("NDX", "^NDX")):
        print(f"\n### {asset} ({sym})")
        df = run_asset(asset, sym, pos_map, frozen_prices[asset])
        for _, r in df.iterrows():
            print(f"\n  -- {r['pozisyon']}  (n={r['n']}, expo {r['expo']*100:.0f}%)")
            print(f"     Sharpe   : kapanis-kapanis {r['Sh_cc']:+.3f} | GECE {r['Sh_gece']:+.3f} "
                  f"| SEANS {r['Sh_seans']:+.3f}")
            print(f"     getiri payi: GECE %{100*r['pay_gece']:.0f} / SEANS %{100*(1-r['pay_gece']):.0f}"
                  f"   [al-tut kontrol: gece %{100*r['BH_pay_gece']:.0f}]")
            print(f"     al-tut Sh  : GECE {r['BH_Sh_gece']:+.3f} | SEANS {r['BH_Sh_seans']:+.3f}")
            print(f"     ZAMANLAMA EDGE (piyasa driftinden ARINDIRILMIS, gunluk bp):")
            print(f"       gece  {r['timing_gece_bp']:+.2f}bp  CI[{r['tg_p5']:+.2f},{r['tg_p95']:+.2f}]  P(>0)={r['tg_P>0']:.2f}")
            print(f"       seans {r['timing_seans_bp']:+.2f}bp  CI[{r['ts_p5']:+.2f},{r['ts_p95']:+.2f}]  P(>0)={r['ts_P>0']:.2f}")
    print("\n" + "=" * 100)
    print("  OKUMA: 'getiri payi' piyasa driftini ICERIR (SPX'te gece-drift bilinen olgu).")
    print("  KARAR-DEGERLI olan ZAMANLAMA EDGE satiri: modelin SECTIGI gunler o segmentte")
    print("  ortalama gunden iyi mi? Edge yalniz gecedeyse -> kapanistan ONCE pozisyonda ol.")
    print("=" * 100)


if __name__ == "__main__":
    main()
