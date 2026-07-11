"""
backtest/slv_gamma_lab — SLV OPSIYON-DEALER kanali gumus fiyatinda ETKEN mi? (on-kayitli, 2026-07-11)

Emir hipotezi: FEAR-fade izi belki dealer-gamma mekanizmasidir (options-flow-leads; fiziksel-lab'in
opsiyon ikizi). Iki parca:

A) BOYUT (bugun, CBOE canli): SLV zinciri gercek OI+gamma → brut gamma-$/%1 → SLV gunluk $-hacmine ve
   SI=F vadeli $-hacmine oran. "Dealer hedge akisi piyasanin kacta kaci" = etkenlik olcusu.
B) TARIHSEL ONCULUK (2016-2023, optionsdx): gunluk HACIM-agirlikli net-gamma (OI YOK — bilinen sinir;
   vendor C_GAMMA/P_GAMMA × C_VOL/P_VOL, dte≤60, naive isaret C+/P−, expanding-z minp 252).
   ON-KAYIT:
     B1-VOL (BIRINCIL): teori kanali YON degil VOL — long-gamma sondurur, short buyutur.
        Test: gex_z ust-tercil vs alt-tercil ertesi-gun |ret|; KONFOUND-KILL: hacim≈vol → RV20-quintil
        ICINDE cift-siralama; blok-bootstrap CI (α=0.05, tek-birincil-test).
     B2-DIR (IKINCIL): corr(gex_z_t, ret_{t+1}) — beklenti ~0 (prior: flip-directional OLU).
     B3-FEAR×GAMMA (KESIFSEL, guc-yetersiz damgali): FEAR-fade rebound'u short-gamma'da daha mi buyuk.
   KARAR: B1 cift-siralamada isaret-tutarli + CI-temiz degilse "dealer-gamma kanali KANITSIZ" yazilir.
Emsal-notlar: SLV ETF-AKISI kader-silver'da test edildi (standalone gercek, incremental ~0);
hacim-GEX duvar/breakout paneli equity'de ARTEFAKT cikti — burada test edilen NESNE farkli (rejim→vol).
  & <kader-macro venv python> backtest/slv_gamma_lab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "screen"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MINP, N_BOOT, BLOCK = 252, 2000, 21


# ── A) BOYUT — canli CBOE SLV zinciri + hacim kiyaslari ─────────────────────────
def part_a():
    import _cboe_lib as L
    import yfinance as yf
    spot, rows = L.load_rows("SLV", band=0.30)
    gross = net = 0.0
    toi = 0
    for r in rows:
        if r["g_cboe"] is None or not r["oi"]:
            continue
        gd = r["g_cboe"] * r["oi"] * 100 * spot * spot * 0.01
        gross += abs(gd)
        net += (1.0 if r["cp"] == "C" else -1.0) * gd
        toi += r["oi"]
    print(f"A) SLV zinciri (CBOE canli, OI GERCEK): spot {spot:.2f}, OI {toi/1e6:.2f}M, "
          f"BRUT gamma ${gross/1e6:.0f}M/%1, NET {net/1e6:+.0f}M ({'LONG' if net >= 0 else 'SHORT'} GAMMA naive)")
    try:
        h = yf.download(["SLV", "SI=F"], period="3mo", progress=False, auto_adjust=False, group_by="ticker")
        adv_slv = float((h["SLV"]["Close"] * h["SLV"]["Volume"]).tail(60).mean())
        adv_si = float((h["SI=F"]["Close"] * h["SI=F"]["Volume"] * 5000).tail(60).mean())
        print(f"   SLV ADV ${adv_slv/1e6:.0f}M/gun | SI=F vadeli ADV ${adv_si/1e9:.1f}bn/gun "
              f"(SLV = kompleksin %{100*adv_slv/(adv_slv+adv_si):.0f}'i)")
        print(f"   → %1'lik harekette dealer brut-hedge ihtiyaci ≈ SLV gunluk hacminin %{100*gross/adv_slv:.1f}'i, "
              f"tum-kompleks hacminin %{100*gross/(adv_slv+adv_si):.2f}'i")
    except Exception as e:
        print(f"   hacim-kiyasi alinamadi: {type(e).__name__}")


# ── B) TARIHSEL — hacim-GEX serisi + on-kayitli testler ─────────────────────────
def build_volgex():
    raw = ROOT / "data" / "raw_optionsdx_slv" / "txt"
    recs = []
    for f in sorted(raw.rglob("*.txt")):
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip().strip("[]").strip() for c in df.columns]
        for c in ("DTE", "C_GAMMA", "P_GAMMA", "C_VOLUME", "P_VOLUME", "UNDERLYING_LAST"):
            if c not in df.columns:
                break
            df[c] = pd.to_numeric(df[c], errors="coerce")
        d = df[(df["DTE"] >= 1) & (df["DTE"] <= 60)]
        for day, g in d.groupby("QUOTE_DATE"):
            S = float(g["UNDERLYING_LAST"].iloc[0])
            cg = float((g["C_GAMMA"] * g["C_VOLUME"]).sum(skipna=True))
            pg = float((g["P_GAMMA"] * g["P_VOLUME"]).sum(skipna=True))
            recs.append({"date": pd.Timestamp(str(day).strip()), "spot": S,
                         "net": (cg - pg) * 100 * S * S * 0.01})
    v = pd.DataFrame(recs).drop_duplicates("date", keep="last").set_index("date").sort_index()
    m = v["net"].expanding(MINP).mean()
    sd = v["net"].expanding(MINP).std()
    v["z"] = (v["net"] - m) / sd.replace(0, np.nan)
    return v


def _bootci(x, rng, alpha=0.05):
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return np.nan, np.nan
    n = len(x); b = min(BLOCK, n); nblk = int(np.ceil(n / b))
    ms = []
    for _ in range(N_BOOT):
        st = rng.integers(0, n - b + 1, nblk)
        idx = (st[:, None] + np.arange(b)[None, :]).ravel()[:n]
        ms.append(np.nanmean(x[idx]))
    return tuple(np.percentile(ms, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def part_b():
    v = build_volgex()
    ret = v["spot"].pct_change()
    a_ret = ret.abs().shift(-1)                    # ertesi-gun |ret|
    rv20 = ret.rolling(20).std()
    ok = v["z"].notna() & a_ret.notna() & rv20.notna()
    z, ar, rv = v["z"][ok], a_ret[ok], rv20[ok]
    print(f"\nB) hacim-GEX serisi: {len(v)} gun, z-tanimli {ok.sum()} "
          f"(OI YOK — hacim-proxy; sinir docstring'de)")

    # B1: cift-siralama — RV-quintil icinde gex-z tercilleri
    rvq = pd.qcut(rv, 5, labels=False)
    diffs = []
    print("  B1 (BIRINCIL) ertesi-gun |ret|, RV-quintil ICINDE long-gamma(T3) − short-gamma(T1):")
    for q in range(5):
        s = rvq == q
        if s.sum() < 60:
            continue
        t = pd.qcut(z[s], 3, labels=False)
        d = float(ar[s][t == 2].mean() - ar[s][t == 0].mean())
        diffs.append(d)
        print(f"     RVq{q+1}: Δ|ret| {1e4*d:+.1f}bp (n={int(s.sum())})")
    rng = np.random.default_rng(23)
    t_hi = pd.qcut(z, 3, labels=False)
    raw_d = ar[t_hi == 2].values - np.nanmean(ar[t_hi == 0].values)
    lo, hi = _bootci(raw_d, rng)
    consistent = all(d < 0 for d in diffs) or all(d > 0 for d in diffs)
    print(f"     ham T3−T1 Δ|ret| {1e4*np.nanmean(raw_d):+.1f}bp, CI [{1e4*lo:+.1f},{1e4*hi:+.1f}] | "
          f"katman-tutarliligi {'EVET' if consistent else 'HAYIR'} ({sum(1 for d in diffs if d < 0)}/{len(diffs)} negatif)")
    b1 = consistent and (hi < 0)                   # teori: long-gamma → daha DUSUK vol → Δ<0

    # B2: yon (ikincil)
    r1 = ret.shift(-1)[ok]
    c = float(np.corrcoef(z, r1.fillna(0))[0, 1])
    print(f"  B2 (IKINCIL) corr(gex_z, ertesi-gun ret) = {c:+.3f} (beklenti ~0)")

    # B3: FEAR × gamma (kesifsel)
    df_sk = pd.read_parquet(ROOT / "data" / "cache" / "rr_skew_slv_2016_2023.parquet")
    df_sk = df_sk[df_sk["t30_ok"]]
    rr = df_sk["t30_rr_skew"].astype(float)
    vals = rr.values
    pct = pd.Series(np.nan, index=rr.index)
    for i in range(MINP, len(vals)):
        pct.iloc[i] = (vals[:i] < vals[i]).mean() * 100
    fear = (pct >= 95).reindex(v.index).fillna(False)
    fwd21 = v["spot"].shift(-21) / v["spot"] - 1
    for tag, mm in (("FEAR & short-gamma(z<0)", fear & (v["z"] < 0)),
                    ("FEAR & long-gamma(z>=0)", fear & (v["z"] >= 0))):
        x = fwd21[mm].dropna()
        print(f"  B3 (KESIFSEL) {tag}: n={len(x)}, 21g ort {1e4*x.mean():+.0f}bp"
              + ("  [GUC-YETERSIZ]" if len(x) < 60 else ""))

    print("\n" + "=" * 96)
    print(f"  B1 VERDICT: {'dealer-gamma vol-kanali IZ VAR (cift-siralama tutarli + CI temiz)' if b1 else 'dealer-gamma kanali KANITSIZ (hacim-proxy ile)'}")


if __name__ == "__main__":
    part_a()
    part_b()
