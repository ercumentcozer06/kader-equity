"""
backtest/volume_gamma_artifact_test — proxy-panel yan bulgusu ARTEFAKT mi GERCEK mi?

Bulgu: hacim-gamma proxy panelinde −γ(dc) gunlerinde duvar-kirilimi (breakout = −MR) net +14-17bps,
iki donem × iki sembolde stabil. Suphe: hacim-duvari fiyatin yakinina yigilir → 'kirilim' = siradan
trend-gunu artefakti olabilir. Battery (hepsi yerel veri, PIT bozulmaz):
  T1 baseline    : gercek duvar + gercek −γ(dc) kosullama (referans) + KOSULSUZ (tum gunler)
  T2 placebo-duvar: duvar-spot OFSETLERI gunler arasi shuffle (mesafe-dagilimi ayni, gun-bilgisi yok)
                    → 500 perm null; p = P(placebo ≥ gercek). Duvar bilgi tasimiyorsa edge placebo'da da cikar.
  T3 etiket-perm : reg_dc shuffle (duvar gercek) → kosullama mi is yapiyor yoksa panel-geneli mi?
  T4 confound    : ayni gunlerde kosullamayi (a) put/call hacim orani>medyan, (b) onceki-gun getiri isareti
                    ile degistir — mekanik ikame edge'i reproduce ediyor mu?
  T5 gap-ayristir: 'dokunus' acilis-gap'iyle mi (o1 duvari gecmis) yoksa seans-ici mi? Gercek seviye
                    davranisi seans-ici dokunusta da gorunmeli.
  T6 saglamlik   : donem-yarilari + %5-trim (outlier suruklemesi).
  & <kader-macro venv python> backtest/volume_gamma_artifact_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from alpaca_panel_extension import build_panel_from, SPLIT       # noqa: E402
from alpaca_chain_backfill import expiry_for                      # noqa: E402
from spine_diagnostic import COST                                 # noqa: E402  (maliyet tek kaynak)

N_PERM = 500


def mr_vec(cw, pw, h1, l1, c1):
    """mean_reversion_return'un vektorize esdegeri (assert ile dogrulanir)."""
    tc = ~np.isnan(cw) & (h1 >= cw)
    tp = ~np.isnan(pw) & (l1 <= pw)
    mc = np.where(tc, (cw - c1) / np.where(cw == 0, np.nan, cw), np.nan)
    mp = np.where(tp, (c1 - pw) / np.where(pw == 0, np.nan, pw), np.nan)
    s = np.nan_to_num(mc) + np.nan_to_num(mp)
    n = tc.astype(int) + tp.astype(int)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def brk_edge(mr, mask):
    """Breakout net edge: E[−MR]−COST, (edge, t, n)."""
    x = -mr[mask]
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return float("nan"), 0.0, len(x)
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if x.std() > 0 else 0.0
    return x.mean() - COST, t, len(x)


def run(sym: str) -> dict:
    lv = pd.read_parquet(ROOT / "data" / "cache" / f"level_series_{sym.lower()}_alpaca.parquet")
    p = build_panel_from(lv, sym)
    A = {c: p[c].to_numpy(dtype=float) for c in ("spot", "call_wall", "put_wall", "o1", "h1", "l1", "c1", "c0")}
    mr = mr_vec(A["call_wall"], A["put_wall"], A["h1"], A["l1"], A["c1"])
    ref = p["mr"].to_numpy(dtype=float)
    ok = ~np.isnan(ref)
    assert np.allclose(mr[ok], ref[ok], atol=1e-12) and np.isnan(mr[~ok]).all(), "mr_vec ≠ spine MR"
    neg = (p["reg_dc"] == -1).to_numpy()

    print(f"=== {sym} (n={len(p)}, −γdc={neg.sum()}) ===")
    e0, t0, n0 = brk_edge(mr, neg)
    ea, ta, na = brk_edge(mr, np.ones(len(p), bool))
    print(f"  T1 baseline: −γdc brk {1e4*e0:+.1f}bps (t{t0:+.1f},n{n0})  |  KOSULSUZ {1e4*ea:+.1f}bps (t{ta:+.1f},n{na})")

    # T2 placebo-duvar (mesafe-eslesmeli): ofset ciftlerini shuffle
    rng = np.random.default_rng(7)
    ocw, opw = A["call_wall"] - A["spot"], A["put_wall"] - A["spot"]
    null2 = []
    for _ in range(N_PERM):
        idx = rng.permutation(len(p))
        m = mr_vec(A["spot"] + ocw[idx], A["spot"] + opw[idx], A["h1"], A["l1"], A["c1"])
        null2.append(brk_edge(m, neg)[0])
    null2 = np.array(null2)
    p2 = float(np.mean(null2 >= e0))
    print(f"  T2 placebo-duvar: null ort {1e4*np.nanmean(null2):+.1f}bps, p(placebo≥gercek)={p2:.3f} "
          f"→ {'duvar BILGI TASIYOR' if p2 < 0.05 else 'duvar bilgi TASIMIYOR (artefakt lehine)'}")

    # T3 etiket-perm: reg_dc==−1 maskesi shuffle, duvar gercek
    null3 = np.array([brk_edge(mr, rng.permutation(neg))[0] for _ in range(N_PERM)])
    p3 = float(np.mean(null3 >= e0))
    print(f"  T3 etiket-perm : null ort {1e4*np.nanmean(null3):+.1f}bps, p={p3:.3f} "
          f"→ {'−γ kosullamasi IS YAPIYOR' if p3 < 0.05 else 'kosullama katki YOK (panel-geneli fenomen)'}")

    # T4 confound ikameleri
    ch = pd.read_parquet(ROOT / "data" / "historical_chains" / f"alpaca_chain_{sym.lower()}.parquet")
    ch = ch[ch["volume"] > 0]
    cal = set(pd.to_datetime(lv.index).date)
    ch = ch[[pd.to_datetime(e).date() == expiry_for(pd.Timestamp(d).date(), cal)
             for d, e in zip(ch["date"], ch["expiration"])]]
    pv = ch.groupby(["date", "right"])["volume"].sum().unstack()
    pcr = (pv["P"] / pv["C"]).reindex(p["D"]).to_numpy()
    m_pcr = pcr > np.nanmedian(pcr)
    prev_neg = np.r_[np.nan, np.diff(A["c0"])] < 0
    for tag, m in (("put/call-vol>medyan", m_pcr), ("onceki-gun getiri<0 ", prev_neg)):
        e, t, n = brk_edge(mr, m & ~np.isnan(mr))
        print(f"  T4 {tag}: brk {1e4*e:+.1f}bps (t{t:+.1f},n{n})")

    # T5 gap-through vs seans-ici dokunus (−γdc gunleri)
    gap_thr = ((A["o1"] >= A["call_wall"]) | (A["o1"] <= A["put_wall"]))
    for tag, m in (("gap-through", neg & gap_thr), ("seans-ici  ", neg & ~gap_thr)):
        e, t, n = brk_edge(mr, m)
        print(f"  T5 {tag}: brk {1e4*e:+.1f}bps (t{t:+.1f},n{n})")

    # T6 yarilar + trim
    D = pd.to_datetime(p["D"])
    ext, ovl = (D < SPLIT).to_numpy(), (D >= SPLIT).to_numpy()
    x = -mr[neg]; x = x[~np.isnan(x)]
    k = max(1, int(0.05 * len(x)))
    trim = np.sort(x)[k:-k].mean() - COST
    e1 = brk_edge(mr, neg & ext)[0]; e2 = brk_edge(mr, neg & ovl)[0]
    print(f"  T6 yarilar: 2024→25 {1e4*e1:+.1f} / 2025→26 {1e4*e2:+.1f}bps; %5-trim {1e4*trim:+.1f}bps\n")
    return dict(e0=e0, p2=p2, p3=p3)


def main() -> int:
    print(f"HACIM-GAMMA BREAKOUT — ARTEFAKT BATTERY (perm={N_PERM}, maliyet {1e4*COST:.1f}bps)\n")
    res = {s: run(s) for s in ("SPY", "QQQ")}
    both = all(r["p2"] < 0.05 and r["p3"] < 0.05 for r in res.values())
    print("  VERDICT girdisi: GERCEK diyebilmek icin T2 VE T3 iki sembolde de p<0.05 olmali; "
          f"durum = {'SAGLANDI' if both else 'SAGLANMADI'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
