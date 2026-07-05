"""
backtest/remeasure/R1_rebuild — FAZ-R / R1: LEVEL SERİSİ REBUILD (onarılmış enstrümanla, iki tanım).
data/raw_chains/ (R0, TÜM-expiry + bid/ask) → iki günlük seviye serisi:
  A) LIVE-MATCH  : gamma_engine.py canlı agregasyonu BİREBİR (front N_EXP=5, düz-toplam, BAND=0.15). ④'ü kapatır.
  B) FULL-SURFACE: tüm unexpired expiry (araştırma bayrağı).
Formüller D0 kod-gerçeği (net_gex/flip/wall/ghost/hvl/max_pain) = gamma_engine._greeks (byte-eş).
IV HİJYENİ (D-FAZ'da tanımlı, YENİ AYAR YOK): V1 bid≤0/crossed(bid>ask) DROP; V2 IV winsorize [0.05,1.50];
  V5 DTE≤2 FLAG (drop değil — gamma_engine 0DTE'yi dışlamaz). Hijyensiz baseline = sensitivity yan-çıktı.
Spot = API underlyingPrice (chain'in kendi referansı; SPX/NDX index, SPY/QQQ ETF). IV = bid/ask-MID'den BS-invert.
  & <venv python> backtest/remeasure/R1_rebuild.py [smoke|build]
    smoke → şimdiye inen ham günlerde sanity (eski %10 vs FULL kapsama)
    build → tüm inen günler için 2 seri × {hijyenli, hijyensiz} parquet + health (eski→yeni)
→ data/cache/level_series_{livematch,fullsurface}_{sym}.parquet
"""
from __future__ import annotations

import glob
import gzip
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "screen"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from _bsiv import implied_vol               # noqa: E402
from gamma_engine import _greeks            # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as CFG                        # noqa: E402  (FAZ-R tek-gerçek-kaynak)

M = CFG.M_CONTRACT
BAND = CFG.BAND
N_EXP = CFG.N_EXP_LIVE
SCAN = np.linspace(CFG.SCAN_LO, CFG.SCAN_HI, CFG.SCAN_N)
IV_LO, IV_HI = CFG.HYG_V2_IV_LO, CFG.HYG_V2_IV_HI    # V2 winsorize (D2 değerleri, config'ten)


def parse_raw(path):
    """Bir ham gz dosyası → DataFrame (tek tarih, tüm expiry). bid/ask/oi/mid/dte/expiration/underlyingPrice."""
    try:
        o = json.load(gzip.open(path, "rt", encoding="utf-8"))
    except Exception:
        return None                                        # yarım-yazılmış (backfill devam ediyor) → atla
    j = o["resp"]
    n = len(j.get("strike", []))
    if not n:
        return None
    df = pd.DataFrame({
        "K": j["strike"], "right": [("C" if s == "call" else "P") for s in j["side"]],
        "oi": j["openInterest"], "bid": j["bid"], "ask": j["ask"], "mid": j["mid"],
        "dte": j["dte"], "exp": j["expiration"], "S": j["underlyingPrice"],
    })
    df["_date"] = o["_date"]
    return df


def _levels(rows, S, mode, hygiene):
    """rows: DataFrame (tek tarih). mode: 'live'(front5) | 'full'. hygiene: V1+V2 uygula."""
    df = rows.copy()
    df = df[(df["oi"].fillna(0) > 0) & df["K"].notna() & (df["dte"] >= 0)]
    if hygiene:                                          # V1: bid≤0 ya da crossed(bid>ask) DROP
        df = df[(df["bid"] > 0) & (df["ask"] >= df["bid"])]
    df["mid2"] = (df["bid"] + df["ask"]) / 2.0
    df = df[df["mid2"] > 0]
    df = df[(df["K"] / S - 1).abs() <= BAND]
    if df.empty:
        return None
    exps = sorted(df["exp"].unique())                    # expiration unix; küçük=yakın
    if mode == "live":
        exps = exps[:N_EXP]
        df = df[df["exp"].isin(exps)]
    if df.empty:
        return None
    # IV invert + V2 winsorize
    recs = []
    for _, r in df.iterrows():
        T = max(int(r["dte"]), 0.5) / 365.0
        iv = implied_vol(float(r["mid2"]), S, float(r["K"]), T, r["right"])
        if not iv or iv <= 0:
            continue
        if hygiene:
            iv = min(max(iv, IV_LO), IV_HI)              # V2 winsorize
        g, *_ = _greeks(S, float(r["K"]), T, iv, r["right"])
        recs.append({"K": float(r["K"]), "oi": float(r["oi"]), "right": r["right"], "iv": iv, "g": g, "T": T})
    if len(recs) < 4:
        return None
    sgn = lambda rt: 1.0 if rt == "C" else -1.0
    net_gex = sum(sgn(x["right"]) * x["g"] * x["oi"] * M * S * S * 0.01 for x in recs)

    def net_g_at(hs):
        tot = 0.0
        for x in recs:
            gg, *_ = _greeks(hs, x["K"], x["T"], x["iv"], x["right"])
            tot += sgn(x["right"]) * gg * x["oi"] * M * hs * hs * 0.01
        return tot
    grid = [(round(S * (1 + p), 2), net_g_at(S * (1 + p))) for p in SCAN]
    flip = None
    for (s0, g0), (s1, g1) in zip(grid, grid[1:]):
        if (g0 <= 0 <= g1) or (g0 >= 0 >= g1):
            flip = round(s0 + (s1 - s0) * (0 - g0) / (g1 - g0), 2) if g1 != g0 else s0
            break
    by_c, by_p, c_oi, by_all = {}, {}, {}, {}
    for x in recs:
        gk = x["g"] * x["oi"]
        if x["right"] == "C":
            by_c[x["K"]] = by_c.get(x["K"], 0.0) + gk
            if x["K"] >= S:
                c_oi[x["K"]] = c_oi.get(x["K"], 0.0) + x["oi"]
        else:
            by_p[x["K"]] = by_p.get(x["K"], 0.0) + gk
        by_all[x["K"]] = by_all.get(x["K"], 0.0) + abs(gk)
    ghost = max((k for k in by_c if k >= S), key=lambda k: by_c[k], default=None)
    call_wall = max(c_oi, key=lambda k: c_oi[k], default=None) if c_oi else None
    put_wall = max((k for k in by_p if k <= S), key=lambda k: by_p[k], default=None)
    hvl = max(by_all, key=lambda k: by_all[k]) if by_all else None
    strikes = sorted({x["K"] for x in recs})
    coi = {}
    for x in recs:
        coi[(x["K"], x["right"])] = coi.get((x["K"], x["right"]), 0.0) + x["oi"]
    pain = lambda P: sum(coi.get((k, "C"), 0) * max(0, P - k) + coi.get((k, "P"), 0) * max(0, k - P) for k in strikes)
    max_pain = min(strikes, key=pain) if strikes else None
    atm = min(recs, key=lambda x: abs(x["K"] - S))
    em1 = S * atm["iv"] * sqrt(1 / 252)
    gamma_dollar = sum(abs(x["g"] * x["oi"]) * M * S * S * 0.01 for x in recs)   # toplam |gamma$| (kapsama için)
    dte2_share = (sum(abs(x["g"] * x["oi"]) * M * S * S * 0.01 for x in recs if x["T"] * 365 <= 2) / gamma_dollar
                  if gamma_dollar else 0.0)
    return dict(net_gex=net_gex, regime=1 if net_gex >= 0 else -1, flip=flip, ghost=ghost,
                call_wall=call_wall, put_wall=put_wall, hvl=hvl, max_pain=max_pain, atm_iv=atm["iv"],
                em1=em1, n_exp=len(exps), n_strikes=len(recs), gamma_dollar=gamma_dollar, dte2_share=dte2_share,
                spot=S)


def build_series(sym, mode, hygiene):
    files = sorted(glob.glob(str(ROOT / "data" / "raw_chains" / sym / "*.json.gz")))
    out = []
    for f in files:
        df = parse_raw(f)
        if df is None or df.empty:
            continue
        S = float(df["S"].median())
        lv = _levels(df, S, mode, hygiene)
        if lv is None:
            continue
        lv["date"] = pd.Timestamp(df["_date"].iloc[0])
        out.append(lv)
    if not out:
        return None
    return pd.DataFrame(out).set_index("date").sort_index()


def smoke():
    print("=== R1 SMOKE — onarım işe yaradı mı? (inen ham günlerde) ===")
    for sym in ("SPY", "QQQ", "SPX", "NDX"):
        files = sorted(glob.glob(str(ROOT / "data" / "raw_chains" / sym / "*.json.gz")))
        if not files:
            print(f"  {sym}: ham yok"); continue
        live = build_series(sym, "live", True)
        full = build_series(sym, "full", True)
        if live is None or full is None:
            print(f"  {sym}: level hesaplanamadı (n_files={len(files)})"); continue
        d = full.index[0]
        print(f"  {sym} ({len(files)}g ham): {d.date()} → "
              f"LIVE n_exp{int(live.loc[d,'n_exp'])} gamma$ {live.loc[d,'gamma_dollar']/1e9:.2f}bn | "
              f"FULL n_exp{int(full.loc[d,'n_exp'])} gamma$ {full.loc[d,'gamma_dollar']/1e9:.2f}bn | "
              f"oran LIVE/FULL %{100*live.loc[d,'gamma_dollar']/full.loc[d,'gamma_dollar']:.0f}")
        # eski tek-expiry (md→level_series) kapsama kıyası
        old_p = ROOT / "data" / "cache" / f"level_series_{sym.lower()}.parquet"
        if old_p.exists() and sym in ("SPY", "QQQ"):
            old = pd.read_parquet(old_p)
            if d in old.index and "n_strikes" in old:
                print(f"       eski tek-expiry n_strikes {int(old.loc[d,'n_strikes'])} vs FULL n_strikes {int(full.loc[d,'n_strikes'])}")
    print("  BEKLENTİ: FULL gamma$ >> LIVE (front5) ve eski-tek-expiry, kapsama büyük artmalı = onarım çalışıyor.")
    return 0


def build():
    print("=== R1 BUILD — 2 mod × {hijyenli,hijyensiz} ===")
    for sym in ("SPY", "QQQ", "SPX", "NDX"):
        for mode in ("livematch", "fullsurface"):
            m = "live" if mode == "livematch" else "full"
            for hyg, tag in ((True, ""), (False, "_nohyg")):
                df = build_series(sym, m, hyg)
                if df is None:
                    print(f"  {sym} {mode}{tag}: veri yok"); continue
                p = ROOT / "data" / "cache" / f"level_series_{mode}_{sym.lower()}{tag}.parquet"
                df.to_parquet(p)
                print(f"  {sym} {mode}{tag}: {len(df)}g → {p.name} (n_exp med {int(df['n_exp'].median())}, "
                      f"gamma$ med {df['gamma_dollar'].median()/1e9:.2f}bn)")
    return 0


def archive():
    """WAVE-1 prologu: mevcut (157g) parquet'leri archive_157g/'ye kopyala (idempotent; provenance + determinizm)."""
    import shutil
    CFG.ARCHIVE_157.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted((ROOT / "data" / "cache").glob("level_series_*match*.parquet")) + \
             sorted((ROOT / "data" / "cache").glob("level_series_*surface*.parquet")):
        dst = CFG.ARCHIVE_157 / p.name
        if not dst.exists():
            shutil.copy2(p, dst); n += 1
    print(f"archive: {n} yeni kopya → {CFG.ARCHIVE_157} (var olanlar atlandı)")
    return 0


def build_one(sym, mode, hyg_tag):
    """WAVE-1 paralel builder: TEK seri. mode: livematch|fullsurface; hyg_tag: hyg|nohyg."""
    import json as _json
    from datetime import datetime, timezone
    m = "live" if mode == "livematch" else "full"
    hyg = hyg_tag == "hyg"
    df = build_series(sym, m, hyg)
    if df is None:
        print(f"{sym} {mode} {hyg_tag}: VERİ YOK"); return 1
    p = CFG.level_path(mode, sym, hygiene=hyg)
    df.to_parquet(p)
    meta = {"config_sha": CFG.config_sha(), "built_utc": datetime.now(timezone.utc).isoformat(),
            "n_days": len(df), "mode": mode, "hygiene": hyg, "code": "R1_rebuild.build_one"}
    Path(str(p) + ".meta.json").write_text(_json.dumps(meta), encoding="utf-8")
    print(f"{sym} {mode} {hyg_tag}: {len(df)}g → {p.name} (n_exp med {int(df['n_exp'].median())}, "
          f"gamma$ med {df['gamma_dollar'].median()/1e9:.2f}bn) config_sha={meta['config_sha']}")
    return 0


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if cmd == "archive":
        return archive()
    if cmd == "build_one":
        return build_one(sys.argv[2].upper(), sys.argv[3], sys.argv[4])
    return smoke() if cmd == "smoke" else build()


if __name__ == "__main__":
    raise SystemExit(main())
