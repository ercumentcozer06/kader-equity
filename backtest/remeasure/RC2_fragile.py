"""
backtest/remeasure/RC2_fragile.py — RC2.3 BAYRAK İSTİKRARI + FRAGILE FINAL (TEŞHİS-ONLY).

(a) ESKİ-bayrak (data/cache/level_series_{spy,qqq}.parquet 'regime' — D-FAZ kırık tek-expiry) vs YENİ
    LIVE-MATCH 'regime' (level_series_livematch_{sym}.parquet): flip-gün % + flip-tarih listesi.
    PRELIM (157g): SPY %32 / QQQ %24 — şimdi 243g FINAL.
(b) FRAGILE-final: baseline = YENİ LIVE-MATCH (hijyenli) regime. Ölçüm-sağlamlık VARYANT İŞARETLERİ
    (yalnız SIGN; P&L'e BAĞLANMAZ; yeni strateji/eşik YOK):
      V1 = hijyensiz parquet işareti (level_series_livematch_{sym}_nohyg.parquet regime)
      V3 = flat-ATM-IV işareti (ham chain'den; günün ATM-IV'ü tüm strike'lara, smile düzleştir)
      V4 = pure-OI-balance işareti (Σ±OI, IV'süz; call:+OI put:−OI, baseline satırları üzerinde)
      V5 = DTE≤2-hariç işareti (ham chain'den; baseline front-5 seçimi İÇİNDE dte>HYG_V5_DTE_FLAG
           kontratları kalır — saf-çıkarma; kalan satır<4 → gün işaretsiz=NaN, D2 konvansiyonu)
    FRAGILE = ≥2 varyant baseline ile çelişen günler (tarih listesi).
    SELF-CHECK: ham-chain'den replike baseline sign == parquet regime HER GÜN olmalı (RC1 determinizm).
(c) MEVCUT gamma_inv P&L (backtest/block_robust.gamma_inv_pnl — ESKİ seri; RECOMPUTE-YALNIZ-OKU,
    YENİ varyant DEĞİL) fragile-günlere konsantrasyon: toplam P&L %payı + holdout top-3 fragile mi.
    Kıyas: D2 eski sayıları (DIAGNOSIS.md §D2: SPY %2 / QQQ %3; top-3 0/3 fragile).

V3/V4/V5 ham-chain replikasyonu R1_rebuild._levels(mode=live, hygiene=True) ile BİREBİR aynı satır
filtreleri/IV-invert/winsorize kullanır (sabitler config.py'den; kod R1'den kopya, tek öğe değişir).
  & <kader-macro venv python> backtest/remeasure/RC2_fragile.py
→ backtest/remeasure/RC2_fragile.json (config_sha'lı) + stdout Türkçe özet.
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "screen"))
sys.path.insert(0, str(ROOT / "backtest"))
sys.path.insert(0, str(HERE))               # EN ÖNDE: 'config' = remeasure/config.py (ROOT config'i gölgelemesin)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import config as CFG                          # noqa: E402  (tek-gerçek-kaynak)
from _bsiv import implied_vol                 # noqa: E402  (kanonik IV)
from gamma_engine import _greeks              # noqa: E402  (byte-eş greeks)
from R1_rebuild import parse_raw              # noqa: E402  (ham gz → DataFrame; R1 ile aynı parser)

M = CFG.M_CONTRACT
BAND = CFG.BAND
N_EXP = CFG.N_EXP_LIVE
IV_LO, IV_HI = CFG.HYG_V2_IV_LO, CFG.HYG_V2_IV_HI
DTE_FLAG = CFG.HYG_V5_DTE_FLAG
PANEL_END = pd.Timestamp(CFG.PANEL_END)
VARIANTS = ["V1", "V3", "V4", "V5"]

# --- KIYAS REFERANSLARI (ölçüm DEĞİL; kaynak belgelerden okunmuş eski sayılar) ---
PRELIM_REF = {  # kaynak: backtest/remeasure/R2_PRELIM.md (157g ön-sonuç, eski-flag vs LIVE-MATCH flip)
    "SPY": 32.0, "QQQ": 24.0}
D2_REF = {      # kaynak: backtest/DIAGNOSIS.md §D2 (eski kırık-seri, n=236)
    "fragile_days": {"SPY": 5, "QQQ": 5}, "fragile_pct": {"SPY": 2.1, "QQQ": 2.1},
    "fragile_pnl_share_pct": {"SPY": 2, "QQQ": 3},
    "fragile_pnl_bps": {"SPY": 29, "QQQ": 36}, "total_pnl_bps": {"SPY": 1170, "QQQ": 1108},
    "holdout_top3_fragile": {"SPY": "0/3", "QQQ": "0/3"},
    "fragile_dates": {
        "SPY": ["2025-06-20", "2025-07-15", "2025-09-09", "2025-11-21", "2026-02-27"],
        "QQQ": ["2025-10-30", "2025-12-12", "2025-12-16", "2025-12-18", "2026-02-27"]}}


def sharpe(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / x.std() * sqrt(252) if len(x) > 2 and x.std() > 0 else 0.0


# ---------------------------------------------------------------- (a) eski-bayrak vs LIVE-MATCH
def flag_stability(sym):
    old = pd.read_parquet(CFG.CACHE / f"level_series_{sym.lower()}.parquet")       # D-FAZ kırık seri (SALT-OKU)
    new = pd.read_parquet(CFG.level_path("livematch", sym))                        # hijyenli LIVE-MATCH
    common = old.index.intersection(new.index)
    common = common[common <= PANEL_END]
    o = old.loc[common, "regime"].astype(int)
    n = new.loc[common, "regime"].astype(int)
    flips = o != n
    dates = [d.strftime("%Y-%m-%d") for d in common[flips]]
    return {"n_common": int(len(common)), "flip_days": int(flips.sum()),
            "flip_pct": round(100 * float(flips.mean()), 1),
            "prelim_157g_flip_pct_ref": PRELIM_REF[sym], "flip_dates": dates}


# ---------------------------------------------------------------- (b) varyant işaretleri (ham chain)
def day_recs(df, S):
    """R1_rebuild._levels(mode='live', hygiene=True) satır-hazırlığının BİREBİR kopyası → recs listesi.
       (filtreler/sıra/IV-invert/winsorize aynı; yalnız level-hesap yerine recs döner)."""
    df = df.copy()
    df = df[(df["oi"].fillna(0) > 0) & df["K"].notna() & (df["dte"] >= 0)]
    df = df[(df["bid"] > 0) & (df["ask"] >= df["bid"])]                  # V1 hijyen (baseline'da AÇIK)
    df["mid2"] = (df["bid"] + df["ask"]) / 2.0
    df = df[df["mid2"] > 0]
    df = df[(df["K"] / S - 1).abs() <= BAND]
    if df.empty:
        return None
    exps = sorted(df["exp"].unique())[:N_EXP]                            # LIVE-MATCH: front-5 expiry
    df = df[df["exp"].isin(exps)]
    if df.empty:
        return None
    recs = []
    for _, r in df.iterrows():
        T = max(int(r["dte"]), 0.5) / 365.0
        iv = implied_vol(float(r["mid2"]), S, float(r["K"]), T, r["right"])
        if not iv or iv <= 0:
            continue
        iv = min(max(iv, IV_LO), IV_HI)                                  # V2 winsorize (baseline hijyen)
        g, *_ = _greeks(S, float(r["K"]), T, iv, r["right"])
        recs.append({"K": float(r["K"]), "oi": float(r["oi"]), "right": r["right"], "iv": iv, "g": g, "T": T})
    if len(recs) < 4:
        return None
    return recs


def _net_sign(recs, S, flat_iv=None, dte_gt=None):
    """Σ ±gamma·OI·M·S²·0.01 işareti. flat_iv: tüm satıra tek IV (V3, greeks yeniden);
       dte_gt: yalnız T·365 > dte_gt satırları (V5, saf-çıkarma; <4 satır → NaN)."""
    rows = recs
    if dte_gt is not None:
        rows = [x for x in rows if x["T"] * 365 > dte_gt]
        if len(rows) < 4:
            return np.nan
    sgn = lambda rt: 1.0 if rt == "C" else -1.0
    if flat_iv is None:
        net = sum(sgn(x["right"]) * x["g"] * x["oi"] * M * S * S * 0.01 for x in rows)
    else:
        net = 0.0
        for x in rows:
            gg, *_ = _greeks(S, x["K"], x["T"], flat_iv, x["right"])
            net += sgn(x["right"]) * gg * x["oi"] * M * S * S * 0.01
    return 1 if net >= 0 else -1


def fragile_battery(sym):
    base = pd.read_parquet(CFG.level_path("livematch", sym))             # baseline (hijyenli)
    nohyg = pd.read_parquet(CFG.level_path("livematch", sym, hygiene=False))
    days = base.index[base.index <= PANEL_END]
    files = {Path(f).stem.replace(".json", ""): f
             for f in glob.glob(str(CFG.RAW_DIR / sym / "*.json.gz"))}
    rows, selfcheck_mismatch, missing_raw = [], 0, []
    for D in days:
        rec = {"date": D, "base": int(base.loc[D, "regime"])}
        rec["V1"] = int(nohyg.loc[D, "regime"]) if D in nohyg.index else np.nan
        f = files.get(D.strftime("%Y-%m-%d"))
        if f is None:
            missing_raw.append(D.strftime("%Y-%m-%d"))
            rec["V3"] = rec["V4"] = rec["V5"] = np.nan
            rows.append(rec); continue
        df = parse_raw(f)
        if df is None or df.empty:
            missing_raw.append(D.strftime("%Y-%m-%d"))
            rec["V3"] = rec["V4"] = rec["V5"] = np.nan
            rows.append(rec); continue
        S = float(df["S"].median())
        recs = day_recs(df, S)
        if recs is None:
            missing_raw.append(D.strftime("%Y-%m-%d"))
            rec["V3"] = rec["V4"] = rec["V5"] = np.nan
            rows.append(rec); continue
        # SELF-CHECK: replike baseline sign == parquet regime (RC1 determinizm garantisi)
        if _net_sign(recs, S) != rec["base"]:
            selfcheck_mismatch += 1
        atm = min(recs, key=lambda x: abs(x["K"] - S))                   # R1 atm_iv tanımı (aynı seçim)
        rec["V3"] = _net_sign(recs, S, flat_iv=atm["iv"])                # flat-ATM-IV (smile düzleşik)
        rec["V4"] = 1 if sum((x["oi"] if x["right"] == "C" else -x["oi"]) for x in recs) >= 0 else -1
        rec["V5"] = _net_sign(recs, S, dte_gt=DTE_FLAG)                  # DTE≤2 hariç
        rows.append(rec)
    sg = pd.DataFrame(rows).set_index("date").sort_index()

    flip_cols, variant_stats = {}, {}
    for v in VARIANTS:
        ok = sg[v].notna()
        flips = (sg.loc[ok, v] != sg.loc[ok, "base"])
        flip_cols[v] = flips.reindex(sg.index, fill_value=False)
        variant_stats[v] = {"flip_days": int(flips.sum()), "n_common": int(ok.sum()),
                            "flip_pct": round(100 * float(flips.mean()), 1) if ok.sum() else None}
    flipmat = pd.DataFrame(flip_cols)
    disagree = flipmat.sum(axis=1)
    fragile_idx = sg.index[disagree >= 2]
    dist = {str(int(k)): int(v) for k, v in disagree.value_counts().sort_index().items()}
    return ({"n": int(len(sg)), "selfcheck_baseline_mismatch": int(selfcheck_mismatch),
             "missing_raw_days": missing_raw, "variant_flip": variant_stats,
             "fragile_n": int(len(fragile_idx)),
             "fragile_pct": round(100 * len(fragile_idx) / len(sg), 1),
             "fragile_dates": [d.strftime("%Y-%m-%d") for d in fragile_idx],
             "disagree_count_dist": dist},
            set(fragile_idx.normalize()), disagree)


# ---------------------------------------------------------------- (c) gamma_inv konsantrasyon
def gamma_inv_concentration(sym, fragile_set, disagree):
    from block_robust import gamma_inv_pnl                               # ESKİ seri; RECOMPUTE-YALNIZ-OKU
    pnl, _ = gamma_inv_pnl(sym)
    pnl.index = pd.to_datetime(pnl.index)
    tot = float(pnl.sum())
    frag_mask = np.asarray(pnl.index.normalize().isin(list(fragile_set)))
    frag_pnl = float(pnl.values[frag_mask].sum())
    ho = pnl[pnl.index >= pd.Timestamp(CFG.HOLDOUT_START)]               # holdout = config tanımı
    ho_tot = float(ho.sum())
    top3 = ho.nlargest(CFG.TOPK_CONC)
    top3_rows, n_frag_top3 = [], 0
    for d, val in top3.items():
        d0 = pd.Timestamp(d).normalize()
        is_frag = d0 in fragile_set
        n_frag_top3 += int(is_frag)
        nd = int(disagree.get(d0, 0)) if d0 in disagree.index else 0
        top3_rows.append({"date": d0.strftime("%Y-%m-%d"), "bps": round(1e4 * float(val), 1),
                          "fragile": bool(is_frag), "n_disagree": nd})
    ho_frag_mask = np.asarray(ho.index.normalize().isin(list(fragile_set)))
    ho_frag_pnl = float(ho.values[ho_frag_mask].sum())
    return {"n": int(len(pnl)), "total_bps": round(1e4 * tot, 0), "full_sharpe": round(sharpe(pnl.values), 2),
            "fragile_days_in_pnl": int(frag_mask.sum()), "fragile_pnl_bps": round(1e4 * frag_pnl, 0),
            "fragile_pnl_share_pct": round(100 * frag_pnl / tot, 1) if tot else None,
            "holdout_start": CFG.HOLDOUT_START, "holdout_n": int(len(ho)),
            "holdout_total_bps": round(1e4 * ho_tot, 0),
            "holdout_top3": top3_rows, "holdout_top3_fragile": f"{n_frag_top3}/{CFG.TOPK_CONC}",
            "holdout_fragile_days": int(ho_frag_mask.sum()),
            "holdout_fragile_pnl_bps": round(1e4 * ho_frag_pnl, 0),
            "holdout_fragile_share_pct": round(100 * ho_frag_pnl / ho_tot, 1) if ho_tot else None}


def main():
    out = {"config_sha": CFG.config_sha(), "run_utc": datetime.now(timezone.utc).isoformat(),
           "script": "backtest/remeasure/RC2_fragile.py", "panel_end": CFG.PANEL_END,
           "baseline": "level_series_livematch_{sym}.parquet (hijyenli) regime",
           "variants": {"V1": "hijyensiz parquet (nohyg) işareti",
                        "V3": "flat-ATM-IV işareti (ham chain; günün ATM-IV'ü tüm strike'a)",
                        "V4": "pure-OI-balance işareti (Σ±OI, IV'süz; baseline satırları)",
                        "V5": f"DTE≤{DTE_FLAG} hariç işareti (front-5 içinde saf-çıkarma; <4 satır→NaN)"},
           "a_flag_stability": {}, "b_fragile_final": {}, "c_gamma_inv_concentration": {},
           "references": {"prelim_157g": "backtest/remeasure/R2_PRELIM.md", "d2": D2_REF}}
    for sym in CFG.TRADE_SYMS:
        print(f"\n{'='*100}\n  {sym} — RC2.3 BAYRAK İSTİKRARI + FRAGILE FINAL (panel ≤ {CFG.PANEL_END})\n{'='*100}")
        a = flag_stability(sym)
        out["a_flag_stability"][sym] = a
        print(f"  (a) ESKİ-bayrak vs LIVE-MATCH: {a['flip_days']}/{a['n_common']} gün flip "
              f"(%{a['flip_pct']}; prelim-157g %{a['prelim_157g_flip_pct_ref']})")
        b, frag_set, disagree = fragile_battery(sym)
        out["b_fragile_final"][sym] = b
        print(f"  (b) self-check baseline-mismatch: {b['selfcheck_baseline_mismatch']} "
              f"(0 olmalı) | ham-eksik gün: {len(b['missing_raw_days'])}")
        for v in VARIANTS:
            s = b["variant_flip"][v]
            print(f"      {v}: {s['flip_days']}/{s['n_common']} flip (%{s['flip_pct']})")
        print(f"      FRAGILE (≥2 varyant çelişik): {b['fragile_n']}/{b['n']} (%{b['fragile_pct']}) "
              f"→ {', '.join(b['fragile_dates']) if b['fragile_dates'] else 'YOK'}")
        print(f"      çelişki-dağılımı: {b['disagree_count_dist']}")
        if b["selfcheck_baseline_mismatch"]:
            print("  !! SELF-CHECK FAIL — replike baseline parquet'le uyuşmuyor; (c) yine raporlanır ama ŞÜPHELİ.")
        c = gamma_inv_concentration(sym, frag_set, disagree)
        out["c_gamma_inv_concentration"][sym] = c
        print(f"  (c) gamma_inv (ESKİ seri, recompute): n={c['n']} toplam {c['total_bps']:+.0f}bps "
              f"Sharpe {c['full_sharpe']:+.2f}")
        print(f"      fragile-gün payı: {c['fragile_days_in_pnl']} gün {c['fragile_pnl_bps']:+.0f}bps "
              f"= toplam'ın %{c['fragile_pnl_share_pct']}  [D2-eski: %{D2_REF['fragile_pnl_share_pct'][sym]}]")
        print(f"      holdout(≥{c['holdout_start']}, n={c['holdout_n']}) toplam {c['holdout_total_bps']:+.0f}bps; "
              f"top-3 fragile: {c['holdout_top3_fragile']}  [D2-eski: {D2_REF['holdout_top3_fragile'][sym]}]")
        for r in c["holdout_top3"]:
            print(f"        {r['date']}  {r['bps']:+.1f}bps  {'FRAGILE' if r['fragile'] else 'stabil'} "
                  f"(çelişen-varyant={r['n_disagree']})")
        print(f"      holdout-içi fragile: {c['holdout_fragile_days']} gün {c['holdout_fragile_pnl_bps']:+.0f}bps "
              f"= holdout'un %{c['holdout_fragile_share_pct']}")
    p = HERE / "RC2_fragile.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n→ {p}  (config_sha={out['config_sha']})")
    print("NOT: TEŞHİS-ONLY — V1/V3/V4/V5 yalnız SIGN serisi, P&L'e bağlanmadı; gamma_inv yalnız recompute/okundu;")
    print("     ham cache'e dokunulmadı (append-only korunur); TIDE/OVERLAYS frozen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
