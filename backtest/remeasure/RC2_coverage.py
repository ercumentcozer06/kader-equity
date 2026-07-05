"""
backtest/remeasure/RC2_coverage.py — RC2.1: gamma$ KAPSAMA + 0DTE-görünmez-dilim (TEŞHİS-ONLY, READ-ONLY).

(a) Yeni serilerin gamma$ kapsama%'si:
    kapsama% = gamma_dollar(seri) / gamma_dollar(FULL-SURFACE, aynı-gün) medyanı; FULL-SURFACE = referans %100.
    ESKİ  : D-FAZ tek-expiry enstrümanı — ham chain'den RE-KONSTRÜKSİYON (eski serinin 'dte' kolonu o günün
            tek-expiry'sini verir; aynı R1 hijyen+band+IV pipeline'ı ile o expiry'nin gamma$'ı hesaplanır).
            Dokümante referans: DIAGNOSIS.md D1.3 tek-gün spot-check %11 SPY / %10 QQQ.
    PRELIM: archive_157g (R2_PRELIM vintage'ı) livematch/fullsurface parquet'leri.
    FINAL : güncel 244-245g parquet'ler (panel penceresi PANEL_START..PANEL_END, config).
(b) 0DTE yapısal-görünmez dilim:
    ham chain'de expiry-günü E (dte==0): aynı-gün volume(exp=E) / o-sabahki OI(exp=E) — gün-bazında oran,
    sembol-başına medyan = intraday-açılan kontrat payının ÜST-SINIR proxy'si.
    CAVEAT: volume açılış+kapanış karışıktır (her trade open mi close mu bilinmez) → oran üst-sınırdır;
    OI o sabahki OCC gece-güncellemesidir → gün-içi açılıp aynı gün expire olan kontratlar EOD-OI'de HİÇ görünmez.
    Ek: dte2_share kolonu medyanı (EOD-OI'nin gördüğü kısa-DTE payı; V5 FLAG eşiği config.HYG_V5_DTE_FLAG).

Hiçbir parquet/raw YAZILMAZ-SİLİNMEZ; tek çıktı = RC2_coverage.json + konsol tablosu.
  & <venv python> backtest/remeasure/RC2_coverage.py
"""
from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "screen"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from _bsiv import implied_vol               # noqa: E402  (kanonik IV; R1 ile aynı)
from gamma_engine import _greeks            # noqa: E402  (kanonik greeks; R1 ile aynı)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as CFG                        # noqa: E402  (TEK-GERÇEK-KAYNAK)

M, BAND = CFG.M_CONTRACT, CFG.BAND
IV_LO, IV_HI = CFG.HYG_V2_IV_LO, CFG.HYG_V2_IV_HI
P0, P1 = pd.Timestamp(CFG.PANEL_START), pd.Timestamp(CFG.PANEL_END)

# Dokümante REFERANS (sabit değil, alıntı): DIAGNOSIS.md D1.3 tek-gün spot-check kapsama yüzdeleri.
D13_DOC_PCT = {"SPY": 11.0, "QQQ": 10.0}


def old_path(sym: str) -> Path:
    """D-FAZ kırık tek-expiry serisi (yalnız SPY/QQQ'da var)."""
    return CFG.CACHE / f"level_series_{sym.lower()}.parquet"


def parse_raw_plus(path: Path) -> pd.DataFrame | None:
    """R1_rebuild.parse_raw BİREBİR + volume kolonu (0DTE dilimi için)."""
    try:
        o = json.load(gzip.open(path, "rt", encoding="utf-8"))
    except Exception:
        return None
    j = o["resp"]
    n = len(j.get("strike", []))
    if not n:
        return None
    df = pd.DataFrame({
        "K": j["strike"], "right": [("C" if s == "call" else "P") for s in j["side"]],
        "oi": j["openInterest"], "bid": j["bid"], "ask": j["ask"], "mid": j["mid"],
        "dte": j["dte"], "exp": j["expiration"], "S": j["underlyingPrice"],
        "volume": j["volume"],
    })
    df["_date"] = o["_date"]
    return df


def gamma_dollar_subset(df: pd.DataFrame, S: float, hygiene: bool = True) -> tuple[float, int]:
    """R1_rebuild._levels'ın gamma$ kısmı BİREBİR (filtreler+V1+band+IV-invert+V2): toplam |gamma$| + kontrat sayısı."""
    d = df[(df["oi"].fillna(0) > 0) & df["K"].notna() & (df["dte"] >= 0)]
    if hygiene:                                          # V1: bid≤0 ya da crossed DROP
        d = d[(d["bid"] > 0) & (d["ask"] >= d["bid"])]
    d = d.assign(mid2=(d["bid"] + d["ask"]) / 2.0)
    d = d[d["mid2"] > 0]
    d = d[(d["K"] / S - 1).abs() <= BAND]
    tot, n = 0.0, 0
    for _, r in d.iterrows():
        T = max(int(r["dte"]), 0.5) / 365.0
        iv = implied_vol(float(r["mid2"]), S, float(r["K"]), T, r["right"])
        if not iv or iv <= 0:
            continue
        if hygiene:
            iv = min(max(iv, IV_LO), IV_HI)              # V2 winsorize
        g, *_ = _greeks(S, float(r["K"]), T, iv, r["right"])
        tot += abs(g * float(r["oi"])) * M * S * S * 0.01
        n += 1
    return tot, n


def cov_stats(ratio: pd.Series) -> dict:
    r = ratio.dropna()
    return {"median_pct": round(float(r.median()) * 100, 2),
            "p25_pct": round(float(r.quantile(0.25)) * 100, 2),
            "p75_pct": round(float(r.quantile(0.75)) * 100, 2),
            "n_days": int(len(r))}


def main() -> int:
    out: dict = {"config_sha": CFG.config_sha(),
                 "generated_utc": datetime.now(timezone.utc).isoformat(),
                 "script": "backtest/remeasure/RC2_coverage.py",
                 "panel": {"start": CFG.PANEL_START, "end": CFG.PANEL_END},
                 "a_gamma_coverage": {
                     "definition": ("kapsama% = gamma_dollar(seri)/gamma_dollar(FULL-SURFACE aynı-gün); "
                                    "FULL-SURFACE=referans %100; medyan + p25/p75; hijyenli seriler; "
                                    "panel penceresi config'ten"),
                     "eski_method": ("ham chain re-konstrüksiyon: eski serinin 'dte' kolonu = o günün tek-expiry'si; "
                                     "o expiry'nin gamma$'ı R1 pipeline'ı (V1+band+V2) ile / FINAL fullsurface "
                                     "gamma_dollar. NOT: D-FAZ orijinal pipeline'ı değil, aynı-pipeline "
                                     "apples-to-apples yapısal-sansür ölçümü"),
                     "per_symbol": {}},
                 "b_0dte_invisible": {
                     "definition": ("ham chain, dte==0 dilimi: sum(volume)/sum(OI) gün-bazında → sembol-başına "
                                    "medyan = intraday-açılan payının ÜST-SINIR proxy'si"),
                     "caveat": ("volume açılış+kapanış KARIŞIK (open/close ayrımı yok) → oran üst-sınır; "
                                "OI=o-sabahki OCC değeri → aynı-gün açılan 0DTE kontratlar EOD-OI'de hiç görünmez; "
                                "oran>1 normal (0DTE'de gün-içi ciro sabah-OI'yi katlar)"),
                     "per_symbol": {}}}

    print("=== RC2.1 KAPSAMA + 0DTE-GÖRÜNMEZ-DİLİM ===")
    print(f"config_sha={out['config_sha']}  panel {CFG.PANEL_START}..{CFG.PANEL_END}\n")

    for sym in CFG.SYMS:
        sy = sym.lower()
        # --- parquet'ler (hijyenli) ---
        lm = pd.read_parquet(CFG.level_path("livematch", sym))
        fs = pd.read_parquet(CFG.level_path("fullsurface", sym))
        lm_p = lm[(lm.index >= P0) & (lm.index <= P1)]
        fs_p = fs[(fs.index >= P0) & (fs.index <= P1)]
        alm_f, afs_f = CFG.ARCHIVE_157 / f"level_series_livematch_{sy}.parquet", \
                       CFG.ARCHIVE_157 / f"level_series_fullsurface_{sy}.parquet"
        rec: dict = {}

        # FINAL kapsama
        j = lm_p[["gamma_dollar"]].join(fs_p[["gamma_dollar"]], how="inner", lsuffix="_lm", rsuffix="_fs")
        rec["final"] = {"livematch": cov_stats(j["gamma_dollar_lm"] / j["gamma_dollar_fs"]),
                        "fullsurface_pct": 100.0,
                        "lm_gamma_med_bn": round(float(lm_p["gamma_dollar"].median()) / 1e9, 2),
                        "fs_gamma_med_bn": round(float(fs_p["gamma_dollar"].median()) / 1e9, 2)}

        # PRELIM (archive_157g) kapsama
        if alm_f.exists() and afs_f.exists():
            alm, afs = pd.read_parquet(alm_f), pd.read_parquet(afs_f)
            ja = alm[["gamma_dollar"]].join(afs[["gamma_dollar"]], how="inner", lsuffix="_lm", rsuffix="_fs")
            rec["prelim_157g"] = {"livematch": cov_stats(ja["gamma_dollar_lm"] / ja["gamma_dollar_fs"]),
                                  "fullsurface_pct": 100.0,
                                  "lm_gamma_med_bn": round(float(alm["gamma_dollar"].median()) / 1e9, 2),
                                  "fs_gamma_med_bn": round(float(afs["gamma_dollar"].median()) / 1e9, 2)}
        else:
            rec["prelim_157g"] = {"status": "ölçülemedi", "neden": f"archive parquet yok: {alm_f.name}/{afs_f.name}"}

        # --- ham chain tek-geçiş: (b) 0DTE + (a)-ESKİ ---
        op = old_path(sym)
        old = pd.read_parquet(op) if op.exists() else None
        raw_files = sorted((CFG.RAW_DIR / sym).glob("*.json.gz"))
        ratios_0dte, no_0dte_days, old_ratios, old_n_contracts, old_skip = [], 0, [], [], 0
        for f in raw_files:
            d = pd.Timestamp(f.name.removesuffix(".json.gz"))
            if not (P0 <= d <= P1):
                continue
            df = parse_raw_plus(f)
            if df is None or df.empty:
                continue
            # (b) 0DTE dilimi
            m0 = df[df["dte"] == 0]
            oi0 = float(m0["oi"].fillna(0).sum())
            if len(m0) == 0 or oi0 <= 0:
                no_0dte_days += 1
            else:
                ratios_0dte.append(float(m0["volume"].fillna(0).sum()) / oi0)
            # (a) ESKİ re-konstrüksiyon (yalnız old-seri olan semboller)
            if old is not None and d in old.index and d in fs.index:
                odte = int(old.loc[d, "dte"])
                sub = df[df["dte"] == odte]
                if sub.empty:                            # konvansiyon kayması: ±1 gün tolerans
                    near = df[(df["dte"] - odte).abs() <= 1]
                    sub = near[near["dte"] == near["dte"].min()] if not near.empty else near
                if sub.empty:
                    old_skip += 1
                else:
                    S = float(sub["S"].median())
                    g_old, n_c = gamma_dollar_subset(sub, S, hygiene=True)
                    if g_old > 0:
                        old_ratios.append((d, g_old / float(fs.loc[d, "gamma_dollar"])))
                        old_n_contracts.append(n_c)
                    else:
                        old_skip += 1

        # ESKİ kaydı
        if old is None:
            rec["eski_tek_expiry"] = {"status": "ölçülemedi",
                                      "neden": ("D-FAZ tek-expiry serisi bu sembolde yok (yalnız SPY/QQQ; "
                                                "index havuzu D-FAZ'da hiç çekilmemişti)")}
        elif old_ratios:
            r = pd.Series({d: v for d, v in old_ratios})
            rec["eski_tek_expiry"] = cov_stats(r)
            rec["eski_tek_expiry"].update({
                "n_skip": old_skip,
                "n_contracts_med": int(np.median(old_n_contracts)),
                "d13_documented_pct": D13_DOC_PCT.get(sym),
                "d13_source": "DIAGNOSIS.md D1.3 (tek-gün spot-check)"})
        else:
            rec["eski_tek_expiry"] = {"status": "ölçülemedi", "neden": f"eşleşen gün yok (skip={old_skip})"}

        out["a_gamma_coverage"]["per_symbol"][sym] = rec

        # (b) kaydı + dte2_share
        b = {}
        if ratios_0dte:
            rs = pd.Series(ratios_0dte)
            b = {"vol_oi_ratio_median": round(float(rs.median()), 2),
                 "vol_oi_ratio_p25": round(float(rs.quantile(0.25)), 2),
                 "vol_oi_ratio_p75": round(float(rs.quantile(0.75)), 2),
                 "n_days_with_0dte": int(len(rs)), "n_days_no_0dte": int(no_0dte_days)}
        else:
            b = {"status": "ölçülemedi", "neden": "panelde dte==0 dilimi olan gün yok",
                 "n_days_no_0dte": int(no_0dte_days)}
        b["dte2_share_median_fullsurface"] = round(float(fs_p["dte2_share"].median()), 4)
        b["dte2_share_median_livematch"] = round(float(lm_p["dte2_share"].median()), 4)
        b["dte2_flag_thresh_days"] = CFG.HYG_V5_DTE_FLAG
        out["b_0dte_invisible"]["per_symbol"][sym] = b

        # --- konsol satırı ---
        e = rec["eski_tek_expiry"]
        e_txt = (f"%{e['median_pct']:.1f} (n={e['n_days']}, D1.3-dok %{e.get('d13_documented_pct')})"
                 if "median_pct" in e else "—")
        p_txt = (f"%{rec['prelim_157g']['livematch']['median_pct']:.1f}"
                 if "livematch" in rec.get("prelim_157g", {}) else "—")
        f_txt = f"%{rec['final']['livematch']['median_pct']:.1f}"
        print(f"[{sym}] (a) kapsama-medyan  ESKİ-tek-expiry {e_txt}  →  PRELIM-LM {p_txt}  →  FINAL-LM {f_txt}"
              f"  (FULL=referans %100; FINAL gamma$ med LM {rec['final']['lm_gamma_med_bn']}bn / "
              f"FS {rec['final']['fs_gamma_med_bn']}bn)")
        if "vol_oi_ratio_median" in b:
            print(f"      (b) 0DTE vol/OI medyan {b['vol_oi_ratio_median']:.2f}"
                  f" [p25 {b['vol_oi_ratio_p25']:.2f} / p75 {b['vol_oi_ratio_p75']:.2f}]"
                  f" (n={b['n_days_with_0dte']}g, 0DTE'siz {b['n_days_no_0dte']}g)"
                  f" | dte2_share med FS %{100*b['dte2_share_median_fullsurface']:.1f}"
                  f" / LM %{100*b['dte2_share_median_livematch']:.1f}")
        else:
            print(f"      (b) 0DTE: ölçülemedi | dte2_share med FS %{100*b['dte2_share_median_fullsurface']:.1f}"
                  f" / LM %{100*b['dte2_share_median_livematch']:.1f}")

    pj = CFG.REMEASURE_DIR / "RC2_coverage.json"
    pj.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {pj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
