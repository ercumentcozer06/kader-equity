"""
backtest/remeasure/RC2_sign.py — RC2.2 SIGN-AGREEMENT FINAL (YOL-A re-confirm, 243g panel).

TEŞHİS-ONLY. Yeni strateji/eşik/parametre YOK. Metodoloji = R2_keydiag.sign_agreement ile BİREBİR
(inner-join, sign(net_gex)==sign(squeeze_gex), tercile = qcut(|net_gex|, 3)); tek fark:
  - PANEL penceresi (config.PANEL_START..config.PANEL_END) filtresi,
  - en-büyük-|net_gex| 10 gün tek-tek dökümü (tarih, bizim-işaret, squeeze-işaret),
  - YOL-A pre-committed koşul değerlendirmesi (SPX-full GENEL >=%75 VE üst-tercile >=%70).

Seriler: SPX-full + SPY-full (asıl) ve zorunlu-kıyas SPY-livematch, QQQ-full, NDX-full.
Çıktı: backtest/remeasure/RC2_sign.json (config_sha dahil) + stdout Türkçe tablo.

  & C:/Users/admin/Downloads/kader-macro/.venv/Scripts/python.exe backtest/remeasure/RC2_sign.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import config  # noqa: E402  (TEK-GERÇEK-KAYNAK)

SQUEEZE_PATH = config.CACHE / "squeeze_dix_gex.parquet"
OUT_PATH = config.REMEASURE_DIR / "RC2_sign.json"

# kıyas-referansları (ÖLÇÜM DEĞİL; kaynak belgelerden alıntı)
REFERENCE = {
    "D-FAZ_SPY_old": {"overall_pct": 48.0, "ust_tercile_pct": 30.0,
                      "source": "C:/Users/admin/Downloads/kader-equity/backtest/DIAGNOSIS.md"},
    "PRELIM_157g_SPX_full": {"overall_pct": 79.0, "ust_tercile_pct": 96.0,
                             "source": "C:/Users/admin/Downloads/kader-equity/backtest/remeasure/R2_PRELIM.md"},
}

# YOL-A pre-committed eşikler (prompt + R2_PRELIM.md'de önceden kilitli)
YOLA_OVERALL_MIN_PCT = 75.0
YOLA_UST_TERCILE_MIN_PCT = 70.0

SERIES = [
    # (etiket, mode, sym)
    ("SPX_full", "fullsurface", "SPX"),
    ("SPY_full", "fullsurface", "SPY"),
    ("SPY_livematch", "livematch", "SPY"),
    ("QQQ_full", "fullsurface", "QQQ"),
    ("NDX_full", "fullsurface", "NDX"),
]

TOP_N = 10


def load_squeeze() -> pd.Series:
    s = pd.read_parquet(SQUEEZE_PATH)["gex"].dropna()
    s.index = pd.to_datetime(s.index)
    return s


def panel_filter(idx: pd.DatetimeIndex) -> pd.Series:
    return (idx >= pd.Timestamp(config.PANEL_START)) & (idx <= pd.Timestamp(config.PANEL_END))


def sign_agreement_final(lv: pd.DataFrame, sq: pd.Series) -> dict | None:
    """R2_keydiag.sign_agreement metodolojisi + panel-pencere + top-N dökümü."""
    df = pd.DataFrame({"ng": lv["net_gex"]}).join(pd.DataFrame({"sq": sq}), how="inner").dropna()
    df = df[panel_filter(df.index)]
    if len(df) < 10:
        return None
    df["ag"] = np.sign(df["ng"]) == np.sign(df["sq"])
    overall = float(df["ag"].mean())
    df["t"] = pd.qcut(df["ng"].abs(), 3, labels=["alt", "orta", "üst"])
    terc = {str(t): {"agree_pct": round(100 * float(g["ag"].mean()), 1), "n": int(len(g))}
            for t, g in df.groupby("t", observed=True)}
    top = df.reindex(df["ng"].abs().sort_values(ascending=False).index).head(TOP_N)
    top_rows = [{
        "date": d.strftime("%Y-%m-%d"),
        "net_gex": float(r["ng"]),
        "squeeze_gex": float(r["sq"]),
        "our_sign": int(np.sign(r["ng"])),
        "squeeze_sign": int(np.sign(r["sq"])),
        "agree": bool(r["ag"]),
    } for d, r in top.iterrows()]
    return {
        "n": int(len(df)),
        "window": [str(df.index.min().date()), str(df.index.max().date())],
        "overall_pct": round(100 * overall, 1),
        "terciles": terc,
        "top10_largest_abs_net_gex": top_rows,
        "top10_agree_n": int(top["ag"].sum()),
    }


def main() -> int:
    sq = load_squeeze()
    print("=" * 100)
    print("  RC2.2 SIGN-AGREEMENT FINAL — net_gex-işaret vs SqueezeMetrics-gex-işaret")
    print(f"  PANEL: {config.PANEL_START} → {config.PANEL_END} | squeeze cache: {len(sq)}g, "
          f"{sq.index.min().date()} → {sq.index.max().date()} | config_sha={config.config_sha()}")
    print("  metodoloji: R2_keydiag.sign_agreement BİREBİR (tercile = qcut(|net_gex|,3)); top-10 = en-büyük-|net_gex|")
    print("=" * 100)

    results: dict[str, dict] = {}
    for label, mode, sym in SERIES:
        p = config.level_path(mode, sym)
        if not p.exists():
            results[label] = {"error": f"parquet yok: {p}"}
            print(f"  {label:14}: ÖLÇÜLEMEDİ — parquet yok ({p})")
            continue
        lv = pd.read_parquet(p)
        lv.index = pd.to_datetime(lv.index)
        r = sign_agreement_final(lv, sq)
        if r is None:
            results[label] = {"error": "squeeze overlap < 10 gün"}
            print(f"  {label:14}: ÖLÇÜLEMEDİ — squeeze overlap yetersiz")
            continue
        results[label] = r
        ts = " / ".join(f"{k} %{v['agree_pct']:.0f} (n{v['n']})" for k, v in r["terciles"].items())
        print(f"  {label:14}: GENEL %{r['overall_pct']:.1f} (n{r['n']}) | tercile {ts} "
              f"| top10 uyum {r['top10_agree_n']}/10")

    # top-10 dökümü (asıl seriler önce)
    for label in [s[0] for s in SERIES]:
        r = results.get(label, {})
        if "top10_largest_abs_net_gex" not in r:
            continue
        print(f"\n  {label} — en-büyük-|net_gex| 10 gün:")
        for row in r["top10_largest_abs_net_gex"]:
            mark = "UYUM" if row["agree"] else "AYKIRI"
            print(f"    {row['date']}  bizim {row['our_sign']:+d} ({row['net_gex']:+.3e})  "
                  f"squeeze {row['squeeze_sign']:+d} ({row['squeeze_gex']:+.3e})  {mark}")

    # ESKİ → PRELIM → FINAL tablosu
    spx = results.get("SPX_full", {})
    spx_overall = spx.get("overall_pct")
    spx_ust = spx.get("terciles", {}).get("üst", {}).get("agree_pct")
    spy = results.get("SPY_full", {})
    spy_overall = spy.get("overall_pct")
    spy_ust = spy.get("terciles", {}).get("üst", {}).get("agree_pct")
    print("\n  ESKİ → PRELIM → FINAL (GENEL % / üst-tercile %):")
    print(f"    D-FAZ  SPY-old (kırık tek-expiry, alıntı) : %48 / %30")
    print(f"    PRELIM SPX-full (157g, alıntı)            : %79 / %96")
    print(f"    FINAL  SPX-full (243g panel, ÖLÇÜLDÜ)     : %{spx_overall} / %{spx_ust}")
    print(f"    FINAL  SPY-full (243g panel, ÖLÇÜLDÜ)     : %{spy_overall} / %{spy_ust}")

    # YOL-A pre-committed değerlendirme
    yola_pass = (spx_overall is not None and spx_ust is not None
                 and spx_overall >= YOLA_OVERALL_MIN_PCT and spx_ust >= YOLA_UST_TERCILE_MIN_PCT)
    print(f"\n  YOL-A (pre-committed): SPX-full GENEL ≥%{YOLA_OVERALL_MIN_PCT:.0f} VE üst-tercile "
          f"≥%{YOLA_UST_TERCILE_MIN_PCT:.0f} → {'GEÇTİ' if yola_pass else 'GEÇMEDİ'}")

    out = {
        "config_sha": config.config_sha(),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "backtest/remeasure/RC2_sign.py",
        "panel": {"start": config.PANEL_START, "end": config.PANEL_END},
        "methodology": ("R2_keydiag.sign_agreement birebir: inner-join(level.net_gex, squeeze.gex), "
                        "agree = sign(net_gex)==sign(gex); tercile = qcut(|net_gex|,3) [alt/orta/üst]; "
                        "top10 = en-büyük-|net_gex| günleri; panel-pencere filtresi uygulanır."),
        "squeeze_source": str(SQUEEZE_PATH),
        "results": results,
        "reference_cited_not_measured": REFERENCE,
        "yol_a": {
            "condition": f"SPX_full overall >= {YOLA_OVERALL_MIN_PCT} AND ust_tercile >= {YOLA_UST_TERCILE_MIN_PCT}",
            "spx_full_overall_pct": spx_overall,
            "spx_full_ust_tercile_pct": spx_ust,
            "pass": bool(yola_pass),
        },
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  JSON → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
