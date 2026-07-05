"""
backtest/remeasure/RC2_events.py — RC2.6 EVENT-EDGE 8-HÜCRE FINAL (TEŞHİS-ONLY; config = tek-gerçek-kaynak).
MR tanımı = spine_diagnostic.mean_reversion_return leg'leri BİREBİR:
  call-leg: D+1 high >= call_wall  → MR = (cw − c1) / cw    (>0 duvar tuttu / geri-döndü)
  put-leg : D+1 low  <= put_wall   → MR = (c1 − pw) / pw
Hücreler: sym(SPY,QQQ) × rejim(+γ/−γ) × duvar(call/put) = 8 hücre; hücre-başına n, ort-MR (bps, GROSS — ölçüm,
P&L değil; maliyet yok, spine M2 ile aynı), t = mean/(std(ddof=1)/√n).
Bayrak setleri (battery build_panel ile birebir hizalama):
  (a) own_livematch : regime + duvarlar = own LIVE-MATCH serisi
  (b) index_flag    : regime = FULL-SURFACE index (SPX→SPY, NDX→QQQ); duvarlar = own FULL-SURFACE
PIT: level[D] (D-EOD) → D+1 RTH (Alpaca 1-dk); panel = config.PANEL_START..PANEL_END.
Tez-yönü işareti: +γ → MR>0 beklenir / −γ → MR<0 beklenir (yorum yok, yalnız sayı+işaret).
ESKİ (D-FAZ D6.3, DIAGNOSIS.md §3 hücre tablosu) ve PRELIM (R2_PRELIM.md ⑦, 157g) referans olarak yan-yana basılır.
  & <venv python> backtest/remeasure/RC2_events.py
→ backtest/remeasure/RC2_events.json (config_sha ile)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "backtest"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import config as CFG                         # noqa: E402
from spine_diagnostic import daily_rth       # noqa: E402

FLAG_SETS = ["own_livematch", "index_flag"]  # RC2.6 kapsamı (görev tanımı a/b; battery FLAG_SETS'in alt-kümesi)

# ---- REFERANS tablolar (karşılaştırma için; parametre DEĞİL — kaynak belgelerden aynen) ----
REF_ESKI = {   # D-FAZ D6.3 — DIAGNOSIS.md "§3 Mevcut veride wall-touch event sayımı" tablosu (kırık tek-expiry seri)
    "source": "backtest/DIAGNOSIS.md (§3, sembol×rejim×duvar tablosu)",
    "cells": {
        "SPY|+g|call": {"n": 16, "mean_bps": -5.4},  "SPY|+g|put": {"n": 15, "mean_bps": -30.2},
        "SPY|-g|call": {"n": 12, "mean_bps": +22.3}, "SPY|-g|put": {"n": 23, "mean_bps": -4.8},
        "QQQ|+g|call": {"n": 22, "mean_bps": -22.5}, "QQQ|+g|put": {"n": 18, "mean_bps": -17.8},
        "QQQ|-g|call": {"n": 15, "mean_bps": -20.6}, "QQQ|-g|put": {"n": 30, "mean_bps": +12.3},
    },
}
REF_PRELIM = {  # R2 ÖN-SONUÇ (157g, own LIVE-MATCH flag+wall) — R2_PRELIM.md ⑦ tablosu (yalnız ort-MR bps)
    "source": "backtest/remeasure/R2_PRELIM.md (⑦ event-edge 8-hücre, 157g LIVE-MATCH)",
    "cells": {
        "SPY|+g|call": {"mean_bps": -23}, "SPY|+g|put": {"mean_bps": -16},
        "SPY|-g|call": {"mean_bps": -11}, "SPY|-g|put": {"mean_bps": -14},
        "QQQ|+g|call": {"mean_bps": -19}, "QQQ|+g|put": {"mean_bps": -17},
        "QQQ|-g|call": {"mean_bps": -1},  "QQQ|-g|put": {"mean_bps": -15},
    },
}


def build_panel(sym: str, flag_set: str) -> pd.DataFrame:
    """own-level (duvar) + flag-src (regime) + D+1 RTH; PANEL_START..PANEL_END (RC2_battery.build_panel hizası)."""
    own_mode = "livematch" if flag_set == "own_livematch" else "fullsurface"
    own = pd.read_parquet(CFG.level_path(own_mode, sym))
    if flag_set == "index_flag":
        fsrc = pd.read_parquet(CFG.level_path("fullsurface", CFG.INDEX_FLAG_MAP[sym]))
    else:
        fsrc = own
    rth = daily_rth(sym)
    sess = list(rth.index)
    start, end = pd.Timestamp(CFG.PANEL_START), pd.Timestamp(CFG.PANEL_END)
    rows = []
    for D in own.index:
        if D < start or D > end or D not in rth.index or D not in fsrc.index:
            continue
        nxt = [s for s in sess if s > D]
        if not nxt:
            continue
        N = nxt[0]
        h1, l1, c1 = rth.loc[N, ["h", "l", "c"]]
        rows.append({"D": D, "regime": int(fsrc.loc[D, "regime"]),
                     "call_wall": own.loc[D, "call_wall"], "put_wall": own.loc[D, "put_wall"],
                     "h1": h1, "l1": l1, "c1": c1})
    return pd.DataFrame(rows).set_index("D").sort_index()


def cell_stats(legs: list[float]) -> dict:
    a = np.asarray(legs, float)
    n = len(a)
    if n == 0:
        return {"n": 0, "mean_bps": None, "t": None}
    m = float(a.mean())
    t = float(m / (a.std(ddof=1) / sqrt(n))) if n >= 2 and a.std(ddof=1) > 0 else None
    return {"n": n, "mean_bps": round(1e4 * m, 1), "t": round(t, 2) if t is not None else None}


def thesis_mark(reg_lab: str, mean_bps) -> str | None:
    """+γ → MR>0 beklenir; −γ → MR<0 beklenir."""
    if mean_bps is None:
        return None
    exp_pos = (reg_lab == "+g")
    ok = (mean_bps > 0) if exp_pos else (mean_bps < 0)
    return "tez-içi" if ok else "tez-dışı"


def eight_cells(p: pd.DataFrame, sym: str) -> dict:
    """sym×rejim×duvar hücreleri: leg'ler spine_diagnostic.mean_reversion_return tanımıyla birebir."""
    buckets: dict[str, list[float]] = {f"{sym}|{r}|{w}": [] for r in ("+g", "-g") for w in ("call", "put")}
    for _, r in p.iterrows():
        lab = "+g" if r["regime"] > 0 else "-g"
        if pd.notna(r["call_wall"]) and r["h1"] >= r["call_wall"]:
            buckets[f"{sym}|{lab}|call"].append((r["call_wall"] - r["c1"]) / r["call_wall"])
        if pd.notna(r["put_wall"]) and r["l1"] <= r["put_wall"]:
            buckets[f"{sym}|{lab}|put"].append((r["c1"] - r["put_wall"]) / r["put_wall"])
    out = {}
    for key, legs in buckets.items():
        st = cell_stats(legs)
        st["thesis_expected"] = "MR>0" if "|+g|" in key else "MR<0"
        st["thesis"] = thesis_mark("+g" if "|+g|" in key else "-g", st["mean_bps"])
        out[key] = st
    return out


def main():
    results = {"config_sha": CFG.config_sha(), "run_utc": datetime.now(timezone.utc).isoformat(),
               "script": "backtest/remeasure/RC2_events.py",
               "mr_definition": "spine_diagnostic.mean_reversion_return (call:(cw-c1)/cw; put:(c1-pw)/pw; gross)",
               "panel": {"start": CFG.PANEL_START, "end": CFG.PANEL_END},
               "flag_sets": {}, "panel_n": {},
               "reference": {"eski_dfaz_d63": REF_ESKI, "prelim_157g": REF_PRELIM}}
    for flag_set in FLAG_SETS:
        cells = {}
        for sym in CFG.TRADE_SYMS:
            p = build_panel(sym, flag_set)
            results["panel_n"][f"{sym}|{flag_set}"] = len(p)
            cells.update(eight_cells(p, sym))
        results["flag_sets"][flag_set] = cells

    out = HERE / "RC2_events.json"
    out.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")

    # ---- stdout: ESKİ → PRELIM → FINAL tablo ----
    print(f"RC2.6 EVENT-EDGE 8-HÜCRE FINAL — config_sha={results['config_sha']}  panel {CFG.PANEL_START}..{CFG.PANEL_END}")
    print(f"panel_n: {results['panel_n']}")
    print("MR = spine_diagnostic.mean_reversion_return (gross, bps). Tez: +γ → MR>0 / −γ → MR<0.")
    hdr = (f"{'hücre':16}{'beklenen':9}| {'ESKİ n':>7}{'ESKİ MR':>9} | {'PRELIM MR':>10} | "
           f"{'OWN-LM n':>9}{'MR':>8}{'t':>7}{'tez':>9} | {'IDX-FLAG n':>11}{'MR':>8}{'t':>7}{'tez':>9}")
    print(hdr); print("-" * len(hdr))
    for sym in CFG.TRADE_SYMS:
        for reg in ("+g", "-g"):
            for wall in ("call", "put"):
                key = f"{sym}|{reg}|{wall}"
                e = REF_ESKI["cells"][key]; pre = REF_PRELIM["cells"][key]
                a = results["flag_sets"]["own_livematch"][key]
                b = results["flag_sets"]["index_flag"][key]
                exp = "MR>0" if reg == "+g" else "MR<0"
                print(f"{key:16}{exp:9}| {e['n']:>7}{e['mean_bps']:>+9.1f} | {pre['mean_bps']:>+10} | "
                      f"{a['n']:>9}{str(a['mean_bps']):>8}{str(a['t']):>7}{str(a['thesis']):>9} | "
                      f"{b['n']:>11}{str(b['mean_bps']):>8}{str(b['t']):>7}{str(b['thesis']):>9}")
    print(f"\n→ {out.name} (config_sha dahil). Ölçüm GROSS (maliyet yok; spine M2 tanımı), teşhis-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
