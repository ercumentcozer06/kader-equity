"""
RC3_synthesis.py — RC3 SENTEZ yardimci olcumleri (TESHIS-ONLY; yeni strateji/esik/parametre YOK).
1) trial_ledger.csv yazar (TEK yazici): prior K=10 nominal + replacement + amendment (K+=10 -> K_CURRENT).
2) D6-guncelleme: battery FINAL uyeleri icin t>=2 & DSR>0(K=K_CURRENT) kalan-N (RC2_battery.build_panel/member_pnl
   + D6_power formulleri IMPORT — tek kaynak, yeni matematik yok).
3) gamma_dollar panel-medyanlari (FINAL seriler) — ①/③ icin son sayilar.
Cikti: trial_ledger.csv + RC3_d6_update.json (config_sha metadata).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "backtest"))
sys.path.insert(0, str(ROOT / "backtest" / "diagnosis"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import config as CFG                                      # noqa: E402
from RC2_battery import build_panel, member_pnl           # noqa: E402  (tek kaynak — byte-ayni P&L)
from D6_power import ann_sharpe, moments, n_for_t2, n_for_dsr_pos, dsr as d6_dsr  # noqa: E402

SHA = CFG.config_sha()
BATTERY_JSON = json.loads((HERE / "RC2_battery_results.json").read_text(encoding="utf-8"))
assert BATTERY_JSON["config_sha"] == SHA, "battery config_sha uyusmuyor"

DIAG_MD = ROOT / "backtest" / "DIAGNOSIS.md"
DFAZ_DECLARED = datetime.fromtimestamp(os.path.getmtime(DIAG_MD), tz=timezone.utc).isoformat(timespec="seconds")
AMD_DECLARED = CFG.AMENDMENT["declared_utc"]


def write_ledger() -> int:
    rows = []
    # (1) prior K=10 nominal: D-FAZ ailesi = flag-bagimli 5 uye x 2 sembol (K_PRIOR ile birebir)
    i = 0
    for sym in CFG.TRADE_SYMS:
        for m in CFG.FLAG_DEPENDENT:
            i += 1
            rows.append([f"P{i:02d}", m, sym, "dfaz_single_expiry", "prior-nominal",
                         "d-faz-original", DFAZ_DECLARED, SHA])
    assert i == CFG.K_PRIOR, f"prior satir sayisi {i} != K_PRIOR {CFG.K_PRIOR}"
    # (2) replacement: battery P&L rows (class=replacement) + olcum-uyeleri (own flag-setleri) re-run'lari
    j = 0
    for r in BATTERY_JSON["rows"]:
        if r.get("class") == "replacement":
            j += 1
            rows.append([f"R{j:02d}", r["member"], r["sym"], r["flag"], "replacement",
                         "instrument-fix", AMD_DECLARED, SHA])
    for sym in CFG.TRADE_SYMS:
        for fs in ("livematch_own", "fullsurface_own"):
            for m in CFG.BATTERY_MEAS:
                j += 1
                rows.append([f"R{j:02d}", m, sym, fs, "replacement",
                             "instrument-fix", AMD_DECLARED, SHA])
    # (3) amendment: INDEX-FLAG flag-bagimli 5 uye x 2 sembol = +K_AMENDMENT (P&L 3 + olcum 2)
    k = 0
    for sym in CFG.TRADE_SYMS:
        for m in CFG.FLAG_DEPENDENT:
            k += 1
            rows.append([f"A{k:02d}", m, sym, "index_flag", "amendment/new-trial",
                         f"instrument-quality ({CFG.AMENDMENT['id']}; K+={CFG.K_AMENDMENT} -> K_CURRENT={CFG.K_CURRENT})",
                         AMD_DECLARED, SHA])
    assert k == CFG.K_AMENDMENT, f"amendment satir sayisi {k} != K_AMENDMENT {CFG.K_AMENDMENT}"
    with open(CFG.TRIAL_LEDGER, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial_id", "member", "sym", "flag", "class", "reason", "declared_utc", "config_sha"])
        w.writerows(rows)
    print(f"trial_ledger.csv: {len(rows)} satir (prior {CFG.K_PRIOR} + replacement {j} + amendment {CFG.K_AMENDMENT}); K_CURRENT={CFG.K_CURRENT}")
    return len(rows)


def d6_update() -> list[dict]:
    out = []
    for sym in CFG.TRADE_SYMS:
        for flag_set in CFG.FLAG_SETS:
            p = build_panel(sym, flag_set)
            if p is None:
                continue
            for m in CFG.BATTERY_PNL:
                if m in CFG.FLAG_FREE and flag_set != "livematch_own":
                    continue
                net = member_pnl(p, m).dropna()
                N = len(net)
                sr_ann = ann_sharpe(net.values)
                sr_d, g3, g4, _n = moments(net.values)
                rec = {"sym": sym, "flag": flag_set if m not in CFG.FLAG_FREE else "none(livematch-panel)",
                       "member": m, "n": N, "sharpe_ann": round(sr_ann, 2),
                       "t_now": round(sr_d * (N ** 0.5), 2),
                       "dsr_now_K": CFG.K_CURRENT,
                       "dsr_now": round(d6_dsr(sr_d, g3, g4, N, CFG.K_CURRENT), 3) if sr_d > 0 else 0.0}
                if sr_d <= 0:
                    rec.update({"n_t2": None, "n_dsr": None, "n_gerekli": None,
                                "kalan_n": None, "forward_yil": None, "parali_tarih_yil": None,
                                "not": "Sharpe<=0: bu yonde N tanimsiz (yon-ters/sifir)"})
                else:
                    import math
                    nt2 = n_for_t2(sr_d)
                    ndsr = n_for_dsr_pos(sr_d, g3, g4, CFG.K_CURRENT)
                    nreq = max(nt2, ndsr)
                    if not math.isfinite(nreq):
                        rec.update({"n_t2": (int(nt2) if math.isfinite(nt2) else None),
                                    "n_dsr": (int(ndsr) if math.isfinite(ndsr) else None),
                                    "n_gerekli": None, "kalan_n": None,
                                    "forward_yil": None, "parali_tarih_yil": None,
                                    "not": "gereken-N > 5M gun (pratikte sonsuz; edge ~0)"})
                    else:
                        kalan = max(0, int(nreq) - N)
                        rec.update({"n_t2": int(nt2), "n_dsr": int(ndsr), "n_gerekli": int(nreq),
                                    "kalan_n": kalan,
                                    "forward_yil": round(kalan / 252.0, 1),
                                    "parali_tarih_yil": round(kalan / 252.0, 1)})
                out.append(rec)
    return out


def gamma_medians() -> dict:
    res = {}
    start, end = pd.Timestamp(CFG.PANEL_START), pd.Timestamp(CFG.PANEL_END)
    for mode in ("fullsurface", "livematch"):
        for sym in CFG.SYMS:
            df = pd.read_parquet(CFG.level_path(mode, sym))
            w = df[(df.index >= start) & (df.index <= end)]
            res[f"{mode}_{sym.lower()}"] = {
                "n": int(len(w)),
                "gamma_dollar_med_bn": round(float(w["gamma_dollar"].median()) / 1e9, 2),
            }
    # eski kirik tek-expiry seriler (kiyas) — gamma_dollar kolonu varsa
    for sym in CFG.TRADE_SYMS:
        f = CFG.CACHE / f"level_series_{sym.lower()}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            if "gamma_dollar" in df.columns:
                w = df[(df.index >= start) & (df.index <= end)]
                res[f"old_dfaz_{sym.lower()}"] = {
                    "n": int(len(w)),
                    "gamma_dollar_med_bn": round(float(w["gamma_dollar"].median()) / 1e9, 2),
                }
            else:
                res[f"old_dfaz_{sym.lower()}"] = {"n": int(len(df)), "gamma_dollar_med_bn": None,
                                                  "not": "kolon yok (D-FAZ semasi)"}
    return res


def main() -> int:
    n_ledger = write_ledger()
    d6 = d6_update()
    gm = gamma_medians()
    out = {"config_sha": SHA, "script": "backtest/remeasure/RC3_synthesis.py",
           "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "K_current": CFG.K_CURRENT, "ledger_rows": n_ledger,
           "dfaz_declared_utc_proxy": DFAZ_DECLARED,
           "d6_update": d6, "gamma_dollar_medians": gm}
    (HERE / "RC3_d6_update.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\nD6 GUNCEL (K=%d) — yalniz Sharpe>0 uyeler N alir:" % CFG.K_CURRENT)
    for r in d6:
        if r.get("n_gerekli"):
            print(f"  {r['sym']} {r['flag']:20} {r['member']:10} SR {r['sharpe_ann']:+5.2f} t {r['t_now']:+5.2f} "
                  f"N {r['n']} -> gerekli {r['n_gerekli']} (t2 {r['n_t2']}/dsr {r['n_dsr']}) "
                  f"kalan {r['kalan_n']} = {r['forward_yil']}y")
        else:
            print(f"  {r['sym']} {r['flag']:20} {r['member']:10} SR {r['sharpe_ann']:+5.2f} t {r['t_now']:+5.2f} -> yon-ters/sifir")
    print("\ngamma$ medyan (panel, bn):")
    for k, v in gm.items():
        print(f"  {k:22} n={v['n']:>3}  med={v.get('gamma_dollar_med_bn')}")
    print("\nRC3_d6_update.json yazildi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
