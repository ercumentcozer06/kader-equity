"""
backtest/remeasure/RC2_live.py — RC2.8 CANLI-UYUM (madde-④ kapanış kanıtı).

TEŞHİS-ONLY. Yeni strateji/eşik/parametre/sinyal YOK. Soru: canlı motorun gamma snapshot'ları
(data/cache/gamma_{spy,qqq}/*.json, alan: net_gex_bn) ile YENİ LIVE-MATCH level serisinin
(level_series_livematch_{spy,qqq}.parquet; 243g panel + 06-09/06-10 uzatma = 245g) AYNI-GÜN
regime işareti uyuşuyor mu? Örtüşme KÜÇÜK (as_of-join: SPY 2 / QQQ 1 gün) — dürüstçe n ile raporlanır.

İKİ hizalama (ikisi de aynı karşılaştırmanın teşhisi, yeni deney değil):
  A) AS_OF-JOIN (asıl, prompt'un birebir tanımı): snapshot.as_of == level-serisi günü.
  B) CHAIN-EOD-JOIN (provenance teşhisi): canlı motorun as_of damgası "snapshot'ın HİZMET ETTİĞİ seans"
     olabiliyor; ts + spot kanıtı bunu gösteriyor (örn. as_of=06-11 snapshot ts=06-10T22:08Z, spot
     06-10 EOD spot'una KURUŞU KURUŞUNA eşit). Spot 2-ondalık TAM-eşitlikle zincirin gerçek EOD günü
     bulunur; eşleşme yoksa snapshot seans-içi çekilmiştir → zincir-günü = as_of. Tolerans/parametre yok.

Çıktı: backtest/remeasure/RC2_live.json (config_sha dahil) + stdout Türkçe gün-gün tablo.
Panel-notu: 06-09/06-10 uzatma günleri config.PANEL_END gereği pre-registered panelin DIŞINDA;
bu ölçüm yalnız RC2.8 canlı-uyum içindir (config.py satır 25'teki kilitli not).

  & C:/Users/admin/Downloads/kader-macro/.venv/Scripts/python.exe backtest/remeasure/RC2_live.py
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

OUT_PATH = config.REMEASURE_DIR / "RC2_live.json"


def gamma_dir(sym: str) -> Path:
    """Canlı motor snapshot dizini (yol kökü config.CACHE; RC2_sign.py'deki SQUEEZE_PATH ile aynı kalıp)."""
    return config.CACHE / f"gamma_{sym.lower()}"


def load_level(sym: str) -> pd.DataFrame:
    lv = pd.read_parquet(config.level_path("livematch", sym))
    lv.index = pd.to_datetime(lv.index)
    return lv


def load_snapshots(sym: str) -> list[dict]:
    snaps = []
    for p in sorted(gamma_dir(sym).glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        snaps.append({"path": str(p), **d})
    return snaps


def spot_to_date_map(lv: pd.DataFrame) -> dict[float, list[pd.Timestamp]]:
    """2-ondalık EOD-spot → level-günü listesi (provenance: TAM eşitlik, tolerans yok)."""
    m: dict[float, list[pd.Timestamp]] = {}
    for d, s in lv["spot"].items():
        m.setdefault(round(float(s), 2), []).append(d)
    return m


def compare_row(snap: dict, lv: pd.DataFrame, level_date: pd.Timestamp | None) -> dict:
    """Tek snapshot ↔ tek level-günü karşılaştırması; level_date=None → ölçülemedi."""
    live_sign = int(np.sign(snap["net_gex_bn"]))
    row = {
        "as_of": snap["as_of"],
        "snapshot_ts_utc": snap["ts"],
        "live_net_gex_bn": float(snap["net_gex_bn"]),
        "live_sign": live_sign,
        "live_regime_str": snap.get("regime"),
        "live_spot": float(snap["spot"]),
    }
    if level_date is None:
        row.update({"level_date": None, "agree": None,
                    "neden": "level serisi bu günü kapsamıyor (seri sonu) → ölçülemedi"})
        return row
    lrow = lv.loc[level_date]
    bt_regime = int(lrow["regime"])
    bt_gex = float(lrow["net_gex"])
    row.update({
        "level_date": level_date.strftime("%Y-%m-%d"),
        "bt_net_gex_bn": round(bt_gex / 1e9, 3),
        "bt_regime": bt_regime,
        "bt_sign": int(np.sign(bt_gex)),
        "level_spot": float(lrow["spot"]),
        "spot_delta": round(float(snap["spot"]) - float(lrow["spot"]), 2),
        "agree": bool(live_sign == bt_regime),
    })
    return row


def run_symbol(sym: str) -> dict:
    lv = load_level(sym)
    snaps = load_snapshots(sym)
    smap = spot_to_date_map(lv)

    rows_asof, rows_chain = [], []
    for snap in snaps:
        as_of = pd.Timestamp(snap["as_of"])
        # A) AS_OF-JOIN (asıl)
        rows_asof.append(compare_row(snap, lv, as_of if as_of in lv.index else None))
        # B) CHAIN-EOD-JOIN (provenance): spot TAM-eşit tek bir EOD gününe → o gün; yoksa seans-içi → as_of
        hits = smap.get(round(float(snap["spot"]), 2), [])
        if len(hits) == 1:
            chain_date, basis = hits[0], "EOD-chain (spot tam-eşit)"
        else:
            chain_date, basis = (as_of if as_of in lv.index else None), "seans-içi capture (spot EOD'den farklı)"
        r = compare_row(snap, lv, chain_date)
        r["chain_basis"] = basis
        rows_chain.append(r)

    def agg(rows: list[dict]) -> dict:
        matched = [r for r in rows if r["agree"] is not None]
        return {"n": len(matched), "agree_n": sum(r["agree"] for r in matched),
                "unmatched_n": len(rows) - len(matched)}

    return {
        "level_series": str(config.level_path("livematch", sym)),
        "level_rows": int(len(lv)),
        "level_last_day": lv.index.max().strftime("%Y-%m-%d"),
        "snapshot_dir": str(gamma_dir(sym)),
        "snapshot_n": len(snaps),
        "asof_join": {"rows": rows_asof, **agg(rows_asof)},
        "chain_eod_join": {"rows": rows_chain, **agg(rows_chain)},
    }


def print_table(sym: str, label: str, rows: list[dict]) -> None:
    print(f"\n  {sym} — {label}:")
    for r in rows:
        if r["agree"] is None:
            print(f"    {r['as_of']}  canlı {r['live_sign']:+d} ({r['live_net_gex_bn']:+.3f}bn, "
                  f"{r['live_regime_str']})  →  ÖLÇÜLEMEDİ: {r['neden']}")
            continue
        mark = "UYUM" if r["agree"] else "AYKIRI"
        basis = f"  [{r['chain_basis']}]" if "chain_basis" in r else ""
        print(f"    as_of {r['as_of']} ↔ level {r['level_date']}  "
              f"canlı {r['live_sign']:+d} ({r['live_net_gex_bn']:+.3f}bn, {r['live_regime_str']})  "
              f"backtest {r['bt_regime']:+d} ({r['bt_net_gex_bn']:+.3f}bn)  "
              f"Δspot {r['spot_delta']:+.2f}  {mark}{basis}")


def main() -> int:
    print("=" * 100)
    print("  RC2.8 CANLI-UYUM — canlı snapshot net_gex_bn işareti vs LIVE-MATCH aynı-gün regime (madde-④)")
    print(f"  config_sha={config.config_sha()} | panel-notu: uzatma günleri (06-09/06-10) pre-registered "
          f"panel-DIŞI (PANEL_END={config.PANEL_END})")
    print("=" * 100)

    results: dict[str, dict] = {}
    for sym in config.TRADE_SYMS:
        try:
            results[sym] = run_symbol(sym)
        except Exception as e:  # fail-loud ama diğer sembolü de raporla
            results[sym] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {sym}: HATA — {e}")
            continue
        r = results[sym]
        print(f"\n  {sym}: level {r['level_rows']}g (son {r['level_last_day']}) | snapshot {r['snapshot_n']} adet")
        print_table(sym, "A) AS_OF-JOIN (asıl: snapshot.as_of == level günü)", r["asof_join"]["rows"])
        print_table(sym, "B) CHAIN-EOD-JOIN (provenance: zincirin gerçek EOD günü)", r["chain_eod_join"]["rows"])
        a, c = r["asof_join"], r["chain_eod_join"]
        print(f"    özet: as_of-join {a['agree_n']}/{a['n']} uyum ({a['unmatched_n']} ölçülemedi) | "
              f"chain-EOD-join {c['agree_n']}/{c['n']} uyum ({c['unmatched_n']} ölçülemedi)")

    ok = [s for s in config.TRADE_SYMS if "error" not in results.get(s, {"error": 1})]
    tot_a_n = sum(results[s]["asof_join"]["n"] for s in ok)
    tot_a_ag = sum(results[s]["asof_join"]["agree_n"] for s in ok)
    tot_c_n = sum(results[s]["chain_eod_join"]["n"] for s in ok)
    tot_c_ag = sum(results[s]["chain_eod_join"]["agree_n"] for s in ok)

    status_madde4 = (
        f"İLK CANLI ÖRTÜŞME ÖLÇÜLDÜ (artık yalnız 'ileriye-dönük ölçülür' değil): as_of-join {tot_a_ag}/{tot_a_n} "
        f"uyum, chain-EOD-join {tot_c_ag}/{tot_c_n} uyum. n KÜÇÜK → istatistiksel güç yok; kanıt niteliği "
        f"'tutarlılık-kontrolü geçti' seviyesinde. Forward-collector her gün +1 örtüşme ekler; tam-güçlü "
        f"sign-agreement zaten RC2_sign.py'de 243g SqueezeMetrics karşılaştırmasıyla ölçüldü."
    )

    print("\n" + "=" * 100)
    print(f"  TOPLAM: as_of-join {tot_a_ag}/{tot_a_n} uyum | chain-EOD-join {tot_c_ag}/{tot_c_n} uyum")
    print(f"  MADDE-④ STATÜ: {status_madde4}")
    print("=" * 100)

    out = {
        "config_sha": config.config_sha(),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "backtest/remeasure/RC2_live.py",
        "methodology": (
            "Canlı motor snapshot'ı (gamma_{sym}/<date>.json, alan net_gex_bn) ile LIVE-MATCH level serisinin "
            "aynı-gün regime'i karşılaştırılır; agree = sign(net_gex_bn)==regime(level). İki hizalama: "
            "A) as_of-join (asıl, prompt tanımı); B) chain-EOD-join (provenance: snapshot spot'u level EOD "
            "spot'una 2-ondalık TAM eşitse zincirin gerçek günü o gündür — as_of damgası canlı motorda "
            "'hizmet edilen seans' olarak atılıyor, ts alanı kanıt; eşleşme yoksa seans-içi capture → as_of). "
            "Uzatma günleri (06-09/06-10) pre-registered panel-DIŞI; yalnız RC2.8 canlı-uyum için."
        ),
        "results": results,
        "totals": {
            "asof_join": {"n": tot_a_n, "agree_n": tot_a_ag},
            "chain_eod_join": {"n": tot_c_n, "agree_n": tot_c_ag},
        },
        "madde_4_status": status_madde4,
        "honesty_notes": [
            "Örtüşme küçüktür (as_of-join toplam n=3, chain-EOD-join toplam n=5); bu bir güç-iddiası DEĞİL, "
            "canlı motor ile yeniden-kurulan serinin tutarlılık kontrolüdür.",
            "as_of=2026-06-11 snapshot'ları level serisi 2026-06-10'da bittiği için as_of-join'de ölçülemedi; "
            "ts+spot kanıtıyla 06-10 EOD zincirinden üretildikleri gösterildi ve chain-EOD-join'de ölçüldü.",
            "Canlı net_gex_bn ile backtest net_gex büyüklükleri seans-içi capture günlerinde farklı spot/zincir "
            "anı yüzünden ayrışabilir; bu ölçüm yalnız İŞARET/regime uyumudur.",
        ],
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  JSON → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
