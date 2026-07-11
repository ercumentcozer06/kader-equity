"""
collect_daily — GÜNLÜK forward-collector. SPX(SPY)+NDX(QQQ) vol-surface+GEX snapshot'larını çeker
(yfinance, mid'den hesaplanan IV) + özet satırı output/gamma_forward_ledger.parquet'e ekler (as_of dedup).
Aylar biriktikçe ince-gamma (flip-distance/vanna/charm) OOS-backtest edilebilir hale gelir.

Çalıştır: python collect_daily.py   (günlük; /schedule ile cron'a bağlanabilir. EOD/kapanış-sonrası ideal.)
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
PY = sys.executable
COLS = ["as_of", "computed_at", "spx_spot", "spx_atm_iv", "spx_gex_flip", "spx_gex_net_bn",
        "ndx_spot", "ndx_atm_iv", "ndx_gex_flip", "ndx_gex_net_bn"]
# G1: gamma-seviyeleri ledger'ı (long-format, ticker-başı satır). Playbook kural-2/3'ün forward hammaddesi.
LEVEL_COLS = ["as_of", "ticker", "snapshot_ts", "spot", "put_wall", "call_wall", "ghost", "hvl", "gamma_flip",
              "regime", "shield_z", "shield_short_gamma", "source"]


def _snap(tick: str) -> dict | None:
    subprocess.run([PY, str(ROOT / "screen" / "surface_yf.py"), tick],
                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    p = ROOT / "data" / "cache" / f"surface_{tick.lower()}" / f"{date.today().isoformat()}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    front = next(iter(d.get("surface", {}).values()), {}) if d.get("surface") else {}
    return {"spot": d.get("spot"), "atm_iv": front.get("atm_iv"),
            "flip": d.get("gamma_flip"), "gex": d.get("gex_net_bn_per_1pct")}


def _gamma_levels(tick: str) -> dict | None:
    """gamma_engine'i çalıştır → gamma_<tick> snapshot'ından seviye satırı (G1: analiz yok, sadece topla)."""
    subprocess.run([PY, str(ROOT / "screen" / "gamma_engine.py"), tick],
                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    p = ROOT / "data" / "cache" / f"gamma_{tick.lower()}" / f"{date.today().isoformat()}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return {"as_of": d.get("as_of"), "ticker": tick, "snapshot_ts": d.get("ts"), "spot": d.get("spot"),
            "put_wall": d.get("put_wall"), "call_wall": d.get("call_wall"), "ghost": d.get("ghost"), "hvl": d.get("hvl"),
            "gamma_flip": d.get("gex_flip"), "regime": d.get("regime"), "shield_z": d.get("shield_z"),
            "shield_short_gamma": d.get("shield_short_gamma"), "source": d.get("source")}


def _append_levels(rows: list) -> int:
    rows = [r for r in rows if r]
    if not rows:
        return 0
    p = ROOT / "output" / "gamma_levels_ledger.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=LEVEL_COLS)
    new = pd.DataFrame([{c: r.get(c) for c in LEVEL_COLS} for r in rows])
    if not df.empty:                                            # (as_of,ticker) dedup
        key = df["as_of"].astype(str) + "|" + df["ticker"].astype(str)
        nkey = set(new["as_of"].astype(str) + "|" + new["ticker"].astype(str))
        df = df[~key.isin(nkey)]
    df = pd.concat([df, new], ignore_index=True).sort_values(["as_of", "ticker"])
    df.to_parquet(p)
    return len(df)


def _written_this_run(kind: str, tick: str, started: float) -> bool:
    """Bugünün snapshot dosyası BU KOŞUDA mı yazıldı (mtime >= başlangıç)? 'Dünden kalan dosya var' ≠ taze."""
    p = ROOT / "data" / "cache" / f"{kind}_{tick.lower()}" / f"{date.today().isoformat()}.json"
    return p.exists() and p.stat().st_mtime >= started


def _is_trading_day(d) -> bool:
    """F7 (denetim 2026-07-05): d NYSE işlem günü mü (hafta sonu + tatil; tek takvim kaynağı opex_calendar)."""
    from modules.opex_calendar import is_market_holiday
    return d.weekday() < 5 and not is_market_holiday(d)


def _append_surface(rec: dict) -> int:
    """Surface-özet satırını gamma_forward_ledger.parquet'e ekle (as_of dedup). Döndürür toplam satır."""
    p = ROOT / "output" / "gamma_forward_ledger.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=COLS)
    df = df[df["as_of"].astype(str) != rec["as_of"]] if not df.empty else df    # dedup
    df = pd.concat([df, pd.DataFrame([{c: rec.get(c) for c in COLS}])], ignore_index=True).sort_values("as_of")
    df.to_parquet(p)
    return len(df)


def main() -> int:
    started = time.time()
    snaps = {"SPY": _snap("SPY"), "QQQ": _snap("QQQ")}
    if not snaps["SPY"] and not snaps["QQQ"]:
        print("  [!] hiçbir snapshot alınamadı (yfinance?).")
        return 1
    # G1: gamma-seviyeleri (put/call-wall, HVL, flip, rejim+z) — SPY+QQQ
    lev = {"SPY": _gamma_levels("SPY"), "QQQ": _gamma_levels("QQQ")}

    # ── TAZELİK GARANTİSİ — defterlere YAZMADAN ÖNCE (GEX-FAIL-2 fix) ─────────────────────────────────
    # Eski sürüm stale-türevli satırları exit-1'den ÖNCE iki parquet'e yazıyordu; gün başarısız bitse bile
    # bayat satır defterde kalıyordu. Şimdi: bu koşuda yazılmamış her artefakt 2 kez daha denenir (kademeli
    # bekleme); hâlâ taze değilse defterlere HİÇBİR ŞEY YAZMADAN exit 1 (run_daily fail-loud yakalar). Sessiz bayat YOK.
    for attempt, wait_s in ((1, 45), (2, 90)):                  # kademeli bekleme: yahoo rate-limit pencereleri 20sn'den uzun
        missing = [(k, t) for k in ("surface", "gamma") for t in ("SPY", "QQQ")
                   if not _written_this_run(k, t, started)]
        if not missing:
            break
        print(f"  [!] taze yazılmamış: {missing} → {attempt}. yeniden deneme ({wait_s}sn)...")
        time.sleep(wait_s)
        for k, t in missing:
            if k == "surface":
                snaps[t] = _snap(t) or snaps[t]
            else:
                lev[t] = _gamma_levels(t) or lev.get(t)
    still = [(k, t) for k in ("surface", "gamma") for t in ("SPY", "QQQ")
             if not _written_this_run(k, t, started)]
    if still:
        print(f"  [!!] TAZELİK GARANTİSİ İHLALİ — bu koşuda yazılamayan: {still} → exit 1 (bayat veri defterlere YAZILMADI)")
        return 1

    # ── G1c: NDX ENDEKS-evreni (2026-07-10, iki-evren diverjans arşivi) — BEST-EFFORT: çekirdek
    # tazelik-kapısına GİRMEZ (ince NDX kompleksi fetch-hatası günü ÖLDÜRMEZ; betimsel bacak) ──
    lev["NDX"] = _gamma_levels("NDX")
    if lev["NDX"] is None:
        print("  [~] NDX-endeks gamma alınamadı (best-effort; SPY/QQQ defterleri etkilenmez)")
    # G1c-2: SLV gerçek-OI GEX arşivi (07-11 gamma-lab: hacim-proxy KANITSIZ → gerçek-OI forward-birikimi)
    lev["SLV"] = _gamma_levels("SLV")
    if lev["SLV"] is None:
        print("  [~] SLV gamma alınamadı (best-effort)")
    # G1d: COMBO (endeks+ETF birleşik, SpotGamma-tarzı — Emir kararı 07-11 "profesyoneller nasıl
    # yapıyorsa"); gamma_combo_{spx,ndx} cache'ini yazar; BEST-EFFORT (çekirdek kapıya girmez)
    try:    # Denetim 07-11 P3 ([33]): timeout'suz best-effort asilirsa TUM zinciri kilitliyordu
        subprocess.run([PY, str(ROOT / "screen" / "gamma_combo.py")],
                       cwd=str(ROOT), encoding="utf-8", errors="replace", timeout=600)
    except subprocess.TimeoutExpired:
        print("  [~] gamma_combo TIMEOUT (600s) — atlandi (best-effort)")
    # G1e: naive-net vs SqueezeMetrics nöbetçisi (07-11) — bizim zincir-okuma SM serisiyle
    # ko-hareket ediyor mu; ~6 ayda korelasyon testi. BEST-EFFORT.
    try:
        subprocess.run([PY, str(ROOT / "screen" / "gex_naive_sentinel.py")],
                       cwd=str(ROOT), encoding="utf-8", errors="replace", timeout=300)
    except subprocess.TimeoutExpired:
        print("  [~] gex_naive_sentinel TIMEOUT (300s) — atlandi (best-effort)")

    # ── F7 (denetim 2026-07-05): HAYALET-SATIR koruması — hafta sonu / NYSE tatili koşusu ÖNCEKİ kapanışın
    # donmuş zincirini işlem-görmemiş bir tarihe damgalayıp deftere yazıyordu (gerçek örnekler:
    # 2026-06-13 Cumartesi, 2026-06-19 Juneteenth [spx_spot 06-22 ile birebir], 2026-07-05 Pazar) ve
    # run-tarihi etiketi verinin gerçek yaşını 0'a kelepçeliyordu. Kapalı günde snapshot'lar yine alınır
    # (brief tüketebilir) ama DEFTERLERE SATIR YAZILMAZ — o verinin işlem-günü satırı zaten kendi gününde
    # yazıldı. Betimsel toplayıcı; sinyal matematiği yok. ──
    if not _is_trading_day(date.today()):
        print("  [i] piyasa kapalı (hafta sonu/NYSE tatili) — defter satırı YAZILMADI "
              "(hayalet as_of önlendi; snapshot'lar alındı, gerçek işlem gününün satırı zaten defterde)")
        return 0

    # ── yalnız HER ŞEY TAZE ise defterlere yaz (stale-türevli satır persist ETMEZ) ──
    spy, qqq = snaps["SPY"], snaps["QQQ"]
    rec = {"as_of": date.today().isoformat(), "computed_at": datetime.now(timezone.utc).isoformat(),
           "spx_spot": (spy or {}).get("spot"), "spx_atm_iv": (spy or {}).get("atm_iv"),
           "spx_gex_flip": (spy or {}).get("flip"), "spx_gex_net_bn": (spy or {}).get("gex"),
           "ndx_spot": (qqq or {}).get("spot"), "ndx_atm_iv": (qqq or {}).get("atm_iv"),
           "ndx_gex_flip": (qqq or {}).get("flip"), "ndx_gex_net_bn": (qqq or {}).get("gex")}
    nsurf = _append_surface(rec)
    print(f"  + {rec['as_of']}: SPX atm {rec['spx_atm_iv']} flip {rec['spx_gex_flip']} | "
          f"NDX atm {rec['ndx_atm_iv']} flip {rec['ndx_gex_flip']}  → surface-ledger {nsurf} satır")
    print(f"  ledger → {ROOT/'output'/'gamma_forward_ledger.parquet'}")
    nlev = _append_levels([r for r in (lev.get("SPY"), lev.get("QQQ"), lev.get("NDX"), lev.get("SLV")) if r])
    print(f"  gamma-levels ledger → {ROOT/'output'/'gamma_levels_ledger.parquet'} ({nlev} satır)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
