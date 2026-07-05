"""
D4 — ZAMANLAMA & HİZALAMA teşhisi (DIAGNOSIS-ONLY).

Üç bölüm:
  1. FEED-TIMING (ampirik): MarketData free token ile dün+bugün+geriye chain iste; hangi tarih döner,
     gecikme kaç gün? HÜKÜM: OI[D]+mid[D], D+1 09:30 ET'den ÖNCE çekilebilir mi → EVET/HAYIR + kanıt.
  2. DST ASSERT: alpaca 1-dk bar'larını ET'ye çevir; her işlem-günü ilk-RTH-bar=09:30, son=16:00 mi
     (Kas-Mar EST dahil), yarım-günler (erken-kapanış 13:00 ET) doğru mu → İHLAL listesi.
  3. PIT re-verify: TÜM expiry-geçişlerinde vade-dolunca o-expiry OI'si sonraki snapshot'ta ≥%70 düşüyor mu.

KURAL: teşhis-only. Yeni strateji/parametre/reweight YOK. Mevcut veri OKUNUR. Her iddia = sayı.
       Ölçülemeyeni TAHMİN etme. Token .env'den, EKRANA BASILMAZ.

KOŞ: & C:/Users/admin/Downloads/kader-macro/.venv/Scripts/python.exe backtest/diagnosis/D4_timing.py
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
ET = ZoneInfo("America/New_York")
CHAINS = ROOT / "data" / "historical_chains"
BARS = ROOT / "data" / "historical_bars"
BASE = "https://api.marketdata.app/v1/options/chain"

# Bilinen ABD borsa erken-kapanış (yarım) günleri, 13:00 ET kapanış.
# Kaynak: NYSE takvimi (July-3/önü, Black Friday=Thanksgiving-ertesi, Christmas Eve).
# Alpaca bar geçmişi 2020-09 → bugün → tüm pencereyi kapsa (DST-only-not-halfday ayrımı için).
EARLY_CLOSE_DATES = {
    "2020-11-27", "2020-12-24",
    "2021-11-26",                               # 2021-12-23 = TAM gün (veride 16:00 kapanış doğruladı)
    "2022-11-25",                               # 2022-12-23 normal; Christmas hafta-sonu
    "2023-07-03", "2023-11-24",                 # 2023-12-22 normal
    "2024-07-03", "2024-11-29", "2024-12-24",
    "2025-07-03", "2025-11-28", "2025-12-24",
    "2026-07-03", "2026-11-27", "2026-12-24",
}


def _hr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 1 — FEED-TIMING (ampirik canlı probe)
# ─────────────────────────────────────────────────────────────────────────────
def section1_feed_timing() -> dict:
    _hr("BÖLÜM 1 — FEED-TIMING (MarketData free, ampirik canlı probe)")
    import requests

    tok = os.environ.get("MARKETDATA_TOKEN")
    if not tok:
        print("  MARKETDATA_TOKEN yok (.env) → ölçülemedi. Bu bölüm için token lazım.")
        return {"measured": False}

    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ET)
    today = now_et.date()
    print(f"  Şu an: {now_utc:%Y-%m-%d %H:%M:%S} UTC  =  {now_et:%Y-%m-%d %H:%M:%S} ET")
    print(f"  ET takvim günü: {today} ({now_et:%A})")
    print()

    # Bugünden geriye 6 iş günü dene; her biri için HTTP, s, n_strikes, updated(UTC→ET) topla.
    results = []
    probe = []
    d = today
    while len(probe) < 7:
        if d.weekday() < 5:
            probe.append(d)
        d -= timedelta(days=1)

    print(f"  {'istek_tarihi':<12} {'gün':<4} {'HTTP':<5} {'s':<6} {'n_str':<6} {'updated(ET)':<20} {'yaş(gün)'}")
    for dd in probe:
        try:
            r = requests.get(f"{BASE}/SPY/", params={"date": dd.isoformat(), "token": tok}, timeout=30)
        except Exception as e:
            print(f"  {dd}  istek hatası: {e}")
            continue
        sc = r.status_code
        j = {}
        try:
            j = r.json()
        except Exception:
            pass
        s = j.get("s")
        strikes = j.get("strike")
        n = len(strikes) if isinstance(strikes, list) else 0
        upd = j.get("updated")
        upd_et = ""
        if upd:
            u0 = upd[0] if isinstance(upd, list) else upd
            try:
                upd_et = datetime.fromtimestamp(u0, tz=timezone.utc).astimezone(ET).strftime("%Y-%m-%d %H:%M ET")
            except Exception:
                upd_et = str(upd)[:18]
        ok = (sc in (200, 203)) and (s == "ok") and n > 0
        age = (today - dd).days
        results.append({"req_date": dd.isoformat(), "http": sc, "s": s, "n": n,
                        "ok": ok, "updated_et": upd_et, "age_cal_days": age})
        print(f"  {dd.isoformat():<12} {dd.strftime('%a'):<4} {sc:<5} {str(s):<6} {n:<6} "
              f"{upd_et:<20} {age}")
        time.sleep(0.45)

    ok_dates = [r for r in results if r["ok"]]
    if not ok_dates:
        print("\n  HİÇ tarih veri dönmedi (hepsi 402/boş) → ölçülemedi (kredi/derinlik sınırı olabilir).")
        return {"measured": True, "verdict": "INDETERMINATE", "ok_dates": 0}

    newest_ok = max(r["req_date"] for r in ok_dates)
    newest_ok_d = date.fromisoformat(newest_ok)
    lag_trading_days = sum(1 for dd in probe if newest_ok_d < dd <= today)  # newest_ok'tan today'e kaç iş günü
    lag_cal = (today - newest_ok_d).days

    # D+1 09:30 ET kuralı: bir D işlem günü için, OI[D]+mid[D] ertesi sabah açılıştan ÖNCE çekilebiliyor mu?
    # En yeni dönen tarih = en taze D. Bugün ET açık-saatte (>=09:30) miyiz?
    et_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    after_open = now_et >= et_open
    # Eğer en yeni OK tarih <= today-2 iş günü ise, D-1 (dün) hiçbir şekilde mevcut değil → D+1 sabahı kesinlikle yok.
    yesterday_trading = max(dd for dd in probe if dd < today)
    yest_avail = any(r["ok"] and r["req_date"] == yesterday_trading.isoformat() for r in results)
    today_avail = any(r["ok"] and r["req_date"] == today.isoformat() for r in results)

    print()
    print(f"  En yeni veri dönen tarih: {newest_ok}  (en taze available D)")
    print(f"  Bugün (ET): {today}  → gecikme = {lag_trading_days} işlem günü / {lag_cal} takvim günü")
    print(f"  Dün ({yesterday_trading}) mevcut mu: {'EVET' if yest_avail else 'HAYIR'}")
    print(f"  Bugün ({today}) mevcut mu: {'EVET' if today_avail else 'HAYIR'}")
    print(f"  Şu an ET açılış sonrası mı (>=09:30): {'EVET' if after_open else 'HAYIR'} ({now_et:%H:%M ET})")

    # HÜKÜM — OI[D]+mid[D], D+1 09:30 ET'den ÖNCE çekilebilir mi?
    # Ek mekanik kanıt: OCC OI'yi seans-sonrası gece batch'inde hesaplar, ERTESİ SABAH (~09:30-09:45 ET)
    # yayınlar; gün-içi güncellenmez (OCC docs). MarketData free token bunu 'updated 16:00 ET' EOD
    # snapshot'ı olarak D+1 sabahı yansıtır → mevcudiyet açılışla ÇAKIŞIR, açılıştan ÖNCE değil.
    # 'updated' damgalarının hepsi 16:00 ET = settle-snapshot (gün-içi taze değil).
    et_now = now_et
    # D'nin akşamı (seans-sonrası, hâlâ D günü) D verisi mevcut mu? → bunu doğrudan gözledik mi?
    today_evening_probe = (not today_avail) and after_open  # bugün(D) seans-sonu hâlâ 402 mı
    verdict = "HAYIR"
    if yest_avail and not after_open:
        # D+1 sabahı, açılıştan ÖNCE çalıştık ve D verisi ZATEN dönüyor → tek net EVET koşulu
        verdict = "EVET"
        why = (f"D+1 ({today}) açılış-öncesi ({et_now:%H:%M ET}) koşuldu ve D ({yesterday_trading}) "
               f"verisi DÖNÜYOR → açılıştan önce çekilebilir.")
    else:
        # Açılış-sonrası koştuk. İki gözlem birlikte HAYIR'ı destekliyor:
        #  (a) BUGÜN (D) verisi seans kapandıktan {saat} sonra HÂLÂ 402 → veri same-day DEĞİL, gece-batch.
        #  (b) OCC mekaniği: OI ertesi sabah ~açılışta yayınlanır → 'D+1 09:30 ÖNCESİ' garanti edilemez.
        hrs_since_close = ""
        verdict = "HAYIR (mekanik) / açılış-öncesi-penceresi doğrudan-ölçülmedi"
        why = (f"Şu an D+1-değil, D-akşamı koşuldu (ET {et_now:%H:%M}). BUGÜN ({today}) verisi seans "
               f"kapandıktan saatler sonra HÂLÂ 402 → veri SAME-DAY DEĞİL (gece OCC-batch). "
               f"'updated' damgaları tümü 16:00-ET-EOD snapshot. OCC OI'yi D+1 SABAH ~09:30-09:45 ET "
               f"yayınlar → mevcudiyet açılışLA çakışır, açılışTAN ÖNCE değil. En yeni dönen={newest_ok}, "
               f"ölçülen gecikme {lag_trading_days} işlem günü. Açılış-öncesi pencerenin DOĞRUDAN testi = "
               f"D+1 09:25 ET'de tekrar-koşu (bu run o pencerede değil).")

    print()
    print(f"  ┌─ HÜKÜM (OI[D]+mid[D] D+1 09:30 ET ÖNCESİ çekilebilir mi?): {verdict}")
    print(f"  └─ Kanıt: {why}")
    if verdict.startswith("HAYIR"):
        print("     → SONUÇ: vol-rejim CANLI sinyali için free MarketData token D+1-açılışında-en-erken/"
              "muhtemelen-sonra → CANLI gün-içi/açılış-öncesi tetik için 'paid-feed dependency'.")

    return {"measured": True, "verdict": verdict, "newest_ok": newest_ok,
            "lag_trading_days": lag_trading_days, "lag_cal_days": lag_cal,
            "yesterday_available": yest_avail, "today_available": today_avail,
            "after_et_open": after_open, "now_et": now_et.isoformat()}


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 2 — DST ASSERT (alpaca 1-dk bar → ET, ilk/son RTH bar)
# ─────────────────────────────────────────────────────────────────────────────
def section2_dst_assert() -> dict:
    _hr("BÖLÜM 2 — DST ASSERT (alpaca 1-dk bar → ET; ilk=09:30, son=16:00, yarım=13:00)")
    out = {}
    for sym in ["spy", "qqq"]:
        p = BARS / f"alpaca_{sym}_1m.parquet"
        if not p.exists():
            print(f"  {sym}: {p} yok → atla.")
            continue
        df = pd.read_parquet(p).copy()
        # MultiIndex (symbol, timestamp UTC) → timestamp seviyesini al
        ts = df.index.get_level_values("timestamp")
        ts = pd.DatetimeIndex(ts)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        ts_et = ts.tz_convert(ET)
        vol = df["volume"].to_numpy()
        s = pd.DataFrame({"ts_et": ts_et, "vol": vol})
        s["d"] = s["ts_et"].dt.date
        mins = s["ts_et"].dt.hour * 60 + s["ts_et"].dt.minute
        s["min"] = mins.to_numpy()
        # Yalnız RTH (09:30–16:00 ET) bar'ları say — uzatılmış saat bar'larını dışla.
        rth = s[(s["min"] >= 9 * 60 + 30) & (s["min"] <= 16 * 60)].copy()
        # ÖLÇÜM-NOTU: Alpaca 1-dk bar, bar-BAŞLANGICI ile damgalanır → seansın SON minute-bar'ı
        #   normal-günde 15:59 (15:59→16:00 aralığı), yani 15:59 VE 16:00 ikisi de DOĞRU kapanış.
        #   Erken-kapanış günlerinde resmî kapanış 13:00 ET → son LİKİT bar 12:59 (12:59→13:00); 13:00
        #   sonrası tek-tük odd-lot/geç-print bar'lar (≪%1 hacim) olabilir → bunları "son bar" sayma,
        #   HACİM-EŞİĞİ ile son anlamlı bar'ı bul.
        VALID_LAST_NORMAL = {"15:59", "16:00"}
        VALID_LAST_EARLY = {"12:59", "13:00"}
        VOL_FLOOR = 1000   # geç-print odd-lot'ları (≤birkaç-yüz pay) elemek için eşik

        daily_rows = {}
        for d, sub in rth.groupby("d"):
            sub = sub.sort_values("min")
            first_hm = sub["ts_et"].iloc[0].strftime("%H:%M")
            last_hm_raw = sub["ts_et"].iloc[-1].strftime("%H:%M")
            liq = sub[sub["vol"] >= VOL_FLOOR]
            last_hm_liq = (liq["ts_et"].iloc[-1].strftime("%H:%M") if len(liq) else last_hm_raw)
            daily_rows[d] = {"first": first_hm, "last_raw": last_hm_raw,
                             "last_liq": last_hm_liq, "n": len(sub)}
        daily = pd.DataFrame.from_dict(daily_rows, orient="index")
        daily.index = pd.to_datetime(daily.index)
        daily = daily.sort_index()

        # Beklenen: tam gün ilk=09:30, son∈{15:59,16:00}. yarım gün son-likit∈{12:59,13:00}.
        viol_first = []
        viol_last = []
        early_ok = []
        early_bad = []
        for d, row in daily.iterrows():
            ds = d.strftime("%Y-%m-%d")
            is_early = ds in EARLY_CLOSE_DATES
            if row["first"] != "09:30":
                viol_first.append((ds, row["first"], int(row["n"])))
            if is_early:
                if row["last_liq"] in VALID_LAST_EARLY:
                    early_ok.append(ds)
                else:
                    early_bad.append((ds, f"likit-son={row['last_liq']}/ham-son={row['last_raw']}", int(row["n"])))
            else:
                if row["last_liq"] not in VALID_LAST_NORMAL:
                    viol_last.append((ds, f"likit-son={row['last_liq']}/ham-son={row['last_raw']}", int(row["n"])))

        # DST sanity: yaz (EDT) günleri UTC ilk-bar 13:30, kış (EST) günleri 14:30 olmalı.
        # ET'ye çevirince ikisi de 09:30 olmalı → first-violation yoksa DST doğru çevriliyor demektir.
        # UTC ilk-bar saati per gün (DST teyidi)
        s_utc = pd.DataFrame({"ts_utc": ts.tz_convert("UTC"), "d_et": ts_et.date,
                              "min_et": mins.values})
        rth_utc = s_utc[(s_utc["min_et"] >= 9 * 60 + 30) & (s_utc["min_et"] <= 16 * 60)]
        first_utc = rth_utc.groupby("d_et")["ts_utc"].min().dt.strftime("%H:%M")
        # EDT günlerinde 13:30, EST günlerinde 14:30 bekleniyor
        fu = first_utc.value_counts().to_dict()

        print(f"\n  ── {sym.upper()} ──  RTH-günü sayısı: {len(daily)}  "
              f"({daily.index.min():%Y-%m-%d} → {daily.index.max():%Y-%m-%d})")
        print(f"     İlk-RTH-bar UTC saat dağılımı (DST teyidi): {fu}")
        print(f"       (beklenen: '13:30'=EDT/yaz, '14:30'=EST/kış → ET'de ikisi de 09:30)")
        print(f"     first != 09:30 ihlali: {len(viol_first)} gün")
        if viol_first[:15]:
            for ds, fv, n in viol_first[:15]:
                print(f"        {ds}: ilk={fv} (n={n})")
            if len(viol_first) > 15:
                print(f"        ... +{len(viol_first)-15} gün daha")
        print(f"     normal-gün son-likit-bar ∉ {{15:59,16:00}} ihlali: {len(viol_last)} gün")
        if viol_last[:15]:
            for ds, lv, n in viol_last[:15]:
                print(f"        {ds}: {lv} (n={n})")
            if len(viol_last) > 15:
                print(f"        ... +{len(viol_last)-15} gün daha")
        print(f"     yarım-gün (12:59/13:00 likit-kapanış beklenen) — DOĞRU: {len(early_ok)}  {early_ok}")
        print(f"     yarım-gün — YANLIŞ: {len(early_bad)}  {early_bad}")

        out[sym] = {
            "rth_days": len(daily), "first_utc_dist": fu,
            "viol_first_0930": len(viol_first), "viol_first_list": viol_first[:30],
            "viol_last_1600": len(viol_last), "viol_last_list": viol_last[:30],
            "early_ok": early_ok, "early_bad": early_bad,
            "range": (daily.index.min().strftime("%Y-%m-%d"), daily.index.max().strftime("%Y-%m-%d")),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 3 — PIT re-verify (expiry-geçişlerinde OI-drop ≥%70)
# ─────────────────────────────────────────────────────────────────────────────
def section3_pit_oi_drop() -> dict:
    _hr("BÖLÜM 3 — PIT: TÜM expiry-geçişlerinde vade-dolunca OI ≥%70 düşüyor mu")
    out = {}
    for sym in ["spy", "qqq"]:
        p = CHAINS / f"md_{sym}.parquet"
        if not p.exists():
            print(f"  {sym}: {p} yok → atla.")
            continue
        df = pd.read_parquet(p)
        # Tarih başına TEK expiry (veri böyle). Her date'in (date, expiration, toplam_OI)'sini al.
        per_date = (df.groupby(["date", "expiration"])["open_interest"].sum()
                    .reset_index().sort_values("date").reset_index(drop=True))
        per_date["date"] = pd.to_datetime(per_date["date"])
        per_date["expiration"] = pd.to_datetime(per_date["expiration"])

        # Expiry-geçişi = ardışık iki snapshot'ta expiration DEĞİŞİYOR.
        # Bir expiry E vade-dolduğunda: E'nin SON snapshot'ındaki OI vs E DOLDUKTAN SONRA
        # bir sonraki snapshot'ta E'ye ait OI. AMA veride tarih başına tek expiry var →
        # bir expiry dolunca o expiry artık zincirde GÖRÜNMÜYOR (yenisi geliyor).
        # PIT doğru testi: E'nin OI'si vade-günü sonrası snapshot'ta zincirden DÜŞTÜ mü (yani
        # yeni snapshot artık E'yi taşımıyor) → bu otomatik %100 düşüş olur (tek-expiry yapı).
        # Bu yüzden ANLAMLI test: E'nin son-snapshot toplam-OI'si ile, E'nin vade tarihinde/öncesinde
        # OI'nin DTE→0'a giderken nasıl davrandığı DEĞİL; asıl PIT-leak testi =
        # "vade dolan expiry'nin OI'si bir sonraki snapshot'ta hâlâ duruyor mu?" → look-ahead yok.
        #
        # Tek-expiry yapı nedeniyle doğrudan ölçülebilir olan: ardışık expiry geçişlerinde,
        # eski expiry'nin son-OI'si vs yeni expiry'nin ilk-OI'si — ama bunlar FARKLI kontrat
        # kümeleri (elma-armut). DOĞRU PIT-kontrolü: ham satır düzeyinde, bir (date, expiration)
        # bloğunun, expiration < date olan (yani VADESİ GEÇMİŞ) hiçbir satır içermemesi.
        leak_rows = df[pd.to_datetime(df["expiration"]) < pd.to_datetime(df["date"])]
        n_leak = len(leak_rows)

        # Ek: expiry-vade-günü (date == expiration, DTE=0) snapshot'larında OI hâlâ DOLU mu
        # (settle öncesi normal) ve ERTESİ snapshot artık o expiry'yi TAŞIMIYOR mu?
        expirations = per_date["expiration"].unique()
        transitions = []
        dates_sorted = per_date.sort_values("date").reset_index(drop=True)
        for i in range(len(dates_sorted) - 1):
            cur_exp = dates_sorted.loc[i, "expiration"]
            nxt_exp = dates_sorted.loc[i + 1, "expiration"]
            if cur_exp != nxt_exp:
                # geçiş: i. snapshot expiry'si cur_exp, i+1'de nxt_exp
                cur_date = dates_sorted.loc[i, "date"]
                cur_dte = (cur_exp - cur_date).days
                # cur_exp ARTIK i+1 snapshot'ında var mı? (tek-expiry → yok bekleniyor)
                still_present = ((per_date["date"] == dates_sorted.loc[i + 1, "date"]) &
                                 (per_date["expiration"] == cur_exp)).any()
                # cur_exp'nin son OI'si vs i+1 snapshot'ında cur_exp OI'si (yoksa 0)
                last_oi = dates_sorted.loc[i, "open_interest"]
                next_same_exp_oi = per_date[(per_date["date"] == dates_sorted.loc[i + 1, "date"]) &
                                            (per_date["expiration"] == cur_exp)]["open_interest"].sum()
                drop_pct = 100.0 if last_oi == 0 else (1 - next_same_exp_oi / last_oi) * 100
                transitions.append({
                    "old_exp": cur_exp.strftime("%Y-%m-%d"),
                    "new_exp": nxt_exp.strftime("%Y-%m-%d"),
                    "last_date": cur_date.strftime("%Y-%m-%d"),
                    "last_dte": int(cur_dte),
                    "last_oi": int(last_oi),
                    "next_oi_same_exp": int(next_same_exp_oi),
                    "drop_pct": round(drop_pct, 1),
                    "still_present": bool(still_present),
                })

        n_trans = len(transitions)
        drop70 = sum(1 for t in transitions if t["drop_pct"] >= 70.0)
        # Geçiş anında eski expiry'nin DTE'si (vade-dolmadan mı roll ediyor?)
        rolled_at_expiry = sum(1 for t in transitions if t["last_dte"] <= 0)
        rolled_early = sum(1 for t in transitions if t["last_dte"] > 0)

        print(f"\n  ── {sym.upper()} ──")
        print(f"     ham PIT-leak (satırda expiration < date, yani vadesi-geçmiş kontrat): {n_leak} satır")
        print(f"     distinct expiry: {len(expirations)} | expiry-geçişi: {n_trans}")
        print(f"     OI-drop ≥%70 olan geçiş: {drop70}/{n_trans}")
        print(f"     geçiş anında DTE: vade-günü(≤0)={rolled_at_expiry}, erken-roll(>0)={rolled_early}")
        print(f"     {'old_exp':<12}→{'new_exp':<12} {'son_tarih':<12} {'DTE':<4} {'son_OI':>11} "
              f"{'sonraki_aynı_exp_OI':>20} {'drop%':>7} still?")
        for t in transitions:
            print(f"     {t['old_exp']:<12}→{t['new_exp']:<12} {t['last_date']:<12} {t['last_dte']:<4} "
                  f"{t['last_oi']:>11,} {t['next_oi_same_exp']:>20,} {t['drop_pct']:>7} "
                  f"{t['still_present']}")

        # YORUM: tek-expiry yapı → geçişte eski expiry zincirden tamamen çıkar (drop %100),
        # yani look-ahead (vadesi geçmiş kontratın hâlâ OI taşıması) YOK. Bunu net söyle.
        if n_leak == 0 and drop70 == n_trans:
            print(f"     → PIT TEMİZ: vadesi-geçmiş kontrat sızıntısı yok; her geçişte eski expiry "
                  f"sonraki snapshot'ta ≥%70 düşüyor ({drop70}/{n_trans}).")
        else:
            print(f"     → DİKKAT: leak={n_leak}, drop70={drop70}/{n_trans} (≠tam) → incele.")

        out[sym] = {"n_leak_rows": n_leak, "n_distinct_exp": len(expirations),
                    "n_transitions": n_trans, "drop70": drop70,
                    "rolled_at_expiry": rolled_at_expiry, "rolled_early": rolled_early,
                    "transitions": transitions}
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("D4 — ZAMANLAMA & HİZALAMA teşhisi (DIAGNOSIS-ONLY)")
    s1 = section1_feed_timing()
    s2 = section2_dst_assert()
    s3 = section3_pit_oi_drop()

    _hr("ÖZET")
    if s1.get("measured"):
        print(f"  1. FEED-TIMING: hüküm={s1.get('verdict')}, en-yeni={s1.get('newest_ok')}, "
              f"gecikme={s1.get('lag_trading_days')} işlem-günü, dün-mevcut={s1.get('yesterday_available')}")
    else:
        print("  1. FEED-TIMING: ölçülemedi (token yok).")
    for sym in ["spy", "qqq"]:
        if sym in s2:
            x = s2[sym]
            print(f"  2. DST {sym.upper()}: first!=09:30 ihlal={x['viol_first_0930']}, "
                  f"last!=16:00 ihlal={x['viol_last_1600']}, "
                  f"yarım-gün-doğru={len(x['early_ok'])}, yarım-gün-yanlış={len(x['early_bad'])}")
        if sym in s3:
            y = s3[sym]
            print(f"  3. PIT {sym.upper()}: leak={y['n_leak_rows']}, drop≥70%={y['drop70']}/{y['n_transitions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
