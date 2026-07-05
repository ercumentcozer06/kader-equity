"""
D7 — VERİ PAZARI MATRİSİ (TEŞHİS-ONLY).

Bu script HİÇBİR strateji/P&L üretmez. İki iş yapar:
  (A) Mevcut tarihsel-chain verisinin (md_spy/md_qqq) ölçülebilir EKSİĞİNİ sayısallaştırır
      -> hangi problem (①truncation ②sign ③havuz ④canlı-uyum) gerçekten AÇIK, sayı ile.
  (B) WebSearch/WebFetch ile toplanan vendor olgularını (fiyat/kapsam/PIT/format) tek
      kaynaktan tablolaştırır ve her vendor'u AÇIK probleme eşler.

Vendor olguları = 2026-06 araştırma snapshot'ı (kaynak-URL'ler raporda). Fiyat/kapsam
zamanla değişir -> 'as-of 2026-06' etiketli. Ölçülemeyen alan = 'belirsiz' yazılır, TAHMİN YOK.
"""
import sys
import io
import pandas as pd
from pathlib import Path

# Windows konsolu cp1254 -> Unicode ok/daire-rakam patlar; UTF-8'e zorla.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
CH = ROOT / "data" / "historical_chains"


# ---------------------------------------------------------------------------
# (A) MEVCUT VERİNİN ÖLÇÜLEBİLİR EKSİĞİ  (repo-içi, gerçek sayı)
# ---------------------------------------------------------------------------
def measure_gap():
    out = {}
    for s in ["spy", "qqq"]:
        df = pd.read_parquet(CH / f"md_{s}.parquet")
        per_date_exp = df.groupby("date")["expiration"].nunique()
        out[s] = {
            "rows": int(len(df)),
            "dates": int(df["date"].nunique()),
            "date_min": str(df["date"].min())[:10],
            "date_max": str(df["date"].max())[:10],
            "uniq_expiry_all": int(df["expiration"].nunique()),
            "expiry_per_date_med": float(per_date_exp.median()),
            "expiry_per_date_max": int(per_date_exp.max()),
            "iv_present": bool(df["iv"].notna().any()),     # sign/IV problemi: IV var mı?
            "delta_present": bool(df["delta"].notna().any()),
            "has_buysell_open_close": False,  # md EOD: yalnız OI; akış-yön sınıfı YOK -> ② kanıtı
        }
    return out


# ---------------------------------------------------------------------------
# (B) VENDOR MATRİSİ  (as-of 2026-06 araştırma; kaynaklar raporda)
#   problem kodları: ①=all-expiry truncation  ②=sign/dealer-envanter  ③=index-havuz  ④=canlı-uyum
# ---------------------------------------------------------------------------
VENDORS = [
    # --- ② SIGN katmanı (EN YÜKSEK ÖNCELİK) ---
    dict(
        name="Cboe Open-Close Volume Summary (DataShop/LiveVol)",
        layer="②sign (kanonik)",
        coverage="TÜM Cboe-borsa option serileri (C1/BZX/C2/EDGX); SPX index + SPY ETF dahil "
                 "(C1 = SPX'in birincil borsası). 'selected exchanges'teki her seri.",
        start_year="C1 EOD 2005-01-03 (format değişimi 2011); intraday 10-dk 2011, 1-dk 2019; "
                   "diğer borsalar 2018-2019.",
        pit="YÜKSEK — borsa-birincil, EOD overnight / intraday +15dk gecikmeli snapshot; "
            "look-ahead yok (gün-sonu kesin hacim).",
        fmt="EOD summary VEYA intraday (1-dk/10-dk) CSV; her satır participant-type"
            "(customer/pro-customer/broker-dealer/MM) × action(buy/sell) × position(open/close), "
            "customer ayrıca contract-size kırılımı.",
        price="fiyat-belirsiz (sayfa fiyat basmıyor; ücretler SEC 'LiveVol Fees' fee-schedule'da; "
              "DataShop'tan teklif). Tipik akademik referanslarda EOD özet ucuz uçta.",
        solves="② — müşteri buy/sell-to-open vs close = dealer-envanter rekonstrüksiyonunun "
               "KANONİK girdisi. naive +call/−put proxy'sini gerçek imza ile değiştirir.",
    ),
    dict(
        name="ThetaData (intraday trades + quotes, Standard/Pro)",
        layer="②sign (proxy: trade-classification)",
        coverage="Tüm OPRA opsiyonları; SPY/QQQ ETF + index. Tick trade+quote.",
        start_year="Value 4y / Standard 8y / Pro 12y geçmiş (as-of 2026-06).",
        pit="YÜKSEK — tick trade/quote zaman-damgalı; trade-sign sınıflama (Lee-Ready/quote-rule) "
            "KULLANICI tarafında, vendor signed-flow vermez.",
        fmt="REST/stream tick; greeks+IV per-contract. İndirilebilir tarihsel seri tier'a bağlı.",
        price="$40 Value / $80 Standard (tick trade+quote) / $160 Pro (full tick, 12y) — aylık.",
        solves="② (dolaylı) — quote-rule ile buy/sell sınıflar; Cboe-Open-Close'un 'cevabı hazır' "
               "halinin DIY'ı. + ① (tüm-vade greeks/IV) bonus.",
    ),
    dict(
        name="Polygon.io / Massive (raw chain + tick)",
        layer="②sign (proxy) + ①",
        coverage="Tüm US options; tam zincir, tick trade/quote, greeks/IV.",
        start_year="2–4y (tier'a bağlı; Advanced uzatılmış). Genelde ThetaData'dan sığ geçmiş.",
        pit="YÜKSEK (tick) ama greeks 'kendin hesapla'; signed-flow vermez.",
        fmt="REST+WebSocket+flat-file; ham zincir.",
        price="$29 Starter / $79 Developer(options add-on) / $199 Advanced — aylık (2025'te Massive rebrand).",
        solves="② (dolaylı, trade-classification) + ① (tüm-vade). Geçmiş sığ -> uzun-backtest zayıf.",
    ),
    dict(
        name="Cboe LiveVol intraday Option Trades/Quotes",
        layer="②sign (proxy) + intraday",
        coverage="OPRA tüm US stock/ETF/index options; SPX+SPY+QQQ.",
        start_year="LiveVol tick geçmişi (yıl-belirsiz, çok-yıllık); intraday +15dk gecikmeli dosya.",
        pit="YÜKSEK — borsa-birincil tick; trade-condition ID map dahil.",
        fmt="Trade-condition'lı tick trade + 1-dk/N-dk quote interval (NBBO+OHLC+IV/greeks opsiyonel).",
        price="fiyat-belirsiz (DataShop teklif).",
        solves="② (trade-classification için en temiz tick) — ama Open-Close zaten signed verir, "
               "bu DIY alternatifi.",
    ),
    # --- ① ALL-EXPIRY (truncation) katmanı ---
    dict(
        name="Cboe DataShop Option EOD Summary / Open-Close OI",
        layer="①all-expiry (full-chain EOD OI+vol)",
        coverage="Tüm seri/vade EOD: OI, hacim, OHLC; Calcs add-on ile IV+greeks. SPX+SPY+QQQ.",
        start_year="EOD 2005+ (sembole bağlı; aktif semboller daha eski).",
        pit="YÜKSEK — gün-sonu kesin.",
        fmt="Günlük CSV, TÜM vade (mevcut md tek-vade'nin aksine).",
        price="fiyat-belirsiz (teklif); akademik referanslarda EOD ucuz.",
        solves="① — mevcut md TEK vade/gün (toplam gamma$'ın ~%10-11'i). Full-chain EOD = "
               "kalan ~%89-90'ı geri getirir.",
    ),
    dict(
        name="ORATS (EOD analytics 2007+ / 1-dk 2020+)",
        layer="①all-expiry + analytics",
        coverage="5000+ sembol; tüm-vade EOD + 98 proprietary gösterge; SPX/SPY/QQQ.",
        start_year="EOD 2007+ (UZUN geçmiş) ; intraday 1-dk 2020-08+.",
        pit="YÜKSEK — smoothed market quotes, gün-sonu.",
        fmt="API EOD/1-dk; greeks+IV önceden-hesaplı (IV-from-mid sorununu çözer).",
        price="$99/ay birey, $299/ay pro (+$50 real-time) — as-of 2026-04.",
        solves="① (full-chain, 2007+ = en uzun temiz EOD) + ⑤ (IV/greeks hazır, penny-mid "
               "instabilitesini bypass). İmza YOK (②'yi çözmez).",
    ),
    dict(
        name="OptionMetrics IvyDB US (akademik, WRDS)",
        layer="①all-expiry (akademik altın-standart)",
        coverage="Tüm US listed equity+index options + underlying; SPX/SPY/QQQ + 3000+ underlying.",
        start_year="1996-01+ (her opsiyon, her gün) — EN UZUN temiz geçmiş.",
        pit="ÇOK YÜKSEK — akademik referans; doğru-hesaplı IV+greeks+gün-sonu.",
        fmt="WRDS pano/flat; günlük tüm-vade.",
        price="fiyat-belirsiz (kurumsal/akademik WRDS aboneliği; birim-fiyat yayınlanmaz).",
        solves="① + ⑤ — 1996+ tüm-vade temiz IV/greeks. uzun-OOS için ideal. İmza YOK (②'yi çözmez). "
               "Erişim akademik-kuruma bağlı.",
    ),
    dict(
        name="iVolatility / HistoricalOptionData.com (DeltaNeutral)",
        layer="①all-expiry (perakende EOD)",
        coverage="iVol: 20y US equity/index options. HistOptData: SPX 1990+ aylık güncel, tüm US equity EOD.",
        start_year="iVol ~2002+; HistOptData SPX 1990+ (35y).",
        pit="YÜKSEK (EOD); gün-sonu fiyat/IV.",
        fmt="iVol pay-per-use indirme (abonelik yok); HistOptData satın-al CSV.",
        price="iVol pay-per-usage (düşük perakende, miktar-bağımlı, sabit-fiyat-belirsiz); "
              "HistOptData 'Bloomberg/Refinitiv'in bir kesri' (kesin-belirsiz).",
        solves="① + ⑤ — perakende-ucuz full-chain EOD + IV. İmza YOK. Backtest-dostu uzun SPX geçmişi.",
    ),
    # --- ③ index-havuz katmanı ---
    dict(
        name="(③) Index option chain — SPX/NDX (yukarıdaki vendor'ların index kapsamı)",
        layer="③index-havuz",
        coverage="^SPX, ^NDX native index options (ETF SPY/QQQ değil). Cboe/ORATS/IvyDB/Theta hepsi taşır.",
        start_year="vendor'a göre (IvyDB 1996, Cboe 2005, ORATS 2007).",
        pit="YÜKSEK.",
        fmt="ayrı sembol (^SPX/^NDX); ETF ile birleştirilir.",
        price="ek-sembol, taşıyıcı vendor fiyatına dahil.",
        solves="③ — ölçülen havuz oranı SPX/SPY gamma$ 0.04× , NDX/QQQ 0.52× : SPX sinyali "
               "neredeyse-tamamı index-havuzda. ETF-only tarihsel veri SPX dealer-pool'unu kaçırır.",
    ),
    # --- DEALER-POSITIONING VENDOR (hazır-GEX; metodoloji şeffaflığı kritik) ---
    dict(
        name="SqueezeMetrics (DIX + GEX, 2011+)",
        layer="dealer-positioning (hazır seri)",
        coverage="4500+ menkul; S&P GEX + DIX dark-pool. SPX seviyesinde.",
        start_year="2011+ (repo'da squeeze_dix_gex.parquet ile uyumlu).",
        pit="YÜKSEK — sabah ~05:30 master-spreadsheet gün-sonrası; look-ahead yok.",
        fmt="CSV/spreadsheet indirme + API; programatik curl-endpoint YOK.",
        price="$720/ay (tek Data plan; $8,640/yıl) — as-of 2026-06.",
        solves="② kısmen (GEX = dealer-net-gamma proxy, naive-sign'dan iyi) ama metodoloji "
               "YARI-ŞEFFAF (2017 white-paper var, kesin-formül kapalı). naive +call/−put'a "
               "göre upgrade, Open-Close'a göre downgrade (türetilmiş, ham-imza değil).",
    ),
    dict(
        name="SpotGamma (GEX/HIRO/levels)",
        layer="dealer-positioning (hazır seri, dashboard)",
        coverage="SPX/SPY/QQQ + tek-isimler; dealer-pozisyon panosu (GEX, HIRO sıralı-flow, levels).",
        start_year="geçmiş-seri-belirsiz (dashboard-odaklı; tarihsel export sınırlı).",
        pit="ORTA — canlı pano; tarihsel PIT-export zayıf belgeli (look-ahead riski export'ta belirsiz).",
        fmt="tarayıcı panosu; 'API-first DEĞİL', programatik erişim sınırlı (Alpha+ tier).",
        price="Standard $89 / Essential $99 / Pro $129 / Alpha $299 / Institutional $1,999+ — aylık as-of 2026-06.",
        solves="② kısmen (HIRO = sıralı-flow, dealer-pozisyon) ama KARA-KUTU metodoloji + "
               "tarihsel-seri zayıf -> backtest için RİSKLİ. canlı-overlay için uygun, backtest-girdi için değil.",
    ),
    dict(
        name="MenthorQ (gamma levels)",
        layer="dealer-positioning (hazır seviye)",
        coverage="tüm-zincir GEX + 0DTE; key gamma levels. Index+futures+ETF.",
        start_year="geçmiş-seri-belirsiz.",
        pit="ORTA — canlı seviye; tarihsel-export belgeli-değil.",
        fmt="pano/seviye; tarihsel-seri satışı belirsiz.",
        price="fiyat-belirsiz (perakende tier'lar; kesin-rakam araştırmada çıkmadı).",
        solves="② kısmen (GEX seviye) ama KARA-KUTU + tarihsel-belirsiz -> backtest-girdi değil.",
    ),
    dict(
        name="Tier1Alpha (daily gamma exposure models)",
        layer="dealer-positioning (hazır model)",
        coverage="günlük GEX modelleri, implied-vol ranges, MBAD; index-seviye.",
        start_year="geçmiş-seri-belirsiz.",
        pit="ORTA — haftalık rapor + günlük model; PIT-export belgeli-değil.",
        fmt="web/app + haftalık webcast; ham-tarihsel-seri satışı belirsiz.",
        price="fiyat-belirsiz (kurumsal; ~SpotGamma-Institutional bandı tahmin-edilebilir ama doğrulanmadı).",
        solves="② kısmen (türetilmiş GEX) ama KARA-KUTU + tarihsel-belirsiz -> backtest-girdi değil; "
               "araştırma/teyit katmanı.",
    ),
]


def main():
    gap = measure_gap()
    print("=" * 78)
    print("(A) MEVCUT VERİ EKSİĞİ — repo-içi gerçek ölçüm (md_{spy,qqq}.parquet)")
    print("=" * 78)
    for s, g in gap.items():
        print(f"\n[{s.upper()}] {g['rows']} satır / {g['dates']} gün "
              f"({g['date_min']}→{g['date_max']})")
        print(f"   uniq_expiry(tüm seri)={g['uniq_expiry_all']}  "
              f"vade/gün med={g['expiry_per_date_med']:.0f} max={g['expiry_per_date_max']} "
              f"-> tek-vade => ① AÇIK")
        print(f"   IV var={g['iv_present']} delta var={g['delta_present']} "
              f"-> IV/greeks YOK (mid-IV türet zorunlu) => ⑤ AÇIK")
        print(f"   buy/sell-open/close imza kolonu={g['has_buysell_open_close']} "
              f"-> akış-yön sınıfı YOK => ② AÇIK (naive +call/−put zorunlu)")

    print("\n" + "=" * 78)
    print("(B) VENDOR MATRİSİ (as-of 2026-06) — öncelik ②sign > ①all-expiry > ③havuz")
    print("=" * 78)
    for v in VENDORS:
        print(f"\n### {v['name']}   [{v['layer']}]")
        print(f"  kapsam     : {v['coverage']}")
        print(f"  başlangıç  : {v['start_year']}")
        print(f"  PIT        : {v['pit']}")
        print(f"  format     : {v['fmt']}")
        print(f"  fiyat      : {v['price']}")
        print(f"  çözer      : {v['solves']}")


if __name__ == "__main__":
    main()
