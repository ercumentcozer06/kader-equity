"""
screen/fetch_net_equity_supply — Constan'in HISSE NET-ARZI borusu (Z.1 Flow of Funds, FRED, bedava).

NE: net hisse ihraci = ihraç − geri-alim/itfa (negatif = arz daraliyor = buyback>ihrac).
Constan tezi: "fiyati belirleyen akis" — arz daralmasi yapisal bid, ihrac patlamasi (2021) yapisal ask.
TEST SONUCU (candidate_net_supply, denetim-sonrasi 2026-06-13): KALICILIK-FAIL / sekuler-artefakt —
tez tarihsel veride DOGRULANAMADI (1984+ icerik ~0, 2005+ isaret ters); panel BETIMSEL arsivdir.

SERI KESFI (FRED search endpoint ile dogrulandi, 2026-06-13):
  • Gorevde anilan NCBEILQ027S = LEVEL serisi (piyasa degeri stoku, Buffett-indikator bileseni) — FLOW DEGIL.
  • Dogru FLOW (transactions) serileri:
      NCBCEBQ027S  Nonfinancial Corporate Business; Corporate Equities; Liability, TRANSACTIONS
                   (Q, Mil USD, SAAR, 1946Q4+)  ← ana motor (Z.1 FA103164103)
      FBCELIQ027S  Domestic Financial Sectors; Corporate Equities; Liability, TRANSACTIONS
                   (Q, Mil USD, SAAR, 1946Q4+)  ← finansal sektor bacagi; toplam = NFC + FIN
                   DIKKAT (fetch sonrasi tespit): Z.1 'corporate equities' enstrumani ETF/kapali-uclu
                   fon PAYLARINI da finansal-sektor ihraci sayar → 2020'lerde fin bacagi +$0.8-1.9tn
                   SAAR = ETF yaratimi (TALEP-tarafi sarmalayici), Constan anlamiyla arz DEGIL.
                   ANA MOTOR = NFC; total KONTAMINE-varyant olarak saklanir/test edilir.
  • GDP          Nominal GDP (Q, Bil USD, SAAR) — %NGDP normalizasyonu icin.

PIT / YAYIN GECIKMESI: Z.1 ceyrek-sonundan ~10-11 hafta sonra yayimlanir; FRED ceyregi BASINDAN
indeksler → pit_date = index + 165 takvim gunu ('pub-lag +165g': ~91g ceyrek + ~74g yayin).
Kaydirmasiz tarih de saklanir (robustluk varyanti).

ROLLING-4Q: seriler SAAR (yillik orana cevrilmis ceyreklik akis) → son-4-ceyrek toplam akis
= 4Q SAAR ORTALAMASI. %NGDP = roll4(net_arz) / roll4(NGDP). z = rolling 40-ceyrek (10y) pencere.

CIKTI:
  data/cache/net_equity_supply.parquet  (raw SAAR + rolling-4Q %NGDP + z10y + pit_date)
  output/net_supply_panel.txt           (Constan grafiginin ikizi: zaman serisi + son 8 ceyrek + yon)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "output"
ENV = Path(r"C:\Users\admin\Downloads\kader-macro\.env")
BASE = "https://api.stlouisfed.org/fred"
PUB_LAG_DAYS = 165          # 'pub-lag +165g' (ceyrek 91g + Z.1 yayin ~74g)

SERIES = {
    "nfc": "NCBCEBQ027S",   # Nonfinancial Corporate Business; Corporate Equities; Liability, Transactions (SAAR $mn)
    "fin": "FBCELIQ027S",   # Domestic Financial Sectors; Corporate Equities; Liability, Transactions (SAAR $mn)
    "gdp": "GDP",           # Nominal GDP (SAAR $bn)
}


def fred_key() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV)
    except Exception:
        pass
    k = os.environ.get("FRED_API_KEY")
    if not k and ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            if line.startswith("FRED_API_KEY"):
                k = line.split("=", 1)[1].strip()
    if not k:
        raise RuntimeError("FRED_API_KEY bulunamadi (kader-macro/.env)")
    return k


def fred_meta(sid: str, key: str) -> dict:
    r = requests.get(f"{BASE}/series", params={"series_id": sid, "api_key": key, "file_type": "json"}, timeout=30)
    r.raise_for_status()
    return r.json()["seriess"][0]


def fred_obs(sid: str, key: str) -> pd.Series:
    r = requests.get(f"{BASE}/series/observations",
                     params={"series_id": sid, "api_key": key, "file_type": "json", "limit": 100000}, timeout=60)
    r.raise_for_status()
    obs = r.json()["observations"]
    idx = pd.to_datetime([o["date"] for o in obs])
    val = pd.to_numeric([o["value"] for o in obs], errors="coerce")
    return pd.Series(val, index=idx, name=sid).dropna()


def build() -> pd.DataFrame:
    key = fred_key()
    raw, meta = {}, {}
    for tag, sid in SERIES.items():
        m = fred_meta(sid, key)
        meta[tag] = m
        raw[tag] = fred_obs(sid, key)
        print(f"  [{tag}] {sid}: {m['title'][:70]}")
        print(f"        {m['frequency_short']} / {m['units_short']} / {m['seasonal_adjustment_short']} "
              f"/ {m['observation_start'][:7]} -> {m['observation_end'][:7]} (n={len(raw[tag])})")
        if tag in ("nfc", "fin"):
            assert "Transactions" in m["title"], f"{sid} flow (transactions) serisi degil!"
            assert m["seasonal_adjustment_short"] == "SAAR" and m["frequency_short"] == "Q"

    df = pd.DataFrame({
        "nfc_saar_mn": raw["nfc"],
        "fin_saar_mn": raw["fin"],
        "ngdp_saar_bn": raw["gdp"],
    }).sort_index()
    df["total_saar_mn"] = df["nfc_saar_mn"] + df["fin_saar_mn"]

    # rolling-4Q toplam akis = SAAR ortalamasi; %NGDP
    for c in ("nfc", "total"):
        roll = df[f"{c}_saar_mn"].rolling(4, min_periods=4).mean()
        gdp4 = df["ngdp_saar_bn"].rolling(4, min_periods=4).mean() * 1000.0
        df[f"ratio4q_{c}_pct"] = 100.0 * roll / gdp4
        r = df[f"ratio4q_{c}_pct"]
        df[f"z10y_{c}"] = (r - r.rolling(40, min_periods=20).mean()) / r.rolling(40, min_periods=20).std()

    df["pit_date"] = df.index + pd.Timedelta(days=PUB_LAG_DAYS)   # 'pub-lag +165g'
    df.index.name = "qdate"
    return df


def write_panel(df: pd.DataFrame) -> None:
    """Constan grafiginin ikizi: rolling-4Q net-ihrac %NGDP zaman serisi (metin), son 8 ceyrek, yon."""
    OUT.mkdir(parents=True, exist_ok=True)
    r = df["ratio4q_total_pct"].dropna()
    rn = df["ratio4q_nfc_pct"].dropna()
    L: list[str] = []
    L.append("=" * 100)
    L.append("  HISSE NET-ARZ PANELI — Constan grafiginin ikizi (Z.1 Flow of Funds, FRED)")
    L.append("  Seri: rolling-4Q net hisse ihraci / NGDP (%) — NEGATIF = arz daraliyor (buyback > ihrac)")
    L.append("  ANA MOTOR = nfc (nonfinansal sirketler, NCBCEBQ027S) = Constan anlamiyla kurumsal arz.")
    L.append("  total = nfc + finansal sektor (FBCELIQ027S); fin bacagi ETF/fon-payi yaratimini da sayar")
    L.append("  (talep-tarafi sarmalayici) -> total KONTAMINE referans-varyanttir, ana okuma DEGILDIR.")
    L.append("=" * 100)
    L.append("")
    L.append("  YILLIK ZAMAN SERISI (yil ortalamasi, %NGDP; bar 0.1 puan/karakter, # = NFC, sinir +-3%)")
    L.append(f"  {'yil':<6}{'nfc':>8}{'total':>8}   {'-3%':<15}{'0':^31}{'+3%':>14}")
    yearly = rn.groupby(rn.index.year).mean()
    yearly_t = r.groupby(r.index.year).mean()
    for y, v in yearly.items():
        pos = int(round(np.clip(v, -3, 3) * 10))
        bar = [" "] * 61
        bar[30] = "|"
        if pos != 0:
            a, b = (31, 31 + pos) if pos > 0 else (31 + pos, 30)
            for i in range(a, b):
                bar[i] = "#"
        L.append(f"  {y:<6}{v:>+8.2f}{yearly_t.get(y, float('nan')):>+8.2f}   {''.join(bar)}")
    L.append("")
    L.append("  SON 8 CEYREK (raw SAAR $mlr + rolling-4Q %NGDP + z10y)")
    L.append(f"  {'ceyrek':<10}{'nfc $mlr':>10}{'fin $mlr':>10}{'tot $mlr':>10}{'r4q-tot%':>10}{'r4q-nfc%':>10}{'z10y-tot':>10}{'PIT-tarih':>12}")
    tail = df.dropna(subset=["ratio4q_total_pct"]).tail(8)
    for q, row in tail.iterrows():
        qlab = f"{q.year}Q{(q.month - 1)//3 + 1}"
        L.append(f"  {qlab:<10}{row['nfc_saar_mn']/1000:>10.0f}{row['fin_saar_mn']/1000:>10.0f}"
                 f"{row['total_saar_mn']/1000:>10.0f}{row['ratio4q_total_pct']:>+10.2f}"
                 f"{row['ratio4q_nfc_pct']:>+10.2f}{row['z10y_total']:>+10.2f}"
                 f"{str(row['pit_date'].date()):>12}")
    last = tail.iloc[-1]
    qlab = f"{tail.index[-1].year}Q{(tail.index[-1].month - 1)//3 + 1}"
    yon = ("arz daraliyor (buyback > ihrac)" if last["ratio4q_nfc_pct"] < 0
           else "arz genisliyor (ihrac > buyback)")
    ztxt = ("10y normalin USTunde (daralma zayifliyor / ihrac artiyor)" if last["z10y_nfc"] > 0.5 else
            "10y normalin ALTinda (daralma derinlesiyor)" if last["z10y_nfc"] < -0.5 else "10y normal bandinda")
    L.append("")
    L.append(f"  SON OKUMA ({qlab}, PIT {last['pit_date'].date()}) — BETIMSEL: NFC rolling-4Q net-arz "
             f"{last['ratio4q_nfc_pct']:+.2f}% NGDP, {yon}; z10y(nfc) {last['z10y_nfc']:+.2f} = {ztxt}.")
    L.append(f"  (kontamine total: {last['ratio4q_total_pct']:+.2f}% NGDP, z {last['z10y_total']:+.2f} — "
             "fin bacagi ETF-yaratimi agirlikli, arz okumasi olarak KULLANMA)")
    L.append("  ICERIK-TESTI NOTU (2026-06-13): on-kayitli test KALICILIK-FAIL / sekuler-artefakt verdi")
    L.append("  (1984+ icerik ~0, 2005+ isaret TERS) -> bu panel yalniz BETIMSELdir; 'yapisal destek/baski'")
    L.append("  MEKANIZMA iddiasi YAPILMAZ. Detay: output/net_supply_report.txt (VERDICT bolumu).")
    L.append("  PIT notu: Z.1 ceyrek-sonundan ~10-11 hafta sonra yayimlanir; pub-lag +165g uygulanmistir.")
    L.append("=" * 100)
    (OUT / "net_supply_panel.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  panel -> {OUT / 'net_supply_panel.txt'}")
    print(f"  SON OKUMA {qlab}: r4q-NFC {last['ratio4q_nfc_pct']:+.2f}% NGDP, z10y(nfc) {last['z10y_nfc']:+.2f}"
          f"  (kontamine-total {last['ratio4q_total_pct']:+.2f}%)")


def main() -> int:
    print("=" * 96)
    print("  FETCH: net hisse arzi (Z.1 transactions, FRED)")
    print("=" * 96)
    df = build()
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "net_equity_supply.parquet"
    df.to_parquet(p)
    ok = df.dropna(subset=["ratio4q_total_pct"])
    print(f"  -> {p}  ({len(df)} ceyrek; ratio kapsami {ok.index.min().date()} -> {ok.index.max().date()})")
    write_panel(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
