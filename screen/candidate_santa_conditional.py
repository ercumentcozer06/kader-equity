"""screen/candidate_santa_conditional — Constan'ın KOŞULLU Noel rallisi (101 Canon, 2026-06-13).

İddia (Constan, 1928+): 1 Kasım'da SPX YTD >= +%10 ise o gün al, yıl sonuna tut →
ort +4.68%, %83 pozitif, maks kayıp −2.01% (n=36). Mekanizma: geride kalan fonların
performans kovalaması (alım) + kazanç realizasyonunun Ocak'a ertelenmesi (satış yok).

ÖN-KAYITLI OKUMA KURALI (sonuçlardan önce):
  1) REPLİKASYON: tam-tarih (1928+) ort/medyan/isabet — Constan sayılarına yakın mı?
  2) STABİLİTE: eski (1928-1979) vs orta (1980-2004) vs modern (2005+) — isabet/ort
     üç dönemde de pozitif mi? Modern dönem ölmüşse takvim-arbitrajı kapanmış demektir.
  3) KOŞULUN DEĞERİ: koşullu (YTD>=10%) vs koşulsuz (her yıl Kas-Ara) farkı — koşul
     gerçek bilgi mi katıyor yoksa Kas-Ara genel pozitifliği mi?
  4) TIDE-ÖRTÜŞME (2019+): koşul tuttuğu yıllarda stack zaten long muydu? (kaba okuma)
Entegrasyon kararı BURADAN ÇIKMAZ — standalone sağlamsa ayrı incremental-over-tide
(strict BH-FDR) testi açılır; bu dosya keşif/replikasyon katmanıdır.

Veri: yfinance ^GSPC günlük (1927-12-30'a kadar, bedava; Stooq JS-duvarlı çıktı) →
data/cache/spx_gspc_long.csv. Doğrulama: Desktop SPX_daily.csv ile 2000+ örtüşme korelasyonu.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CACHE = ROOT / "data" / "cache"
DESK = Path(r"C:\Users\admin\Desktop\backtesting")


def load_long_spx() -> pd.Series:
    p = CACHE / "spx_gspc_long.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")
    else:
        import yfinance as yf
        df = yf.Ticker("^GSPC").history(period="max", auto_adjust=False)
        if df is None or df.empty:
            raise RuntimeError("yfinance ^GSPC bos dondu")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        CACHE.mkdir(parents=True, exist_ok=True)
        df[["Close"]].to_csv(p)
        df = df[["Close"]]
    s = df["Close"].dropna().sort_index()
    # sağlama: yerel günlük CSV ile örtüşme korelasyonu (veri-bütünlüğü; repo yükleyicisi)
    sys.path.insert(0, str(ROOT))
    from screen._util import load_price_csv
    local = load_price_csv(str(DESK / "SPX_daily.csv"))
    both = pd.concat([s, local], axis=1, join="inner").dropna()
    corr = both.iloc[:, 0].pct_change().corr(both.iloc[:, 1].pct_change())
    print(f"  stooq uzun-tarih: {s.index.min().date()} -> {s.index.max().date()} ({len(s)} gun)")
    print(f"  yerel-CSV sagllama (2000+ gunluk-getiri corr): {corr:.4f}  (>=0.99 beklenir)")
    return s


def year_rows(s: pd.Series, start_year: int = 1928) -> pd.DataFrame:
    rows = []
    for y in sorted(set(s.index.year)):
        if y < start_year or y >= s.index.max().year + 1:
            continue
        yr = s[s.index.year == y]
        if len(yr) < 100:
            continue
        nov = yr[yr.index >= f"{y}-11-01"]
        if nov.empty or yr[yr.index < f"{y}-11-01"].empty:
            continue
        first = yr.iloc[0]                      # yılın ilk kapanışı
        nov1 = nov.iloc[0]                      # 1 Kasım (veya sonraki ilk işgünü)
        end = yr.iloc[-1]                       # yıl sonu
        ytd = nov1 / first - 1
        fwd = end / nov1 - 1
        rows.append({"year": y, "ytd_nov1": ytd, "novdec": fwd})
    return pd.DataFrame(rows).set_index("year")


def stats(x: pd.Series) -> str:
    if len(x) == 0:
        return "n=0"
    return (f"n={len(x):>3}  ort {x.mean()*100:+.2f}%  medyan {x.median()*100:+.2f}%  "
            f"isabet {100*(x > 0).mean():.0f}%  min {x.min()*100:+.2f}%  maks {x.max()*100:+.2f}%")


def main() -> int:
    s = load_long_spx()
    df = year_rows(s)
    cond = df[df["ytd_nov1"] >= 0.10]["novdec"]
    rest = df[df["ytd_nov1"] < 0.10]["novdec"]
    print("\n  [1] REPLIKASYON (1928+, kosul: 1 Kas'ta YTD >= +10%)")
    print(f"      kosullu  : {stats(cond)}")
    print(f"      kalan    : {stats(rest)}")
    print(f"      kosulsuz : {stats(df['novdec'])}   (her yil Kas-Ara)")

    print("\n  [2] STABILITE")
    for lab, a, b in (("1928-1979", 1928, 1979), ("1980-2004", 1980, 2004), ("2005+", 2005, 2100)):
        sub = df[(df.index >= a) & (df.index <= b)]
        c = sub[sub["ytd_nov1"] >= 0.10]["novdec"]
        print(f"      {lab:<10} kosullu: {stats(c)}")

    print("\n  [3] KOSULUN DEGERI: kosullu-ort - kosulsuz-ort = "
          f"{(cond.mean() - df['novdec'].mean())*100:+.2f} puan; "
          f"kosullu-ort - kalan-ort = {(cond.mean() - rest.mean())*100:+.2f} puan")
    # basit permutasyon-p: yil etiketlerini karistir, kosul rastgele atansa fark dagilimi
    rng = np.random.default_rng(77)
    n_c = len(cond)
    obs = cond.mean() - rest.mean()
    perm = [df["novdec"].sample(n_c, random_state=int(rng.integers(1e9))).mean()
            - df["novdec"].drop(df["novdec"].sample(n_c, random_state=int(rng.integers(1e9))).index, errors="ignore").mean()
            for _ in range(2000)]
    # daha temiz: indeks bazlı
    vals = df["novdec"].values
    perm = []
    for _ in range(5000):
        idx = rng.permutation(len(vals))
        perm.append(vals[idx[:n_c]].mean() - vals[idx[n_c:]].mean())
    p = float((np.array(perm) >= obs).mean())
    print(f"      permutasyon p (kosullu>kalan tesadufu): {p:.3f}  (5000 perm, seed 77)")

    print("\n  [4] KOSULLU YILLAR (son 15):")
    last = df[df["ytd_nov1"] >= 0.10].tail(15)
    for y, r in last.iterrows():
        print(f"      {y}: YTD@Kas1 {r['ytd_nov1']*100:+.1f}% -> KasAra {r['novdec']*100:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
