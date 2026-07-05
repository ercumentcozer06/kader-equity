"""
screen/audit_cor1m_lead — READ-ONLY denetim (model değişmez). Kitap-sentezi A2'yi test eder:
COR1M-froth GERÇEKTEN öncül mü (forward downside'ı önceden bilir mi), yoksa sadece düşük-realized-vol
ile EŞ-ZAMANLI sakinlik okuması mı? (Shiller: balon=dikkatsizlik değil coşku; Taleb: stres-koşullu;
Hull: eksik-tanımlı). + kısa-pencere/era-stabilite (A3) + stres-koşullu (Taleb).

Hiçbir şeyi yazmaz/değiştirmez; sadece rapor basar. corr_pc.parquet (COR1M 2006+) + Desktop fiyat CSV'leri.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from screen._util import load_price_csv  # noqa: E402

DESK = Path(r"C:\Users\admin\Desktop\backtesting")
PRICES = {"SPX": "SPX_daily.csv", "NDX": "NASDAQ_daily.csv"}
ERAS = [("pre-2014", "1900", "2013-12-31"), ("2014-2018", "2014-01-01", "2018-12-31"),
        ("2019-2023", "2019-01-01", "2023-12-31"), ("2024+", "2024-01-01", "2100")]


def _q(x):  # standardize
    return (x - x.mean()) / x.std()


def main():
    cor = pd.read_parquet(ROOT / "data" / "cache" / "corr_pc.parquet")["COR1M"].dropna().sort_index()
    print("=" * 100)
    print(f"  COR1M aralık: {cor.index.min().date()}..{cor.index.max().date()}  n={len(cor)}")
    # 1) KISA-PENCERE (A3): düşük-COR1M gözlemleri hangi çağda?
    print("\n  [1] DÜŞÜK-COR1M GÖZLEM SAYISI (çağ × eşik) — sinyal hangi rejimde yaşıyor?")
    print(f"    {'çağ':<12}{'n':>7}{'<8':>7}{'<9':>7}{'<10':>7}{'<11':>8}   medyan COR1M")
    for name, s, e in ERAS:
        w = cor[(cor.index >= pd.Timestamp(s)) & (cor.index <= pd.Timestamp(e))]
        if len(w) == 0:
            print(f"    {name:<12}{0:>7}"); continue
        print(f"    {name:<12}{len(w):>7}{int((w<8).sum()):>7}{int((w<9).sum()):>7}"
              f"{int((w<10).sum()):>7}{int((w<11).sum()):>8}   {w.median():.1f}")

    for a, fn in PRICES.items():
        close = load_price_csv(DESK / fn)
        idx = cor.index.intersection(close.index)
        c = cor.reindex(idx); px = close.reindex(idx)
        r = np.log(px).diff()
        rv21 = r.rolling(21).std() * np.sqrt(252)                      # trailing realized vol (t'de bilinir)
        cb = px
        fwd_ret = cb.shift(-21) / cb - 1.0                              # forward 21g getiri
        # forward 21g en-kötü-yol (downside proxy): sonraki 21 günde min kümülatif getiri
        fwd_dd = pd.Series(index=idx, dtype=float)
        v = px.values
        for i in range(len(v) - 1):
            j = min(i + 21, len(v) - 1)
            seg = v[i + 1:j + 1]
            fwd_dd.iloc[i] = (seg.min() / v[i] - 1.0) if len(seg) else np.nan
        df = pd.DataFrame({"cor": c, "rv": rv21, "fret": fwd_ret, "fdd": fwd_dd}).dropna()

        print("\n" + "=" * 100)
        print(f"  [{a}]  hizalı n={len(df)}   "
              f"corr(COR1M, trailing-RV) = {df['cor'].corr(df['rv']):+.2f}   "
              f"(yüksekse COR1M büyük oranda VOL-EŞZAMANLI okuma)")

        # 2) HAM bucket: COR1M quintile -> forward getiri/downside (FINDING 4 re-confirm)
        df["cq"] = pd.qcut(df["cor"], 5, labels=False, duplicates="drop")
        print("\n  [2] HAM COR1M quintile → forward-21g (düşük q = düşük korelasyon = 'froth' iddiası)")
        print(f"    {'q':<4}{'COR1M~':>9}{'fwd ret%':>10}{'%neg':>7}{'fwd maxDD%':>12}{'n':>7}")
        for q in sorted(df["cq"].dropna().unique()):
            g = df[df["cq"] == q]
            print(f"    {int(q):<4}{g['cor'].mean():>9.1f}{100*g['fret'].mean():>+10.2f}"
                  f"{100*(g['fret']<0).mean():>7.0f}{100*g['fdd'].mean():>+12.2f}{len(g):>7}")

        # 3) LEAD vs COINCIDE: COR1M'i trailing-RV'ye göre ORTOGONALLE, REZİDÜYÜ bucketle
        #    rezidü hâlâ forward-downside'ı bilirse → gerçek öncül; bilmezse → sadece sakinlik-eşzamanlı
        b = np.polyfit(_q(df["rv"]), _q(df["cor"]), 1)
        resid = _q(df["cor"]) - (b[0] * _q(df["rv"]) + b[1])
        df["rq"] = pd.qcut(resid, 5, labels=False, duplicates="drop")
        print("\n  [3] RV-ARINDIRILMIŞ COR1M-rezidü quintile → forward-21g  (öncül-mü-eşzamanlı-mı TESTİ)")
        print(f"    {'q':<4}{'fwd ret%':>10}{'%neg':>7}{'fwd maxDD%':>12}{'n':>7}")
        for q in sorted(df["rq"].dropna().unique()):
            g = df[df["rq"] == q]
            print(f"    {int(q):<4}{100*g['fret'].mean():>+10.2f}{100*(g['fret']<0).mean():>7.0f}"
                  f"{100*g['fdd'].mean():>+12.2f}{len(g):>7}")

        # 4) DOUBLE-SORT (Taleb stres-koşullu): RV tercile × COR1M düşük/yüksek → forward downside
        df["rvt"] = pd.qcut(df["rv"], 3, labels=["RV-düşük", "RV-orta", "RV-yüksek"])
        df["clow"] = df["cor"] < df["cor"].median()
        print("\n  [4] DOUBLE-SORT: RV-tercile × COR1M(düşük/yüksek) → forward maxDD% (stres-koşullu mu?)")
        piv = df.pivot_table(index="rvt", columns="clow", values="fdd", aggfunc="mean", observed=True) * 100
        piv.columns = ["COR1M-yüksek", "COR1M-düşük"]
        print(piv.round(2).to_string())

        # 5) ERA-STABİLİTE: en-düşük COR1M quintile'ın forward getirisi çağ-çağ
        print("\n  [5] EN-DÜŞÜK COR1M quintile forward-21g — ÇAĞ-ÇAĞ (sinyal 2024+ artefaktı mı?)")
        lowq = df[df["cq"] == 0]
        print(f"    {'çağ':<12}{'n':>6}{'fwd ret%':>10}{'%neg':>7}{'fwd maxDD%':>12}")
        for name, s, e in ERAS:
            g = lowq[(lowq.index >= pd.Timestamp(s)) & (lowq.index <= pd.Timestamp(e))]
            if len(g) == 0:
                print(f"    {name:<12}{0:>6}        —"); continue
            print(f"    {name:<12}{len(g):>6}{100*g['fret'].mean():>+10.2f}"
                  f"{100*(g['fret']<0).mean():>7.0f}{100*g['fdd'].mean():>+12.2f}")
    print("\n" + "=" * 100)
    print("  OKU: [3] rezidü-bucket monoton+negatifse COR1M gerçek öncül; düzse sadece sakinlik-eşzamanlı.")
    print("       [4] etki sadece RV-yüksek sütununda güçlüyse Taleb haklı (stres-koşullu). [5] tüm değer")
    print("       2024+'taysa kısa-pencere artefaktı riski (A3). HİÇBİR MODEL DEĞİŞMEDİ — sadece denetim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
