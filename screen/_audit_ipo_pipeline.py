"""Adversarial audit: ipo_pipeline parquet schema + CLI-number reproduction + live-edge apples-to-apples."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"

pq = pd.read_parquet(CACHE / "ipo_pipeline.parquet")
mon = pd.read_parquet(CACHE / "ipo_pipeline_monthly.parquet")
live = json.loads((CACHE / "ipo_pipeline_live.json").read_text(encoding="utf-8"))
mega = json.loads((CACHE / "mega_ipo_hits.json").read_text(encoding="utf-8"))

print("=== 1) PARQUET SEMASI ===")
print("quarterly index:", pq.index.name, pq.index.dtype, "rows:", len(pq))
print("quarterly cols :", list(pq.columns))
print("dtypes:\n", pq.dtypes.to_string())
print("monthly index:", mon.index.name, mon.index.dtype, "rows:", len(mon))
print("monthly cols :", list(mon.columns))

SPEC_Q = {"s1_new_n", "s1_amend_n", "f1_new_n", "f1_amend_n", "total_new_n", "total_all_n",
          "sb2_new_n", "sb2_amend_n", "total_new_adj_n", "mega_hits_n",
          "roll4q", "z10y", "roll4q_adj", "z10y_adj", "partial", "pit_date"}
missing = SPEC_Q - set(pq.columns)
extra = set(pq.columns) - SPEC_Q
print("spec-eksik:", missing or "YOK", "| spec-fazla:", extra or "YOK")

print("\n=== 2) CLI/RAPOR SAYILARINI PARQUET'TEN YENIDEN URET ===")
# roll4q son kapanmis ceyrek
closed = pq[pq["partial"] == 0.0]
last4 = closed["total_new_n"].dropna().tail(4)
print("son-4-kapanmis total_new_n:", list(last4.values), "toplam:", last4.sum(), "(rapor: 1523)")
# z10y yeniden hesapla
s = pq["total_new_n"].rolling(4, min_periods=4).sum()
s[pq["partial"] == 1.0] = np.nan
z = (s - s.rolling(40, min_periods=20).mean()) / s.rolling(40, min_periods=20).std()
zlast = z.dropna().iloc[-1]
stored = pq["z10y"].dropna().iloc[-1]
print(f"z10y yeniden-hesap son-kapanmis: {zlast:+.4f} vs parquet {stored:+.4f} (rapor +0.69)")
# mega_hits_n toplam = strict ilk-dosyalama sayisi (json'dan)
strict_first = [h for h in mega["historical_all"] if h["tier"] == "strict" and not h["form"].endswith("/A")]
print("parquet mega_hits_n toplam:", pq["mega_hits_n"].sum(), "| json strict ilk-dosyalama:", len(strict_first))
# 2026Q2 satiri rapordaki tabloyla ayni mi
r = pq.iloc[-1]
print("2026Q2 satiri:", {c: (None if not np.isfinite(r[c]) else round(float(r[c]), 2))
                          for c in ["s1_new_n", "s1_amend_n", "f1_new_n", "f1_amend_n",
                                    "total_new_n", "mega_hits_n", "partial"]})
print("2026Q2 roll4q/z10y NaN mi:", bool(np.isnan(r["roll4q"]) and np.isnan(r["z10y"])))
print("pit_date son satir:", r["pit_date"], "| sondan-2:", pq.iloc[-2]["pit_date"])

print("\n=== 3) CANLI-UC ELMA-ELMA: son-90g sayimi ham idx'ten bagimsiz yeniden ===")
sys.path.insert(0, str(ROOT))
from screen.fetch_ipo_pipeline import parse_form_idx, RAW_EDGAR

today = pd.Timestamp(live["as_of"])
parts = []
for f in sorted(RAW_EDGAR.glob("form_202[5-6]_QTR*.idx")):
    parts.append(parse_form_idx(f))
recent = pd.concat(parts, ignore_index=True)
w90 = recent[recent["date"] >= today - pd.Timedelta(days=90)]
n90 = {"s1_new": int((w90["form"] == "S-1").sum()),
       "s1_amend": int((w90["form"] == "S-1/A").sum()),
       "f1_new": int((w90["form"] == "F-1").sum()),
       "f1_amend": int((w90["form"] == "F-1/A").sum())}
n90["total_new"] = n90["s1_new"] + n90["f1_new"]
print("yeniden-sayim 90g:", n90)
print("live.json       :", live["window_90d_counts"])
# z90 yeniden
qq = closed["total_new_n"].dropna().tail(40)
z90 = (n90["total_new"] - qq.mean()) / qq.std()
print(f"z90 yeniden: {z90:+.4f} vs live.json {live['z90_vs_40q']:+.4f}")
print(f"taban: 40-kapanmis-c ort {qq.mean():.1f} std {qq.std():.1f}; pencere 90g vs ceyrek-ort ~91.3g")
# pencere-uzunlugu yanlilik tahmini
bias = n90["total_new"] * (91.3 / 90.0 - 1.0)
print(f"90g-vs-91.3g olcek yanliligi ~{bias:.1f} dosyalama (~{bias / qq.std():+.3f} z)")
# canli ucta bugunun dosyalamalari idx'te var mi (SEC gece gunceller)
print("idx icindeki son dosyalama tarihi:", recent["date"].max().date(), "| as_of:", today.date())

print("\n=== 4) MONTHLY tutarlilik: aylik toplam == ceyreklik toplam ===")
mq = mon.copy()
mq["q"] = mq.index.to_period("Q").start_time
agg = mq.groupby("q")[["s1_new_n", "f1_new_n"]].sum()
j = agg.join(pq[["s1_new_n", "f1_new_n"]], how="inner", lsuffix="_m", rsuffix="_q").dropna()
mism = j[(j["s1_new_n_m"] != j["s1_new_n_q"]) | (j["f1_new_n_m"] != j["f1_new_n_q"])]
print("aylik-vs-ceyreklik uyusmayan ceyrek:", len(mism), "/", len(j))

print("\n=== 5) NaN / sahte-sifir kontrolu ===")
print("quarterly NaN'li satir sayisi (total_new_n):", pq["total_new_n"].isna().sum())
print("partial==1 satir sayisi:", int(pq["partial"].sum()), "(beklenen 1)")
print("kapanmis ceyreklerde total_new_n==0 olan:", int((closed["total_new_n"] == 0).sum()))
