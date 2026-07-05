"""screen/probe_cot — CFTC TFF şeması + ES/NQ market isimleri."""
import sys
import requests
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
r = requests.get(URL, params={"$limit": 1}, timeout=40, headers={"User-Agent": "Mozilla/5.0"})
print("status:", r.status_code)
if r.status_code == 200 and r.json():
    keys = list(r.json()[0].keys())
    print("alanlar (lev/asset/oi/date/market içerenler):")
    for k in keys:
        if any(t in k.lower() for t in ("lev", "asset", "open_interest", "date", "market", "dealer")):
            print("   ", k)
# market isimleri
r2 = requests.get(URL, params={"$select": "market_and_exchange_names", "$group": "market_and_exchange_names",
                               "$limit": 2000}, timeout=40, headers={"User-Agent": "Mozilla/5.0"})
if r2.status_code == 200:
    names = sorted(set(x.get("market_and_exchange_names", "") for x in r2.json()))
    print("\nES/NQ/S&P/NASDAQ içeren market isimleri:")
    for n in names:
        if any(t in n.upper() for t in ("S&P 500", "NASDAQ", "E-MINI")):
            print("   ", n)
else:
    print("market list status:", r2.status_code)
