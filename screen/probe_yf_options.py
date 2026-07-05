"""screen/probe_yf_options — yfinance SPY opsiyon zinciri free olarak IV + OI veriyor mu?"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import yfinance as yf

t = yf.Ticker("SPY")
try:
    spot = t.fast_info["lastPrice"]
except Exception:
    spot = t.history(period="1d")["Close"].iloc[-1]
print(f"  SPY spot ≈ {spot:.2f}")

exps = t.options
print(f"  expiries: {len(exps)}  ilk6={exps[:6]}  son={exps[-1] if exps else None}")
if not exps:
    print("  [!] yfinance expiry listesi boş (Yahoo API bloğu olabilir).")
    raise SystemExit(1)

oc = t.option_chain(exps[min(3, len(exps) - 1)])
print(f"  calls kolonları: {list(oc.calls.columns)}")
cols = [c for c in ("strike", "impliedVolatility", "openInterest", "volume", "bid", "ask", "lastPrice") if c in oc.calls.columns]
near = oc.calls.iloc[(oc.calls['strike'] - spot).abs().argsort()[:6]]
print(near[cols].to_string(index=False))
print(f"\n  IV var mı: {'impliedVolatility' in oc.calls.columns}  |  OI var mı: {'openInterest' in oc.calls.columns}")
print(f"  ATM call IV ≈ {near['impliedVolatility'].median():.3f}  OI toplam(yakın6) ≈ {int(near['openInterest'].sum())}")
