"""screen/probe_ibkr — IBKR (TWS/Gateway) bağlı mı? + SPX opsiyon zinciri erişilebilir mi?"""
from __future__ import annotations

import sys
from pathlib import Path

KMR = Path(r"C:\Users\admin\Downloads\kader-macro")
sys.path.insert(0, str(KMR))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import yaml                              # noqa: E402
from dotenv import load_dotenv           # noqa: E402
from modules import _ibkr               # noqa: E402

load_dotenv(KMR / ".env")
cfg = yaml.safe_load((KMR / "config.yaml").read_text(encoding="utf-8"))
hc = _ibkr.health_check(cfg)
print("  IBKR health_check:", hc)

if hc.get("reachable"):
    from ib_insync import IB, Index      # noqa: E402
    ib = _ibkr.connect(cfg)
    try:
        ib.reqMarketDataType(cfg.get("ibkr", {}).get("market_data_type", 3))
        spx = Index("SPX", "CBOE", "USD")
        q = ib.qualifyContracts(spx)
        print("  SPX qualify:", q[0] if q else None)
        if q:
            chains = ib.reqSecDefOptParams(spx.symbol, "", "IND", spx.conId)
            print(f"  option-param chains: {len(chains)}")
            for c in chains[:4]:
                print(f"    exch={c.exchange} tradingClass={c.tradingClass} "
                      f"#exp={len(c.expirations)} #strk={len(c.strikes)}")
    except Exception as e:
        print(f"  SPX chain probe FAILED: {type(e).__name__}: {e}")
    finally:
        _ibkr._safe_disconnect(ib)
else:
    print("  → TWS/IB Gateway kapalı ya da erişilemiyor. Gerçek yüzey için Gateway açık olmalı.")
