"""screen/probe_mktdata — market-data duvarı global mi SPY'a-özel mi? SLV (silver) vs SPY delayed quote."""
from __future__ import annotations

import sys
from pathlib import Path

KMR = Path(r"C:\Users\admin\Downloads\kader-macro")
sys.path.insert(0, str(KMR))
import yaml                                              # noqa: E402
from dotenv import load_dotenv                           # noqa: E402
from modules import _ibkr                                # noqa: E402

load_dotenv(KMR / ".env")
cfg = yaml.safe_load((KMR / "config.yaml").read_text(encoding="utf-8"))
cfg.setdefault("ibkr", {})["client_id"] = 49
for mdt in (3, 4):                                        # 3=delayed, 4=delayed-frozen
    cfg["ibkr"]["market_data_type"] = mdt
    ib = _ibkr.connect(cfg)
    if ib is None:
        print(f"  mdt={mdt}: bağlanamadı"); continue
    try:
        ib.reqMarketDataType(mdt)
        from ib_insync import Stock
        for sym in ("SLV", "SPY"):
            c = Stock(sym, "SMART", "USD")
            ib.qualifyContracts(c)
            tk = ib.reqMktData(c, "", False, False)
            ib.sleep(8)
            vals = {f: getattr(tk, f, None) for f in ("last", "close", "bid", "ask", "marketPrice")}
            vals = {k: (v() if callable(v) else v) for k, v in vals.items()}
            ib.cancelMktData(c)
            print(f"  mdt={mdt} {sym}: {vals}")
    finally:
        _ibkr._safe_disconnect(ib)
