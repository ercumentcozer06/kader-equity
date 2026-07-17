import csv
import gzip
import json

from screen import snapshot_cboe_chains as sc


def _payload(symbol: str, n: int = 1001) -> dict:
    option = {
        "option": f"{symbol.replace('_', '')}260717C07500000",
        "open_interest": 10,
        "iv": 0.2,
        "gamma": 0.001,
    }
    return {
        "timestamp": "2026-07-16 17:20:00",
        "symbol": symbol,
        "data": {"current_price": 100.0, "options": [dict(option) for _ in range(n)]},
    }


def _isolate(tmp_path, monkeypatch):
    out = tmp_path / "raw_chains_cboe"
    monkeypatch.setattr(sc, "OUT", out)
    monkeypatch.setattr(sc, "LEDGER", out / "pit_ledger.csv")
    monkeypatch.setattr(sc, "RUNS", out / "runs")
    return out


def test_archives_all_symbols_append_only(tmp_path, monkeypatch):
    out = _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(sc, "fetch", lambda symbol, timeout=60: _payload(symbol))

    assert sc.main() == 0
    first_bytes = {}
    for symbol in sc.SYMBOLS:
        path = out / symbol / "2026-07-16.json.gz"
        first_bytes[symbol] = path.read_bytes()
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            saved = json.load(fh)
        assert saved["symbol"] == symbol
        assert len(saved["response"]["data"]["options"]) == 1001

    assert sc.main() == 0
    for symbol in sc.SYMBOLS:
        assert (out / symbol / "2026-07-16.json.gz").read_bytes() == first_bytes[symbol]
    with sc.LEDGER.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 8
    assert {row["status"] for row in rows[:4]} == {"ok"}
    assert {row["status"] for row in rows[4:]} == {"exists_valid"}


def test_one_failure_does_not_block_other_symbols(tmp_path, monkeypatch):
    out = _isolate(tmp_path, monkeypatch)

    def fake_fetch(symbol, timeout=60):
        if symbol == "QQQ":
            raise TimeoutError("vendor timeout")
        return _payload(symbol)

    monkeypatch.setattr(sc, "fetch", fake_fetch)
    assert sc.main() == 1
    assert not (out / "QQQ" / "2026-07-16.json.gz").exists()
    for symbol in ("SPY", "SPX", "NDX"):
        assert (out / symbol / "2026-07-16.json.gz").exists()
