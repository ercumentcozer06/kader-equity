"""EQ-1/EQ-2 onarım testleri (denetim 2026-07-04).

EQ-1: H4 bayat-fiyat bekçisi import'suz datetime yüzünden her çağrıda NameError üretip
except-pass ile ölüyordu → price_stale hiçbir zaman True olamıyordu. Bu test CANLI dalı
(closes=None) çalıştırır — guard NameError'suz hesaplanmalı ve bayat kaynakta STALE damgalamalı.

EQ-2: latest.json artık run.write_latest ile otomasyon yolundan da yazılıyor.
"""
from __future__ import annotations

import json

import pandas as pd


def _seed_call(L, as_of: str):
    L.append_call({"as_of": as_of, "computed_at": "t", "model_tag": "test",
                   "call_status": "current", "position_target": 0.5, "direction": "LONG",
                   "size": 0.5, "tide_dir": 1, "tide_score": 1.0,
                   "active_overlays": "", "data_source": "live"})


def test_h4_guard_alive_and_marks_stale(tmp_path, monkeypatch, capsys):
    from validation import ledger as L
    monkeypatch.setattr(L, "ledger_path", lambda: tmp_path / "fl.parquet")
    # M1: bu test H4 bayat-fiyat bekçisini İZOLE eder — KALICI-GAP sidecar'ından bağımsız olmalı
    # (aksi halde date-relative pencere gap-penceresine kayınca yanlış-negatif verir).
    monkeypatch.setattr(L, "gaps_path", lambda: tmp_path / "no_gaps.json")
    # bayat 'canlı' fiyat serisi: son kapanış haftalar önce (bugüne göre >2 işlem günü)
    idx = pd.bdate_range("2026-06-01", periods=9)
    closes = pd.Series([100.0 + i for i in range(9)], index=idx)
    monkeypatch.setattr(L, "_index_closes", lambda asset=L.REF_ASSET: closes)
    _seed_call(L, str(idx[-1].date()))
    df = L.mark_to_market()            # closes=None → CANLI dal (kırık olan buydu)
    out = capsys.readouterr().out
    assert "H4 bekçisi bu koşuda KÖR" not in out, "guard hâlâ exception yutuyor"
    assert "LEDGER ALARM" in out, "bayat kaynak alarmı basılmadı"
    assert bool(df["price_stale"].iloc[-1]) is True, "bayat kaynakta price_stale=True damgalanmadı"


def test_h4_guard_fresh_source_no_alarm(tmp_path, monkeypatch, capsys):
    from validation import ledger as L
    monkeypatch.setattr(L, "ledger_path", lambda: tmp_path / "fl2.parquet")
    # M1: KALICI-GAP sidecar'ından izole et (date-relative pencere gap-penceresine kayabilir).
    monkeypatch.setattr(L, "gaps_path", lambda: tmp_path / "no_gaps.json")
    # taze seri: bugüne kadar kapanışlar → alarm yok, guard sessiz-temiz
    idx = pd.bdate_range(end=pd.Timestamp.now(tz="UTC").tz_localize(None).normalize(), periods=9)
    closes = pd.Series([100.0 + i for i in range(9)], index=idx)
    monkeypatch.setattr(L, "_index_closes", lambda asset=L.REF_ASSET: closes)
    _seed_call(L, str(idx[-2].date()))
    df = L.mark_to_market()
    out = capsys.readouterr().out
    assert "LEDGER ALARM" not in out
    assert "H4 bekçisi bu koşuda KÖR" not in out
    assert not bool(df["price_stale"].iloc[-1])
    assert df["signal_pnl"].iloc[-1] is not None       # taze kaynakta işaretleme çalışıyor


def test_write_latest_updates_artifact(tmp_path, monkeypatch):
    import run
    monkeypatch.setattr(run, "ROOT", tmp_path)
    d = {"as_of": "2026-07-04", "call_status": "current", "position_target": 0.42}
    run.write_latest(d)
    got = json.loads((tmp_path / "output" / "kader_equity_latest.json").read_text(encoding="utf-8"))
    assert got["as_of"] == "2026-07-04" and got["position_target"] == 0.42
