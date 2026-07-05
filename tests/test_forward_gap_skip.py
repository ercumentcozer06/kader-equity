"""M1 (ops-fix 2026-07-06) — KALICI-GAP forward-skorlama testleri.

2026-06-22..07-04 makine-KAPALI penceresinde canlı opsiyon-zinciri snapshot'ı ALINMADI
(gamma_spy/gamma_qqq cache'i 06-22 → 07-05 atlıyor) → o 8 işlem günü (06-23..07-02)
KURTARILAMAZ point-in-time. Sidecar output/forward_ledger_gaps.json bunu KALICI kaydeder.

Bu testler İKİ YÖNÜ de doğrular:
  • GAP tarihi bir defter satırıysa → forward-skorlama onu AÇIKÇA atlar (signal_pnl=None), UYDURMAZ,
    saymaz; GÖRÜNÜR not basar. (Geçerli ertesi-gün kapanışı OLSA BİLE skorlanmaz = uydurma-koruması.)
  • GAP-DIŞI tarih → normal skorlanır (skip yalnız kalıcı-boşluk günlerine uygulanır).
  • Sidecar sadece recoverable=false boşlukları atlar; recoverable=true atlanmaz.
"""
from __future__ import annotations

import json

import pandas as pd


GAP_DATES = ["2026-06-23", "2026-06-24", "2026-06-25", "2026-06-26",
             "2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02"]


def _seed_call(L, as_of: str, pos: float = 0.5):
    L.append_call({"as_of": as_of, "computed_at": "t", "model_tag": "test",
                   "call_status": "current", "position_target": pos, "direction": "LONG",
                   "size": pos, "tide_dir": 1, "tide_score": 1.0,
                   "active_overlays": "", "data_source": "live"})


def _write_gaps_sidecar(path, dates=GAP_DATES, recoverable=False):
    path.write_text(json.dumps({"gaps": [{
        "span": "2026-06-23..2026-07-02", "dates": list(dates),
        "reason": "machine-off 2026-06-22..07-04; no live option-chain snapshot captured",
        "recoverable": recoverable, "logged_by": "ops-fix M1 2026-07-06"}]}), encoding="utf-8")


def test_real_sidecar_lists_the_eight_permanent_gap_dates():
    """CANLI sidecar dosyası (repo'daki) tam 8 kalıcı-gap gününü içeriyor mu."""
    from validation import ledger as L
    got = L.load_permanent_gaps()
    assert got == set(GAP_DATES), f"sidecar kalıcı-gap kümesi beklenenden farklı: {sorted(got)}"


def test_gap_row_is_skipped_not_fabricated(tmp_path, monkeypatch, capsys):
    """GAP tarihi defterde + geçerli ertesi-gün kapanışı VAR → yine de signal_pnl=None (uydurulmadı),
    price_stale=False (bayat-kaynak DEĞİL), GÖRÜNÜR KALICI-GAP notu basıldı."""
    from validation import ledger as L
    monkeypatch.setattr(L, "ledger_path", lambda: tmp_path / "fl.parquet")
    monkeypatch.setattr(L, "gaps_path", lambda: tmp_path / "gaps.json")
    _write_gaps_sidecar(tmp_path / "gaps.json")
    # kapanış serisi gap gününü ve ertesi işlem gününü KAPSIYOR → normalde skorlanabilir olurdu
    idx = pd.bdate_range("2026-06-22", "2026-07-06")
    closes = pd.Series([100.0 + i for i in range(len(idx))], index=idx)
    monkeypatch.setattr(L, "_index_closes", lambda asset=L.REF_ASSET: closes)
    _seed_call(L, "2026-06-24")                       # bir GAP günü
    df = L.mark_to_market(closes=closes)              # closes verildi → H4 canlı-dalı devre dışı, izole test
    out = capsys.readouterr().out
    assert "KALICI-GAP atlandı" in out, "görünür KALICI-GAP notu basılmadı"
    row = df[df["as_of"].astype(str) == "2026-06-24"].iloc[0]
    assert row["signal_pnl"] is None, "GAP satırı skorlandı (UYDURMA) — atlanmalıydı"
    assert bool(row["price_stale"]) is False, "GAP satırı bayat-kaynak damgalandı; kalıcı-boşluk ayrı kategoridir"


def test_gap_dates_not_counted_in_summary(tmp_path, monkeypatch):
    """drag_summary: GAP satırı n_signal_marked'a KATILMAZ (aggregate sessizce şişmez);
    n_permanent_gap alanı boşluğu AÇIKÇA raporlar."""
    from validation import ledger as L
    monkeypatch.setattr(L, "ledger_path", lambda: tmp_path / "fl.parquet")
    monkeypatch.setattr(L, "gaps_path", lambda: tmp_path / "gaps.json")
    _write_gaps_sidecar(tmp_path / "gaps.json")
    idx = pd.bdate_range("2026-06-22", "2026-07-06")
    closes = pd.Series([100.0 + i for i in range(len(idx))], index=idx)
    monkeypatch.setattr(L, "_index_closes", lambda asset=L.REF_ASSET: closes)
    _seed_call(L, "2026-06-24")                       # GAP → atlanmalı
    _seed_call(L, "2026-06-22")                       # GAP-DIŞI → skorlanmalı
    L.mark_to_market(closes=closes)
    ds = L.drag_summary()
    assert ds["n_calls"] == 2
    assert ds["n_signal_marked"] == 1, "yalnız GAP-DIŞI satır işaretlenmeli (GAP sayılmamalı)"
    assert ds["n_permanent_gap"] == len(GAP_DATES), "kalıcı-boşluk sayısı açıkça raporlanmalı"


def test_non_gap_date_still_scored(tmp_path, monkeypatch):
    """GAP-DIŞI tarih normal skorlanır — skip YALNIZ kalıcı-boşluk günlerine uygulanır (yön-2)."""
    from validation import ledger as L
    monkeypatch.setattr(L, "ledger_path", lambda: tmp_path / "fl.parquet")
    monkeypatch.setattr(L, "gaps_path", lambda: tmp_path / "gaps.json")
    _write_gaps_sidecar(tmp_path / "gaps.json")
    idx = pd.bdate_range("2026-06-15", "2026-06-22")
    closes = pd.Series([100.0 + i for i in range(len(idx))], index=idx)
    monkeypatch.setattr(L, "_index_closes", lambda asset=L.REF_ASSET: closes)
    _seed_call(L, "2026-06-16")                       # GAP-DIŞI, ertesi kapanış var
    df = L.mark_to_market(closes=closes)
    row = df[df["as_of"].astype(str) == "2026-06-16"].iloc[0]
    assert row["signal_pnl"] is not None, "GAP-DIŞI tarih skorlanmadı — skip fazla geniş"


def test_recoverable_gap_not_skipped(tmp_path, monkeypatch):
    """recoverable=true işaretli boşluk ATLANMAZ (yalnız kalıcı/kurtarılamaz olanlar atlanır)."""
    from validation import ledger as L
    monkeypatch.setattr(L, "gaps_path", lambda: tmp_path / "gaps.json")
    _write_gaps_sidecar(tmp_path / "gaps.json", dates=["2026-06-24"], recoverable=True)
    assert L.load_permanent_gaps() == set(), "recoverable=true boşluk yanlışlıkla kalıcı-gap sayıldı"


def test_missing_sidecar_is_safe_no_skip(tmp_path, monkeypatch):
    """Sidecar yoksa → boş küme, hiçbir tarih atlanmaz (mevcut davranış bozulmaz)."""
    from validation import ledger as L
    monkeypatch.setattr(L, "gaps_path", lambda: tmp_path / "does_not_exist.json")
    assert L.load_permanent_gaps() == set()
