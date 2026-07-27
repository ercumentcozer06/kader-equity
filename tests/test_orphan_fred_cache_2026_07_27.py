# -*- coding: utf-8 -*-
"""YETİM FRED CACHE regresyon kilidi — 2026-07-27.

BULGU: `_audit_input_freshness` cache_dir'i CWD'ye göre çözüyordu; `_fred.fetch_series` ise
07-11'den beri REPO-KÖKLÜ (kader-macro/data/fred_cache). Ayrışma sonucu: gözlem tarihi DOĞRU
cache'ten okunurken MTS donma-dedektörünün `last_updated` meta'sı kader-equity altındaki
07-11'den kalma YETİM dosyadan okunuyordu (last_updated=2026-06-10 -> 47 takvim günü > 45 ->
sahte 'kaynak DONMUŞ') -> call_status=STALE -> deira GLOBAL HALT. FRED'in gerçek last_updated'ı
2026-07-13'tü; veri elimizde TAZEYDİ.

`_fred` yalnız kader-macro sys.path penceresi içinde import edilebildiği için burada
KAYNAK-DÜZEYİ kilit + saf karar-çekirdeği testi kullanılır (dei-ra write-sandbox testiyle aynı desen).
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "spine" / "reconstruct.py"


def _audit_fn_source() -> str:
    txt = SRC.read_text(encoding="utf-8")
    i = txt.index("def _audit_input_freshness")
    j = txt.index("\ndef ", i + 1)
    return txt[i:j]


def test_cache_dir_fredin_repo_kokunden_cozulur():
    """Yol çözümü _fred.__file__'a bağlı olmalı — CWD'ye göreli Path(...) TEK BAŞINA yasak."""
    body = _audit_fn_source()
    assert "_fred.__file__" in body, (
        "cache_dir, _fred'in FİİLEN yazdığı dizinden çözülmeli (yetim cache = sahte 'kaynak "
        "DONMUŞ' = sahte GLOBAL HALT). `Path(_fred.__file__).resolve().parents[1] / _cc` kullan.")
    assert re.search(r"is_absolute\(\)", body), "mutlak cfg yolu korunmalı (yapılandırma üstün)"


def test_mutlak_yol_korunur_goreli_yol_repo_kokune_baglanir():
    """Çözüm kuralının SAF testi (fonksiyonu çalıştırmadan aynı ifadeyi doğrular)."""
    fred_file = Path("C:/Users/admin/Downloads/kader-macro/modules/_fred.py")

    def _resolve(cc: str) -> Path:
        return Path(cc) if Path(cc).is_absolute() else fred_file.resolve().parents[1] / cc

    assert _resolve("data/fred_cache") == Path("C:/Users/admin/Downloads/kader-macro/data/fred_cache")
    assert _resolve("C:/mutlak/yol") == Path("C:/mutlak/yol")


def test_mts_donma_dedektoru_gercek_metayla_temiz_yetimle_bayat():
    """Bulgunun kendisi: AYNI gözlem tarihi, İKİ farklı last_updated -> iki farklı hüküm."""
    import spine.reconstruct as R
    gercek = R._check_spec("MTSO133FMS", "m9", None, date(2026, 6, 1), date(2026, 7, 27),
                           last_updated="2026-07-13 14:04:17-05")
    yetim = R._check_spec("MTSO133FMS", "m9", None, date(2026, 6, 1), date(2026, 7, 27),
                          last_updated="2026-06-10 14:04:38-05")
    assert gercek is None, "FRED'in gerçek meta'sıyla bacak TEMİZ olmalı"
    assert yetim is not None and "DONMUŞ" in yetim["reason"], (
        "yetim meta gerçekten sahte-donma üretiyor — bu yüzden yol çözümü kilitlendi")
