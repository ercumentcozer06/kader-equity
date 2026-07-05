"""pytest: repo kökünü sys.path'e ekle + KANONİK-ORTAM guard'ı (H8) + frozen-spine pin'i."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def _pin_frozen_spine(monkeypatch):
    """config.yaml default 'live' (günlük canlı çağrı) olsa bile testler frozen koşar → ağsız,
    deterministik, byte-identik reprodüksiyon (canlı reconstruct ~3dk FRED-fetch + non-deterministik).
    load_config()'in DÖNÜŞ cfg'sinde spine.source='frozen' sabitlenir. Canlı yolu test eden bir test
    cfg'yi tekrar 'live' yapabilir; build_state zaten kendi içinde live'ı zorlar (H1 fallback testi etkilenmez)."""
    import config as _cfg_mod
    _orig = _cfg_mod.load_config

    def _frozen(*a, **k):
        c = _orig(*a, **k)
        c.setdefault("spine", {})["source"] = "frozen"
        return c

    monkeypatch.setattr(_cfg_mod, "load_config", _frozen)

# H8: parquet engine (pyarrow) yoksa FAIL-FAST — "21/33 artefaktı"nı önler. Kanonik ortam = kader-macro venv.
try:
    import pyarrow  # noqa: F401
except ImportError:
    import pytest
    pytest.exit(
        "KANONİK ORTAM GEREKLİ: pyarrow YOK → parquet testleri patlar (sahte 21/33 artefaktı). "
        "kader-macro venv ile koş: "
        r'"C:/Users/admin/Downloads/kader-macro/.venv/Scripts/python.exe" -m pytest tests/',
        returncode=2)
