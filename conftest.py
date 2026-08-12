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


# ── KARDEŞ-REPO SIZINTI BARİYERİ (2026-08-11, KANONİK — 6 repoda eş) ────────────────────────
# BULGU: bazı modüller import ANINDA `sys.path.insert(0, <kardeş repo>)` yapar (ör. gold
# spine/phys_stress/sge.py -> kader-macro'nun modules._fred'ine MEŞRU ihtiyaç). Bir test bunu
# gövdesinde import edince kardeş repo süreç genelinde sys.path'in ÖNÜNE geçer ve o andan sonra
# çıplak `import notify` / `config` / `modules` YANLIŞ REPONUN modülünü bulur.
# Ölçülen zarar (silver, 2026-08-11): "SILVER veri bayat" alarmı kader-macro/output/'a yazıldı;
# silver'ın kendi output'unda iz YOKTU -> gerçek bir bayatlıkta alarm bakılan yerde görünmezdi.
# Enjeksiyonun kendisi meşru olabilir (subprocess yolunda gerekli) -> silmek yerine KAPSA:
# her testten sonra sys.path eski haline döner ve o test sırasında YENİ yüklenmiş, BAŞKA repoya
# ait üst-seviye modüller sys.modules'tan atılır (sonraki test temiz başlasın).
# Sızıntı yoksa fixture hiçbir şey yapmaz = mevcut davranış birebir aynı.
@pytest.fixture(autouse=True)
def _sibling_repo_leak_barrier():
    _self = Path(__file__).resolve().parent
    if _self.name == "tests":
        _self = _self.parent
    _path_before = list(sys.path)
    _mods_before = set(sys.modules)
    yield
    if sys.path != _path_before:
        sys.path[:] = _path_before
    for _name in list(set(sys.modules) - _mods_before):
        if "." in _name:
            continue                          # yalnız üst-seviye ad çakışır
        _m = sys.modules.get(_name)
        _f = getattr(_m, "__file__", None)
        if not _f:
            continue
        _fp = str(_f)
        if "Downloads" in _fp and "kader-" in _fp and str(_self) not in _fp:
            del sys.modules[_name]            # kardeş repodan gelen gölge modülü at
