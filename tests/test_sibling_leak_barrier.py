# -*- coding: utf-8 -*-
"""KARDES-REPO SIZINTI BARIYERI regresyon testi (2026-08-11, KANONIK — 6 repoda es).

BULGU: bazi moduller import ANINDA `sys.path.insert(0, <kardes repo>)` yapar (gold
spine/phys_stress/sge.py -> kader-macro modules._fred; silver build_panel/phys_stress ayni).
Enjeksiyon subprocess yolunda MESRU, ama bir test onu govdesinde import edince kardes repo
surec genelinde sys.path'in ONUNE geciyor ve o andan sonra ciplak `import notify` / `config` /
`modules` YANLIS REPONUN modulunu buluyor.

Olculen zarar (silver, ayni gun): "SILVER veri bayat" alarmi kader-macro/output/'a yazildi;
silver'in kendi output'unda iz YOKTU -> gercek bir bayatlikta alarm bakilan yerde GORUNMEZDI.
Gold'da ayni sizinti sessizce mevcuttu (KMPRESENT=True) ama henuz patlamamisti.

conftest'teki autouse `_sibling_repo_leak_barrier` her testten sonra sys.path'i eski haline
dondurur ve o test sirasinda yuklenmis BASKA repoya ait ust-seviye modulleri sys.modules'tan atar.
Bu iki test SIRALI calisir: ilki kirletir, ikincisi bariyerin temizledigini dogrular.
"""
import sys
import types
from pathlib import Path

SELF = Path(__file__).resolve().parents[1]
SIBLING = Path(r"C:\Users\admin\Downloads\kader-macro")
_FAKE = "_kader_leak_probe_mod"


def test_a_inject_sibling_path_and_module():
    """Kirletici test — gercek bir enjekte-eden modulun yaptigini birebir taklit eder."""
    sys.path.insert(0, str(SIBLING))
    m = types.ModuleType(_FAKE)
    m.__file__ = str(SIBLING / "notify.py")          # kardes repoya ait gibi gorunsun
    sys.modules[_FAKE] = m
    assert str(SIBLING) in sys.path and _FAKE in sys.modules


def test_b_barrier_restored_everything():
    """REGRESYON: bariyer yoksa bu test kirmizi olur (kardes repo hala sys.path'te)."""
    leaked = [p for p in sys.path if str(SIBLING) == str(p)]
    assert not leaked, f"kardes repo sys.path'te KALDI: {{leaked}} — bariyer devre disi mi?"
    assert _FAKE not in sys.modules, "kardes repo modulu sys.modules'ta KALDI (golge riski surer)"


def test_c_own_modules_are_not_evicted():
    """Bariyer KENDI repomuzun modullerini ATMAMALI (yanlis-pozitif olmasin).

    Repo-agnostik: her repoda `notify` yok (kader-btc'de alarm baska yoldan) -> mevcut olan
    ilk ust-seviye repo modulu denenir; hicbiri yoksa test anlamsizdir, atlanir."""
    import importlib
    import pytest
    for name in ("notify", "config", "run"):
        if not (SELF / f"{name}.py").exists():
            continue
        importlib.import_module(name)
        assert Path(sys.modules[name].__file__).resolve().parent == SELF,             f"{name} kendi repomuzdan cozulmedi (bariyer yanlis-pozitif mi?)"
        return
    pytest.skip("bu repoda ust-seviye tekil modul yok (spine/modules paketleri kullaniliyor)")
