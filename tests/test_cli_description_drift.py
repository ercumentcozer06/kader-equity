# -*- coding: utf-8 -*-
"""CLI aciklamasi RAPOR-DOGRULUGU / BAYATLIK regresyon testi (2026-08-11).

BULGU (kader-oil spare-desize denetiminin equity kolu): `--help` aciklamasi canli overlay
yiginini ELLE sayiyordu ("tide x dispersion-ensemble x GEX-shield"). es_basis_unwind
2026-08-04'te 4. overlay olarak DEPLOY edilince metin guncellenmedi -> CLI, gercekte
uygulanan bir overlay'i hic saymayan bir stack tarif ediyordu (yanlis-aktif degil,
EKSIK-rapor; ayni sinif: etiket uygulamayla ayristi).

Onarim = sabit listeyi kaldirmak. Bu test o kurali kilitler: aciklama ya HIC overlay adi
saymaz, ya da config'te ACIK olan HEPSINI sayar. Boylece bir sonraki overlay deploy/rollback'i
metni sessizce yalanci yapamaz.

MODEL/SIZING'e DOKUNULMADI — yalniz cikti metni.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import run as R
from config import load_config

# CLI metninde gecebilecek overlay takma adlari (config anahtari -> metinde arayacagimiz parcalar)
_ALIASES = {
    "cor1m_froth": ("cor1m",),
    "dispersion_ensemble": ("dispersion",),
    "gex_shield": ("gex",),
    "es_basis_unwind": ("es_basis", "es-basis"),
}


def _description() -> str:
    import argparse
    parsers = []
    orig = argparse.ArgumentParser.__init__

    def spy(self, *a, **kw):
        parsers.append(kw.get("description") or (a[0] if a else ""))
        return orig(self, *a, **kw)

    argparse.ArgumentParser.__init__ = spy
    try:
        try:
            R.main(["--validate"])          # parser kurulur; --validate ag/fetch yapmaz
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.__init__ = orig
    assert parsers, "argparse aciklamasi yakalanamadi"
    return (parsers[0] or "").lower()


def _enabled_overlays():
    ov = (load_config().get("overlays", {}) or {})
    return [k for k, v in ov.items() if (v or {}).get("enabled")]


def test_description_names_all_enabled_overlays_or_none():
    desc = _description()
    enabled = _enabled_overlays()
    named = [k for k in _ALIASES if any(a in desc for a in _ALIASES[k])]
    if not named:
        return                              # sabit liste yok = bayatlayamaz (tercih edilen hal)
    missing = sorted(set(enabled) - set(named))
    assert not missing, (
        f"CLI aciklamasi ACIK overlay'leri eksik sayiyor: {missing} "
        f"(aciklama={desc!r}). Ya hepsini say ya da hicini sayma.")


def test_description_does_not_name_disabled_overlays():
    """Ters yon: KAPALI bir overlay'i canli stack'in parcasiymis gibi anmasin."""
    desc = _description()
    off = [k for k in _ALIASES if k not in _enabled_overlays()]
    named_off = [k for k in off if any(a in desc for a in _ALIASES[k])]
    assert not named_off, f"kapali overlay CLI aciklamasinda canli gibi geciyor: {named_off}"


def test_live_stack_is_reported_per_run():
    """Aciklama liste saymiyorsa, fiili stack'in BASKA bir yerde basildigini garanti et."""
    import inspect
    src = inspect.getsource(R._render)
    assert 'active_overlays' in src and 'overlay        :' in src, \
        "fiili overlay listesi artik raporlanmiyor — CLI aciklamasi tek kaynak kalmis olur"


# ── PER-ASSET SATIRI: ifade katmani GORUNURLUGU (2026-08-11, 2. tur) ──────────────────────
# BULGU: `per-asset` satiri `if ox:` (OpEx blogu) ICINE gomulmustu. opex_gate kapatilirsa
# K2-trim ve uc-gun-boost `asset_deploy`i DEGISTIRMEYE devam eder ama satir HIC basilmaz ->
# fiili sleeve boyutu rapordan okunamazdi. Ayrica manset "deploy %X" frozen stack'i gosterir;
# ifade katmani onu degistirdiginde manset sessizce farkli kaliyordu.
# FIX rapor-only: satir OpEx'ten BAGIMSIZ basilir + manset farki isaretler.

def _decision(deploy=0.80, asset_deploy=None, opex=None):
    return {"call_status": "current", "freshness": {"as_of": "2026-08-11", "age_days": 0},
            "data_source": "live", "as_of": "2026-08-11", "market_open": True,
            "tide_score": 1.0, "tide_dir": 1, "direction": "LONG",
            "position_target": deploy, "deploy_fraction": deploy,
            "active_overlays": [], "overlays": {}, "spine": {"recipe": "r"},
            "opex_gate": opex, "asset_deploy": asset_deploy}


def _render(d, capsys):
    R._render(d)
    return capsys.readouterr().out


def test_per_asset_printed_without_opex_gate(capsys):
    """REGRESYON: opex_gate KAPALI ama K2/uc-gun deploy'u degistirdi -> satir yine basilmali."""
    out = _render(_decision(asset_deploy={"SPX": 0.85, "NDX": 0.68}, opex=None), capsys)
    assert "per-asset" in out, "opex_gate kapaliyken sleeve boyutu rapordan kayboldu"
    assert "SPX %85" in out and "NDX %68" in out


def test_headline_flags_expression_layer_divergence(capsys):
    out = _render(_decision(deploy=0.80, asset_deploy={"SPX": 0.68, "NDX": 0.0}), capsys)
    poz = [ln for ln in out.splitlines() if "POZİSYON" in ln][0]
    assert "ifade katmanı DEĞİŞTİRDİ" in poz, f"manset farki gizliyor: {poz!r}"
    assert "manşetten FARKLI" in out


def test_no_false_divergence_flag_when_equal(capsys):
    out = _render(_decision(deploy=0.80, asset_deploy={"SPX": 0.80, "NDX": 0.80}), capsys)
    poz = [ln for ln in out.splitlines() if "POZİSYON" in ln][0]
    assert "DEĞİŞTİRDİ" not in poz
    assert "(= manşet deploy)" in out


def test_no_per_asset_line_when_layer_inactive(capsys):
    """asset_deploy bos (hicbir ifade katmani calismadi) -> satir basilmaz = eski cikti."""
    out = _render(_decision(asset_deploy=None), capsys)
    assert "per-asset" not in out


# ── ICRA SOZLESMESI satiri (2026-08-11, Adim 1) ────────────────────────────────────────────
# OLCUM (backtest/research/execution_timing.py, n=1857, blok-bootstrap):
#   erken (acilista rebalans) - base : dSharpe +0.03  CI[-0.25,+0.26]  -> FARK YOK
#   gec   (bir gun gec)       - base : dSharpe -0.11  CI[-0.37,+0.16]  -> ANLAMSIZ
#   seans (geceyi tutma)      - base : dSharpe -0.69  CI[-1.09,-0.30] P=0.00 -> ANLAMLI KAYIP
# Yani gun-ici zamanlama serbest, GECE TASIMAK zorunlu. Rapor bunu soylemeli ki ileride kimse
# icrayi "gun icine optimize" etmesin. (Ilk pas kendi labimda cakisan-pencere hatasi vardi ve
# ters sonuc uretmisti; duzeltme docstring'de kayitli.)

def test_execution_contract_line_present(capsys):
    out = _render(_decision(asset_deploy={"SPX": 0.5, "NDX": 0.5}), capsys)
    line = [ln for ln in out.splitlines() if "icra sözleşmesi" in ln]
    assert line, "icra sozlesmesi satiri kayboldu"
    assert "GECE" in line[0] and "betimsel" in line[0]


def test_execution_contract_does_not_claim_intraday_timing():
    """REGRESYON: olcum gun-ici zamanlamada FARK YOK dedi — satir 'kapanistan once gir' gibi
    olculmemis bir kural IDDIA ETMEMELI (ilk hatali pasin tekrari)."""
    import inspect
    src = inspect.getsource(R._render)
    i = src.find("icra sözleşmesi")
    assert i > 0
    seg = src[i:i + 400]
    for banned in ("kapanıştan önce", "kapanistan once", "açılışta gir"):
        assert banned not in seg, f"olculmemis zamanlama kurali iddia ediliyor: {banned}"
