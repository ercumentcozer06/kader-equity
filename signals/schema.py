"""
schema — ModuleSignal: downstream overlay'lerin (Faz 2: vol-surface, COT, breadth...) yaydığı standart şema.

forecast −CAP..+CAP (Carver idiom, kader-macro/btc ±20 ile tutarlı), conviction 0..1.
valid=False → overlay beyne "YOK" girer (combiner renormalize) — sessizce 0 DEĞİL (eksik ≠ nötr).
Pass 0'da hiç overlay yok → karar = tide_dir (faithful invariant).
"""
from __future__ import annotations

from dataclasses import dataclass, field

CAP = 20.0


@dataclass
class ModuleSignal:
    module: str
    forecast: float = 0.0                         # −CAP..+CAP (yön + güç)
    conviction: float = 0.0                       # 0..1 (ekstremite × teyit)
    valid: bool = False
    regime_validity: dict = field(default_factory=dict)
    falsified: bool = False
    note: str = ""

    def __post_init__(self):
        self.forecast = max(-CAP, min(CAP, float(self.forecast)))
        self.conviction = max(0.0, min(1.0, float(self.conviction)))

    @property
    def usable(self) -> bool:
        return bool(self.valid and not self.falsified)


def disabled(module: str) -> ModuleSignal:
    return ModuleSignal(module=module, forecast=0.0, conviction=0.0, valid=False, note="disabled")


def invalid(module: str, reason: str = "") -> ModuleSignal:
    return ModuleSignal(module=module, forecast=0.0, conviction=0.0, valid=False,
                        note=f"invalid: {reason}" if reason else "invalid")


def neutral(module: str, conviction: float = 0.0, regime_validity: dict | None = None) -> ModuleSignal:
    return ModuleSignal(module=module, forecast=0.0, conviction=conviction, valid=True,
                        regime_validity=regime_validity or {}, note="neutral")
