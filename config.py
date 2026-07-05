"""
config — kader-equity yapılandırma + ortam (env). Düz yapı, kök modül (kader-btc ile aynı idiom).

İki .env:
  - kader-equity/.env   : equity veri-kaynak key'leri (Faz 1 fetcher'ları; yoksa sorun değil).
  - kader-macro/.env    : FRED key (yalnız canlı RAW m2 rekonstrüksiyonu için; frozen path FRED gerektirmez).
Key GÖMÜLMEZ. Eksik key → ilgili katman "backtestlenemez (key bekliyor)" işaretlenir.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
_ENV_LOADED = False


@lru_cache(maxsize=1)
def load_config() -> dict:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    _load_envs(cfg)
    return cfg


def _load_envs(cfg: dict) -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    load_dotenv(ROOT / ".env")
    macro_repo = Path((cfg.get("macro", {}) or {}).get("repo_path", ""))
    if (macro_repo / ".env").exists():
        load_dotenv(macro_repo / ".env")                  # FRED (canlı raw-m2 için)
    _ENV_LOADED = True


def get_key(env_var: str) -> str | None:
    val = os.environ.get(env_var)
    return val if val else None
