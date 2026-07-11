"""modules/_netutil — bayatlık-ÖNLEME: geçici ağ/endpoint hıçkırıklarını KAYNAKTA yutan retry-get.

Bayatlığın en sık nedeni kalıcı outage DEĞİL — geçici timeout/5xx (endpoint canlı ama o an cevap vermedi).
Tek-deneme fetch → hemen bayat-fallback/no-trade'e düşüyordu. 3-deneme üstel-backoff bunların ~%80'ini
kurtarır → veri taze kalır, gate tetiklenmeden. time.sleep MODEL sürecinde çalışır (Bash-sandbox değil).

Kullanım: cor1m_froth + gex_shield fetch'leri + ileride eklenen her canlı-veri bacağı buradan geçmeli.
"""
from __future__ import annotations

import time


def http_get_retry(url, *, timeout, attempts=3, backoff=(0.0, 3.0, 10.0), headers=None, params=None):
    """N-deneme üstel-backoff GET (requests.Response döndürür). Hepsi patlarsa SON exception'ı fırlatır
    (fail-loud korunur — sessizce None dönmez). 200 dışı = raise_for_status ile hata sayılır → retry.
    params: sorgu-parametreleri (COT/dealer Socrata gibi params'lı API'ler için)."""
    import requests
    headers = headers or {"User-Agent": "Mozilla/5.0"}
    last = None
    for i in range(attempts):
        w = backoff[i] if i < len(backoff) else backoff[-1]
        if w:
            time.sleep(w)
        try:
            r = requests.get(url, timeout=timeout, headers=headers, params=params)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001 — bir sonraki denemeye
            last = e
    raise last if last is not None else RuntimeError(f"GET başarısız (tüm denemeler): {url}")
