import sys
sys.path.insert(0, r"C:\Users\admin\Downloads\kader-equity")
from modules import cor1m_froth as cf

LO, HI, FL = 8.0, 11.0, 0.0  # canlı config

try:
    s = cf.fetch_cor1m_live()
except Exception as e:
    print("FETCH HATASI:", type(e).__name__, str(e)[:200]); raise SystemExit

print("CANLI COR1M (CBOE CDN) — son 8 gün:")
print("-" * 48)
for dt, v in s.tail(8).items():
    f = cf.froth_factor(float(v), LO, HI, FL)
    zone = "DANGER (<8, taban)" if v <= LO else ("FROTH (<11, trim)" if v < HI else "normal")
    print(f"  {str(dt.date())}   COR1M {float(v):6.2f}   faktor {f:.3f}   {zone}")

c = float(s.iloc[-1]); asof = str(s.index[-1].date())
f = cf.froth_factor(c, LO, HI, FL)
print("-" * 48)
print(f"CANLI: COR1M {c:.2f} @ {asof}  ->  froth faktor {f:.3f}")
print(f"  (Pazartesi snapshot 10.74 -> 0.913 idi; degisim = {f-0.913:+.3f})")
if c <= LO:
    print("  >>> 8 ALTI: model equity tide-long'u TABANA (x0) ceker = TAM DE-RISK")
elif c < HI:
    print(f"  >>> 8-11 arasi: kismi trim, pozisyon x{f:.3f}")
else:
    print("  >>> 11 ustu: normal, tam long")
