# Newton's law of cooling and radiocarbon dating
# Applied ODE & PDE with Python, Ch. 1 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np

# --- Newton's law of cooling: T' = -k (T - A),  T(t) = A + (T0 - A) exp(-k t)
A, T0, T5 = 30.0, 90.0, 75.0          # room, initial and 5-minute temperatures (deg C)
k = np.log((T0 - A) / (T5 - A)) / 5.0  # from T(5) = 75
t60 = np.log((T0 - A) / (60.0 - A)) / k
print(f"cooling constant k        = {k:.5f} per minute")
print(f"time to cool to 60 C      = {t60:.2f} minutes")
for t in (0, 5, 10, 20, 30):
    print(f"   T({t:2d}) = {A + (T0 - A)*np.exp(-k*t):6.2f} C")

# --- Radioactive decay: N' = -lam N,  N(t) = N0 exp(-lam t)
half_life = 5730.0                    # carbon-14 half-life in years
lam = np.log(2.0) / half_life
print(f"\ndecay constant of C-14    = {lam:.4e} per year")
for frac in (0.90, 0.50, 0.30, 0.10):
    age = np.log(1.0 / frac) / lam
    print(f"   {100*frac:4.0f}% of C-14 left  ->  age = {age:8.0f} years")
