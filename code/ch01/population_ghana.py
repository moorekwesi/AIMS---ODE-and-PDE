# Malthus and logistic models for the population of Ghana
# Applied ODE & PDE with Python, Ch. 1 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Ghana Statistical Service census counts (persons)
year = np.array([1960, 1970, 1984, 2000, 2010, 2021], dtype=float)
pop = np.array([6726815, 8559313, 12296081, 18912079, 24658823, 30832019]) / 1e6
s = year - 1960.0                      # time in years since 1960

# Malthus: P = P0 exp(r s)  <=>  ln P = ln P0 + r s  (straight-line fit)
r_m, lnP0 = np.polyfit(s, np.log(pop), 1)
malthus = lambda s: np.exp(lnP0) * np.exp(r_m * s)

def logistic(s, P0, r, K):
    """Solution of P' = r P (1 - P/K) with P(0) = P0."""
    return K / (1.0 + (K / P0 - 1.0) * np.exp(-r * s))

(P0_l, r_l, K_l), _ = curve_fit(logistic, s, pop, p0=(6.7, 0.03, 60.0))

print(f"Malthus : P0 = {np.exp(lnP0):6.3f} M, r = {r_m:.4f} per year")
print(f"Logistic: P0 = {P0_l:6.3f} M, r = {r_l:.4f} per year, K = {K_l:.1f} M")
print(" year   census  Malthus  logistic   (millions)")
for yr, p in zip(year, pop):
    print(f" {yr:4.0f}  {p:7.2f}  {malthus(yr-1960):7.2f}  {logistic(yr-1960, P0_l, r_l, K_l):8.2f}")
for yr in (2030, 2050):
    print(f" {yr:4d}     --    {malthus(yr-1960):7.2f}  {logistic(yr-1960, P0_l, r_l, K_l):8.2f}")

tt = np.linspace(1950, 2060, 300)
plt.figure(figsize=(7, 4))
plt.plot(year, pop, "ko", label="census")
plt.plot(tt, malthus(tt - 1960), "--", label="Malthus (exponential)")
plt.plot(tt, logistic(tt - 1960, P0_l, r_l, K_l), "-", label="logistic")
plt.xlabel("year"); plt.ylabel("population (millions)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("ch01_population_ghana.pdf", bbox_inches="tight")
plt.close()
