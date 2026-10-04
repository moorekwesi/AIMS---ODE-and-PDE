# Regions of absolute stability of Euler, RK2, RK4 and backward Euler
# Applied ODE & PDE with Python, Ch. 4 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# stability functions R(z): y_{n+1} = R(z) y_n for y' = lambda y, z = h lambda
R = {"Euler": lambda z: 1 + z,
     "RK2 (Heun)": lambda z: 1 + z + z**2 / 2,
     "RK4": lambda z: 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24,
     "backward Euler": lambda z: 1 / (1 - z)}

print("left end of the real stability interval [x*, 0]:")
for name in ["Euler", "RK2 (Heun)", "RK4"]:
    xs = brentq(lambda x: abs(R[name](x)) - 1, -4.0, -1.0)
    print(f"  {name:12s}: x* = {xs:.4f}")
print("  backward Euler: stable for every Re(z) <= 0 (A-stable)")

x, y = np.meshgrid(np.linspace(-5, 3, 500), np.linspace(-4, 4, 500))
Z = x + 1j * y
fig, axs = plt.subplots(1, 4, figsize=(12, 3.3), sharey=True)
for ax, (name, Rf) in zip(axs, R.items()):
    ax.contourf(x, y, abs(Rf(Z)), levels=[0, 1], colors=["#9ecae1"])
    ax.contour(x, y, abs(Rf(Z)), levels=[1], colors="k", linewidths=1)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_title(name)
    ax.set_xlabel("Re z")
    ax.set_aspect("equal")
axs[0].set_ylabel("Im z")
plt.tight_layout()
plt.savefig("ch04_stability_regions.pdf", bbox_inches="tight")
plt.close()
