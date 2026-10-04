# The multivalued solution of the inviscid Burgers equation after breaking
# Applied ODE & PDE with Python, Ch. 7 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

u0 = lambda x: np.exp(-x**2)        # same Gaussian hump as before, t_b = 1.1658

xi = np.linspace(-3, 3, 2001)       # parameter: foot of each characteristic
fig, ax = plt.subplots(figsize=(7, 4))
for t in [0.0, 1.1658, 2.0, 3.0]:
    # the graph of the "solution" is the curve (xi + u0(xi) t, u0(xi))
    ax.plot(xi + u0(xi) * t, u0(xi), label=f"t = {t}")
ax.set_xlabel("x"); ax.set_ylabel("u")
ax.set_title("Parametric solution curves: single-valued only for t <= t_b")
ax.legend()
plt.tight_layout()
plt.savefig("ch07_burgers_multivalued.pdf", bbox_inches="tight")


def preimages(x, t):
    """All feet xi with xi + u0(xi) t = x (located by sign changes, refined by brentq)."""
    F = lambda s: s + u0(s) * t - x
    s = np.linspace(-3, 3, 6001)
    Fs = F(s)
    idx = np.where(np.sign(Fs[:-1]) != np.sign(Fs[1:]))[0]
    return [brentq(F, s[i], s[i + 1]) for i in idx]


for t, x in [(1.0, 1.8), (2.0, 1.8), (3.0, 2.5)]:
    roots = preimages(x, t)
    vals = ", ".join(f"{u0(r):.4f}" for r in roots)
    print(f"t = {t:.1f}, x = {x:.1f}: {len(roots)} characteristic(s), u = {vals}")
