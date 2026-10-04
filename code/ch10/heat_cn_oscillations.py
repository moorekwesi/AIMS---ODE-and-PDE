# Crank-Nicolson with discontinuous data: oscillations and the Rannacher start
# Applied ODE & PDE with Python, Ch. 10 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt

N, dt = 40, 0.01
h = 1.0 / N
r = dt / h**2                                   # kappa = 1, r = 16
x = np.linspace(0.0, 1.0, N + 1)[1:-1]
u0 = np.where(np.abs(x - 0.5) < 0.25, 1.0, 0.0)
u0[np.isclose(np.abs(x - 0.5), 0.25)] = 0.5     # mean value at the jumps
D2 = sps.diags([1, -2, 1], [-1, 0, 1], shape=(N - 1, N - 1), format="csc") / h**2
I = sps.identity(N - 1, format="csc")


def exact(x, t):
    n = np.arange(1, 2001)[:, None]
    bn = 2 / (n * np.pi) * (np.cos(n * np.pi / 4) - np.cos(3 * n * np.pi / 4))
    return np.sum(bn * np.exp(-(n * np.pi)**2 * t) * np.sin(n * np.pi * x), axis=0)


def theta_stepper(theta, k):
    """Return a function U -> U^{n+1} for the theta-method with step k."""
    lu = splu((I - theta * k * D2).tocsc())
    B = (I + (1 - theta) * k * D2).tocsr()
    return lambda U: lu.solve(B @ U)


cn, be_half = theta_stepper(0.5, dt), theta_stepper(1.0, dt / 2)
be = theta_stepper(1.0, dt)
print(f"r = {r:.0f};  max errors")
print("   t      CN         BTCS       CN + Rannacher start")
U = {"cn": u0.copy(), "be": u0.copy(), "ra": u0.copy()}
snap = {}
for n in range(1, 21):
    U["cn"], U["be"] = cn(U["cn"]), be(U["be"])
    if n <= 2:                                  # two CN steps -> four BTCS half-steps
        U["ra"] = be_half(be_half(U["ra"]))
    else:
        U["ra"] = cn(U["ra"])
    t = n * dt
    if n in (1, 2, 5, 10, 20):
        e = {k: np.max(np.abs(v - exact(x, t))) for k, v in U.items()}
        print(f"  {t:.2f}   {e['cn']:.2e}   {e['be']:.2e}   {e['ra']:.2e}")
    if n == 5:
        snap = {k: v.copy() for k, v in U.items()}

xf = np.linspace(0, 1, 400)
plt.figure(figsize=(7, 4))
plt.plot(xf, exact(xf, 0.05), "k-", label="exact")
plt.plot(x, snap["cn"], "o--", ms=3, label="Crank-Nicolson")
plt.plot(x, snap["ra"], "s-", ms=3, label="CN with Rannacher start")
plt.xlabel("x"); plt.ylabel("u(x, 0.05)"); plt.legend()
plt.title(f"Box initial data, N = {N}, dt = {dt}, r = {r:.0f}")
plt.tight_layout(); plt.savefig("ch10_cn_oscillations.pdf", bbox_inches="tight")
plt.close()
