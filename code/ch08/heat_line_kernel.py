# The heat kernel: convolution, erf solution and infinite speed
# Applied ODE & PDE with Python, Ch. 8 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import erf, erfc

kappa = 1.0
Phi = lambda x, t: np.exp(-x**2 / (4 * kappa * t)) / np.sqrt(4 * np.pi * kappa * t)


def u_conv(x, t):
    """u = (Phi * g)(x,t) for the box data g = 1 on [-1,1], by quadrature."""
    return quad(lambda y: Phi(x - y, t), -1, 1, points=[x])[0]


def u_erf(x, t):
    """Closed form of the same convolution."""
    s = np.sqrt(4 * kappa * t)
    return 0.5 * (erf((x + 1) / s) - erf((x - 1) / s))


print("   x     t    convolution     erf formula      difference")
for x, t in [(0.0, 0.01), (0.9, 0.01), (1.0, 0.1), (2.0, 0.5), (0.0, 2.0)]:
    a, b = u_conv(x, t), u_erf(x, t)
    print(f"{x:4.1f}  {t:4.2f}  {a:.12f}  {b:.12f}  {abs(a - b):.1e}")

print("\nInfinite speed: u(x, t) > 0 everywhere for every t > 0")
for x in [2.0, 3.0, 5.0]:
    t = 0.01
    val = 0.5 * (erfc((x - 1) / np.sqrt(4 * kappa * t))
                 - erfc((x + 1) / np.sqrt(4 * kappa * t)))
    print(f"u({x:.0f}, {t}) = {val:.3e}")

x = np.linspace(-4, 4, 801)
plt.figure(figsize=(7, 3.6))
plt.plot(x, (abs(x) <= 1).astype(float), "k", lw=1, label="$t=0$")
for t in [0.01, 0.1, 0.5, 2.0]:
    plt.plot(x, u_erf(x, t), label=f"$t={t}$")
plt.xlabel("$x$"); plt.ylabel("$u(x,t)$"); plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("ch08_heat_kernel.pdf", bbox_inches="tight")
