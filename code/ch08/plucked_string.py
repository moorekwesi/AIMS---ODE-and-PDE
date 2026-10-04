# The plucked string: Fourier series versus d'Alembert, and energy
# Applied ODE & PDE with Python, Ch. 8 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

L, c, h, a = 1.0, 1.0, 1.0, 0.2          # string length, speed, pluck height, position
f = lambda x: np.where(x < a, h * x / a, h * (L - x) / (L - a))
n = np.arange(1, 801)
bn = 2 * h * L**2 * np.sin(n * np.pi * a / L) / (n**2 * np.pi**2 * a * (L - a))


def u_series(x, t, N=800):
    k = n[:N]
    return np.sin(np.outer(x, k) * np.pi / L) @ (bn[:N] * np.cos(k * np.pi * c * t / L))


def u_dalembert(x, t):
    """Odd, 2L-periodic extension of f in d'Alembert's formula (g = 0)."""
    def F(s):
        s = np.mod(s, 2 * L)
        return np.where(s <= L, f(s), -f(2 * L - s))
    return 0.5 * (F(x - c * t) + F(x + c * t))


print("harmonic amplitudes |b_n|, n = 1..10:")
print(" ".join(f"{abs(b):.4f}" for b in bn[:10]))
x = np.linspace(0, L, 1001)
for t in [0.1, 0.35, 0.8, 1.5]:
    err = np.max(np.abs(u_series(x, t) - u_dalembert(x, t)))
    print(f"t = {t:4.2f}: max|series(800) - dAlembert| = {err:.2e}")

E_exact = 0.5 * c**2 * h**2 * (1 / a + 1 / (L - a))     # (1/2) int c^2 f'^2
for N in [10, 100, 800]:
    E = (np.pi**2 * c**2 / (4 * L)) * np.sum(n[:N]**2 * bn[:N]**2)
    print(f"energy from {N:3d} modes = {E:.6f}   (exact {E_exact:.6f})")
print(f"kora string: L = 0.6 m, f_1 = 220 Hz  =>  c = 2 L f_1 = {2 * 0.6 * 220:.0f} m/s")

plt.figure(figsize=(7, 3.6))
for t in [0, 0.1, 0.3, 0.5, 0.75, 1.0]:
    plt.plot(x, u_dalembert(x, t), label=f"$t={t}$")
plt.xlabel("$x$"); plt.ylabel("$u(x,t)$"); plt.legend(fontsize=8, ncol=3)
plt.tight_layout()
plt.savefig("ch08_plucked.pdf", bbox_inches="tight")
