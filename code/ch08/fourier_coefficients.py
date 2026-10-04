# Fourier coefficients by quadrature and Parseval's identity
# Applied ODE & PDE with Python, Ch. 8 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
from scipy.integrate import quad


def coefficients(f, n):
    """Return (a_n, b_n) of the 2*pi-periodic function f on (-pi, pi)."""
    a = quad(lambda x: f(x) * np.cos(n * x), -np.pi, np.pi, limit=200)[0] / np.pi
    b = quad(lambda x: f(x) * np.sin(n * x), -np.pi, np.pi, limit=200)[0] / np.pi
    return a, b


saw = lambda x: x            # odd: sine series only
tent = lambda x: abs(x)      # even: cosine series only

print(" n   b_n(x) quad   exact     a_n(|x|) quad   exact")
for n in range(1, 7):
    _, b = coefficients(saw, n)
    a, _ = coefficients(tent, n)
    b_ex = 2 * (-1) ** (n + 1) / n
    a_ex = 2 * ((-1) ** n - 1) / (np.pi * n**2)
    print(f"{n:2d}  {b:11.8f} {b_ex:11.8f}  {a:12.8f} {a_ex:11.8f}")

# Parseval for f(x) = x:  (1/pi) int x^2 = 2 pi^2/3 = sum b_n^2 = 4 sum 1/n^2
lhs = quad(lambda x: x**2, -np.pi, np.pi)[0] / np.pi
print(f"\n(1/pi)*int f^2 = {lhs:.10f}")
for N in [10, 100, 1000, 10000]:
    n = np.arange(1, N + 1)
    s = np.sum((2.0 / n) ** 2)
    print(f"N = {N:5d}: sum b_n^2 = {s:.10f}   deficit = {lhs - s:.3e}")
print(f"=> sum 1/n^2 = pi^2/6 = {np.pi**2 / 6:.10f}")
