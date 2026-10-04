# Heat equation with a source term by eigenfunction expansion
# Applied ODE & PDE with Python, Ch. 8 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
from scipy.integrate import quad

f = lambda x: x * (1 - x) * (10 - 22 * x)            # heat source
# steady state: -w'' = f, w(0) = w(1) = 0  (solved by hand)
w = lambda x: x / 10 - 5 * x**3 / 3 + 8 * x**4 / 3 - 11 * x**5 / 10

NMAX = 60
n = np.arange(1, NMAX + 1)
fn = np.array([2 * quad(lambda x: f(x) * np.sin(k * np.pi * x), 0, 1)[0]
               for k in n])                         # sine coefficients of f
lam = (n * np.pi) ** 2
print("first sine coefficients f_n:", np.round(fn[:4], 6))


def u(x, t, N=NMAX):
    """u_t = u_xx + f(x), u(0,t) = u(1,t) = 0, u(x,0) = 0."""
    T = fn[:N] / lam[:N] * (1 - np.exp(-lam[:N] * t))   # T_n(t)
    return np.sin(np.pi * np.outer(x, n[:N])) @ T


x = np.linspace(0, 1, 201)
print("\n N   max|u(x,inf) - w(x)|")
for N in [1, 2, 4, 8, 16, 32]:
    print(f"{N:2d}   {np.max(np.abs(u(x, np.inf, N) - w(x))):.3e}")

print("\n   t     u(0.25,t)   u(0.75,t)")
for t in [0.01, 0.05, 0.1, 0.2, 0.5, np.inf]:
    print(f"{t:5.2f}  {u(0.25, t)[0]: .6f}  {u(0.75, t)[0]: .6f}")
print(f" w     {w(0.25): .6f}  {w(0.75): .6f}")
