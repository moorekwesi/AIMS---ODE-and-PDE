# One initial value problem solved three ways
# Applied ODE & PDE with Python, Ch. 1 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# The IVP  y' = -2 t y + t,  y(0) = 2  on  0 <= t <= 3
# 1. Exact solution with SymPy
t = sp.symbols("t")
y = sp.Function("y")
ode = sp.Eq(y(t).diff(t), -2*t*y(t) + t)
exact = sp.dsolve(ode, y(t), ics={y(0): 2})
print("SymPy   :", exact)
print("check   :", sp.checkodesol(ode, exact))
y_exact = sp.lambdify(t, exact.rhs, "numpy")      # symbolic -> NumPy function

# 2. Numerical solution with SciPy
def f(t, y):
    return -2*t*y + t

tt = np.linspace(0, 3, 61)
for rtol in (1e-3, 1e-6, 1e-10):
    sol = solve_ivp(f, (0, 3), [2.0], t_eval=tt, rtol=rtol, atol=1e-12)
    err = np.max(np.abs(sol.y[0] - y_exact(tt)))
    print(f"solve_ivp rtol = {rtol:.0e}: {sol.nfev:4d} f-evaluations, max error {err:.2e}")

# 3. Plot the exact and numerical solutions
sol = solve_ivp(f, (0, 3), [2.0], rtol=1e-3)          # default RK45, few steps
plt.figure(figsize=(7, 4))
plt.plot(tt, y_exact(tt), "-", label="exact: 1/2 + (3/2) exp(-t^2)")
plt.plot(sol.t, sol.y[0], "o", label=f"solve_ivp steps ({sol.t.size} points)")
plt.xlabel("t"); plt.ylabel("y"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ch01_three_ways.pdf", bbox_inches="tight")
plt.close()
