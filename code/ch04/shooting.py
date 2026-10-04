# The shooting method for a nonlinear boundary value problem
# Applied ODE & PDE with Python, Ch. 4 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# y'' = 1.5 y^2 on [0, 1], y(0) = 4, y(1) = 1.  One exact solution: y = 4/(1+x)^2.


def rhs(x, z):
    return [z[1], 1.5 * z[0]**2]


def shoot(s):
    """Solve the IVP with y(0) = 4, y'(0) = s and return the miss F(s) = y(1; s) - 1."""
    sol = solve_ivp(rhs, (0, 1), [4.0, s], rtol=1e-11, atol=1e-11)
    return sol.y[0, -1] - 1.0


slopes = np.linspace(-45, -2, 44)
F = np.array([shoot(s) for s in slopes])
print("sign changes of F(s) between consecutive trial slopes:")
roots = []
for i in range(len(slopes) - 1):
    if F[i] * F[i + 1] < 0:
        s_star = brentq(shoot, slopes[i], slopes[i + 1], xtol=1e-12)
        roots.append(s_star)
        print(f"  bracket [{slopes[i]:.0f}, {slopes[i + 1]:.0f}]  ->  y'(0) = {s_star:.8f}")

xx = np.linspace(0, 1, 101)
plt.figure(figsize=(6.5, 3.6))
for s_star in roots:
    sol = solve_ivp(rhs, (0, 1), [4.0, s_star], t_eval=xx, rtol=1e-11, atol=1e-11)
    plt.plot(xx, sol.y[0], label=f"y'(0) = {s_star:.4f}")
    if abs(s_star + 8) < 1e-6:
        err = np.max(abs(sol.y[0] - 4 / (1 + xx)**2))
        print(f"max error against 4/(1+x)^2 for the first solution: {err:.2e}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("ch04_shooting.pdf", bbox_inches="tight")
plt.close()
