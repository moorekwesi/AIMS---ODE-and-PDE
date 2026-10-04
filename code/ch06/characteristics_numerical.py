# Tracing characteristics numerically
# Applied ODE & PDE with Python, Ch. 6 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def solve_by_characteristics(v, k, f, x, T):
    """Value u(x, T) for all grid points x.  Each characteristic through
    (x, T) is followed backwards to t = 0 while the decay is accumulated:
        dX/ds = -v(X),  dI/ds = k(X),  s in [0, T]   (s = T - t),
    then u(x, T) = f(X(T)) exp(-I(T))."""
    n = x.size

    def rhs(s, z):
        X = z[:n]
        return np.concatenate([-v(X), k(X)])

    z0 = np.concatenate([x, np.zeros(n)])
    sol = solve_ivp(rhs, [0, T], z0, rtol=1e-9, atol=1e-12)
    X0, I = sol.y[:n, -1], sol.y[n:, -1]
    return f(X0)*np.exp(-I)


# 1. validation on the exam problem: v = 1 + x^2, k = 0, f = 1/(1+x^2)
x = np.linspace(-1, 3, 81)
T = 0.5
u_num = solve_by_characteristics(lambda X: 1 + X**2, lambda X: 0*X,
                                 lambda X: 1/(1 + X**2), x, T)
u_ex = (np.cos(T) + x*np.sin(T))**2/(1 + x**2)
print(f"exam problem, T = {T}: max error = {np.max(np.abs(u_num - u_ex)):.2e}")

# 2. a river with a slow, wide stretch and a wetland that removes pollutant
v = lambda X: 1.0 - 0.6*np.exp(-(X - 5)**2)          # speed drops near x = 5
k = lambda X: 0.05 + 0.5*np.exp(-(X - 5)**2/0.5)     # strong decay in wetland
f = lambda X: np.exp(-4*(X - 1)**2)                  # initial plume at x = 1
x = np.linspace(0, 12, 601)
fig, ax = plt.subplots(figsize=(7, 3.8))
for T in [0, 2, 4, 6, 8]:
    uT = solve_by_characteristics(v, k, f, x, T)
    i = np.argmax(uT)
    m1, m2 = np.trapz(uT, x), np.trapz(uT/v(x), x)   # see the text
    print(f"T = {T}: peak {uT[i]:.4f} at x = {x[i]:.2f}, "
          f"int u dx = {m1:.4f}, int u/v dx = {m2:.4f}")
    ax.plot(x, uT, label=f"t = {T}")
ax.axvspan(4, 6, color="0.9")
ax.set_xlabel("x (km)"); ax.set_ylabel("u"); ax.legend()
plt.tight_layout()
plt.savefig("ch06_variable_river.pdf", bbox_inches="tight")
