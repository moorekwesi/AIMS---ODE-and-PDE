# Operators, lambdify and the heat kernel
# Applied ODE & PDE with Python, Ch. 5 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x, t, kappa = sp.symbols("x t kappa", positive=True)


def heat_op(u, k=kappa):
    """The heat operator L[u] = u_t - k u_xx."""
    return sp.diff(u, t) - k*sp.diff(u, x, 2)


candidates = {
    "heat kernel": sp.exp(-x**2/(4*kappa*t))/sp.sqrt(4*sp.pi*kappa*t),
    "Fourier mode": sp.exp(-kappa*sp.pi**2*t)*sp.sin(sp.pi*x),
    "x^2 + 2 kappa t": x**2 + 2*kappa*t,
    "x^2 + kappa t": x**2 + kappa*t,
}
for name, u in candidates.items():
    print(f"{name:16s} L[u] = {sp.simplify(heat_op(u))}")

# turn the symbolic heat kernel into a fast NumPy function
Phi = sp.lambdify((x, t, kappa), candidates["heat kernel"], "numpy")
X = np.linspace(-12, 12, 2401)
fig, ax = plt.subplots(figsize=(7, 4))
for tt in [0.05, 0.2, 0.5, 1.0, 2.0]:
    vals = Phi(X, tt, 1.0)
    mass = np.trapz(vals, X)
    print(f"t = {tt:4.2f}: max = {vals.max():.4f}, integral = {mass:.6f}")
    ax.plot(X, vals, label=f"t = {tt}")
ax.set_xlim(-5, 5); ax.set_xlabel("x"); ax.set_ylabel("u(x,t)")
ax.set_title("Heat kernel, kappa = 1"); ax.legend()
plt.tight_layout()
plt.savefig("ch05_heat_kernel.pdf", bbox_inches="tight")
