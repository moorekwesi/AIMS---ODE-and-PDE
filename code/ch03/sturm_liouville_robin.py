# Eigenvalues of y'' + lambda y = 0, y(0) = 0, y'(1) + y(1) = 0: equation, shooting, orthogonality
# Applied ODE & PDE with Python, Ch. 3 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import solve_ivp, quad

# (1) eigenvalue equation: y = sin(mu x) and mu cos(mu) + sin(mu) = 0, i.e. tan(mu) = -mu
F = lambda mu: mu * np.cos(mu) + np.sin(mu)
mus = [brentq(F, (k - 0.5) * np.pi + 1e-9, k * np.pi) for k in range(1, 6)]


# (2) shooting: solve y'' = -lam y, y(0) = 0, y'(0) = 1 and adjust lam until
#     the boundary residual B(lam) = y'(1) + y(1) vanishes
def B(lam):
    s = solve_ivp(lambda x, u: [u[1], -lam * u[0]], (0, 1), [0.0, 1.0],
                  rtol=1e-11, atol=1e-12)
    return s.y[1, -1] + s.y[0, -1]


print(" k   mu_k        lambda_k (tan)   lambda_k (shooting)  (k-1/2)^2 pi^2")
for k, mu in enumerate(mus, start=1):
    lam_shoot = brentq(B, (mu - 0.3) ** 2, (mu + 0.3) ** 2)
    print(f"{k:2d}  {mu:.6f}  {mu**2:15.6f}  {lam_shoot:18.6f}  {((k-0.5)*np.pi)**2:14.4f}")

# (3) orthogonality of the eigenfunctions sin(mu_k x) on [0, 1]
G = np.array([[quad(lambda x: np.sin(a * x) * np.sin(b * x), 0, 1)[0] for b in mus[:4]]
              for a in mus[:4]])
G = np.round(G, 12) + 0.0                       # tidy: no "-0."
print("Gram matrix of sin(mu_k x), k = 1..4:")
print(np.array2string(G, precision=6, suppress_small=True))

x = np.linspace(0, 1, 300)
plt.figure(figsize=(7, 3.6))
for k, mu in enumerate(mus[:4], start=1):
    plt.plot(x, np.sin(mu * x), label=f"k = {k}, lambda = {mu**2:.3f}")
plt.axhline(0, color="k", lw=0.6)
plt.xlabel("x")
plt.ylabel("y_k(x)")
plt.legend(fontsize=8, loc="lower left")
plt.tight_layout()
plt.savefig("ch03_robin_eigenfunctions.pdf", bbox_inches="tight")
plt.close()
