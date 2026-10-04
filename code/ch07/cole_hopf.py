# Viscous Burgers equation by the Cole-Hopf formula and the vanishing viscosity limit
# Applied ODE & PDE with Python, Ch. 7 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

y = np.linspace(-5, 5, 10001)                    # quadrature grid for the y-integrals


def cole_hopf(x, t, nu, U0):
    """u(x,t) = int (x-y)/t e^{-G/(2nu)} dy / int e^{-G/(2nu)} dy,
    with G(y) = U0(y) + (x-y)^2/(2t) and U0 an antiderivative of u0."""
    u = np.empty_like(x)
    for k in range(0, len(x), 100):              # 100 x-values at a time
        X = x[k:k + 100, None]
        G = U0(y) + (X - y)**2 / (2 * t)
        w = np.exp(-(G - G.min(axis=1, keepdims=True)) / (2 * nu))  # avoids overflow
        u[k:k + 100] = np.trapz((X - y) / t * w, y, axis=1) / np.trapz(w, y, axis=1)
    return u


U0_riemann = lambda s: np.minimum(s, 0.0)        # u0 = 1 (x<0), 0 (x>0)
U0_gauss = lambda s: 0.5 * np.sqrt(np.pi) * erf(s)   # u0 = exp(-x^2)

t = 1.0
x = np.linspace(-1, 2, 751)
shock = np.where(x < 0.5 * t, 1.0, 0.0)          # inviscid entropy solution, s = 1/2
print("    nu      L1 distance to shock   L1/nu    max|u - tanh wave|")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for nu in [0.1, 0.05, 0.025, 0.0125]:
    u = cole_hopf(x, t, nu, U0_riemann)
    L1 = np.trapz(np.abs(u - shock), x)
    tw = 0.5 - 0.5 * np.tanh((x - 0.5 * t) / (4 * nu))     # travelling wave
    print(f"{nu:8.4f}  {L1:18.6f}  {L1 / nu:9.4f}  {np.abs(u - tw).max():14.2e}")
    ax1.plot(x, u, label=f"nu = {nu}")
ax1.plot(x, shock, "k--", label="inviscid shock")
ax1.set_xlabel("x"); ax1.set_ylabel("u"); ax1.legend(); ax1.set_title("Riemann data, t = 1")

xg = np.linspace(-2, 4, 301)
for nu in [0.2, 0.05, 0.01]:
    ax2.plot(xg, cole_hopf(xg, 3.0, nu, U0_gauss), label=f"nu = {nu}")
ax2.plot(xg, np.exp(-xg**2), "k:", label="u0")
ax2.set_xlabel("x"); ax2.set_ylabel("u"); ax2.legend()
ax2.set_title("Gaussian data, t = 3 (after breaking)")
plt.tight_layout()
plt.savefig("ch07_cole_hopf.pdf", bbox_inches="tight")
