# The damped wave equation u_tt + beta u_t = c^2 u_xx (AIMS Senegal 2024 project)
# Applied ODE & PDE with Python, Ch. 11 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
from scipy.fft import dst
import matplotlib.pyplot as plt

beta, c, T, nu = 1.0, 1.0, 0.6, 0.8
f1 = lambda x: np.exp(-(x - 0.7)**2)                       # the project's data
f2 = lambda x: f1(x) - (f1(0) * (1 - x) + f1(1) * x)       # compatible: f2(0)=f2(1)=0


def series(f, K=4000, M=2**16):
    """Exact Fourier sine series solution (g = 0), coefficients by a DST."""
    xm = np.arange(1, M) / M
    b = dst(f(xm), type=1)[:K] / M                        # b_k = 2 int f sin(k pi x)
    k = np.arange(1, K + 1); w = np.sqrt((c * k * np.pi)**2 - beta**2 / 4)
    def u(x, t):
        amp = b * np.exp(-beta * t / 2) * (np.cos(w * t) + beta / (2 * w) * np.sin(w * t))
        return amp @ np.sin(np.outer(k, np.pi * x))
    return u


def damped_leapfrog(f, N):
    h = 1.0 / N; x = np.linspace(0, 1, N + 1)
    nsteps = int(np.ceil(T / (nu * h / c))); dt = T / nsteps; r2 = (c * dt / h)**2
    d2 = lambda V: V[2:] - 2 * V[1:-1] + V[:-2]
    Uold = f(x); Uold[[0, -1]] = 0.0                       # impose u = 0 at the ends
    U = Uold.copy(); U[1:-1] += 0.5 * r2 * d2(Uold)        # starting step (g = 0)
    for n in range(1, nsteps):
        Unew = np.zeros_like(U)
        Unew[1:-1] = (2 * U[1:-1] - (1 - beta * dt / 2) * Uold[1:-1]
                      + r2 * d2(U)) / (1 + beta * dt / 2)
        Uold, U = U, Unew
    return x, U


ex = {name: series(f) for name, f in (("f1", f1), ("f2", f2))}
print("  n      h      L2 error (f1)  rate  |  L2 error (f2)  rate")
prev = None
for n in range(1, 8):
    N = 2**n; errs = []
    for name, f in (("f1", f1), ("f2", f2)):
        x, U = damped_leapfrog(f, N)
        errs.append(np.sqrt(np.sum((U - ex[name](x, T))**2) / N))
    rates = ["  -- " if prev is None else f"{np.log2(p / e):5.2f}" for p, e in zip(prev or errs, errs)]
    print(f"{n:3d}  {1 / N:.5f}    {errs[0]:.3e}    {rates[0]} |   {errs[1]:.3e}    {rates[1]}")
    prev = errs

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
xf = np.linspace(0, 1, 400)
for a, (name, f) in zip(ax, (("f1", f1), ("f2", f2))):
    for t in (0.0, 0.25, 0.5, 1.0, 2.0):
        a.plot(xf, ex[name](xf, t), label=f"t = {t}")
    a.set_xlabel("x"); a.legend(fontsize=8)
ax[0].set_title("f(x) = exp(-(x-0.7)^2), u = 0 imposed at the ends")
ax[1].set_title("compatible data f2 = f - linear interpolant")
plt.tight_layout(); plt.savefig("ch11_damped_wave.pdf", bbox_inches="tight")
plt.close()
