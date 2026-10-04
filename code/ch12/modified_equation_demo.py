# Numerical diffusion and dispersion: schemes versus their modified equations
# Applied ODE & PDE with Python, Ch. 12 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

c, N, nu, Tend = 1.0, 200, 0.5, 1.0          # one period on [0, 1)
h = 1.0 / N
dt = nu * h / c
x = np.arange(N) * h
u0 = np.exp(-((x - 0.5) / 0.03) ** 2)         # Gaussian pulse (6 points wide)
k = 2 * np.pi * np.fft.fftfreq(N, d=h)        # wave numbers on the grid


def evolve_pde(symbol):
    """Exact solution of v_t = P(d/dx) v on the periodic grid via FFT,
    where symbol(k) = P(ik)."""
    return np.real(np.fft.ifft(np.fft.fft(u0) * np.exp(symbol(k) * Tend)))


def run(scheme):
    U = u0.copy()
    for _ in range(int(round(Tend / dt))):
        Up, Um = np.roll(U, -1), np.roll(U, 1)   # U_{j+1}, U_{j-1} (periodic)
        if scheme == "upwind":
            U = U - nu * (U - Um)
        else:                                     # Lax-Wendroff
            U = U - nu / 2 * (Up - Um) + nu**2 / 2 * (Up - 2 * U + Um)
    return U


ik = lambda k: 1j * k
a2 = c * h / 2 * (1 - nu)                     # upwind: numerical diffusion
a3 = -c * h**2 / 6 * (1 - nu**2)              # Lax-Wendroff: dispersion
a4 = -c * h**3 / 8 * nu * (1 - nu**2)         # Lax-Wendroff: weak damping
models = {"upwind": lambda k: -c * ik(k) + a2 * ik(k)**2,
          "Lax-Wendroff": lambda k: -c * ik(k) + a3 * ik(k)**3 + a4 * ik(k)**4}
exact = u0                                    # after one period the pulse returns
print(f"N = {N}, nu = {nu}, t = {Tend}")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for a, (name, P) in zip(ax, models.items()):
    U, V = run(name), evolve_pde(P)
    print(f"{name:13s} max|U - u| = {np.max(np.abs(U - exact)):.3e},"
          f"  max|U - v_modified| = {np.max(np.abs(U - V)):.3e}")
    a.plot(x, exact, "k-", lw=1, label="exact u")
    a.plot(x, U, "o", ms=2.5, label=name)
    a.plot(x, V, "-", lw=1.2, label="modified equation")
    a.set_xlim(0.25, 0.75); a.set_xlabel("x"); a.legend(fontsize=8)
print(f"upwind diffusion coefficient a2 = {a2:.3e}")
plt.tight_layout()
plt.savefig("ch12_modified_equation.pdf", bbox_inches="tight")
plt.close()
