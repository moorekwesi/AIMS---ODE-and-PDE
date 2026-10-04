# The leapfrog scheme for the plucked string u_tt = c^2 u_xx
# Applied ODE & PDE with Python, Ch. 11 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

c, N = 1.0, 100
h = 1.0 / N
x = np.linspace(0, 1, N + 1)
pluck = lambda s: np.where(s < 0.3, s / 0.3, (1 - s) / 0.7)    # plucked at x = 0.3


def F(s):
    """Odd, 2-periodic extension of the initial shape."""
    s = np.mod(s, 2.0)
    return np.where(s <= 1, pluck(s), -pluck(2 - s))


exact = lambda x, t: 0.5 * (F(x - c * t) + F(x + c * t))      # d'Alembert, u_t(x,0) = 0


def leapfrog(nu, T, energy=False):
    dt = nu * h / c; nsteps = int(round(T / dt))
    Uold = pluck(x)
    U = Uold.copy()                             # starting step with g = u_t(x,0) = 0
    U[1:-1] = Uold[1:-1] + 0.5 * nu**2 * (Uold[2:] - 2 * Uold[1:-1] + Uold[:-2])
    E = []
    for n in range(1, nsteps):
        Unew = np.zeros_like(U)                 # u = 0 at both ends
        Unew[1:-1] = (2 * U[1:-1] - Uold[1:-1]
                      + nu**2 * (U[2:] - 2 * U[1:-1] + U[:-2]))
        if energy:                              # discrete energy at t^{n+1/2}
            kin = 0.5 * h * np.sum(((Unew - U) / dt)**2)
            pot = 0.5 * c**2 * h * np.sum(np.diff(Unew) * np.diff(U)) / h**2
            E.append(kin + pot)
        Uold, U = U, Unew
    return U, nsteps * dt, np.array(E)


print("  nu     T    max error    max|U|")
for nu in (1.0, 0.9, 0.5, 1.02):
    for T in (0.5, 2.0):
        U, t, _ = leapfrog(nu, T)
        print(f"{nu:5.2f}  {t:4.1f}   {np.max(abs(U - exact(x, t))):.3e}   {np.max(abs(U)):.3e}")
_, _, E = leapfrog(0.9, 2.0, energy=True)
print(f"nu = 0.9: discrete energy min = {E.min():.12f}, max = {E.max():.12f}")
print(f"exact energy (1/2) int f'(x)^2 dx = {0.5 * (1 / 0.3 + 1 / 0.7):.12f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
xf = np.linspace(0, 1, 801)
ax[0].plot(xf, exact(xf, 0.9), "k-", label="exact")
for nu in (1.0, 0.5):
    U, t, _ = leapfrog(nu, 0.9)
    ax[0].plot(x, U, ".-", ms=3, lw=0.8, label=f"leapfrog, nu = {nu}")
    if nu < 1:
        ax[1].plot(x, U - exact(x, t), ".-", ms=3, lw=0.8, label=f"nu = {nu}")
U, t, _ = leapfrog(0.9, 0.9)
ax[1].plot(x, U - exact(x, t), ".-", ms=3, lw=0.8, label="nu = 0.9")
ax[0].set_xlim(0.4, 0.8); ax[0].set_ylim(-0.95, -0.4); ax[0].set_title("zoom at t = 0.9")
ax[1].set_title("error U - u at t = 0.9")
for a in ax:
    a.set_xlabel("x"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("ch11_wave_leapfrog.pdf", bbox_inches="tight")
plt.close()