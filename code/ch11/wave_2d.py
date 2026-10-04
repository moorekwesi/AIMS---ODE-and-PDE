# Vibrating square membrane: leapfrog for u_tt = c^2 (u_xx + u_yy)
# Applied ODE & PDE with Python, Ch. 11 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

c = 1.0


def membrane(N, nu, T, u0, snaps=()):
    """Leapfrog with Courant number nu = c dt / h, u = 0 on the boundary, u_t(.,0) = 0."""
    h = 1.0 / N; x = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")
    nsteps = int(np.ceil(T / (nu * h / c))); dt = T / nsteps; r2 = (c * dt / h)**2

    def lap(V):                                  # h^2 * 5-point Laplacian (interior)
        return (V[2:, 1:-1] + V[:-2, 1:-1] + V[1:-1, 2:] + V[1:-1, :-2]
                - 4 * V[1:-1, 1:-1])

    Uold = u0(X, Y); U = Uold.copy()
    U[1:-1, 1:-1] += 0.5 * r2 * lap(Uold)        # starting step
    out = {}
    for n in range(1, nsteps):
        Unew = np.zeros_like(U)
        Unew[1:-1, 1:-1] = 2 * U[1:-1, 1:-1] - Uold[1:-1, 1:-1] + r2 * lap(U)
        Uold, U = U, Unew
        for ts in snaps:
            if abs((n + 1) * dt - ts) < dt / 2:
                out[ts] = U.copy()
    return X, Y, U, out


mode = lambda X, Y: np.sin(np.pi * X) * np.sin(2 * np.pi * Y)
w = c * np.pi * np.sqrt(5)                       # frequency of the (1,2) mode
T = 1.0
print("  N    nu     max error    rate")
prev = None
for N in (10, 20, 40, 80, 160):
    X, Y, U, _ = membrane(N, 0.5, T, mode)
    e = np.max(np.abs(U - np.cos(w * T) * mode(X, Y)))
    print(f"{N:4d}  0.50   {e:.3e}   " + ("  --" if prev is None else f"{np.log2(prev / e):5.2f}"))
    prev = e
for nu in (0.70, 0.71, 0.72):                   # CFL limit: nu <= 1/sqrt(2) = 0.7071
    X, Y, U, _ = membrane(40, nu, 4.0, mode)
    print(f"  N = 40, nu = {nu:.2f}: max|U| at t = 4 is {np.max(np.abs(U)):.3e}")

bump = lambda X, Y: np.exp(-200 * ((X - 0.35)**2 + (Y - 0.4)**2))   # a struck drum
times = (0.0, 0.15, 0.3, 0.6)
X, Y, U, snaps = membrane(160, 0.5, 0.6, bump, snaps=times[1:])
snaps[0.0] = bump(X, Y)
fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
for a, t in zip(ax, times):
    a.imshow(snaps[t].T, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r",
             vmin=-0.3, vmax=0.3)
    a.set_title(f"t = {t}"); a.set_xticks([0, 1]); a.set_yticks([0, 1])
plt.tight_layout(); plt.savefig("ch11_membrane.pdf", bbox_inches="tight")
plt.close()
