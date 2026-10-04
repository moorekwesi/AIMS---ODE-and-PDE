# Fisher-KPP travelling waves with an IMEX (semi-implicit) scheme
# Applied ODE & PDE with Python, Ch. 10 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt

L, h, dt, T = 200.0, 0.1, 0.05, 80.0
N = int(round(L / h)); x = np.linspace(0, L, N + 1)


def neumann_laplacian(N, h):
    """Second difference with zero-flux (ghost point) conditions at both ends."""
    main = -2 * np.ones(N + 1)
    up = np.ones(N); lo = np.ones(N); up[0] = 2; lo[-1] = 2
    return sps.diags([lo, main, up], [-1, 0, 1], format="csc") / h**2


def fisher(D, rho, snap_times=()):
    """IMEX Euler: (I - dt D A) U^{n+1} = U^n + dt rho U^n (1 - U^n)."""
    lu = splu((sps.identity(N + 1) - dt * D * neumann_laplacian(N, h)).tocsc())
    U = np.where(x < 10, 1.0, 0.0)                 # a population confined to x < 10
    times, fronts, snaps = [], [], {}
    for n in range(1, int(round(T / dt)) + 1):
        U = lu.solve(U + dt * rho * U * (1 - U))
        t = n * dt
        if n % 20 == 0:                            # front = where U crosses 1/2
            j = np.argmax(U < 0.5)
            times.append(t)
            fronts.append(x[j - 1] + h * (U[j - 1] - 0.5) / (U[j - 1] - U[j]))
        for ts in snap_times:
            if abs(t - ts) < dt / 2:
                snaps[ts] = U.copy()
    times, fronts = np.array(times), np.array(fronts)
    late = times >= T / 2                          # fit the speed on the second half
    speed = np.polyfit(times[late], fronts[late], 1)[0]
    return speed, snaps


print("   D     rho   2 sqrt(D rho)   Bramson-corrected   measured speed")
for D, rho in ((1.0, 1.0), (0.25, 1.0), (1.0, 0.25), (2.0, 0.5)):
    c, _ = fisher(D, rho)
    cstar, lam = 2 * np.sqrt(D * rho), np.sqrt(rho / D)
    corr = cstar - 1.5 / lam * np.log(2) / (T / 2)   # mean of 3/(2 lam t), t in [T/2, T]
    print(f"{D:5.2f}  {rho:5.2f}     {cstar:6.3f}          {corr:6.3f}          {c:6.3f}")

_, snaps = fisher(1.0, 1.0, snap_times=(10, 30, 50, 70))
plt.figure(figsize=(7, 3.5))
for ts, U in snaps.items():
    plt.plot(x, U, label=f"t = {ts}")
plt.xlim(0, 160); plt.xlabel("x"); plt.ylabel("u(x,t)"); plt.legend()
plt.title("Fisher-KPP front, D = r = 1 (speed close to 2)")
plt.tight_layout(); plt.savefig("ch10_fisher_kpp.pdf", bbox_inches="tight")
plt.close()
