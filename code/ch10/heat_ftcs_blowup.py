# FTCS just below and just above the stability limit r = 1/2
# Applied ODE & PDE with Python, Ch. 10 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

N = 20
h = 1.0 / N
x = np.linspace(0.0, 1.0, N + 1)
u0 = 1.0 - np.abs(2 * x - 1)          # hat function, kink at x = 1/2


def ftcs_history(r, nsteps, every):
    """Run FTCS with mesh ratio r; return max|U| every `every` steps and final U."""
    U = u0.copy()
    peaks = [np.max(np.abs(U))]
    for n in range(1, nsteps + 1):
        U[1:-1] = U[1:-1] + r * (U[2:] - 2 * U[1:-1] + U[:-2])
        if n % every == 0:
            peaks.append(np.max(np.abs(U)))
    return np.array(peaks), U


nsteps, every = 400, 50
res = {r: ftcs_history(r, nsteps, every) for r in (0.48, 0.52)}
print(" step    max|U| (r=0.48)   max|U| (r=0.52)")
for k in range(len(res[0.48][0])):
    print(f"{k * every:5d}    {res[0.48][0][k]:.6e}     {res[0.52][0][k]:.6e}")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for a, r in zip(ax, (0.48, 0.52)):
    _, U = ftcs_history(r, 160, 1)
    a.plot(x, u0, "k--", lw=0.8, label="u(x,0)")
    a.plot(x, U, "o-", ms=3, label="FTCS, 160 steps")
    a.set_title(f"r = {r}"); a.set_xlabel("x"); a.legend()
plt.tight_layout(); plt.savefig("ch10_ftcs_blowup.pdf", bbox_inches="tight")
plt.close()
