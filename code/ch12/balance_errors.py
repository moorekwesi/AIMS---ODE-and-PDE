# Balancing time and space errors: FTCS, BTCS and Crank-Nicolson
# Applied ODE & PDE with Python, Ch. 12 | (c) 2026 Stephen E. Moore | MIT Licence
import time
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt

Tend = 0.1
exact = lambda x, t: (np.exp(-np.pi**2 * t) * np.sin(np.pi * x)
                      + 0.5 * np.exp(-9 * np.pi**2 * t) * np.sin(3 * np.pi * x))


def theta_method(N, dt, theta):
    """theta = 0: FTCS, 1: BTCS, 1/2: Crank-Nicolson, for u_t = u_xx on (0,1)."""
    h = 1.0 / N
    x = np.linspace(0, 1, N + 1)[1:-1]
    A = sps.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(N - 1, N - 1),
                  format="csc") / h**2
    I = sps.identity(N - 1, format="csc")
    nsteps = int(round(Tend / dt))
    U = exact(x, 0.0)
    B = (I + (1 - theta) * dt * A).tocsr()
    lu = splu((I - theta * dt * A).tocsc()) if theta > 0 else None
    for _ in range(nsteps):
        U = B @ U if lu is None else lu.solve(B @ U)
    return np.max(np.abs(U - exact(x, Tend))), nsteps


# (a) temporal order of Crank-Nicolson: h tiny, dt halved
print("(a) Crank-Nicolson, N = 2000, dt -> dt/2")
prev = None
for dt in Tend / 2.0 ** np.arange(1, 7):
    e, _ = theta_method(2000, dt, 0.5)
    print(f"   dt = {dt:.5f}  error = {e:.3e}" +
          (f"  p = {np.log2(prev / e):.2f}" if prev else ""))
    prev = e

# (b) refine h and dt together, as each scheme requires
cfgs = {"FTCS 0.4h^2": (0.0, lambda h: 0.4 * h**2),
        "BTCS h": (1.0, lambda h: h),
        "BTCS h^2": (1.0, lambda h: h**2),
        "CN h": (0.5, lambda h: h),
        "CN h/8": (0.5, lambda h: h / 8)}
print("(b) max error when h and dt (column heading) are refined together")
print(f"{'N':>5}" + "".join(f"{name:>13}" for name in cfgs))
res, steps = {name: [] for name in cfgs}, {}
for N in [10, 20, 40, 80, 160, 320]:
    row = f"{N:5d}"
    for name, (th, dtf) in cfgs.items():
        t0 = time.perf_counter()
        e, n = theta_method(N, Tend / round(Tend / dtf(1 / N)), th)
        res[name].append((time.perf_counter() - t0, e))
        row += f"{e:13.2e}"
        steps[name] = n
    print(row)
print("steps at N=320: " + ", ".join(f"{v}" for v in steps.values()))

plt.figure(figsize=(7, 4))
for name, m in zip(cfgs, "o^vsd"):
    tt, ee = np.array(res[name]).T
    plt.loglog(tt, ee, m + "-", label=name)
plt.xlabel("CPU time (s)"); plt.ylabel("max error at t = 0.1")
plt.legend(); plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("ch12_efficiency.pdf", bbox_inches="tight")
plt.close()
