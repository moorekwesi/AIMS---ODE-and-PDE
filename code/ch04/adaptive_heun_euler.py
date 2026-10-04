# Adaptive step size control with the embedded Heun-Euler pair
# Applied ODE & PDE with Python, Ch. 4 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

a = 2500.0                                            # sharp peak of width ~0.02 at t = 1
exact = lambda t: 1.0 / (1.0 + a * (t - 1.0)**2)
f = lambda t, y: -2 * a * (t - 1.0) * exact(t)**2     # y' = f(t), y(0) = exact(0)


def heun_euler_adaptive(f, t0, y0, T, tol, h=0.1):
    """Heun (order 2) propagates, Euler (order 1) is used only to estimate the error."""
    t, y = t0, y0
    ts, ys, rejected = [t], [y], 0
    while t < T:
        h = min(h, T - t)
        k1 = f(t, y)
        k2 = f(t + h, y + h * k1)
        err = abs(0.5 * h * (k2 - k1))          # |Heun - Euler| = local error estimate
        if err <= tol:                          # accept the step
            t, y = t + h, y + 0.5 * h * (k1 + k2)
            ts.append(t)
            ys.append(y)
        else:
            rejected += 1
        h *= min(5.0, max(0.2, 0.9 * np.sqrt(tol / max(err, 1e-16))))   # new step
    return np.array(ts), np.array(ys), rejected


print("   tol      accepted  rejected   max error")
for tol in [1e-3, 1e-4, 1e-5, 1e-6]:
    ts, ys, rej = heun_euler_adaptive(f, 0.0, exact(0.0), 2.0, tol)
    print(f"{tol:8.0e}   {len(ts) - 1:6d}    {rej:5d}     {np.max(abs(ys - exact(ts))):.2e}")

ts, ys, rej = heun_euler_adaptive(f, 0.0, exact(0.0), 2.0, 1e-4)
N = len(ts) - 1                                 # same number of steps, but uniform
h, y, tu = 2.0 / N, exact(0.0), np.linspace(0, 2, N + 1)
yu = [y]
for n in range(N):
    k1 = f(tu[n], y)
    y = y + 0.5 * h * (k1 + f(tu[n] + h, y + h * k1))
    yu.append(y)
print(f"uniform Heun with the same {N} steps: max error {np.max(abs(np.array(yu) - exact(tu))):.2e}")

fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
tt = np.linspace(0, 2, 2000)
ax[0].plot(tt, exact(tt), "k", label="exact")
ax[0].plot(ts, ys, "o", ms=2.5, label="adaptive Heun-Euler, tol = 1e-4")
ax[0].set_ylabel("y")
ax[0].legend()
ax[1].semilogy(ts[1:], np.diff(ts), ".-")
ax[1].set_xlabel("t")
ax[1].set_ylabel("accepted step h")
plt.tight_layout()
plt.savefig("ch04_adaptive_steps.pdf", bbox_inches="tight")
plt.close()
