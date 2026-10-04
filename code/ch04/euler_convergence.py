# Convergence of Euler's method: error table and log-log plot
# Applied ODE & PDE with Python, Ch. 4 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt


def euler(f, t0, y0, T, N):
    """Return the Euler approximation of y(T) using N steps."""
    h = (T - t0) / N
    t, y = t0, y0
    for n in range(N):
        y = y + h * f(t, y)
        t = t + h
    return y


f = lambda t, y: y - t**2 + 1
exact = lambda t: (t + 1)**2 - 0.5 * np.exp(t)
T = 2.0

hs, errs = [], []
print("    N        h         |E(T)|      ratio    order")
for k in range(1, 9):
    N = 5 * 2**k
    h = T / N
    err = abs(euler(f, 0.0, 0.5, T, N) - exact(T))
    if errs:
        ratio = errs[-1] / err
        print(f"{N:5d}  {h:.3e}  {err:.4e}  {ratio:6.3f}  {np.log2(ratio):6.3f}")
    else:
        print(f"{N:5d}  {h:.3e}  {err:.4e}")
    hs.append(h)
    errs.append(err)

hs, errs = np.array(hs), np.array(errs)
p = np.polyfit(np.log(hs), np.log(errs), 1)[0]     # slope of the log-log line
print(f"least-squares slope of log(error) vs log(h): {p:.4f}")

plt.figure(figsize=(6, 4))
plt.loglog(hs, errs, "o-", label="Euler error at t = 2")
plt.loglog(hs, errs[0] * hs / hs[0], "k--", label="slope 1 reference")
plt.xlabel("step size h")
plt.ylabel("|y_N - y(2)|")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("ch04_euler_convergence.pdf", bbox_inches="tight")
plt.close()
