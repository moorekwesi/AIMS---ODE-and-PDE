# Convergence of difference quotients for f(x) = cos(x)
# Applied ODE & PDE with Python, Ch. 9 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt


def quotients(f, x, h):
    """Return forward, backward, central first differences and the
    second difference of f at the points x with step h."""
    fp, f0, fm = f(x + h), f(x), f(x - h)
    return {"forward": (fp - f0) / h,
            "backward": (f0 - fm) / h,
            "central": (fp - fm) / (2 * h),
            "second": (fp - 2 * f0 + fm) / h**2}


x = np.linspace(0, 2 * np.pi, 201)          # points where we differentiate
exact = {"forward": -np.sin(x), "backward": -np.sin(x),
         "central": -np.sin(x), "second": -np.cos(x)}
hs = 2.0 ** -np.arange(1, 11)               # h = 1/2, 1/4, ..., 1/1024
names = list(exact)
err = {k: [] for k in names}
for h in hs:
    D = quotients(np.cos, x, h)
    for k in names:
        err[k].append(np.max(np.abs(D[k] - exact[k])))   # max error on grid

print(f"{'h':>9} " + " ".join(f"{k:>10} {'p':>5}" for k in names))
for i, h in enumerate(hs):
    row = f"{h:9.2e} "
    for k in names:
        e = err[k]
        p = np.log2(e[i - 1] / e[i]) if i > 0 else np.nan
        row += f"{e[i]:10.3e} {p:5.2f} "
    print(row)

plt.figure(figsize=(7, 4.2))
for k, m in zip(names, "ov^s"):
    plt.loglog(hs, err[k], m + "-", label=k)
plt.loglog(hs, 0.5 * hs, "k--", lw=0.8, label="slope 1")
plt.loglog(hs, 0.1 * hs**2, "k:", lw=1.0, label="slope 2")
plt.xlabel("step size h")
plt.ylabel("max error")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("ch09_fd_convergence.pdf", bbox_inches="tight")
plt.close()
