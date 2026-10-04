# Explicit Runge-Kutta methods from Butcher tableaux and empirical orders
# Applied ODE & PDE with Python, Ch. 4 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

TABLEAUX = {   # name: (A, b, c)
    "Euler":    ([[0]], [1], [0]),
    "Heun":     ([[0, 0], [1, 0]], [1/2, 1/2], [0, 1]),
    "Kutta3":   ([[0, 0, 0], [1/2, 0, 0], [-1, 2, 0]], [1/6, 2/3, 1/6], [0, 1/2, 1]),
    "RK4":      ([[0, 0, 0, 0], [1/2, 0, 0, 0], [0, 1/2, 0, 0], [0, 0, 1, 0]],
                 [1/6, 1/3, 1/3, 1/6], [0, 1/2, 1/2, 1]),
}


def rk_solve(f, t0, y0, T, N, A, b, c):
    """Explicit RK method with tableau (A, b, c); y may be a scalar or a vector."""
    A, b, c = np.array(A, float), np.array(b, float), np.array(c, float)
    s, h = len(b), (T - t0) / N
    t, y = t0, np.atleast_1d(np.array(y0, float))
    for n in range(N):
        K = np.zeros((s, y.size))
        for i in range(s):
            K[i] = f(t + c[i] * h, y + h * A[i, :i] @ K[:i])
        y = y + h * b @ K
        t = t + h
    return y


f = lambda t, y: y * np.cos(t)          # y' = y cos t, y(0) = 1
exact = np.exp(np.sin(5.0))             # y(5) = exp(sin 5)
Ns = 10 * 2**np.arange(7)
plt.figure(figsize=(6, 4))
print(" method   error(N=10)   error(N=640)   observed order")
for name, (A, b, c) in TABLEAUX.items():
    err = np.array([abs(rk_solve(f, 0, 1.0, 5.0, N, A, b, c)[0] - exact) for N in Ns])
    order = np.log2(err[-2] / err[-1])
    print(f"{name:8s}  {err[0]:.3e}     {err[-1]:.3e}      {order:6.3f}")
    plt.loglog(5.0 / Ns, err, "o-", label=name)
plt.xlabel("step size h")
plt.ylabel("error at t = 5")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("ch04_rk_orders.pdf", bbox_inches="tight")
plt.close()
