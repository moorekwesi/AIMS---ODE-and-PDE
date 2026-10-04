# Sample project: experimental stability limit and cell Peclet number
# Applied ODE & PDE with Python, Ch. 13 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt


def ftcs(N, dt, eps, T):
    """Explicit scheme for u_t = eps u_xx + u_x, u = 0 at x = 0, 1, u(x,0) = x(1-x)."""
    h = 1.0 / N
    x = np.linspace(0, 1, N + 1)
    lam, mu = eps * dt / h**2, dt / (2 * h)
    U = x * (1 - x)
    hist = [np.max(np.abs(U))]                     # history of max|U^n|
    for n in range(int(round(T / dt))):
        U[1:-1] = U[1:-1] + lam * (U[2:] - 2 * U[1:-1] + U[:-2]) + mu * (U[2:] - U[:-2])
        hist.append(np.max(np.abs(U)))
    return x, U, np.array(hist)


eps, N, T = 0.1, 32, 1.0
h = 1.0 / N
print(f"Experiment 1: eps = {eps}, h = 1/{N}, T = {T}; theory: dt <= h^2/(2 eps) = "
      f"{h**2 / (2 * eps):.3e}")
print("   r = eps dt/h^2      dt       steps    max|U(T)|")
results = {}
for r in [0.25, 0.45, 0.50, 0.52, 0.55]:
    dt = T / np.ceil(T / (r * h**2 / eps))        # T/dt must be an integer
    x, U, hist = ftcs(N, dt, eps, T)
    results[r] = (dt * np.arange(hist.size), hist)
    print(f"      {eps * dt / h**2:.4f}      {dt:.3e}   {int(round(T / dt)):5d}    "
          f"{np.max(np.abs(U)):.3e}")

print("Experiment 2: eps = 0.01, r = 0.4, T = 0.25 (exact solution is >= 0)")
print("   h       cell Peclet h/(2 eps)    min U      max U")
small = {}
for N2 in [16, 32, 64, 128]:
    h2 = 1.0 / N2
    dt = 0.25 / np.ceil(0.25 / (0.4 * h2**2 / 0.01))
    x, U, hist = ftcs(N2, dt, 0.01, 0.25)
    small[N2] = (x, U)
    print(f" 1/{N2:<4d}        {h2 / 0.02:5.2f}            {U.min():+.4f}   {U.max():.4f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for r, (tn, hist) in results.items():
    ax[0].semilogy(tn, hist, label=f"r = {r}")
ax[0].set_title("eps = 0.1, h = 1/32")
ax[0].set_xlabel("t")
ax[0].set_ylabel("max |U^n|")
ax[0].legend()
for N2 in [16, 128]:
    ax[1].plot(*small[N2], "o-" if N2 == 16 else "-", ms=3, label=f"h = 1/{N2}")
ax[1].set_title("eps = 0.01, t = 0.25")
ax[1].set_xlabel("x")
ax[1].legend()
plt.tight_layout()
plt.savefig("ch13_cd_stability.pdf", bbox_inches="tight")
plt.close()
