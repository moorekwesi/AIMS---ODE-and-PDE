# Hadamard's example of an ill-posed problem
# Applied ODE & PDE with Python, Ch. 5 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

# Part 1: exact solutions u_n = sinh(n y) sin(n x)/n^2 with data u_y(x,0) = sin(nx)/n
print("  n   sup|data|    sup|u(.,1)|")
for n in [1, 5, 10, 20, 40]:
    print(f"{n:3d}   {1/n:.3e}   {np.sinh(n)/n**2:.3e}")


# Part 2: march u_xx + u_yy = 0 in y, starting from u = 0, u_y = sin x
def march(N, ymax, noise, seed=1):
    """Explicit marching in y on a periodic grid with k = h.  Returns the
    y values and the max error against the exact solution sinh(y) sin(x)."""
    h = 2*np.pi/N
    k = h
    x = h*np.arange(N)
    rng = np.random.default_rng(seed)
    g = np.sin(x) + noise*rng.standard_normal(N)    # slightly noisy data
    u_old, u = np.zeros(N), k*g + k**3/6*(-np.sin(x))  # Taylor start
    ys, errs = [k], [np.max(np.abs(u - np.sinh(k)*np.sin(x)))]
    for j in range(1, int(round(ymax/k))):
        lap_x = (np.roll(u, -1) - 2*u + np.roll(u, 1))/h**2
        u_old, u = u, 2*u - u_old - k**2*lap_x      # u_yy = -u_xx
        y = (j + 1)*k
        ys.append(y)
        errs.append(np.max(np.abs(u - np.sinh(y)*np.sin(x))))
    return np.array(ys), np.array(errs)


fig, ax = plt.subplots(figsize=(7, 4))
print("  N   error at y=0.5   y=1.0       y=1.5")
for N in [32, 64, 128]:
    ys, errs = march(N, 1.5, noise=1e-10)
    pick = [errs[np.argmin(abs(ys - yy))] for yy in (0.5, 1.0, 1.5)]
    print(f"{N:4d}   {pick[0]:.3e}      {pick[1]:.3e}   {pick[2]:.3e}")
    ax.semilogy(ys, errs, label=f"N = {N}")
ax.set_xlabel("y"); ax.set_ylabel("max error")
ax.set_title("Marching Laplace's equation from noisy Cauchy data (noise 1e-10)")
ax.legend(); plt.tight_layout()
plt.savefig("ch05_hadamard.pdf", bbox_inches="tight")
