# Upwind scheme for u_t + (1 + x^2) u_x = 0 with an inflow boundary
# Applied ODE & PDE with Python, Ch. 11 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

L, T = 2.0, 1.0
a = lambda x: 1 + x**2                          # speed a(x) > 0: inflow at x = 0
u0 = lambda x: 1 / (1 + x**2)
exact = lambda x, t: u0(np.tan(np.arctan(x) - t))   # constant along characteristics
inflow = lambda t: np.cos(t)**2                 # = exact(0, t)


def upwind(N, nu_max=0.9, T=T):
    h = L / N; x = np.linspace(0, L, N + 1); A = a(x)
    dt = nu_max * h / A.max(); nsteps = int(np.ceil(T / dt)); dt = T / nsteps
    U = u0(x)
    for n in range(nsteps):
        U[1:] = U[1:] - dt / h * A[1:] * (U[1:] - U[:-1])
        U[0] = inflow((n + 1) * dt)             # no condition needed at x = L
    return x, U


print("    N    max error    rate")
prev = None
for N in (50, 100, 200, 400, 800):
    x, U = upwind(N)
    e = np.max(np.abs(U - exact(x, T)))
    print(f"{N:5d}   {e:.3e}   " + ("  --" if prev is None else f"{np.log2(prev / e):5.2f}"))
    prev = e

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
t = np.linspace(0, T, 200)
for x0 in np.linspace(0, 1.5, 7):               # characteristics x = tan(t + arctan x0)
    xc = np.tan(t + np.arctan(x0)); ok = (t + np.arctan(x0) < np.pi / 2) & (xc <= L)
    ax[0].plot(xc[ok], t[ok], "b-", lw=0.8)
for t0 in np.linspace(0.1, 1, 6):               # characteristics entering at x = 0
    tt = np.linspace(t0, T, 100); ax[0].plot(np.tan(tt - t0), tt, "r-", lw=0.8)
ax[0].set_xlim(0, L); ax[0].set_xlabel("x"); ax[0].set_ylabel("t")
ax[0].set_title("characteristics (red: from the inflow boundary)")
xf = np.linspace(0, L, 300)
for tt in (0.0, 0.5, 1.0):
    x, U = upwind(40, T=tt) if tt > 0 else (np.linspace(0, L, 41), u0(np.linspace(0, L, 41)))
    l, = ax[1].plot(xf, exact(xf, tt), label=f"t = {tt}")
    ax[1].plot(x, U, "o", ms=3, color=l.get_color())
ax[1].set_xlabel("x"); ax[1].legend(); ax[1].set_title("exact (lines), upwind N = 40 (dots)")
plt.tight_layout(); plt.savefig("ch11_variable_speed.pdf", bbox_inches="tight")
plt.close()
