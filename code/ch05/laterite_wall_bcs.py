# Laterite wall: Dirichlet versus Neumann
# Applied ODE & PDE with Python, Ch. 5 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

L, kappa = 0.30, 5.0e-7             # wall thickness (m), diffusivity (m^2/s)
x = np.linspace(0, L, 601)
s = x/L
u0 = 25 + 80*s**2*(1 - s)          # initial temperature (deg C) after a hot day
nmodes = 80


def series(t, kind):
    """Fourier series solution of u_t = kappa u_xx at time t (seconds)."""
    n = np.arange(1, nmodes + 1)[:, None]
    decay = np.exp(-kappa*(n*np.pi/L)**2*t)
    if kind == "Dirichlet":          # u(0,t) = u(L,t) = 25
        modes = np.sin(n*np.pi*x/L)
        b = 2/L*np.trapz((u0 - 25)*modes, x, axis=1)[:, None]
        return 25 + np.sum(b*decay*modes, axis=0)
    modes = np.cos(n*np.pi*x/L)      # Neumann: u_x(0,t) = u_x(L,t) = 0
    a0 = np.trapz(u0, x)/L
    a = 2/L*np.trapz(u0*modes, x, axis=1)[:, None]
    return a0 + np.sum(a*decay*modes, axis=0)


hours = [0, 1, 3, 6, 12, 24]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
print(" t (h)   mean_D   max_D    mean_N   max_N")
for h in hours:
    uD, uN = series(3600*h, "Dirichlet"), series(3600*h, "Neumann")
    mD, mN = np.trapz(uD, x)/L, np.trapz(uN, x)/L
    print(f"{h:5d}  {mD:7.3f}  {uD.max():7.3f}  {mN:7.3f}  {uN.max():7.3f}")
    axes[0].plot(100*x, uD, label=f"t = {h} h")
    axes[1].plot(100*x, uN, label=f"t = {h} h")
for ax, title in zip(axes, ["Dirichlet: faces at 25 C", "Neumann: insulated faces"]):
    ax.set_title(title); ax.set_xlabel("x (cm)")
axes[0].set_ylabel("temperature (C)"); axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig("ch05_laterite_wall.pdf", bbox_inches="tight")
