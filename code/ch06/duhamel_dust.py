# Duhamel's principle in 2D: a Harmattan dust plume
# Applied ODE & PDE with Python, Ch. 6 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

b = np.array([-400.0, -200.0])     # wind (km/day), blowing towards the south-west
k = 0.3                            # deposition rate (1/day)
Q, sig = 1.0, 100.0                # source strength and radius (km)


def f(x, y):
    """Dust emission rate, centred on the source at the origin."""
    return Q*np.exp(-(x**2 + y**2)/sig**2)


def u(x, y, t, ns=400):
    """Duhamel: u = int_0^t exp(-k(t-s)) f(x - (t-s) b) ds   (g = 0)."""
    s = np.linspace(0, t, ns + 1)
    lag = (t - s)[:, None, None] if np.ndim(x) else (t - s)
    integrand = np.exp(-k*lag)*f(x - lag*b[0], y - lag*b[1])
    return np.trapz(integrand, s, axis=0)


city = (-1000.0, -500.0)           # about 1100 km downwind of the source
print(" t (days)   u at source   u at city")
for t in [0.5, 1, 2, 3, 4, 6]:
    print(f"{t:8.1f}   {u(0.0, 0.0, t):10.4f}   {u(*city, t):10.4f}")

# check that the formula satisfies the PDE (central differences)
x0, y0, t0, h = -600.0, -250.0, 3.0, 1e-3
res = ((u(x0, y0, t0 + h) - u(x0, y0, t0 - h))/(2*h)
       + b[0]*(u(x0 + h, y0, t0) - u(x0 - h, y0, t0))/(2*h)
       + b[1]*(u(x0, y0 + h, t0) - u(x0, y0 - h, t0))/(2*h)
       + k*u(x0, y0, t0) - f(x0, y0))
print(f"PDE residual at (x,y,t) = ({x0}, {y0}, {t0}): {res:.1e}")

X, Y = np.meshgrid(np.linspace(-2000, 300, 231), np.linspace(-1100, 300, 141))
fig, ax = plt.subplots(figsize=(7, 4.3))
cs = ax.contourf(X, Y, u(X, Y, 4.0), levels=15, cmap="YlOrBr")
fig.colorbar(cs, ax=ax, label="dust concentration")
ax.plot(0, 0, "k^", label="source"); ax.plot(*city, "ks", label="city")
ax.set_xlabel("east (km)"); ax.set_ylabel("north (km)"); ax.legend()
ax.set_aspect("equal"); ax.set_title("Dust plume after 4 days")
plt.tight_layout()
plt.savefig("ch06_dust_plume.pdf", bbox_inches="tight")
