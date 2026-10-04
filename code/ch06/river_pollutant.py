# A pollutant spill in a river
# Applied ODE & PDE with Python, Ch. 6 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

v = 0.5*86.4            # river speed: 0.5 m/s = 43.2 km/day
k = 0.2                 # decay rate (1/day)
tau = 2/24              # the spill lasts 2 hours (in days)
M = 10.0                # peak concentration at the outfall (mg/L)


def g(t):
    """Concentration released at x = 0: a smooth 2-hour pulse."""
    return np.where((t > 0) & (t < tau), M*np.sin(np.pi*t/tau)**2, 0.0)


def u(x, t):
    """Solution of u_t + v u_x = -k u, u(x,0) = 0, u(0,t) = g(t)."""
    return g(t - x/v)*np.exp(-k*x/v)


t = np.linspace(0, 3, 30001)
print(" x (km)   arrival of peak (h)   peak (mg/L)   predicted peak")
fig, ax = plt.subplots(figsize=(7, 3.8))
for x in [0, 20, 40, 60, 100]:
    c = u(x, t)
    i = np.argmax(c)
    print(f"{x:6d}   {24*t[i]:14.2f}        {c[i]:8.4f}      "
          f"{M*np.exp(-k*x/v):8.4f}")
    ax.plot(24*t, c, label=f"x = {x} km")
ax.set_xlim(0, 60); ax.set_xlabel("time (hours)")
ax.set_ylabel("concentration (mg/L)"); ax.legend()
plt.tight_layout()
plt.savefig("ch06_river_pollutant.pdf", bbox_inches="tight")
