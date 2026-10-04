# Snapshots and space-time plot of transport
# Applied ODE & PDE with Python, Ch. 6 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

c = 1.5                                         # wave speed


def f(x):
    """Initial profile: a smooth bump plus a small plateau."""
    return np.exp(-8*(x + 3)**2) + 0.5*((x > -1.5) & (x < -0.5))


def u(x, t):
    """Exact solution u(x,t) = f(x - c t)."""
    return f(x - c*t)


x = np.linspace(-5, 6, 1101)
print("  t    position of max   predicted -3 + c t")
for t in [0.0, 1.0, 2.0, 3.0]:
    print(f"{t:4.1f}   {x[np.argmax(u(x, t))]:10.3f}     {-3 + c*t:10.3f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
for t in [0, 1, 2, 3]:
    a1.plot(x, u(x, t), label=f"t = {t}")
a1.set_xlabel("x"); a1.set_ylabel("u"); a1.legend(); a1.set_title("Snapshots")
T = np.linspace(0, 3, 301)
XX, TT = np.meshgrid(x, T)
a2.contourf(XX, TT, u(XX, TT), levels=20, cmap="viridis")
for x0 in np.arange(-5, 6, 1.0):                # characteristics x = x0 + c t
    a2.plot(x0 + c*T, T, "w-", lw=0.6)
a2.set_xlim(-5, 6); a2.set_xlabel("x"); a2.set_ylabel("t")
a2.set_title("Space-time plot with characteristics")
plt.tight_layout()
plt.savefig("ch06_transport.pdf", bbox_inches="tight")
