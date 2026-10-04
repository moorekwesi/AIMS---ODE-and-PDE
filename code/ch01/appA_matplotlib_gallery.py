# A Matplotlib gallery for differential equations
# Applied ODE & PDE with Python, App. A | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 101)
t = np.linspace(0, 0.2, 81)
X, T = np.meshgrid(x, t)                       # rows: time, columns: space
U = np.exp(-np.pi**2 * T) * np.sin(np.pi * X)  # solution of u_t = u_xx

fig = plt.figure(figsize=(10, 7.5))

ax1 = fig.add_subplot(2, 2, 1)                 # 1. line plots of snapshots
for n in (0, 20, 40, 80):
    ax1.plot(x, U[n], label=f"t = {t[n]:.2f}")
ax1.set_xlabel("x"); ax1.set_ylabel("u"); ax1.legend(); ax1.set_title("plot")

ax2 = fig.add_subplot(2, 2, 2)                 # 2. filled contours in the (x,t) plane
cs = ax2.contourf(X, T, U, levels=20, cmap="viridis")
ax2.contour(X, T, U, levels=[0.25, 0.5, 0.75], colors="w", linewidths=0.8)
fig.colorbar(cs, ax=ax2)
ax2.set_xlabel("x"); ax2.set_ylabel("t"); ax2.set_title("contourf")

ax3 = fig.add_subplot(2, 2, 3, projection="3d")   # 3. surface plot
ax3.plot_surface(X, T, U, cmap="viridis", linewidth=0, rstride=4, cstride=4)
ax3.set_xlabel("x"); ax3.set_ylabel("t"); ax3.set_zlabel("u")
ax3.set_title("plot_surface")

ax4 = fig.add_subplot(2, 2, 4)                 # 4. vector field of a pendulum
th, om = np.meshgrid(np.linspace(-2*np.pi, 2*np.pi, 41), np.linspace(-3, 3, 31))
ax4.streamplot(th, om, om, -np.sin(th), density=1.2, linewidth=0.7, color="gray")
ax4.quiver(th[::3, ::3], om[::3, ::3], om[::3, ::3], -np.sin(th[::3, ::3]),
           color="tab:blue", width=0.004)
ax4.set_xlabel("theta"); ax4.set_ylabel("omega"); ax4.set_title("streamplot + quiver")

plt.tight_layout()
plt.savefig("ch01_appA_gallery.pdf", bbox_inches="tight")
plt.close()
