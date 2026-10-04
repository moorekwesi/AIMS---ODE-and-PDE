# An integral surface woven from characteristic curves
# Applied ODE & PDE with Python, Ch. 6 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def f(theta):
    """Cauchy data u = f(theta) prescribed on the unit circle."""
    return 1 + 0.3*np.cos(3*theta)


def char_system(s, z):
    """dx/ds = a = x, dy/ds = b = y, du/ds = c = u."""
    return z                                  # (x, y, u)' = (x, y, u)


fig = plt.figure(figsize=(7, 5.5))
ax = fig.add_subplot(projection="3d")
rho, th = np.meshgrid(np.linspace(0.3, 1.8, 30), np.linspace(0, 2*np.pi, 73))
ax.plot_surface(rho*np.cos(th), rho*np.sin(th), rho*f(th), alpha=0.35,
                color="tab:blue", linewidth=0)
worst = 0.0
for r0 in np.linspace(0, 2*np.pi, 12, endpoint=False):
    z0 = [np.cos(r0), np.sin(r0), f(r0)]      # a point of the initial curve
    for s_end in (np.log(0.3), np.log(1.8)):  # integrate backward and forward
        sol = solve_ivp(char_system, [0, s_end], z0, dense_output=True,
                        rtol=1e-10, atol=1e-12)
        X, Y, U = sol.sol(np.linspace(0, s_end, 40))
        worst = max(worst, np.max(np.abs(U - np.hypot(X, Y)*f(r0))))
        ax.plot(X, Y, U, "r", lw=1.5)
t = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(t), np.sin(t), f(t), "k", lw=3, label="initial curve")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("u")
ax.legend(loc="upper left"); ax.view_init(elev=28, azim=-60)
plt.tight_layout()
plt.savefig("ch06_integral_surface.pdf", bbox_inches="tight")
print(f"max |u_numerical - rho f(theta)| along 12 characteristics: {worst:.2e}")
