# Projectile with quadratic air drag: events and dense output in solve_ivp
# Applied ODE & PDE with Python, Ch. 4 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

g = 9.81                                   # m/s^2
rho, Cd, d, m = 1.2, 0.25, 0.22, 0.43      # air density, drag coeff., ball diameter, mass
k = 0.5 * rho * Cd * np.pi * (d / 2)**2 / m   # drag constant (1/m)


def rhs(t, z):
    """z = (x, y, vx, vy); drag force opposite to the velocity, size k |v|^2."""
    x, y, vx, vy = z
    speed = np.hypot(vx, vy)
    return [vx, vy, -k * speed * vx, -g - k * speed * vy]


def hit_ground(t, z):
    return z[1]                  # y = 0
hit_ground.terminal = True       # stop the integration
hit_ground.direction = -1        # only when y is decreasing


def apex(t, z):
    return z[3]                  # vy = 0 at the highest point


def kick(v0, angle_deg):
    a = np.radians(angle_deg)
    z0 = [0.0, 0.0, v0 * np.cos(a), v0 * np.sin(a)]
    return solve_ivp(rhs, (0, 20), z0, events=[hit_ground, apex],
                     dense_output=True, rtol=1e-9, atol=1e-9)


v0, ang = 25.0, 35.0
sol = kick(v0, ang)
tf, xf = sol.t_events[0][0], sol.y_events[0][0][0]
ta, ya = sol.t_events[1][0], sol.y_events[1][0][1]
print(f"drag constant k = {k:.5f} 1/m")
print(f"with drag : flight time {tf:.3f} s, range {xf:.2f} m, apex {ya:.2f} m at t = {ta:.3f} s")
R0 = v0**2 * np.sin(2 * np.radians(ang)) / g
print(f"no drag   : flight time {2 * v0 * np.sin(np.radians(ang)) / g:.3f} s, range {R0:.2f} m")
angles = np.arange(20.0, 55.0, 1.0)
ranges = [kick(v0, a).y_events[0][0][0] for a in angles]
i = int(np.argmax(ranges))
print(f"best launch angle with drag (1 degree grid): {angles[i]:.0f} deg, range {ranges[i]:.2f} m")

tt = np.linspace(0, tf, 300)
X = sol.sol(tt)
plt.figure(figsize=(7, 3.5))
plt.plot(X[0], X[1], label="with drag")
x0 = v0 * np.cos(np.radians(ang)) * np.linspace(0, 2 * v0 * np.sin(np.radians(ang)) / g, 300)
plt.plot(x0, np.tan(np.radians(ang)) * x0 - g * x0**2 / (2 * (v0 * np.cos(np.radians(ang)))**2),
         "--", label="no drag")
plt.plot(xf, 0, "ko")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.tight_layout()
plt.savefig("ch04_projectile.pdf", bbox_inches="tight")
plt.close()
