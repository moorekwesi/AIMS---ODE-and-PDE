# Characteristics, breaking time and profiles of the inviscid Burgers equation
# Applied ODE & PDE with Python, Ch. 7 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar


def u0(x):
    """Initial profile: a Gaussian hump."""
    return np.exp(-x**2)


def du0(x):
    return -2 * x * np.exp(-x**2)


# breaking time t_b = -1 / min u0'(x) and the foot of the first breaking characteristic
res = minimize_scalar(du0, bounds=(0, 3), method="bounded", options={"xatol": 1e-12})
xi_b = res.x
t_b = -1.0 / res.fun
x_b = xi_b + u0(xi_b) * t_b
print(f"min u0' = {res.fun:.6f} at xi = {xi_b:.6f}")
print(f"breaking time t_b = {t_b:.6f},  breaking point x_b = {x_b:.6f}")


def burgers_solution(x, t):
    """Solve u = u0(x - u t) pointwise with brentq (valid for t < t_b)."""
    if t == 0:
        return u0(x)
    # root of F(u) = u - u0(x - u t); u lies between 0 and 1 for this u0
    return np.array([brentq(lambda w: w - u0(xx - w * t), -1e-12, 1 + 1e-12)
                     for xx in x])


x = np.linspace(-3, 4, 1401)
xis = np.linspace(-3, 3, 600001)     # fine grid of characteristic feet
print("     t   x_steep   -min u_x    1/(t_b - t)")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for t in [0.0, 0.5, 0.9, 1.1, 1.15]:
    u = burgers_solution(x, t)
    # exact slope along the characteristic from xi: u_x = u0'(xi) / (1 + u0'(xi) t)
    ux = du0(xis) / (1 + du0(xis) * t)
    k = np.argmin(ux)                    # steepest negative slope
    x_steep = xis[k] + u0(xis[k]) * t
    print(f"{t:6.2f}  {x_steep:8.4f}  {abs(ux[k]):11.4f}  {1/(t_b - t):11.4f}")
    ax2.plot(x, u, label=f"t = {t}")
# characteristic diagram: x = xi + u0(xi) t
for xi in np.linspace(-3, 3, 41):
    tt = np.array([0, 2.0])
    ax1.plot(xi + u0(xi) * tt, tt, "b-", lw=0.7)
ax1.plot(x_b, t_b, "ro", label="first breaking point")
ax1.set_xlim(-3, 4); ax1.set_ylim(0, 2)
ax1.set_xlabel("x"); ax1.set_ylabel("t"); ax1.legend(loc="upper left")
ax1.set_title("Characteristics x = xi + u0(xi) t")
ax2.set_xlabel("x"); ax2.set_ylabel("u"); ax2.legend()
ax2.set_title("Solution profiles before breaking")
plt.tight_layout()
plt.savefig("ch07_burgers_breaking.pdf", bbox_inches="tight")
