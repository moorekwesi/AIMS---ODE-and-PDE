# Solving a quasilinear Cauchy problem by integrating the characteristic system
# Applied ODE & PDE with Python, Ch. 7 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# PDE:  u u_x + y u_y = x,   data u(x0, 1) = 2 x0 on the line y = 1


def char_rhs(s, z):
    """Characteristic system dx/ds = a, dy/ds = b, du/ds = c with z = (x, y, u)."""
    x, y, u = z
    return [u, y, x]


def exact(x, y):
    """Solution found by hand in the text (valid for y > 1/sqrt(3))."""
    return x * (3 * y**2 + 1) / (3 * y**2 - 1)


fig, ax = plt.subplots(figsize=(7, 4))
print("   x0      s       x(s)       y(s)       u(s)     exact u   |error|")
for x0 in np.linspace(-1.0, 1.0, 9):
    # follow the characteristic both forwards (s > 0) and backwards (s < 0)
    for s_end in (1.0, -0.5):
        sol = solve_ivp(char_rhs, (0.0, s_end), [x0, 1.0, 2 * x0],
                        rtol=1e-10, atol=1e-12, dense_output=True)
        s = np.linspace(0.0, s_end, 100)
        x, y, u = sol.sol(s)
        ax.plot(x, y, "b-", lw=1)
    if x0 in (-1.0, 0.5, 1.0):          # print a few end points (s = 1)
        x, y, u = solve_ivp(char_rhs, (0, 1), [x0, 1.0, 2 * x0],
                            rtol=1e-10, atol=1e-12).y[:, -1]
        print(f"{x0:6.2f}  {1.0:5.2f}  {x:9.5f}  {y:9.5f}  {u:9.5f}  "
              f"{exact(x, y):9.5f}  {abs(u - exact(x, y)):.1e}")

ax.axhline(1.0, color="r", lw=2, label="initial curve y = 1")
ax.axhline(1 / np.sqrt(3), color="k", ls="--", label="y = 1/sqrt(3): blow-up")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Projected characteristics of u u_x + y u_y = x")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("ch07_quasilinear_chars.pdf", bbox_inches="tight")
