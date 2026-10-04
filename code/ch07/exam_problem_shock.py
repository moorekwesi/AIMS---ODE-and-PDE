# Shock path for Burgers' equation with piecewise linear initial data
# Applied ODE & PDE with Python, Ch. 7 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# u_t + u u_x = 0,  u0 = 0 (x<0), x (0<=x<=1/2), 1/2 (1/2<=x<=1), 0 (x>1)


def s_exact(t):
    """Shock position derived in the text."""
    t = np.asarray(t, dtype=float)
    return np.where(t <= 2, 1 + t / 4, np.sqrt(3 * (1 + t)) / 2)


def u_left(x, t):
    """Smooth solution to the left of the shock (from the characteristics)."""
    return np.where(x < 0, 0.0, np.where(x <= (1 + t) / 2, x / (1 + t), 0.5))


def shock_speed(t, s):
    """Rankine-Hugoniot: s' = (u_minus + u_plus)/2 with u_plus = 0."""
    return 0.5 * (u_left(s, t) + 0.0)


sol = solve_ivp(shock_speed, (0, 10), [1.0], max_step=0.01, rtol=1e-10, atol=1e-12,
                dense_output=True)
print("    t    s numerical   s exact      u_minus   mass")
for t in [0.0, 1.0, 2.0, 4.0, 6.0, 10.0]:
    s = sol.sol(t)[0]
    x = np.linspace(0, s, 200001)
    mass = np.trapz(u_left(x, t), x)               # should stay 3/8 = 0.375
    print(f"{t:5.1f}  {s:11.6f}  {float(s_exact(t)):10.6f}  {float(u_left(s, t)):9.5f}"
          f"  {mass:.6f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
t = np.linspace(0, 6, 601)
S = s_exact(t)
for x0 in np.linspace(-0.5, 2.5, 31):
    if 0 <= x0 <= 0.5:
        xc = x0 * (1 + t)
    elif 0.5 < x0 <= 1:
        xc = x0 + t / 2
    else:
        xc = x0 + 0 * t
    keep = xc <= S if x0 <= 1 else xc >= S      # characteristics end on the shock
    ax1.plot(xc[keep], t[keep], "b-", lw=0.7)
ax1.plot(S, t, "r-", lw=2.5, label="shock x = s(t)")
ax1.plot(1.5, 2.0, "ko", label="(3/2, 2): shock reaches the fan")
ax1.set_xlim(-0.5, 2.5); ax1.set_ylim(0, 6)
ax1.set_xlabel("x"); ax1.set_ylabel("t"); ax1.legend(loc="upper left")
x = np.linspace(-0.5, 3, 2001)
for tt in [0, 1, 2, 4, 8]:
    ax2.plot(x, np.where(x < s_exact(tt), u_left(x, tt), 0.0), label=f"t = {tt}")
ax2.set_xlabel("x"); ax2.set_ylabel("u"); ax2.legend()
plt.tight_layout()
plt.savefig("ch07_exam_shock.pdf", bbox_inches="tight")
