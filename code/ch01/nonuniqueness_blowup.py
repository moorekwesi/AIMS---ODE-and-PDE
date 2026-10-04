# Non-uniqueness and blow-up of solutions
# Applied ODE & PDE with Python, Ch. 1 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))

# (a) y' = 3 y^(2/3), y(0) = 0: y = 0 and y = (t - a)^3 for t > a, any a >= 0
t = np.linspace(-0.5, 2, 400)
ax[0].plot(t, 0*t, "k", lw=2, label="y = 0")
for a in (0.0, 0.5, 1.0):
    ax[0].plot(t, np.where(t > a, (t - a)**3, 0.0), label=f"a = {a}")
ax[0].set_xlabel("t"); ax[0].set_ylabel("y"); ax[0].set_ylim(-0.2, 3)
ax[0].set_title("y' = 3 y^(2/3),  y(0) = 0"); ax[0].legend(); ax[0].grid(alpha=0.3)

# (b) y' = y^2, y(0) = y0 > 0: y = y0 / (1 - y0 t) blows up at t* = 1/y0
for y0 in (0.5, 1.0, 2.0):
    ts = 1.0 / y0
    tt = np.linspace(0, 0.98 * ts, 300)
    ax[1].plot(tt, y0 / (1 - y0 * tt), label=f"y0 = {y0}, t* = {ts:.2f}")
    ax[1].axvline(ts, ls=":", color="gray")
ax[1].set_ylim(0, 20); ax[1].set_xlabel("t"); ax[1].set_title("y' = y^2")
ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ch01_nonuniq_blowup.pdf", bbox_inches="tight")
plt.close()

# What does a numerical solver do with the blow-up?  (exact t* = 1)
sol = solve_ivp(lambda t, y: y**2, (0, 2), [1.0], rtol=1e-8, atol=1e-10)
print("solver status :", sol.status)
print("message       :", sol.message)
print(f"last time     : t = {sol.t[-1]:.10f}")
print(f"last value    : y = {sol.y[0, -1]:.3e}")

# The numerical solver also misses the non-unique solutions of y' = 3 y^(2/3)
sol2 = solve_ivp(lambda t, y: 3*np.cbrt(y)**2, (0, 2), [0.0])
print(f"y' = 3 y^(2/3), y(0) = 0: solver returns y(2) = {sol2.y[0, -1]}")
