# Phase line of an autonomous equation with an Allee effect
# Applied ODE & PDE with Python, Ch. 1 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

a = 0.3                                    # Allee threshold
Y = sp.symbols("y")
fsym = Y * (1 - Y) * (Y - a)              # y' = f(y)
equilibria = sorted(sp.solve(fsym, Y))
fprime = sp.diff(fsym, Y)
for ys in equilibria:
    slope = float(fprime.subs(Y, ys))
    kind = "stable (sink)" if slope < 0 else "unstable (source)"
    print(f"y* = {float(ys):.2f}:  f'(y*) = {slope:+.3f}  ->  {kind}")

f = sp.lambdify(Y, fsym, "numpy")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
yy = np.linspace(-0.2, 1.2, 300)
ax[0].plot(yy, f(yy)); ax[0].axhline(0, color="k", lw=0.8)
for ys in equilibria:
    filled = float(fprime.subs(Y, ys)) < 0
    ax[0].plot(float(ys), 0, "o", ms=8, color="k", mfc="k" if filled else "w")
for y0 in (-0.1, 0.15, 0.65, 1.1):         # arrows show the direction of motion
    ax[0].annotate("", xy=(y0 + 0.1*np.sign(f(y0)), 0), xytext=(y0, 0),
                   arrowprops=dict(arrowstyle="->", lw=2, color="tab:red"))
ax[0].set_xlabel("y"); ax[0].set_ylabel("f(y)"); ax[0].set_title("phase line")

t = np.linspace(0, 30, 300)
for y0 in (0.05, 0.2, 0.28, 0.32, 0.4, 0.7, 1.2):
    sol = solve_ivp(lambda t, y: f(y), (0, 30), [y0], t_eval=t)
    ax[1].plot(t, sol.y[0])
for ys in equilibria:
    ax[1].axhline(float(ys), ls=":", color="gray")
ax[1].set_xlabel("t"); ax[1].set_ylabel("y(t)"); ax[1].set_title("solutions")
plt.tight_layout()
plt.savefig("ch01_phase_line.pdf", bbox_inches="tight")
plt.close()
