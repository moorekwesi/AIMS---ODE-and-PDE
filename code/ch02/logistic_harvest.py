# Phase line and solutions of the logistic equation with constant harvesting
# Applied ODE & PDE with Python, Ch. 2 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

r, K, H = 0.5, 1000.0, 100.0      # growth rate (1/yr), capacity (t), catch (t/yr)


def f(y):
    return r * y * (1 - y / K) - H


def fprime(y):
    return r * (1 - 2 * y / K)


# equilibria: roots of -(r/K) y^2 + r y - H = 0
eq = np.sort(np.roots([-r / K, r, -H]).real)
for e in eq:
    kind = "stable" if fprime(e) < 0 else "unstable"
    print(f"equilibrium y* = {e:8.3f} tonnes, f'(y*) = {fprime(e):+.4f} -> {kind}")
print(f"maximum sustainable yield rK/4 = {r * K / 4:.1f} tonnes/year")


def extinct(t, y):              # event: the stock reaches zero
    return y[0]


extinct.terminal = True
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
yy = np.linspace(0, K, 400)
ax1.plot(yy, f(yy), "C0")
ax1.axhline(0, color="k", lw=0.8)
ax1.plot(eq, [0, 0], "o", mfc="w", mec="k")
ax1.plot(eq[1], 0, "ko")                       # filled dot = stable
for a, b in ((0, eq[0]), (eq[0], eq[1]), (eq[1], K)):
    m = 0.5 * (a + b)
    d = np.sign(f(m)) * 0.25 * (b - a)
    ax1.annotate("", xy=(m + d, 0), xytext=(m - d, 0),
                 arrowprops=dict(arrowstyle="->", color="C3", lw=2))
ax1.set_xlabel("stock y (tonnes)")
ax1.set_ylabel("f(y) = dy/dt")
ax1.set_title("phase line")

for y0 in (100, 250, 300, 500, 900, 1000):
    s = solve_ivp(lambda t, y: f(y), (0, 30), [y0], events=extinct,
                  max_step=0.1)
    ax2.plot(s.t, s.y[0])
    if s.t_events[0].size:
        print(f"y0 = {y0:4d}: stock collapses at t = {s.t_events[0][0]:.3f} years")
for e in eq:
    ax2.axhline(e, color="0.5", ls="--", lw=0.8)
ax2.set_xlabel("t (years)")
ax2.set_ylabel("y(t)")
ax2.set_title("solutions, H = 100")
plt.tight_layout()
plt.savefig("ch02_logistic_harvest.pdf", bbox_inches="tight")
plt.close()
