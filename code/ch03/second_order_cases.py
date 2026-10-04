# The three cases of a y'' + b y' + c y = 0: distinct, repeated and complex roots
# Applied ODE & PDE with Python, Ch. 3 | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

t, r = sp.symbols("t r")
y = sp.Function("y")
cases = [(1, 3, 2, "distinct real"), (1, 4, 4, "repeated"), (1, 2, 5, "complex")]

fig, ax = plt.subplots(figsize=(7, 3.8))
for a, b, c, name in cases:
    roots = sp.roots(a * r**2 + b * r + c, r)            # {root: multiplicity}
    ode = a * y(t).diff(t, 2) + b * y(t).diff(t) + c * y(t)
    sol = sp.dsolve(ode, y(t), ics={y(0): 1, y(t).diff(t).subs(t, 0): 0})
    print(f"{name:13s} roots {roots}:  y = {sol.rhs}")
    f = sp.lambdify(t, sol.rhs, "numpy")
    tt = np.linspace(0, 6, 400)
    ax.plot(tt, f(tt), label=f"{name}: y''+{b}y'+{c}y=0")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("t")
ax.set_ylabel("y(t)")
ax.set_title("y(0) = 1, y'(0) = 0")
ax.legend()
plt.tight_layout()
plt.savefig("ch03_second_order_cases.pdf", bbox_inches="tight")
plt.close()
