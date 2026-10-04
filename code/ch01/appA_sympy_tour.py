# A short tour of SymPy for differential equations
# Applied ODE & PDE with Python, App. A | (c) 2026 Stephen E. Moore | MIT Licence
import numpy as np
import sympy as sp

x, t = sp.symbols("x t", real=True)
k, w = sp.symbols("k omega", positive=True)
y = sp.Function("y")

expr = sp.exp(-x**2) * sp.sin(k*x)
print("diff      :", sp.diff(expr, x))
print("integrate :", sp.integrate(sp.exp(-k*x), (x, 0, sp.oo)))
print("series    :", sp.series(sp.cos(x), x, 0, 7))
print("limit     :", sp.limit(sp.sin(x)/x, x, 0))

# Ordinary differential equations
print("dsolve 1  :", sp.dsolve(y(t).diff(t, 2) + w**2*y(t), y(t)))
ivp = sp.dsolve(y(t).diff(t) - y(t)*(1 - y(t)), y(t), ics={y(0): sp.Rational(1, 2)})
print("dsolve 2  :", sp.simplify(ivp.rhs))

# A first-order linear PDE with constant coefficients:  u_x + 2 u_t = 0
u = sp.Function("u")
print("pdsolve   :", sp.pdsolve(u(x, t).diff(x) + 2*u(x, t).diff(t)))

# From symbols to numbers and to LaTeX
f = sp.lambdify((x, k), expr, "numpy")
print("lambdify  :", np.round(f(np.array([0.0, 0.5, 1.0]), 2.0), 6))
print("latex     :", sp.latex(sp.Integral(expr, (x, 0, 1))))
